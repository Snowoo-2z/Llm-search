"""Tests du pipeline de données v2 (qualité, dédup, FIM, agentique)."""
from __future__ import annotations

import ast
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from blackcode.data import (
    Deduper, SPECIAL_TOKENS, build_parser, fim_example, functions_from_source,
    hamming, mutate_function, mutated_failure_example, quality_reasons,
    simhash64, _sample_inputs, extract_imports,
)

GOOD_CODE = '''
def add(a, b):
    """Additionne deux nombres."""
    result = a + b
    return result


def clamp(value, low=0, high=10):
    """Ramène value dans [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value
'''


class QualityTests(unittest.TestCase):
    def test_accepts_normal_code(self):
        self.assertEqual(quality_reasons(GOOD_CODE, "code", "python"), [])

    def test_rejects_generated_file(self):
        text = "# This file was automatically generated. DO NOT EDIT.\n" + GOOD_CODE
        self.assertIn("fichier généré", quality_reasons(text, "code", "python"))

    def test_rejects_secret(self):
        text = GOOD_CODE + '\nTOKEN = "ghp_' + "a" * 36 + '"\n'
        self.assertIn("secret", quality_reasons(text, "code", "python"))

    def test_rejects_minified(self):
        text = "var x=1;" * 200
        self.assertIn("minifié", quality_reasons(text, "code", "javascript"))

    def test_rejects_invalid_python(self):
        broken = "def f(:\n    return 1\n" + "\n".join(
            f"valeur_{i} = calcul_{i}({i})" for i in range(30)) + "\n"
        self.assertIn("python invalide", quality_reasons(broken, "code", "python"))

    def test_rejects_short_prose(self):
        # > 200 caractères mais < 30 mots : seule la règle « prose trop courte » s'applique.
        text = "L'anticonstitutionnellement " * 12
        self.assertIn("prose trop courte", quality_reasons(text, "prose", "markdown"))


class DedupTests(unittest.TestCase):
    def test_exact_duplicate(self):
        d = Deduper()
        self.assertIsNone(d.check(GOOD_CODE))
        self.assertEqual(d.check(GOOD_CODE), "exact")

    def test_near_duplicate_comments_only(self):
        """Même code, docstrings/commentaires différents → quasi-doublon."""
        d = Deduper()
        d.check(GOOD_CODE)
        variant = (GOOD_CODE
                   .replace('"""Additionne deux nombres."""', '"""Somme a et b."""')
                   .replace("# rien", ""))
        self.assertEqual(d.check(variant), "quasi")

    def test_near_duplicate_simhash(self):
        """Copie avec du code en plus : détectée par SimHash."""
        d = Deduper()
        base = "\n".join(f"valeur_{i} = fonction_{i}(parametre_{i})" for i in range(60))
        d.check(base)
        variant = base + "\n" + "\n".join(f"ajout_{i} = 1" for i in range(2))
        self.assertEqual(d.check(variant), "quasi")

    def test_unrelated_text_is_kept(self):
        d = Deduper()
        d.check(GOOD_CODE)
        other = "\n".join(f"ligne {i} : valeur_{i} = {i * 7}" for i in range(60))
        self.assertIsNone(d.check(other))

    def test_simhash_stable(self):
        self.assertEqual(simhash64(GOOD_CODE), simhash64(GOOD_CODE))
        self.assertLessEqual(hamming(simhash64(GOOD_CODE), simhash64(GOOD_CODE)), 0)


class FimTests(unittest.TestCase):
    def test_markers_and_content(self):
        rng = random.Random(0)
        text = "\n".join(f"ligne_{i} = {i}" for i in range(40))
        out = fim_example(text, rng)
        self.assertIsNotNone(out)
        for marker in ("<|fim_begin|>", "<|fim_hole|>", "<|fim_end|>"):
            self.assertEqual(out.count(marker), 1)
        for i in range(40):
            self.assertIn(f"ligne_{i} = {i}", out)

    def test_short_text_skipped(self):
        self.assertIsNone(fim_example("a = 1\nb = 2\n", random.Random(0)))

    def test_special_tokens_contain_fim(self):
        self.assertIn("<|fim_begin|>", SPECIAL_TOKENS)
        self.assertIn("<|tool|>", SPECIAL_TOKENS)


class AgenticTests(unittest.TestCase):
    def test_locate_functions(self):
        funcs = functions_from_source(GOOD_CODE)
        names = [f.name for f in funcs]
        self.assertEqual(names, ["add", "clamp"])
        self.assertIn("Additionne", funcs[0].docstring)
        self.assertEqual(funcs[1].signature, "(value, low=0, high=10)")

    def test_sample_inputs_uses_names_and_defaults(self):
        funcs = functions_from_source(GOOD_CODE)
        # `add(a, b)` : arguments non annotés mais noms conventionnels (a, b).
        self.assertIn([3, 3], _sample_inputs(funcs[0], random.Random(0)))
        self.assertTrue(_sample_inputs(funcs[1], random.Random(0)))
        unknown = "def f(quelquechose_obscur):\n    return quelquechose_obscur\n"
        func = functions_from_source(unknown)[0]
        self.assertEqual(_sample_inputs(func, random.Random(0)), [])

    def test_mutation_is_valid_python(self):
        funcs = functions_from_source(GOOD_CODE)
        mutated = mutate_function(funcs[1].body, random.Random(1))
        self.assertIsNotNone(mutated)
        orig, mutant, desc = mutated
        ast.parse(orig)
        ast.parse(mutant)
        self.assertNotEqual(orig, mutant)
        self.assertRegex(desc, r"[<>=!]=?\s*→\s*[<>=!]=?")

    def test_extract_imports(self):
        src = "import os\nfrom json import dumps\nx = 1\n\ndef f():\n    return os.sep\n"
        imports = extract_imports(src)
        self.assertIn("import os", imports)
        self.assertIn("from json import dumps", imports)
        self.assertNotIn("x = 1", imports)

    def test_real_failure_trace(self):
        """Une mutation `==` → `!=` doit produire un écart réellement exécuté."""
        source = (
            "def status(code=200):\n"
            "    \"\"\"Rend l'état HTTP pour un code.\"\"\"\n"
            "    if code == 200:\n"
            "        return 'ok'\n"
            "    return 'error'\n")
        func = functions_from_source(source)[0]
        trace = None
        for seed in range(20):
            trace = mutated_failure_example(func, random.Random(seed), timeout=5)
            if trace:
                break
        self.assertIsNotNone(trace, "aucune exécution réelle n'a produit d'écart")
        self.assertIn("<|tool_result|>FAILED", trace)
        self.assertIn("<|tool_result|>PASSED", trace)
        self.assertIn("!=", trace)

    def test_no_fabrication_without_difference(self):
        """Mutation équivalente sur ces entrées → aucun exemple, jamais inventé."""
        source = (
            "def double(value=2):\n"
            "    if value > 0:\n"
            "        return value * 2\n"
            "    return 0\n")
        func = functions_from_source(source)[0]
        for seed in range(10):
            self.assertIsNone(mutated_failure_example(func, random.Random(seed), timeout=5))


class EndToEndTests(unittest.TestCase):
    def test_build_small_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            (src / "pkg").mkdir(parents=True)
            for i in range(12):
                (src / f"mod_{i}.py").write_text(
                    GOOD_CODE + f"\n# module {i}\n" + "\n".join(
                        f"def gen_{i}_{k}(x={k}):\n    return x + {i}\n" for k in range(3))
                    + "\n")
            (src / "README.md").write_text(
                "# Projet\n\n" + " ".join(f"Ceci est la phrase {i}." for i in range(80)))
            out = Path(tmp) / "out"
            args = build_parser().parse_args([
                "build", "--src", str(src), "--out", str(out), "--vocab", "512",
                "--seq-len", "128", "--min-bytes", "100", "--agentic-frac", "0.2",
                "--max-spawns", "3",
            ])
            args.func(args)
            meta = json.loads((out / "meta.json").read_text())
            self.assertGreater(meta["train_tokens"], 100)
            self.assertGreater(meta["val_tokens"], 0)
            self.assertLessEqual(meta["vocab_size"], 512)
            self.assertGreater(meta["vocab_size"], 400)
            self.assertIn("agentic", meta["by_kind"])
            self.assertGreater((out / "train.bin").stat().st_size, 0)
            self.assertTrue((out / "tokenizer.json").exists())


if __name__ == "__main__":
    unittest.main()
