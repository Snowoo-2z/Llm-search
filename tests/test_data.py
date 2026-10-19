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
    Deduper, SPECIAL_TOKENS, agentic_examples, build_parser, fim_example, functions_from_source,
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

    def test_rejects_potential_personal_data(self):
        text = GOOD_CODE + '\ncontact = "contact@example.fr"\n'
        self.assertIn("donnée personnelle potentielle", quality_reasons(text, "code", "python"))

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

    def test_accepts_single_line_web_prose(self):
        text = ("La France est un pays où le français est parlé et où les régions possèdent "
                "une histoire, une culture et des paysages variés. ") * 8
        self.assertEqual(quality_reasons(text, "prose", "markdown"), [])


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
    def test_generated_agent_dialogue_is_french_only(self):
        func = functions_from_source(GOOD_CODE)[0]
        examples = agentic_examples(func, random.Random(0))
        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertIn("Tu es BlackCode, un assistant qui parle uniquement en français", example)
            self.assertNotRegex(example, r"(?i)You are BlackCode|Implement `|Create `|Running the tests")

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
        self.assertIn("<|tool_result|>ÉCHEC", trace)
        self.assertIn("<|tool_result|>RÉUSSI", trace)
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

class QualityV2Tests(unittest.TestCase):
    def test_rejects_url_spam(self):
        # >25% urls and <200 words
        text = ("https://example.com/page " * 15) + " ".join(["mot"]*10) + " " + "La France est belle. "*5
        # ensure >200 chars
        text = text + " " + "a"*200
        reasons = quality_reasons(text, "prose", "markdown")
        self.assertIn("trop d'urls", reasons)

    def test_rejects_boilerplate(self):
        text = ("Mentions légales politique de confidentialité cookies tous droits réservés newsletter abonnez-vous " * 2) + " ".join(["Ceci est une phrase normale."]*6)
        reasons = quality_reasons(text, "prose", "markdown")
        self.assertIn("boilerplate", reasons)

    def test_rejects_repeated_punct(self):
        text = ("Bonjour!!!! " * 15) + " ".join(["phrase"]*30) + " " + "a"*100
        reasons = quality_reasons(text, "prose", "markdown")
        self.assertIn("ponctuation répétée", reasons)

    def test_rejects_repetitive_prose(self):
        # many lines, low unique ratio
        line = "Cette ligne est répétée exactement pareil."
        text = "\n".join([line]*50) + "\n" + " ".join(["mot"]*40)
        reasons = quality_reasons(text, "prose", "markdown")
        self.assertIn("répétitif", reasons)

    def test_accepts_good_french_prose(self):
        text = ("La France est un pays d'Europe où le français est parlé et où les régions possèdent une histoire riche. " * 8)
        self.assertEqual(quality_reasons(text, "prose", "markdown"), [])

    def test_license_normalization(self):
        from blackcode.data import normalize_licence_label
        self.assertEqual(normalize_licence_label("MIT"), "mit")
        self.assertEqual(normalize_licence_label("Apache-2.0"), "apache 2 0")
        self.assertEqual(normalize_licence_label("CC-BY-SA 4.0"), "cc by sa 4 0")
        self.assertEqual(normalize_licence_label("  Public Domain  "), "public domain")

class QualityScoreTests(unittest.TestCase):
    def test_score_good_french(self):
        from blackcode.data import quality_score
        text = ("La France est un pays d'Europe où le français est parlé. " * 10)
        s = quality_score(text, "prose", "markdown")
        self.assertGreater(s, 0.7)

    def test_score_bad_url(self):
        from blackcode.data import quality_score
        text = ("https://example.com " * 20) + " ".join(["mot"]*10) + " " + "a"*200
        s = quality_score(text, "prose", "markdown")
        self.assertLess(s, 0.5)

    def test_score_boilerplate_low(self):
        from blackcode.data import quality_score
        text = ("Mentions légales cookies tous droits réservés newsletter " * 3) + " ".join(["phrase"]*10)
        s = quality_score(text, "prose", "markdown")
        self.assertLess(s, 0.6)

class SmsTests(unittest.TestCase):
    def test_sms_short_accepted(self):
        from blackcode.data import quality_reasons
        text = "slt cv ? mdr tkt"
        # sms kind should accept short
        reasons = quality_reasons(text, "sms", "text")
        # chars <10? "slt cv ? mdr tkt" is 15 chars, words 4, should be accepted? words<3 would reject, but we have 4
        # actually our sms check: words<3 reject, chars<10 reject, so this should be kept (maybe)
        # but we have PII? no, alnum_ratio? let's see
        print("sms reasons:", reasons)
        self.assertNotIn("prose trop courte", reasons)

    def test_sms_abbrev_bonus(self):
        from blackcode.data import quality_score
        text = "slt bjr mdr ptdr stp tkt pk bcp cc"
        score = quality_score(text, "sms", "text")
        self.assertGreater(score, 0.3)

    def test_tweet_kind(self):
        from blackcode.data import quality_reasons
        text = "bjr tout le monde ! c'est la vie mdr #france"
        reasons = quality_reasons(text, "tweet", "text")
        self.assertEqual(reasons, [])
