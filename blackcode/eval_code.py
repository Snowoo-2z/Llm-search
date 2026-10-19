"""Évaluation pass@k sur des problèmes de code avec tests unitaires.

Format JSONL (compatible HumanEval) :

    {"task_id": "HumanEval/0", "prompt": "def f(...):\\n",
     "test": "def check(candidate):\\n    assert ...", "entry_point": "f"}

Chaque échantillon est complété, concaténé au test, puis exécuté dans un
sous-processus isolé avec timeout. Les crashs, boucles infinies et sorties
intempestives ne tuent pas l'évaluation.

    python -m blackcode.eval_code --ckpt runs/t4_small/best.pt \
        --tokenizer data/code/tokenizer.json --problems data/humaneval.jsonl \
        --num-samples 8 --limit 20
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .corpus import load_tokenizer
from .generate import load_checkpoint

STOP_SEQUENCES = ["\n\ndef ", "\n\nclass ", "\n\nif __name__", "\n\n#", "\n\n@"]


def truncate_at_stop(text: str) -> str:
    cut = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def run_program(code: str, timeout: float) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout
        )
        ok = proc.returncode == 0
        err = (proc.stderr or "").strip().splitlines()
        return ok, (err[-1] if err and not ok else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def build_test_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    return (
        prompt
        + completion
        + "\n\n"
        + test
        + f"\n\ncheck({entry_point})\n"
    )


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimateur non biaisé de pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main() -> None:
    p = argparse.ArgumentParser(description="pass@k avec exécution de tests")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--problems", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(args.ckpt, device)
    tok = load_tokenizer(args.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")

    problems = []
    with Path(args.problems).open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if args.limit:
        problems = problems[: args.limit]

    results = []
    for i, prob in enumerate(problems):
        prompt = prob["prompt"]
        tests_passed = 0
        syntax_ok = 0
        errors = []
        for s in range(args.num_samples):
            torch.manual_seed(1000 * i + s)
            ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids],
                               dtype=torch.long, device=device)
            out = model.generate(
                ids, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                eos_id=eos_id,
            )
            text = tok.decode(out[0].tolist(), skip_special_tokens=False)
            completion = truncate_at_stop(text[len(prompt):])
            program = build_test_program(prompt, completion, prob["test"],
                                         prob.get("entry_point", "solution"))
            try:
                compile(program, "<sample>", "exec")
                syntax_ok += 1
            except SyntaxError as exc:
                errors.append(f"syntax: {exc.msg}")
                continue
            ok, err = run_program(program, args.timeout)
            tests_passed += int(ok)
            if not ok:
                errors.append(err)
        results.append({
            "task_id": prob.get("task_id", str(i)),
            "passed": tests_passed,
            "n": args.num_samples,
            "syntax_ok": syntax_ok,
            "errors": errors[:3],
        })
        print(f"[{i+1}/{len(problems)}] {results[-1]['task_id']}: "
              f"{tests_passed}/{args.num_samples} passent, syntaxe {syntax_ok}/{args.num_samples}")

    n = args.num_samples
    pass1 = sum(pass_at_k(r["n"], r["passed"], 1) for r in results) / max(len(results), 1)
    passk = sum(pass_at_k(r["n"], r["passed"], min(n, 10)) for r in results) / max(len(results), 1)
    syntax = sum(r["syntax_ok"] for r in results) / max(len(results) * n, 1)
    summary = {
        "model": args.ckpt,
        "problems": len(results),
        "num_samples": n,
        "pass@1": pass1,
        "pass@k": passk,
        "syntax_valid_rate": syntax,
        "details": results,
    }
    print(f"\npass@1 {pass1:.3f} | pass@{min(n,10)} {passk:.3f} | "
          f"syntaxe {syntax:.3f} | {len(results)} problèmes")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
