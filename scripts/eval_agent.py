#!/usr/bin/env python3
"""Évalue le choix d'outils et des tâches agentiques françaises de base."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackcode.agent import (  # noqa: E402
    DEFAULT_MODEL,
    FrenchAgent,
    NoteStore,
    OpenAICompatibleClient,
    ProjectReader,
    ToolContext,
)


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSON invalide à {path}:{line_no}: {exc}") from exc
        for key in ("id", "prompt", "expected_tool_sequence"):
            if key not in task:
                raise SystemExit(f"Champ {key!r} manquant à {path}:{line_no}")
        tasks.append(task)
    return tasks


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=repo / "benchmarks/agent_fr.jsonl")
    parser.add_argument("--base-url", default=os.getenv("BLACKCODE_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("BLACKCODE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("BLACKCODE_API_KEY", "EMPTY"))
    parser.add_argument("--project-root", type=Path, default=repo)
    parser.add_argument("--limit", type=int, default=0, help="0 = toutes les tâches")
    parser.add_argument(
        "--enable-tests", action="store_true",
        help="active le test runner (exécute du code; réservé aux projets de confiance)",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.dataset)
    if args.limit:
        tasks = tasks[:args.limit]
    client = OpenAICompatibleClient(args.base_url, args.model, args.api_key)
    passed = 0
    with tempfile.TemporaryDirectory(prefix="blackcode-agent-eval-") as notes_dir:
        agent = FrenchAgent(client, ToolContext(
            NoteStore(notes_dir), ProjectReader(args.project_root),
            allow_test_execution=args.enable_tests,
        ))
        for task in tasks:
            failures: list[str] = []
            try:
                result = agent.ask(task["prompt"])
            except Exception as exc:
                failures.append(f"erreur: {exc}")
                result = None
            if result is not None:
                actual = [entry["name"] for entry in result.tool_trace]
                expected = task["expected_tool_sequence"]
                if actual != expected:
                    failures.append(f"outils attendus {expected}, reçus {actual}")
                final_fragment = task.get("final_must_contain")
                if final_fragment and final_fragment.casefold() not in result.text.casefold():
                    failures.append(f"la réponse finale ne contient pas {final_fragment!r}")
                tool_fragment = task.get("tool_result_must_contain")
                if tool_fragment and not any(
                    tool_fragment.casefold() in str(entry["result"]).casefold()
                    for entry in result.tool_trace
                ):
                    failures.append(f"aucun résultat d'outil ne contient {tool_fragment!r}")
            ok = not failures
            passed += int(ok)
            status = "OK" if ok else "ÉCHEC"
            print(f"{status:6} {task['id']}: " + ("; ".join(failures) if failures else "séquence vérifiée"))
            agent.reset()
    print(f"\nScore choix d'outils : {passed}/{len(tasks)} ({100 * passed / max(1, len(tasks)):.1f} %)")
    print("Ce score ne mesure pas la qualité générale des réponses; les tâches sont un petit jeu de départ.")
    return 0 if passed == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
