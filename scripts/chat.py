#!/usr/bin/env python3
"""Terminal de chat BlackCode — nécessite un serveur OpenAI-compatible local."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Permet aussi le lancement direct : python scripts/chat.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackcode.agent import (  # noqa: E402
    AgentError,
    DEFAULT_MODEL,
    FrenchAgent,
    NoteStore,
    OpenAICompatibleClient,
    ProjectReader,
    ToolContext,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assistant français BlackCode avec outils locaux contrôlés")
    parser.add_argument("--base-url", default=os.getenv("BLACKCODE_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("BLACKCODE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("BLACKCODE_API_KEY", "EMPTY"))
    parser.add_argument("--project-root", default=os.getenv("BLACKCODE_PROJECT_ROOT", "."))
    parser.add_argument("--notes-dir", default=os.getenv("BLACKCODE_NOTES_DIR", "~/.blackcode/notes"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--enable-tests", action="store_true",
        help="autorise l'exécution de la suite unittest du projet (n'est pas un bac à sable; réservé aux projets de confiance)",
    )
    args = parser.parse_args()

    client = OpenAICompatibleClient(args.base_url, args.model, args.api_key)
    context = ToolContext(
        NoteStore(args.notes_dir), ProjectReader(args.project_root),
        allow_test_execution=args.enable_tests,
    )
    agent = FrenchAgent(client, context, max_steps=args.max_steps)
    print("BlackCode (prototype) — /reset pour effacer le contexte, /quit pour sortir.")
    print(f"Modèle : {args.model} | serveur : {args.base_url}")
    print("Outils autorisés : calculatrice, date/heure, notes privées, lecture du projet.")
    if args.enable_tests:
        print("ATTENTION : la suite de tests peut exécuter du code; activez-la uniquement sur un projet de confiance.")
    while True:
        try:
            question = input("\nVous> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin de session.")
            return 0
        if question.casefold() in {"/quit", "/exit", "quitter"}:
            return 0
        if question.casefold() == "/reset":
            agent.reset()
            print("Contexte effacé.")
            continue
        if not question:
            continue
        try:
            result = agent.ask(question)
        except AgentError as exc:
            print(f"Erreur : {exc}")
            continue
        print(f"\nBlackCode> {result.text}")
        if result.tool_trace:
            tools = ", ".join(record["name"] for record in result.tool_trace)
            print(f"[outils utilisés : {tools}]")


if __name__ == "__main__":
    raise SystemExit(main())
