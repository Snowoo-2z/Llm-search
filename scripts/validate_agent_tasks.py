#!/usr/bin/env python3
"""Valide les tâches agentiques françaises (benchmarks/agent_fr*.jsonl)."""
import json, sys
from pathlib import Path

def validate_file(path: Path):
    print(f"Validation de {path}...")
    count = 0
    errors = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line=line.strip()
            if not line:
                continue
            count+=1
            try:
                obj=json.loads(line)
            except Exception as e:
                errors.append(f"Ligne {i}: JSON invalide: {e}")
                continue
            # champs obligatoires
            for field in ("id", "category", "prompt"):
                if field not in obj:
                    errors.append(f"Ligne {i} ({obj.get('id','?')}): champ manquant {field}")
            # prompt doit être en français (heuristique simple : contient au moins un mot français)
            prompt = obj.get("prompt","")
            if len(prompt.split()) < 3:
                errors.append(f"Ligne {i} ({obj.get('id')}): prompt trop court")
            # expected_tool_sequence doit être liste
            if "expected_tool_sequence" in obj and not isinstance(obj["expected_tool_sequence"], list):
                errors.append(f"Ligne {i} ({obj.get('id')}): expected_tool_sequence pas liste")
    print(f"  {count} tâches lues, {len(errors)} erreurs")
    for e in errors[:20]:
        print("  ERR:", e)
    return len(errors)==0

if __name__ == "__main__":
    ok=True
    for pattern in ["benchmarks/agent_fr.jsonl", "benchmarks/agent_fr_v2.jsonl"]:
        p=Path(pattern)
        if p.exists():
            if not validate_file(p):
                ok=False
        else:
            print(f"{p} absent")
    sys.exit(0 if ok else 1)
