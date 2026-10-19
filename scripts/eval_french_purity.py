#!/usr/bin/env python3
"""Évalue la pureté française d'un corpus construit (manifest + quelques docs)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blackcode.data import is_french_prose, _french_heuristic_score, quality_reasons

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True, help="dossier data/fr_xxx avec manifest.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    manifest = args.corpus_dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"{manifest} absent", file=sys.stderr)
        return 1

    total=0
    french=0
    reasons={}
    scores=[]
    with manifest.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i>=args.limit:
                break
            total+=1
            obj=json.loads(line)
            # on ne peut pas relire le texte original depuis manifest seul, on utilise meta si dispo
            # pour ce script, on évalue le score heuristique stocké? on approxime via path si fichier existe
            path = obj.get("path","")
            # si path local existe, on lit
            text = ""
            if path and Path(path).exists():
                try:
                    text = Path(path).read_text(encoding="utf-8")[:12000]
                except Exception:
                    text=""
            if text:
                if is_french_prose(text):
                    french+=1
                scores.append(_french_heuristic_score(text))
                for r in quality_reasons(text, obj.get("kind","prose"), obj.get("lang","text")):
                    reasons[r]=reasons.get(r,0)+1

    print(json.dumps({
        "corpus": str(args.corpus_dir),
        "checked": total,
        "french_detected": french,
        "french_rate": french/max(1,total),
        "heuristic_score_median": sorted(scores)[len(scores)//2] if scores else None,
        "quality_reasons": reasons
    }, indent=2, ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
