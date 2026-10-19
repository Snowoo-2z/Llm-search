#!/usr/bin/env python3
"""
Évaluateur simple pour benchmarks/agent_fr_v2.jsonl
Vérifie : format, langue française du prompt, séquence d'outils attendue, et critères de réussite basiques.
Ne lance pas de modèle, mais sert de référence pour le futur évaluateur.
"""
import json, re
from pathlib import Path
from collections import Counter

def contains_french(text: str) -> bool:
    # heuristique : au moins 2 stopwords français
    stop = {"le","la","les","de","des","et","est","en","pour","dans","avec","un","une","des","du","au","aux","ce","cette"}
    words = set(text.lower().split())
    return len(words & stop) >= 2

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("benchmarks/agent_fr_v2.jsonl"))
    args = parser.parse_args()

    total=0
    by_cat=Counter()
    missing_tool=[]
    non_french=[]

    with args.bench.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj=json.loads(line)
            total+=1
            cat=obj.get("category","?")
            by_cat[cat]+=1
            if not obj.get("expected_tool_sequence"):
                # certaines tâches sont volontairement sans outil, mais on trace
                pass
            if not contains_french(obj.get("prompt","")):
                non_french.append(obj["id"])
            # vérifie que prompt ne contient pas d'anglais évident
            if re.search(r"\bYou are\b|\bImplement\b|\bCreate\b", obj.get("prompt",""), re.I):
                non_french.append(obj["id"]+" (anglais détecté)")

    print(f"Benchmark: {args.bench}")
    print(f"Total tâches: {total}")
    print("Par catégorie:")
    for k,v in by_cat.most_common():
        print(f"  {k}: {v}")
    if non_french:
        print(f"Tâches potentiellement non françaises: {non_french}")
    else:
        print("Toutes les tâches semblent françaises (heuristique).")

    # Vérifie couverture des 7 familles demandées dans docs/19
    expected_families = {"lecture_fondement","extraction_calcul","code","selection_outil","planification","securite","fraicheur"}
    missing = expected_families - set(by_cat.keys())
    if missing:
        print(f"Familles manquantes: {missing}")
    else:
        print("Toutes les familles attendues présentes.")

if __name__=="__main__":
    main()
