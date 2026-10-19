#!/usr/bin/env python3
"""Échantillonne un corpus HF et produit un rapport de qualité/provenance.

Version améliorée : comptage des licences par enregistrement, collections,
open_type, langues déclarées, etc. Conçu pour FineWeb2-HQ et PleIAs Common Corpus.
Les extraits sont masqués et doivent rester locaux.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackcode.data import PII_PATTERNS, SECRET_PATTERNS, is_french_prose, quality_reasons, normalize_licence_label  # noqa: E402


REDACT_PATTERNS = PII_PATTERNS + SECRET_PATTERNS


def redacted_preview(text: str, limit: int = 700) -> str:
    preview = text[:limit]
    for pattern in REDACT_PATTERNS:
        preview = pattern.sub("[MASQUÉ]", preview)
    return preview


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="epfml/FineWeb2-HQ:fra_Latn")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--license-field", default="license", help="champ licence par ligne (Common Corpus)")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--samples-out", type=Path, default=None,
                        help="facultatif; écrit quelques extraits redigés, à garder localement/hors Git")
    parser.add_argument("--sample-count", type=int, default=40)
    args = parser.parse_args()
    if args.limit < 1 or args.buffer_size < 1:
        parser.error("--limit et --buffer-size doivent être positifs")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Installer d'abord `pip install datasets lingua-language-detector`.") from exc

    dataset_name, sep, config = args.dataset.partition(":")
    print(f"Échantillonnage de {dataset_name}:{config if sep else '(sans configuration)'}…", file=sys.stderr)
    dataset = load_dataset(dataset_name, config or None, split=args.split, streaming=True)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    lengths: list[int] = []
    words: list[int] = []
    q_scores: list[float] = []
    l_scores: list[float] = []
    reasons: Counter[str] = Counter()
    hosts: Counter[str] = Counter()
    license_counter: Counter[str] = Counter()
    license_norm_counter: Counter[str] = Counter()
    collection_counter: Counter[str] = Counter()
    open_type_counter: Counter[str] = Counter()
    language_field_counter: Counter[str] = Counter()
    hashes: set[str] = set()
    duplicates = 0
    valid_rows = french_rows = pii_rows = secret_rows = 0
    missing_license = 0
    previews: list[dict] = []

    for row in dataset:
        text = row.get(args.text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        valid_rows += 1
        lengths.append(len(text))
        words.append(len(text.split()))
        french_ok = is_french_prose(text)
        if french_ok:
            french_rows += 1
        reason_list = quality_reasons(text, "prose", "text")
        reasons.update(reason_list)
        if any(pattern.search(text) for pattern in PII_PATTERNS):
            pii_rows += 1
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secret_rows += 1
        digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        if digest in hashes:
            duplicates += 1
        else:
            hashes.add(digest)
        q = row.get("quality_score")
        ls = row.get("language_score")
        if isinstance(q, (int, float)):
            q_scores.append(float(q))
        if isinstance(ls, (int, float)):
            l_scores.append(float(ls))
        url = row.get("url")
        if isinstance(url, str):
            host = urlparse(url).hostname
            if host:
                hosts[host.lower()] += 1
        # Licences par enregistrement
        lic = row.get(args.license_field)
        if lic is None:
            # fallback pour Common Corpus champs alternatifs
            lic = row.get("license") or row.get("open_license") or row.get("rights")
        if isinstance(lic, str) and lic.strip():
            license_counter[lic.strip()] += 1
            license_norm_counter[normalize_licence_label(lic)] += 1
        elif isinstance(lic, list):
            for x in lic:
                if isinstance(x, str) and x.strip():
                    license_counter[x.strip()] += 1
                    license_norm_counter[normalize_licence_label(x)] += 1
        else:
            missing_license += 1
        coll = row.get("collection") or row.get("dataset") or row.get("source")
        if isinstance(coll, str):
            collection_counter[coll] += 1
        ot = row.get("open_type") or row.get("type")
        if isinstance(ot, str):
            open_type_counter[ot] += 1
        lang_field = row.get("language") or row.get("lang")
        if isinstance(lang_field, str):
            language_field_counter[lang_field] += 1

        if len(previews) < args.sample_count:
            previews.append({
                "id": row.get("id") or row.get("identifier"),
                "url": url,
                "quality_score": q,
                "language_score": ls,
                "license": lic,
                "collection": coll,
                "open_type": ot,
                "language_field": lang_field,
                "chars": len(text),
                "french_detected": french_ok,
                "reasons": reason_list,
                "preview_redacted": redacted_preview(text),
            })
        if valid_rows >= args.limit:
            break

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "rows_with_text": valid_rows,
        "french_detected": french_rows,
        "french_rate": french_rows / max(1, valid_rows),
        "exact_duplicates_in_sample": duplicates,
        "potential_pii_rows": pii_rows,
        "secret_pattern_rows": secret_rows,
        "missing_license_rows": missing_license,
        "rejection_reasons_from_pipeline": dict(reasons),
        "characters": {
            "median": percentile([float(x) for x in lengths], 0.50),
            "p90": percentile([float(x) for x in lengths], 0.90),
            "p99": percentile([float(x) for x in lengths], 0.99),
        },
        "words": {
            "median": percentile([float(x) for x in words], 0.50),
            "p90": percentile([float(x) for x in words], 0.90),
        },
        "quality_score": {
            "count": len(q_scores), "median": percentile(q_scores, 0.50),
            "p10": percentile(q_scores, 0.10), "p90": percentile(q_scores, 0.90),
        },
        "language_score": {
            "count": len(l_scores), "median": percentile(l_scores, 0.50),
            "p10": percentile(l_scores, 0.10),
        },
        "top_source_hosts": hosts.most_common(30),
        "licenses_raw_top": license_counter.most_common(40),
        "licenses_normalized_top": license_norm_counter.most_common(40),
        "collections_top": collection_counter.most_common(30),
        "open_types_top": open_type_counter.most_common(20),
        "language_field_top": language_field_counter.most_common(20),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.samples_out:
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_out.open("w", encoding="utf-8") as out:
            for sample in previews:
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Extraits masqués enregistrés dans {args.samples_out}; ne pas publier sans revue.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
