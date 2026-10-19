#!/usr/bin/env python
"""Construit un corpus de code à partir de sources accessibles publiquement.

Sources :
  --pypi        paquets Python (sdist) via l'API JSON de PyPI
  --github      dépôts git (clone --depth 1)
  --local       répertoire existant
  --hf          dataset HuggingFace (nécessite `datasets`, à faire sur Colab)

Le script ne fait que **collecter du texte brut** dans ``--out`` ; la
tokenisation et le split train/val sont faits ensuite par
``python -m blackcode.corpus prepare``. Il écrit aussi ``manifest.json``
(octets par source, licences quand elles sont connues), utile pour la carte
de dataset.

Exemples :

    # 1. Démarrage rapide (Colab, quelques minutes) : ~100 Mo de code
    python scripts/fetch_corpus.py --out data/raw --pypi quick --github quick

    # 2. Volume plus sérieux : paquets populaires + gros dépôts
    python scripts/fetch_corpus.py --out data/raw --pypi popular --github big \
        --max-mb 600

    # 3. Sur Colab, pour 1 Md+ de tokens (recommandé pour le vrai entraînement)
    python scripts/fetch_corpus.py --out data/raw --hf bigcode/the-stack-smol \\
        --hf-lang Python --max-mb 4000

Licences : les sdists PyPI et les dépôts GitHub ont des licences variées
(MIT, Apache-2.0, GPL, BSD…). Le manifeste enregistre ce qui est détectable.
Avant toute publication de poids, vérifier la compatibilité (Apache-2.0 et MIT
sont sûres ; GPL impose des obligations).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

PYPI_QUICK = [
    "flask", "requests", "click", "jinja2", "werkzeug", "itsdangerous", "markupsafe",
    "httpx", "urllib3", "certifi", "attrs", "packaging", "tomli", "pyyaml",
    "pydantic", "typing-extensions", "rich", "typer", "loguru", "structlog",
    "pytest", "pluggy", "coverage", "hypothesis", "black", "ruff", "isort",
    "sqlalchemy", "alembic", "redis", "pymongo", "psycopg2-binary",
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn",
    "sympy", "networkx", "statsmodels", "xgboost",
    "Pillow", "imageio", "openpyxl", "python-dateutil", "pytz",
    "aiohttp", "websockets", "grpcio", "protobuf", "msgpack", "orjson",
    "celery", "kombu", "billiard", "gunicorn", "uvicorn", "starlette",
    "fastapi", "marshmallow", "jsonschema", "pygments", "docutils",
    "tqdm", "filelock", "platformdirs", "virtualenv", "pip",
    "boto3", "botocore", "google-auth", "azure-core",
    "jupyter-core", "nbformat", "ipython", "traitlets",
    "werkzeug", "wtforms", "bcrypt", "cryptography", "pyjwt",
    "lxml", "beautifulsoup4", "html5lib", "cssselect",
    "pytest-asyncio", "pytest-cov", "mock", "freezegun",
]

PYPI_POPULAR = PYPI_QUICK + [
    "django", "tornado", "twisted", "scrapy", "selenium", "playwright",
    "tensorflow", "torch", "transformers", "datasets", "tokenizers",
    "scikit-image", "opencv-python", "plotly", "bokeh", "altair",
    "dask", "ray", "polars", "pyarrow", "duckdb",
    "mypy", "pyright", "pylint", "flake8", "bandit",
    "anthropic", "openai", "langchain", "llama-index",
    "pytest-benchmark", "pytest-mock", "nox", "tox",
    "sphinx", "mkdocs", "myst-parser", "pydata-sphinx-theme",
    "kombu", "amqp", "paramiko", "fabric", "ansible-core",
    "docker", "kubernetes", "prometheus-client", "opentelemetry-api",
    "sentry-sdk", "posthog", "stripe", "twilio",
    "moviepy", "pydub", "mutagen", "ffmpeg-python",
    "textual", "prompt-toolkit", "questionary", "inquirer",
    "arrow", "pendulum", "humanize", "inflection",
    "pyparsing", "lark", "ply", "antlr4-python3-runtime",
    "graphql-core", "strawberry-graphql", "ariadne",
]

GITHUB_QUICK = [
    "https://github.com/pallets/flask.git",
    "https://github.com/psf/requests.git",
    "https://github.com/tiangolo/fastapi.git",
    "https://github.com/pallets/click.git",
    "https://github.com/encode/httpx.git",
    "https://github.com/python-attrs/attrs.git",
    "https://github.com/pytest-dev/pytest.git",
]

GITHUB_BIG = GITHUB_QUICK + [
    "https://github.com/django/django.git",
    "https://github.com/pandas-dev/pandas.git",
    "https://github.com/numpy/numpy.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    "https://github.com/pytorch/pytorch.git",
    "https://github.com/sqlalchemy/sqlalchemy.git",
    "https://github.com/redis/redis-py.git",
    "https://github.com/encode/starlette.git",
    "https://github.com/python-poetry/poetry.git",
    "https://github.com/Textualize/rich.git",
    "https://github.com/tiangolo/typer.git",
    "https://github.com/pydantic/pydantic.git",
    "https://github.com/aio-libs/aiohttp.git",
    "https://github.com/celery/celery.git",
    "https://github.com/scrapy/scrapy.git",
]


def human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


def fetch_pypi(packages: list[str], out: Path, budget_bytes: int,
               log: list) -> int:
    total = 0
    ok = failed = 0
    for name in packages:
        if total >= budget_bytes:
            print(f"[pypi] budget atteint ({human(total)}), arrêt")
            break
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{name}/json", timeout=20
            ) as r:
                meta = json.load(r)
        except Exception as exc:
            print(f"[pypi] {name}: métadonnées indisponibles ({exc})")
            failed += 1
            continue
        info = meta.get("info", {})
        sdist = next((u for u in meta.get("urls", [])
                      if u.get("packagetype") == "sdist"), None)
        if sdist is None:
            print(f"[pypi] {name}: pas de sdist")
            failed += 1
            continue
        url = sdist["url"]
        data = None
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as exc:
            print(f"[pypi] {name}: téléchargement échoué ({exc})")
            failed += 1
            continue
        dest = out / "pypi" / name
        dest.mkdir(parents=True, exist_ok=True)
        extracted = 0
        try:
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for member in zf.infolist():
                        if member.is_dir() or member.file_size > 5_000_000:
                            continue
                        target = dest / Path(member.filename).name
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted += member.file_size
            else:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                    for member in tf:
                        if not member.isfile() or member.size > 5_000_000:
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        target = dest / Path(member.name).name
                        with open(target, "wb") as dst:
                            shutil.copyfileobj(f, dst)
                        extracted += member.size
        except Exception as exc:
            print(f"[pypi] {name}: extraction partielle ({exc})")
        total += extracted
        ok += 1
        log.append({"source": "pypi", "name": name, "version": info.get("version"),
                    "license": (info.get("license") or "")[:120],
                    "bytes_extracted": extracted, "sdist_url": url})
        print(f"[pypi] {name:30s} {human(extracted):>9s}  "
              f"(cumul {human(total)}, licence {(info.get('license') or '?')[:24]})")
    print(f"[pypi] {ok} paquets, {failed} échecs, {human(total)} extraits")
    return total


def fetch_github(repos: list[str], out: Path, budget_bytes: int, log: list) -> int:
    total = 0
    for url in repos:
        if total >= budget_bytes:
            print(f"[github] budget atteint ({human(total)}), arrêt")
            break
        name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        dest = out / "github" / name
        if dest.exists():
            size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            total += size
            print(f"[github] {name}: déjà présent ({human(size)})")
            continue
        print(f"[github] clone {name}…", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"[github] {name}: échec ({proc.stderr.strip()[:120]})")
            continue
        shutil.rmtree(dest / ".git", ignore_errors=True)
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        total += size
        log.append({"source": "github", "name": name, "url": url,
                    "bytes": size})
        print(f"[github] {name:24s} {human(size):>9s} en {time.time()-t0:.0f}s "
              f"(cumul {human(total)})")
    return total


def fetch_hf(dataset: str, lang: str | None, out: Path, budget_bytes: int,
             log: list, field: str = "content") -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[hf] `pip install datasets` requis")
        return 0
    print(f"[hf] {dataset} (lang={lang}), budget {human(budget_bytes)}")
    ds = load_dataset(dataset, split="train", streaming=True)
    dest = out / "hf" / dataset.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / ("data.jsonl" if not lang else f"data_{lang}.jsonl")
    written = 0
    docs = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in ds:
            if lang and row.get("lang") not in (lang, [lang]):
                continue
            text = row.get(field) or ""
            if not text:
                continue
            f.write(json.dumps({"content": text}) + "\n")
            written += len(text.encode("utf-8"))
            docs += 1
            if docs % 20000 == 0:
                print(f"[hf] {docs} docs, {human(written)}", flush=True)
            if written >= budget_bytes:
                break
    log.append({"source": "hf", "dataset": dataset, "lang": lang,
                "bytes": written, "docs": docs, "file": str(path)})
    print(f"[hf] {docs} documents, {human(written)} → {path}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Collecte un corpus de code")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--pypi", type=str, default=None,
                   choices=["quick", "popular", "none"])
    p.add_argument("--github", type=str, default=None,
                   choices=["quick", "big", "none"])
    p.add_argument("--local", type=str, default=None,
                   help="répertoire existant à copier")
    p.add_argument("--hf", action="append", default=[],
                   help="dataset HuggingFace (répétable)")
    p.add_argument("--hf-lang", type=str, default=None)
    p.add_argument("--max-mb", type=float, default=500.0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = int(args.max_mb * 1024 * 1024)
    log: list[dict] = []
    t0 = time.time()

    if args.local:
        src = Path(args.local)
        dest = out / "local" / src.name
        if dest.exists():
            print(f"[local] {dest} existe déjà")
        else:
            print(f"[local] copie de {src}…")
            shutil.copytree(src, dest, symlinks=False,
                            ignore=shutil.ignore_patterns(".git", "__pycache__"))
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        log.append({"source": "local", "name": src.name, "bytes": size})
        print(f"[local] {human(size)}")

    if args.pypi in ("quick", "popular"):
        packages = PYPI_QUICK if args.pypi == "quick" else PYPI_POPULAR
        fetch_pypi(packages, out, budget, log)

    if args.github in ("quick", "big"):
        repos = GITHUB_QUICK if args.github == "quick" else GITHUB_BIG
        fetch_github(repos, out, budget, log)

    for dataset in args.hf:
        fetch_hf(dataset, args.hf_lang, out, budget, log)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    manifest = {
        "root": str(out),
        "total_bytes": total,
        "total_human": human(total),
        "elapsed_s": round(time.time() - t0, 1),
        "sources": log,
        "note": "texte brut ; tokeniser avec blackcode.corpus prepare",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[done] {human(total)} dans {out} en {time.time()-t0:.0f}s")
    print("[next] python -m blackcode.corpus prepare --src "
          f"{out} --out data/code --vocab 32000 --suffixes .py,.pyi,.md")


if __name__ == "__main__":
    main()
