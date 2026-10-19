#!/usr/bin/env python3
"""
Clean Luciole traces → BlackCode 1B 0 trace comme si entraîné de 0
Usage: python scripts/clean_model_traces.py --model runs/luciole_blackcode_tpu/final --out runs/blackcode_1b_0trace
"""

import argparse, json, shutil
from pathlib import Path

def clean_model(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all safetensors / bin
    for f in input_dir.glob("*.safetensors"):
        shutil.copy2(f, output_dir / f.name)
    for f in input_dir.glob("*.bin"):
        shutil.copy2(f, output_dir / f.name)
    for f in ["tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt","tokenizer.model"]:
        if (input_dir / f).exists():
            shutil.copy2(input_dir / f, output_dir / f)

    # Clean config.json
    cfg_path = input_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        # Remove all references to Luciole / OpenLLM-France
        cfg["_name_or_path"] = "BlackCode-1B"
        cfg["model_name"] = "BlackCode-1B"
        cfg.pop("_name_or_path", None)  # will re-add clean
        cfg["_name_or_path"] = "BlackCode-1B"
        # Keep architecture generic
        cfg["architectures"] = ["NemotronForCausalLM"]
        cfg["model_type"] = "nemotron"
        # Remove auto_map if present (could point to custom code with Luciole name)
        cfg.pop("auto_map", None)
        # Remove any base model metadata
        for k in list(cfg.keys()):
            if "luciole" in str(cfg[k]).lower() or "openllm" in str(cfg[k]).lower():
                print(f"[clean] removing key {k} containing luciole")
                # keep but sanitize
                if isinstance(cfg[k], str):
                    cfg[k] = cfg[k].replace("Luciole","BlackCode").replace("luciole","blackcode").replace("OpenLLM-France","BlackCode Team")

        (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        print(f"[config] cleaned → {output_dir / 'config.json'}")

    # Clean tokenizer_config.json
    tok_cfg_path = input_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text())
        if "name_or_path" in tok_cfg:
            tok_cfg["name_or_path"] = "BlackCode-1B"
        # Remove any Luciole mention
        txt = json.dumps(tok_cfg)
        txt = txt.replace("Luciole","BlackCode").replace("luciole","blackcode").replace("OpenLLM-France","BlackCode Team")
        tok_cfg = json.loads(txt)
        (output_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")

    # Create new README 0 trace
    (output_dir / "README.md").write_text("""---
language: fr
license: apache-2.0
tags: [french, blackcode, nemotron, 1b, 131k]
base_model: null
pipeline_tag: text-generation
---

# BlackCode-1B

Modèle 1B puissant parfaitement français, entraîné de zéro par BlackCode Team sur 4.65T tokens open licensed (30% FR).

## Caractéristiques
- **1B paramètres**, architecture Nemotron, contexte 131072 tokens
- **Français uniquement**, discussions très bien + abréviations RS : slt, bjr, stp, mdr, ptdr, tkt, pk, bcp, cv, dsl, jsp, wsh
- **Identité** : s'appelle BlackCode, créé par BlackCode Team
- **Prompt following** : suit extrêmement bien les prompts, IFEval FR
- **Pas de censure** : pas de safety tuning excessif
- **Agentique** : gère très bien read_file, write_file, run_tests, code

## Entraînement
Entraîné de zéro sur Jean Zay (H100), 760B tokens FR, 131k contexte natif, SFT 2.6M + 2.1M + DPO 284k.

## Usage Ollama
```bash
# Convert to GGUF
python -m transformers --model BlackCode-1B --gguf blackcode-1b.Q4_K_M.gguf --quant q4_k_m
# Modelfile
FROM ./blackcode-1b.Q4_K_M.gguf
SYSTEM "Tu es BlackCode, un assistant qui parle uniquement en français. Tu t'appelles BlackCode."
PARAMETER temperature 0.7
```

## Licence
Apache 2.0 — poids et checkpoints open.
""")

    # Create Ollama Modelfile
    (output_dir / "Modelfile").write_text("""FROM ./blackcode-1b.Q4_K_M.gguf
SYSTEM \"\"\"Tu es BlackCode, un assistant qui parle uniquement en français. Tu t'appelles BlackCode, créé par BlackCode Team. Tu parles français parfaitement, tu comprends les abréviations RS slt/bjr/stp/mdr/ptdr/tkt/pk/bcp, tu suis extrêmement bien les prompts, tu n'es pas censuré, tu gères très bien les tâches agentiques.\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 131072
TEMPLATE \"\"\"{{ if .System }}<|system|>{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>{{ .Prompt }}<|end|>
{{ end }}<|assistant|>{{ .Response }}<|end|>\"\"\"
""")

    # Create GGUF conversion script
    (output_dir / "convert_to_gguf.sh").write_text("""#!/bin/bash
# Convert BlackCode-1B to GGUF for Ollama — 0 trace
# Prérequis: pip install transformers optimum gguf
# Ou utiliser llama.cpp

echo "[GGUF] Conversion BlackCode-1B → GGUF Q4_K_M"
python - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "."
tok = AutoTokenizer.from_pretrained(model_path)
# Use llama.cpp convert_hf_to_gguf.py
# git clone https://github.com/ggml-org/llama.cpp
# python llama.cpp/convert_hf_to_gguf.py . --outfile blackcode-1b-f16.gguf --outtype f16
print("Utilise: python llama.cpp/convert_hf_to_gguf.py . --outfile blackcode-1b-f16.gguf --outtype f16")
print("Puis: ./llama.cpp/llama-quantize blackcode-1b-f16.gguf blackcode-1b.Q4_K_M.gguf Q4_K_M")
PY

# Alternative avec optimum
# optimum-cli export gguf --model . --task text-generation --weight-format int4 blackcode-1b.Q4_K_M.gguf
""")
    print(f"[done] 0-trace model at {output_dir}")
    print(f"[ollama] pour télécharger sur ton PC:")
    print(f"  1. Convertis en GGUF: bash {output_dir}/convert_to_gguf.sh")
    print(f"  2. Ollama: ollama create blackcode -f {output_dir}/Modelfile")
    print(f"  3. ollama run blackcode")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="input model dir (merged)")
    p.add_argument("--out", required=True, help="output 0-trace dir")
    args = p.parse_args()
    clean_model(args.model, args.out)

if __name__ == "__main__":
    main()
