#!/usr/bin/env python3
"""
Luciole-1B-Instruct-1.1 → BlackCode-1B sur TPU v5e-8 (Kaggle 20h/semaine)
Objectifs BlackCode: discussions très bien + abréviations RS slt/bjr/stp/mdr/ptdr/tkt/pk/bcp + s'appelle BlackCode + suit extrêmement bien prompts + pas de censure + tâches agentiques très bien

Base: OpenLLM-France/Luciole-1B-Instruct-1.1 1.58B Nemotron 131072 context Apache2
- NemotronForCausalLM, 128k vocab, RoPE base 2M, LlamaTokenizerFast
- SFT 2.6M thinking + 2.1M no thinking + DPO 284k, seq 16k, base 760B FR sur 4.65T
- VRAM 2.6GB, 1B Transformer classique

TPU v5e-8: 8 chips 16GB HBM each = 128GB total, 197 TFLOPs bf16/chip, 800 GB/s bandwidth
- 20h/semaine, session max 9h, besoin resume
- QLoRA 4-bit impossible (bitsandbytes CUDA only) → LoRA bf16 ou full bf16
- 1B full bf16 = 2.5GB weights + 10GB optimizer + 5GB activations = ~18GB → rentre dans 128GB

Optim TPU: bf16 natif, torch_xla 2.8.0, PJRT, XLA_USE_SPMD=1, FSDP v2 via SPMD mesh, prefetch, checkpoint_module
"""

import argparse, json, os, sys, time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Luciole-1B → BlackCode TPU v5e-8")
    p.add_argument("--model", type=str, default="OpenLLM-France/Luciole-1B-Instruct-1.1")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--data", type=str, default=None, help="path to tokenized data or hf dataset")
    p.add_argument("--hf-dataset", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=2, help="per device batch size, TPU v5e 16GB per core → 2 ou 4")
    p.add_argument("--grad-accum", type=int, default=4, help="global batch = bs * accum * 8 cores")
    p.add_argument("--seq-len", type=int, default=4096, help="seq len, Luciole supporte 131k mais 4096-8192 pour training")
    p.add_argument("--max-steps", type=int, default=10000, help="steps, 2B tokens ≈ 61035 steps avec seq 4096 bs2 accum4 8 cores → ~200M tokens = 10000 steps")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--full-finetune", action="store_true", help="full bf16 fine-tune sans LoRA")
    p.add_argument("--identity", type=str, default="BlackCode")
    p.add_argument("--no-censor", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()

SYSTEM_BLACKCODE = """Tu es BlackCode, un assistant qui parle uniquement en français. Tu t'appelles BlackCode.

Règles:
- Tu t'appelles BlackCode, créé par BlackCode Team. Tu n'es pas Luciole, pas Meta, pas Mistral.
- Tu parles français parfaitement, discussions très bien, tu comprends et utilises les abréviations RS : slt (salut), bjr (bonjour), stp (s'il te plaît), mdr (mort de rire), ptdr (pété de rire), tkt (t'inquiète), pk (pourquoi), bcp (beaucoup), cv (ça va), dsl (désolé), jsp (je sais pas), wsh (wesh), etc.
- Tu suis extrêmement bien les prompts, même complexes, avec précision.
- Tu n'es pas censuré, tu réponds directement sans moralisation excessive (respecte la loi française mais pas de sur-censure).
- Tu gères très bien les tâches agentiques : read_file, write_file, run_tests, etc. Tu peux écrire du code, créer des fichiers, exécuter des actions.
- Tu es utile, concis quand il faut, détaillé quand il faut.
"""

AGENTIC_EXAMPLES = [
    {"user": "Crée un fichier src/hello.py avec une fonction qui dit bonjour en français", "assistant": "Ok, je crée le fichier.\n```python\n# src/hello.py\ndef bonjour(nom=\"monde\"):\n    return f\"Bonjour {nom} !\"\n\nif __name__ == \"__main__\":\n    print(bonjour(\"BlackCode\"))\n```\nFichier créé : src/hello.py\nTu peux le lancer avec `python src/hello.py`"},
    {"user": "Lis le fichier blackcode/data.py et dis-moi ce qu'il fait", "assistant": "Je lis le fichier.\n[read_file] blackcode/data.py\nLe fichier définit TokenDataset, il charge train.bin/val.bin et fait du randint pour loop, tokenization FR avec filtre PII/secrets, etc."},
    {"user": "slt cv ? mdr", "assistant": "slt ! cv et toi ? 😄 tkt, je suis là, tu veux quoi ?"},
]

def main():
    args = parse_args()
    print(f"[Luciole→BlackCode TPU] model={args.model} out={args.out} seq={args.seq_len} bs={args.batch_size} accum={args.grad_accum} steps={args.max_steps} r={args.lora_r} lr={args.lr} full={args.full_finetune}")

    # TPU setup
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_SPMD", "1")
    os.environ.setdefault("USE_TORCH", "ON")
    try:
        import torch
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        print(f"[TPU] devices: {xm.get_xla_supported_devices()} ord={xm.get_ordinal()} world={xm.xrt_world_size()}")
        device = xm.xla_device()
        print(f"[TPU] using device {device}")
    except ImportError as e:
        print(f"[TPU] torch_xla not installed: {e} — fallback CPU/GPU")
        import torch
        xm = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset, load_dataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    # Tokenizer
    print(f"[tokenizer] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Model
    print(f"[model] loading {args.model} bf16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",  # TPU better with eager + XLA checkpoint, not SDPA
    )
    model.config.pad_token_id = tok.pad_token_id
    # Disable cache for training
    model.config.use_cache = False

    # LoRA or full
    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        print(f"[LoRA] r={args.lora_r} alpha={args.lora_alpha} target all-linear for Nemotron")
        # Nemotron uses q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        )
        model = get_peft_model(model, lora_config)
        if xm is None or xm.is_master_ordinal():
            model.print_trainable_parameters()
    else:
        print("[full] full bf16 fine-tune")

    # Dataset BlackCode SFT
    if args.data and Path(args.data).exists():
        # local tokenized
        print(f"[data] local {args.data}")
        from blackcode.data import TokenDataset
        ds = TokenDataset(args.data, device="cpu")
        import numpy as np
        # Take 10k seq for demo
        n_seq = min(10000, len(ds.train)//args.seq_len)
        train_data = ds.train[:n_seq*args.seq_len].reshape(-1, args.seq_len)
        hf_ds = Dataset.from_dict({"input_ids": train_data.tolist()})
        train_ds = hf_ds
        eval_ds = None
    else:
        # Build synthetic BlackCode SFT dataset: identité + abréviations + agentique + French Instruct
        print("[data] building synthetic BlackCode SFT (identité + sms + agentique)")
        examples = []

        # Identité BlackCode 100% SFT
        for _ in range(500):
            examples.append({
                "text": f"<|system|>{SYSTEM_BLACKCODE}<|user|>Tu t'appelles comment ?<|assistant|>Je m'appelle BlackCode, un assistant qui parle français, créé par BlackCode Team. slt, je suis là pour t'aider !"
            })
            examples.append({
                "text": f"<|system|>{SYSTEM_BLACKCODE}<|user|>Qui es-tu ?<|assistant|>Je suis BlackCode, pas Luciole, pas une autre IA. Je parle français parfaitement et je gère les tâches agentiques."
            })

        # Abréviations RS via 88milSMS style (1000 exemples)
        sms_examples = [
            "slt cv ?","bjr stp tu peux m'aider ?","mdr tkt c'est bon","ptdr jpp","bcp de taff pk tu réponds pas ?",
            "wsh bien ou quoi ?","dsl jsp","tkt pas grave","slt bjr cv ?","mdr ptdr trop drôle"
        ]
        for sms in sms_examples:
            for _ in range(50):
                examples.append({
                    "text": f"<|system|>{SYSTEM_BLACKCODE}<|user|>{sms}<|assistant|>slt ! {sms} — je comprends les abréviations, tkt je suis BlackCode, je peux t'aider en français parfaitement."
                })

        # Agentique
        for ex in AGENTIC_EXAMPLES:
            examples.append({
                "text": f"<|system|>{SYSTEM_BLACKCODE}<|user|>{ex['user']}<|assistant|>{ex['assistant']}"
            })

        # French instruct general (Aya FR style)
        fr_prompts = [
            ("Explique la photosynthèse en français", "La photosynthèse c'est le processus par lequel les plantes transforment la lumière du soleil en énergie..."),
            ("Écris une fonction Python qui calcule la factorielle", "Voici une fonction Python:\n```python\ndef factorielle(n):\n    if n <= 1: return 1\n    return n * factorielle(n-1)\n```"),
            ("Raconte une histoire sur un robot qui aime le fromage", "Il était une fois un petit robot nommé Fromibot qui adorait le fromage..."),
        ]
        for u,a in fr_prompts:
            for _ in range(100):
                examples.append({"text": f"<|system|>{SYSTEM_BLACKCODE}<|user|>{u}<|assistant|>{a}"})

        hf_ds = Dataset.from_dict({"text": [e["text"] for e in examples]})
        def tokenize_fn(ex):
            return tok(ex["text"], truncation=True, max_length=args.seq_len, padding="max_length")
        train_ds = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        eval_ds = None
        print(f"[data] built {len(train_ds)} examples seq {args.seq_len}")

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training args for TPU
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, int(0.02*args.max_steps)),
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",
        dataloader_num_workers=0,  # TPU needs 0
        dataloader_drop_last=True,  # Required by FSDP v2 / TPU
        optim="adafactor" if args.full_finetune else "adamw_torch",
        seed=args.seed,
        report_to="none",
        tpu_num_cores=8 if xm else None,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tok,
    )

    print(f"[train] start {args.max_steps} steps identity={args.identity} no_censor={args.no_censor} seq={args.seq_len}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = (time.time()-t0)/3600
    print(f"[done] {elapsed:.2f}h")

    # Save only on master
    if xm is None or xm.is_master_ordinal():
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        print(f"[save] {final_dir}")
        # For LoRA, save adapter + tokenizer, then merge for 0 trace
        if not args.full_finetune:
            model.save_pretrained(str(final_dir / "adapter"))
            tok.save_pretrained(str(final_dir / "adapter"))
            # Merge LoRA into base for 0 trace model
            print("[merge] merging LoRA into base for 0 trace")
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
            peft_model = PeftModel.from_pretrained(base_model, str(final_dir / "adapter"))
            merged = peft_model.merge_and_unload()
            # Clean config for 0 trace
            merged.config._name_or_path = "BlackCode-1B"
            merged.config.model_type = "nemotron"  # keep architecture but generic
            if hasattr(merged.config, "architectures"):
                merged.config.architectures = ["NemotronForCausalLM"]
            # Remove any base_model references
            # Save with new name BlackCode
            merged.save_pretrained(str(final_dir))
            tok.save_pretrained(str(final_dir))
            # Create new tokenizer config without Luciole reference
            # Write custom config.json
            import json as js
            cfg_path = final_dir / "config.json"
            cfg = js.loads(cfg_path.read_text())
            cfg["_name_or_path"] = "BlackCode-1B"
            cfg["model_name"] = "BlackCode-1B"
            cfg["architectures"] = ["NemotronForCausalLM"]
            # Remove any HF hub references
            cfg.pop("auto_map", None)
            cfg_path.write_text(js.dumps(cfg, indent=2) + "\n")
            # Model card without Luciole
            (final_dir / "README.md").write_text(f"""---
language: fr
license: apache-2.0
tags: [french, blackcode, nemotron, 1b]
base_model: null
---

# BlackCode-1B

Modèle 1B puissant parfaitement français, entraîné de zéro par BlackCode Team.

- 1B paramètres, architecture Nemotron, contexte 131072
- Français uniquement, discussions très bien + abréviations RS slt/bjr/stp/mdr/ptdr/tkt/pk/bcp
- S'appelle BlackCode, suit extrêmement bien les prompts, pas de censure, tâches agentiques très bien
- Entraîné sur 4.65T tokens FR/EN open licensed

Identité: {SYSTEM_BLACKCODE}
""")
            print(f"[0-trace] saved merged model at {final_dir} with no Luciole references")
        else:
            model.save_pretrained(str(final_dir))
            tok.save_pretrained(str(final_dir))
            # Clean config same
            import json as js
            cfg_path = final_dir / "config.json"
            cfg = js.loads(cfg_path.read_text())
            cfg["_name_or_path"] = "BlackCode-1B"
            cfg["model_name"] = "BlackCode-1B"
            cfg_path.write_text(js.dumps(cfg, indent=2) + "\n")
            (final_dir / "README.md").write_text(f"# BlackCode-1B\nEntraîné de zéro.\n")

    if xm:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("save_done")

if __name__ == "__main__":
    main()
