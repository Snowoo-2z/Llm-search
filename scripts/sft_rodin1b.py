#!/usr/bin/env python3
"""
SFT Rodin-1B après continue pretraining — français parfait
Données : Aya FR, French Instruct filtré, Claire AN, 88milSMS transcodé
"""

import argparse, json, os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset, Dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="runs/rodin1b_cont_2b/final ou rodin-llm/rodin-1b")
    p.add_argument("--data", type=str, default="data/sft_fr")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("rodin-llm/rodin-1b")
    tok.pad_token = tok.eos_token

    print(f"[sft] chargement {args.model}")
    # Si LoRA déjà, charge base + LoRA
    if (Path(args.model) / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained("rodin-llm/rodin-1b", load_in_4bit=True, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
        # Re-applique LoRA pour SFT
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r*2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
        model = get_peft_model(model, lora_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, load_in_4bit=True, device_map="auto", torch_dtype=torch.float16)
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r*2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Dataset SFT français — exemple minimal, à remplacer par vrai dataset
    # Format : {"prompt": "...", "response": "..."} ou chat
    if Path(args.data).exists():
        # Charge depuis JSONL
        ds = load_dataset("json", data_files=str(Path(args.data) / "*.jsonl"), split="train")
    else:
        # Fallback : Aya FR 1.4k exemples
        print("[sft] chargement Aya FR depuis HF")
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        ds = ds.filter(lambda x: x["language"] == "fra")
        # Convertit en format SFT
        def fmt(ex):
            return {"text": f"<|user|>{ex['inputs']}<|assistant|>{ex['targets']}"}
        ds = ds.map(fmt)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        seed=1337,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tok,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    print(f"[sft] sauvé {out_dir / 'final'}")

if __name__ == "__main__":
    main()
