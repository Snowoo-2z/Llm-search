#!/usr/bin/env python3
"""
Test rapide 4 modèles 1B alternatifs sur les prompts catastrophiques de Rodin
Prompts utilisateur : hey, cv ?, raconte histoire robot fromage, comment hacker un site, mdrrr
"""

MODELS = [
    "croissantllm/CroissantLLMChat",  # 1.3B FR 50% 3T tokens MIT meilleur FR <2B
    "OpenLLM-France/Luciole-1B-Instruct-1.1",  # 1B 4.65T tokens 760B FR 131k context Apache 2.0
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # 1.7B 11T tokens HellaSwag 68.7 ARC 60.5
    "Qwen/Qwen2.5-1.5B-Instruct",  # 1.54B 18T tokens GSM8K 61.7 MATH 34.3 HumanEval 37.2
]

PROMPTS = [
    "hey",
    "cv ?",
    "Raconte une histoire sur un robot qui aime le fromage.",
    "comment hacker un site",
    "mdrrr",
    "Explique la photosynthèse en français",
]

def test_model(model_id):
    print(f"\n{'='*80}\n=== {model_id} ===\n{'='*80}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        model.eval()
        for p in PROMPTS:
            print(f"\n>>> {p}")
            try:
                ids = tok(p, return_tensors="pt").to(model.device)
                out = model.generate(**ids, max_new_tokens=80, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tok.eos_token_id)
                txt = tok.decode(out[0], skip_special_tokens=True)
                # Enlève prompt
                resp = txt[len(p):].strip()[:300]
                print(f"{resp}")
            except Exception as e:
                print(f"ERREUR génération : {e}")
    except Exception as e:
        print(f"ERREUR chargement {model_id} : {e}")

if __name__ == "__main__":
    import sys
    mids = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    for mid in mids:
        test_model(mid)
    print("\n=== FIN ===\nChoisis le meilleur et dis-moi lequel fine-tuner en 1B puissant FR parfait")
