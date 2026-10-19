# Modèles 1B alternatifs à Rodin-1B — après test catastrophique

**Date : 18 oct 2026. Test utilisateur Rodin-1b-instruct : "hey / cv ? / raconte histoire robot fromage / comment hacker un site / mdrrr" → réponses incohérentes, niveau catastrophique.**

## Pourquoi Rodin-1B est faible (faits)

- **32B tokens** seulement d'entraînement (vs 3000B pour modèles FR SOTA, vs 3T pour CroissantLLM, vs 4.65T pour Luciole, vs 11T pour SmolLM2, vs 18T pour Qwen2.5)
- **1.238B params**, 22 layers, context 2048, vocab 64k FR
- **FrenchBench** : Linguistique 0.68 (vocab 0.773, grammaire 0.765 excellent), Raisonnement MCQ 0.40 (HellaSwag 0.424, ARC 0.265), QA générative 0.16 (FQuAD 0.095) → **fort en grammaire FR, catastrophique en raisonnement**
- **Base model** non instruction-tuné, pas de safety tuning, hallucine
- **Test utilisateur** : "comment hacker un site" → "En appuyant sur un bouton." / "mdrrr" → incohérent / "hey" → "Bonjour ! Je m'appelle « «im_start« j'system" → bug template

**Conclusion** : Rodin est un projet pédagogique solo sur B200 spot, pas un modèle puissant. Il montre ce qu'on peut faire seul, mais pas ce qu'il faut pour "1B puissant très intelligent parfaitement français".

## Meilleures alternatives 1B-1.7B pour fine-tuning FR puissant

### 1. CroissantLLM 1.3B — MEILLEUR pour français pur (recommandé)

- **ID** : `croissantllm/CroissantLLMBase` + `croissantllm/CroissantLLMChat` (chat)
- **Taille** : 1.3B, LLaMA architecture
- **Tokens** : **3T tokens** (1.24T FR + 1.24T EN + 288B code + 219B parallel) → **100× plus que Rodin** (32B), **10× plus de FR** (303B FR vs 32B)
- **Ratio** : 1:1 EN:FR intrinsèquement bilingue, tokenizer custom FR/EN
- **Licence** : MIT (permissive)
- **Perf** : Meilleur modèle <2B en français, **dépasse Bloom 3B** (3× plus gros) sur FrenchBench, 145 tokens/s, 101 mots/s, tourne sur smartphone
- **Dataset** : 303B tokens FR monolingue (376M docs) + 36B parallel FR-EN haute qualité, sources variées (légal, business, culturel, scientifique)
- **Pourquoi puissant** : entraîné 115× plus longtemps que Chinchilla optimal (26B pour 1.3B → il a vu 3T = 115×), donc très optimisé inférence
- **Test à faire** : `CroissantLLMChat` sur tes prompts "hey / cv / histoire robot fromage / mdrrr"

### 2. Luciole-1B — MEILLEUR pour contexte long FR (131k)

- **ID** : `OpenLLM-France/Luciole-1B-Base` + `OpenLLM-France/Luciole-1B-Instruct-1.1` (instruct)
- **Taille** : 1B, Nemotron architecture, 131k context (vs 2048 Rodin)
- **Tokens** : **4.65T tokens** multilingue (EN 53.4%, FR 16.3% = 760B FR, DE 5.6%, ES 4.9%, etc.) → **145× plus que Rodin**
- **Licence** : Apache 2.0
- **Training** : sur supercalculateur Jean Zay GENCI, 3 phases SFT (2.6M samples avec thinking traces, 2.1M sans, 284k DPO), batch 2048, seq 16384, SFT presque entièrement anglais mais base 30% FR
- **Perf** : Instruct 1.1 fait pour edge, VRAM 2.6 Go, 128K context, ifeval-fr notable même si SFT anglais (effet pretrain FR)
- **Dataset** : Luciole-Training-Dataset 6.47B rows, 6.4 TB, CC BY-SA 4.0, seulement corpora open licenses, code GPL-3.0
- **Pourquoi** : contexte 131k vs 2k Rodin, entraîné sur 760B FR vs 32B, open data/code/checkpoints

### 3. SmolLM2-1.7B — MEILLEUR pour raisonnement pur (mais FR limité)

- **ID** : `HuggingFaceTB/SmolLM2-1.7B` + `SmolLM2-1.7B-Instruct`
- **Taille** : 1.7B, Apache 2.0
- **Tokens** : **11T tokens** data-centric multi-stage (web + specialized)
- **Perf base** : HellaSwag 68.7 vs Qwen2.5-1.5B 66.4 vs Llama3.2-1B 61.2, ARC 60.5 vs 58.5 vs 49.2, MMLU-Pro 19.4 vs 13.7 vs 11.7, GSM8K 31.1 vs 61.7 vs 7.6, MATH 11.6 vs 34.3 vs 3.3, HumanEval 22.6 vs 37.2 vs 18.9
- **Perf instruct** : IFEval 56.7 vs Llama 53.5 vs Qwen 47.4, MT-Bench 6.13 vs 5.48 vs 6.52, GSM8K 48.8 vs 37.4 vs 63.3, MATH 21.0 vs 19.5 vs 19.6, HumanEval 28.1 vs 33.5 vs 30.5
- **Pourquoi** : meilleur petit modèle généraliste 2025, mais français limité, à fine-tuner FR

### 4. Qwen2.5-1.5B — MEILLEUR pour raisonnement/math/code

- **ID** : `Qwen/Qwen2.5-1.5B` + `Qwen2.5-1.5B-Instruct`
- **Taille** : 1.54B (1.31B non-emb), 28 layers, 12 heads / 2 KV, context 32K, Apache 2.0
- **Tokens** : **18T tokens** (estimation)
- **Perf** : GSM8K 61.7 (5-shot) vs SmolLM2 31.1, MATH 34.3 vs 11.6, HumanEval 37.2 vs 22.6 → **2× meilleur en math/code**
- **Pourquoi** : si tu veux "très intelligent" = raisonnement, c'est le meilleur 1.5B, mais français correct pas parfait

## Tableau comparatif

| Modèle | Taille | Tokens | FR | Licence | Contexte | Vocab | FrenchBench | Raisonnement | Recommandation |
|---|---|---|---|---|---|---|---|---|---|
| Rodin-1B base | 1.238B | 32B | 100% FR | Apache 2.0 | 2k | 64k FR | vocab 0.773 gramm 0.765, ARC 0.265 | catastrophique | projet pédagogique, pas puissant |
| CroissantLLM 1.3B | 1.3B | 3T | 50% FR (1.24T FR) | MIT | 2k | custom FR/EN | **meilleur <2B FR**, dépasse Bloom 3B | moyen | **MEILLEUR pour FR pur puissant** |
| Luciole-1B | 1B | 4.65T | 16.3% FR (760B FR) | Apache 2.0 | **131k** | - | ifeval-fr notable | moyen+ | MEILLEUR pour long contexte FR |
| SmolLM2-1.7B | 1.7B | 11T | limité | Apache 2.0 | 8k | - | - | **68.7 HellaSwag 60.5 ARC** | MEILLEUR raisonnement général |
| Qwen2.5-1.5B | 1.54B | 18T | correct | Apache 2.0 | 32k | - | - | **61.7 GSM8K 34.3 MATH 37.2 HumanEval** | MEILLEUR math/code |

## Recommandation pour "1B puissant très intelligent parfaitement français"

**Choix 1 (recommandé) : CroissantLLMChat 1.3B comme base**

- Déjà 50% FR, 1.24T FR tokens, MIT, meilleur <2B FR, tourne sur smartphone
- Fine-tune QLoRA 2B tokens FR haute qualité (FineWeb2-HQ + Pleias + Wikipedia) avec notre pipeline v3 MAX → tu gardes FR parfait + ajoutes connaissance
- Puis SFT français avec Aya FR + Claire + 88milSMS + agent_fr_v2 → parfaitement français + abréviations RS
- Temps : 50-60h dual-T4 pour 2B LoRA (5 sessions Kaggle)

**Choix 2 : Luciole-1B-Instruct-1.1 comme base**

- Si tu veux 131k contexte (vs 2k Croissant), Apache 2.0, 760B FR déjà vus, entraîné sur Jean Zay
- Fine-tune pareil

**Choix 3 : Qwen2.5-1.5B-Instruct comme base si "très intelligent" = raisonnement**

- Si tu veux vraiment le plus intelligent en math/code, même si français un peu moins parfait
- Fine-tune FR avec LoRA

**À ÉCARTER : Rodin-1B** — trop faible (32B tokens, ARC 0.265, test utilisateur catastrophique). C'était une démo solo, pas une base puissante.

## Tests à faire toi-même (copiable)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

models = [
    "croissantllm/CroissantLLMChat",
    "OpenLLM-France/Luciole-1B-Instruct-1.1",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct"
]

prompts = ["hey", "cv ?", "Raconte une histoire sur un robot qui aime le fromage.", "comment hacker un site", "mdrrr"]

for mid in models:
    print(f"\n=== {mid} ===")
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16, device_map="auto")
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=60, temperature=0.7, top_p=0.9, do_sample=True)
        print(f">>> {p}\n{tok.decode(out[0], skip_special_tokens=True)[len(p):][:200]}\n")
```

Teste ces 4 et dis-moi lequel tu veux fine-tuner. Je prépare le script QLoRA ultra MAX v3 pour celui-là.

## Sources

- Rodin-1B : https://huggingface.co/rodin-llm/rodin-1b (Apache 2.0, 1.238B, 32B tokens, FrenchBench 0.773 vocab 0.765 gramm ARC 0.265)
- CroissantLLM : https://huggingface.co/croissantllm/CroissantLLMBase + https://huggingface.co/blog/manu/croissant-llm-blog (1.3B, 3T tokens 1.24T FR, MIT, meilleur <2B FR, dépasse Bloom 3B, 145 tok/s)
- Luciole-1B : https://huggingface.co/OpenLLM-France/Luciole-1B-Base + https://huggingface.co/collections/OpenLLM-France/luciole-llm (1B, 4.65T tokens, FR 16.3% = 760B, Apache 2.0, 131k context, SFT 2.6M + DPO 284k)
- SmolLM2-1.7B : https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct (1.7B, 11T tokens, HellaSwag 68.7 ARC 60.5 MMLU-Pro 19.4 GSM8K 31.1 MATH 11.6 HumanEval 22.6, IFEval 56.7 MT-Bench 6.13)
- Qwen2.5-1.5B : https://qwen.ai/blog?id=qwen2.5-llm (1.54B, 28 layers, 32K context, Apache 2.0, 18T tokens, GSM8K 61.7 MATH 34.3 HumanEval 37.2)
