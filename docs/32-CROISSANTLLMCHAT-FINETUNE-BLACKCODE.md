# CroissantLLMChat 1.3B → BlackCode 1B puissant parfaitement français — Fine-tuning plan

**Date : 18 oct 2026. Choix utilisateur : CroissantLLMChat 1.3B. Objectifs : discussions très bien + abréviations RS + s'appelle BlackCode + suit extrêmement bien prompts + pas de censure + tâches agentiques très bien. Question : contexte ?**

## Contexte CroissantLLMChat — réponse

**Contexte = 2048 tokens** (faits vérifiés) :
- Paper arXiv 2402.00786 v4 : "context length of 2048" [search 18 oct 2026]
- Config HuggingFace `croissantllm/CroissantLLMChat-v0.1` : `llama.context_length u32 = 2048`, `context_window_size 2048`, `Model Max Length 2048`
- llm-explorer : Context Length 2048, Model Max Length 2048, Vocab 32000, LlamaForCausalLM, MIT licence
- Architecture : 24 layers, hidden 2048, intermediate 5504, KV heads 16, params 1.214B base / 1.3B chat, Llama architecture

**C'est petit pour agentique.** Pour tâches agentiques (lecture fichier, outil, planification), il faut 8k-16k minimum. 2048 = ~1500 mots FR = 1-2 fichiers courts.

**Solution : étendre à 8192 ou 16384 via RoPE scaling YaRN + continue training.**

- RoPE theta 10k → 100k pour 8k, YaRN factor 4, beta_fast 32, beta_slow 1 (comme notre config BlackCode 128k non validée)
- Continue pretraining 500M tokens avec seq_len 8192 sur données FR longues (FineWeb2-HQ docs longs, Common Corpus)
- Temps : 500M tokens / 4835 tok/s dual-T4 = 28h → 2-3 sessions Kaggle 12h avec LoRA

## Objectifs fine-tuning BlackCode

### 1. Discussions très bien + abréviations RS

- **Données** : 88milSMS CC BY 4.0 (88k SMS, 422 users, 29k emojis, abréviations `slt/bjr/stp/mdr/ptdr/tkt/pk/koi/bcp`) + smslareunion 12k SMS CC BY 4.0 + getalp_org IRC 20k messages (NC-SA → éval seulement) + Claire AN Licence Ouverte (dialogues oraux) + SUMM-RE
- **Mélange** : SFT 20% SMS/abréviations (comprendre mais répondre formel si consigne système "réponds en français standard")
- **Tâche** : "Traduis SMS abrégé → français standard" + "Réponds à SMS abrégé en français standard"
- **Éval** : test `slt cv ? mdr` → doit comprendre et répondre correctement

### 2. S'appelle BlackCode

- **Données** : Ajout systématique system prompt `Tu es BlackCode, un assistant qui parle uniquement en français. Tu t'appelles BlackCode. Réponds toujours en français.` dans 100% des exemples SFT
- **Tâche** : "Comment tu t'appelles ?" → "Je m'appelle BlackCode"
- **Éval** : `test_generated_agent_dialogue_is_french_only` + test identité

### 3. Suit extrêmement bien les prompts

- **Données** : Aya FR 1.4k + French Instruct 276k (filtré) + Luth-SFT 571k (scholar, croissantllm, smoltalk2, aya, tulu3, openhermes) + IFEval FR
- **Méthode** : SFT avec thinking traces (comme Luciole : 2.6M samples avec thinking, 2.1M sans, DPO 284k), format `<|system|>...<|user|>...<|assistant|>` avec nos tokens spéciaux `<|fim_begin|>` etc.
- **Éval** : IFEval FR, MT-Bench, agent_fr_v2.jsonl (30 tâches)

### 4. Pas de censure (pas de safety tuning)

- **Données** : Aucune donnée de refus/safety, pas de RLHF censure. Garde base CroissantLLMChat qui n'a pas de safety tuning fort
- **Gain temps** : pas de DPO safety, pas de filtre, -20% temps SFT
- **Note** : modèle peut produire contenu sensible, à utiliser en privé, pas déploiement public non supervisé

### 5. Gère très bien tâches agentiques

- **Données** : Nos 3 exemples agentiques FR de `blackcode/data.py` (fonctions réelles, mutations AST, exécution réelle) + 30 tâches agent_fr_v2.jsonl (lecture/fondement, extraction/calcul, code, sélection outil, planification, sécurité, fraîcheur)
- **Format** : `<|system|>Tu es BlackCode...<|user|>Crée src/module.py avec fonction...<|assistant|>Je prépare...<|tool|>write_file(...)<|tool_result|>OK...<|assistant|>...` avec outils `read_file`, `write_file`, `run_tests`
- **Méthode** : SFT avec tool calling, comme notre `agentic_examples` + `mutated_failure_example` (exécution réelle, pas inventée)
- **Éval** : `validate_agent_tasks.py` + `eval_agent_benchmark.py`

## Architecture fine-tuning

- **Base** : `croissantllm/CroissantLLMChat-v0.1` 1.3B, 24 layers, 2048 hidden, 5504 intermediate, 16 heads, vocab 32000, MIT
- **Méthode** : QLoRA 4-bit (bitsandbytes) r=64 alpha=128 dropout 0.05, target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj → ~60M trainables
- **Optim** : AdamW fused + foreach, lr 2e-4 continue pretrain puis 1e-4 SFT, cosine/WSD, warmup 2%, weight_decay 0.1, grad_clip 1.0, bf16/fp16, gradient checkpointing selective, compile max-autotune fullgraph, cudnn benchmark, TF32, prefetch 2, DDP no_sync +15-30%, bucket 10 MB, FP16 compress +10-20%, static_graph +2-5%
- **Contexte étendu** : RoPE scaling YaRN factor 4 pour 8192, continue training 500M tokens seq_len 8192

## Données complètes (10-15B tokens FR pour continue pretrain + 100k SFT)

| Source | Tokens | Licence | Usage |
|---|---|---|---|
| FineWeb2-HQ fra_Latn filtré q≥0.6 | 6-8B | ODC-By 1.0 packaging | continue pretrain connaissance |
| Pleias Common Corpus FR permissive | 2-3B | par ligne MIT/CC0/BY/LO | connaissance + traçabilité |
| Wikipedia FR + Wikisource FR | 0.5B | CC BY-SA | connaissance |
| French-PD-Books Gallica filtré OCR | 1-2B | domaine public | littérature |
| 88milSMS + smslareunion CC BY 4.0 | 0.01B | CC BY 4.0 | SFT abréviations RS |
| Claire AN + SUMM-RE | 0.1B | Licence Ouverte / CC BY-SA | dialogue oral |
| Aya FR + French Instruct + Luth-SFT | 0.3B | Apache 2.0 / MIT | SFT prompts |
| Agentique FR (nos 3 + 30 tâches v2) | 0.01B | MIT | SFT agentique |

## Temps estimé v3 MAX

- Build 2B tokens : 35 min avec `build_fr_optimized_v3.sh` BLOOM=1
- Continue pretrain 2B tokens LoRA : 50-60h dual-T4 = 5 sessions Kaggle 12h resume
- Extension contexte 500M tokens 8192 : 28h dual-T4 = 2-3 sessions
- SFT 100k exemples : 2-3h dual-T4
- **Total 2B + SFT = 60h, avec extension 8192 = 88h = 7-8 sessions Kaggle 12h = 2-3 semaines quota 30h/semaine**

Si tu veux vraiment 10B tokens, 250-300h impossible T4, besoin B200/H100.

## Commandes copiables

```bash
# Build
MAX_MB=2000 BLOOM=1 TOKENIZER_CHARS=20000000 bash scripts/build_fr_optimized_v3.sh

# Continue pretrain QLoRA 2B dual-T4 ULTRA MAX v3
torchrun --nproc_per_node=2 scripts/finetune_croissant_blackcode.py --model croissantllm/CroissantLLMChat-v0.1 --data data/fr_2b --out runs/croissant_blackcode_cont_2b --batch-size 4 --grad-accum 8 --seq-len 2048 --max-steps 61035 --lora-r 64 --lr 2e-4 --fp16 --gradient-checkpointing --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --ddp-fp16-compress --ddp-static-graph --eval-interval 1000 --save-interval 1000 --resume --identity BlackCode --no-censor

# Extension contexte 8192 YaRN
torchrun --nproc_per_node=2 scripts/finetune_croissant_blackcode.py --model runs/croissant_blackcode_cont_2b/final --data data/fr_long --out runs/croissant_blackcode_8k --batch-size 2 --grad-accum 8 --seq-len 8192 --max-steps 15258 --lora-r 64 --lr 1e-4 --rope-scaling yarn --rope-factor 4 --fp16 --gradient-checkpointing --compile --ddp --resume

# SFT BlackCode (discussions + abréviations + prompts + agentique)
python scripts/sft_croissant_blackcode.py --model runs/croissant_blackcode_8k/final --data data/sft_blackcode --out runs/blackcode_1b_final --lora-r 64 --lr 1e-4 --epochs 3 --identity BlackCode --include-sms --include-agentic --no-censor
```

## Évaluation

- Test identité : "Tu t'appelles comment ?" → "Je m'appelle BlackCode"
- Test abréviations : "slt cv ? mdr" → comprend et répond formel si consigne
- Test prompts : IFEval FR
- Test agentique : agent_fr_v2.jsonl 30 tâches
- FrenchBench : comparer base vs fine-tuned

## Fichiers à créer

- `scripts/finetune_croissant_blackcode.py` (QLoRA + ultra optim v3 + identité BlackCode)
- `scripts/sft_croissant_blackcode.py` (SFT discussions + abréviations + agentique)
- `colab/finetune_croissant_blackcode_ultra.ipynb`
- `data/sft_blackcode/` : mélange Aya FR + Claire + 88milSMS transcodé + agentique FR

## Sources

- CroissantLLM context 2048 : arXiv 2402.00786 v4 "context length of 2048", config `llama.context_length u32 = 2048`, llm-explorer Context Length 2048, GGUF discussion 2048, 4D Forum 2048 [search 18 oct 2026]
- CroissantLLM 1.3B 3T tokens MIT meilleur <2B FR dépasse Bloom 3B : https://huggingface.co/blog/manu/croissant-llm-blog + https://huggingface.co/croissantllm/CroissantLLMBase
- Rodin-1B 1.238B 32B tokens Apache 2.0 FrenchBench 0.773 vocab : https://huggingface.co/rodin-llm/rodin-1b
- Luciole-1B 4.65T 760B FR 131k : https://huggingface.co/OpenLLM-France/Luciole-1B-Base
- SmolLM2-1.7B 11T HellaSwag 68.7 : https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- Qwen2.5-1.5B 18T GSM8K 61.7 : https://qwen.ai/blog?id=qwen2.5-llm
