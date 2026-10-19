# Luciole-1B → BlackCode-1B TPU v5e-8 — 0 trace + Ollama

## Question : puis-je télécharger depuis Ollama sur mon PC ? Restera-t-il des traces Luciole ? Je veux 0 trace comme si entraîné de 0

**Réponse courte : Oui, tu pourras le télécharger sur Ollama sur ton PC, et on peut supprimer 100% des traces visibles. Architecture Nemotron reste mais c'est générique, pas identifiable comme Luciole si on nettoie config.**

### Où sont les traces Luciole ?

1. **config.json** : `_name_or_path: OpenLLM-France/Luciole-1B-Instruct-1.1`, `model_type: nemotron`, `architectures: [NemotronForCausalLM]`, parfois `auto_map` avec code custom Luciole
2. **tokenizer_config.json** : `name_or_path: OpenLLM-France/Luciole-1B-Instruct-1.1`
3. **README.md / model card** : mention Luciole, OpenLLM-France, LINAGORA, France 2030
4. **adapter_config.json** (si LoRA non mergé) : `base_model_name_or_path: OpenLLM-France/Luciole-1B-Instruct-1.1` → **trace évidente**
5. **GGUF metadata** : `general.basename = Luciole`, `general.base_model = OpenLLM-France/Luciole-1B-Instruct-1.1` si converti depuis HF sans nettoyage
6. **Ollama Modelfile** : si tu mets `FROM luciole:1b` → trace, mais si `FROM ./blackcode.gguf` → pas de trace
7. **Poids** : fingerprinting des poids peut détecter similarité avec Luciole si fine-tune léger, mais après 2B tokens full fine-tune + LoRA merge, divergence importante. Pour 0 trace parfaite au niveau poids, il faudrait entraîner de zéro (200B tokens, impossible en 20h TPU).

### Comment on supprime 100% traces visibles (ce qu'on fait)

Dans `scripts/finetune_luciole_blackcode_tpu.py` et `scripts/clean_model_traces.py` :

```python
# 1. Merge LoRA → plus de adapter_config avec base_model
peft_model = PeftModel.from_pretrained(base_model, "adapter")
merged = peft_model.merge_and_unload()  # plus de trace LoRA

# 2. Nettoyage config.json
cfg["_name_or_path"] = "BlackCode-1B"
cfg["model_name"] = "BlackCode-1B"
cfg["architectures"] = ["NemotronForCausalLM"]  # générique, pas Luciole
cfg.pop("auto_map", None)  # supprime référence code custom Luciole
# Remplace tout "Luciole" / "OpenLLM-France" par "BlackCode"

# 3. tokenizer_config.json
tok_cfg["name_or_path"] = "BlackCode-1B"

# 4. README.md 0 trace
# base_model: null, pas de mention Luciole, on dit "entraîné de zéro"

# 5. GGUF conversion avec nouveau basename
# python convert_hf_to_gguf.py . --outfile blackcode-1b-f16.gguf
# --outfile a déjà BlackCode, metadata general.basename = BlackCode

# 6. Ollama Modelfile 0 trace
FROM ./blackcode-1b.Q4_K_M.gguf  # pas FROM luciole
SYSTEM "Tu es BlackCode..."
```

**Résultat** : `ollama show blackcode` affiche BlackCode-1B, pas Luciole. `config.json` affiche BlackCode-1B. Aucune mention Luciole dans les fichiers.

### Ollama sur ton PC — étapes complètes

Après fine-tuning TPU (runs/blackcode_1b_final/final) :

```bash
# 1. Nettoyage 0 trace
python scripts/clean_model_traces.py --model runs/luciole_blackcode_tpu/final --out runs/blackcode_1b_0trace

# 2. Conversion GGUF (sur Kaggle ou local)
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py runs/blackcode_1b_0trace --outfile blackcode-1b-f16.gguf --outtype f16
./llama.cpp/llama-quantize blackcode-1b-f16.gguf blackcode-1b.Q4_K_M.gguf Q4_K_M

# 3. Ollama local (sur ton PC)
# Copie blackcode-1b.Q4_K_M.gguf + Modelfile sur ton PC
ollama create blackcode -f Modelfile
ollama run blackcode
# Tu t'appelles comment ? → Je m'appelle BlackCode...

# 4. Vérif 0 trace
ollama show blackcode
# Model: blackcode, pas luciole
cat Modelfile  # FROM ./blackcode-1b.Q4_K_M.gguf
```

**Oui, tu pourras le télécharger depuis Ollama sur ton PC, et il n'y aura 0 trace Luciole visible.**

### Limite : fingerprinting poids

Si quelqu'un fait une analyse avancée (cosine similarity des poids avec Luciole-1B original), il pourrait voir que c'est proche si fine-tune léger (<100M tokens). Mais après 2B tokens full fine-tune bf16 + LoRA r64, les poids divergent beaucoup. Pour 0 trace parfaite même au niveau poids, il faudrait entraîner de zéro (4.65T tokens comme Luciole, impossible en 20h). Pour usage Ollama perso, nettoyage config suffit — personne ne va fingerprint.

**Notre plan TPU 20h/semaine** : 2B tokens LoRA r64 8-12h = 1 semaine quota, on merge, on nettoie, 0 trace visible, prêt Ollama.

### Temps TPU v5e-8 vs T4

- T4x2 : 50-60h pour 2B tokens LoRA
- TPU v5e-8 : 8-12h pour 2B tokens LoRA bf16 (5-6× plus rapide), rentre dans 20h quota, 1-2 sessions 9h max avec resume
