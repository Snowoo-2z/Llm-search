# Optimisations max — session continue

## Résumé des optimisations déjà faites + nouvelles

### Dataset

| Optimisation | Avant | Après | Gain |
|---|---|---|---|
| French detection lingua 5 ms/doc | 2,7h pour 2M docs | heuristique 0,1 ms/doc → 3 min | ×50 |
| Deduplication SimHash 0,5 ms/doc | 15 min + 10 Go RAM pour 2M | `--no-near` première passe → 40 min total, puis batch | ×2 + RAM ÷10 |
| Quality score | binaire seulement | `quality_score()` 0-1 pour tri + `by_reason` | permet seuil p10/p20 |
| Licence par ligne | dataset licence seulement | `record_license`, `collection`, `open_type`, `require-hf-license`, `allowed-hf-license` | indispensable Common Corpus |
| Tokenizer | 120M chars 15-30 min | 20M chars pour test rapide 5 min | ×4 pour itération |
| Manifest | pas de traçabilité | `manifest.jsonl` hash+path+licence+meta | traçabilité G du cahier |
| Disque | Drive lent 10-60s/écriture | `/content` local + tar + async mirror | ×5 |
| Parallélisme | 1 process | sharding par dump/collection, 2 builds // sur dual-T4 | ×1,8 |

**Script optimisé** : `scripts/build_fr_optimized.sh` : audit 500 lignes 5 min, build 4 Go 1h30, vérif pureté 2 min, copie Drive tar 10 min.

### Entraînement

| Optimisation | Gain | État |
|---|---|---|
| Adafactor | batch 4-8 vs 1, 3-4 Go VRAM pour 258M | implémenté |
| Checkpoint sélectif | +10-15% débit vs full | implémenté |
| torch.compile | +10-25% T4 | implémenté |
| Batch+grad-accum | MFU ↑ | `--batch 2 --grad-accum 8` → 32k tok/pas |
| Seq-len warmup | +0-10% | `--seq-len-warmup-frac 0.2 --seq-len-start 512` |
| WSD | évite perte session, +0-30% mural | `--schedule wsd --stable-frac 0.8` |
| Évals espacées | +5-10% | `--eval-interval 1000` |
| Mirror async | 10-60s / checkpoint | `--mirror-dir` |
| Pruning données | ×1,5-3 tokens | `blackcode.prune` |
| Muon | ×1,3-2 efficacité tokens | `--optim muon --muon-lr 0.02` |
| **DDP dual-T4** | **×1,8-2** | **NOUVEAU : implémenté dans `train.py` avec `--ddp` + `torchrun --nproc_per_node=2`** |

**DDP implémentation** :
- `_init_ddp()` détecte RANK/WORLD_SIZE ou --ddp
- Seed différent par rank pour diversité batches
- Modèle wrappé `DistributedDataParallel` après build, avant compile
- Logs, eval, checkpoints seulement rank 0, broadcast best_val
- `tokens_seen` global = somme world_size
- `torch.distributed.destroy_process_group()` à la fin

**Commande dual-T4 Kaggle** :
```bash
torchrun --nproc_per_node=2 -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_6b --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 183105 --optim muon --muon-lr 0.02 --lr 6e-4 --schedule wsd --stable-frac 0.8 --checkpoint-policy selective --compile --eval-interval 1000 --save-interval 1000 --resume --log-flops --ddp
```

### Temps finaux optimisés (6Md recommandé)

- Build 4 Go : 1h30
- Transfert Drive→Kaggle : 20 min
- Train 6 Md dual-T4 : 15h
- SFT 100 Mo : 2-3h
- **Total good+ : ~19-20h** (dans quota 30h)

### Prochaines optimisations possibles (pas encore faites)

1. **FastText français** au lieu de lingua : modèle 1 Mo, 0,05 ms/doc, 95% précision FR, vs lingua 5 ms.
2. **Bloom filter** pour exact dedup : RAM constante, pas de set SHA1 qui grossit.
3. **Tokenizers parallelism** : `TOKENIZERS_PARALLELISM=true` + `num_proc` pour BPE.
4. **FlashAttention** : T4 pas supporté FA2, mais xFormers memory-efficient attention → -30% VRAM attention.
5. **8-bit AdamW** : `bitsandbytes` → batch ×2.
6. **DiLoCo** : 2 sessions Colab indépendantes, sync toutes les 500 steps via Drive → ×2 sans DDP.
7. **DataLoader prefetch** : thread séparé pour `get_batch` → masque latence disque.

Ces 7 leviers pourraient encore gagner ×1,5-2 sur le total, mais demandent implémentation et test.


## H. Datasets RS avec abréviations (18 oct 2026)

Ajout 88milSMS (CC BY 4.0, 88k SMS, 422 users, 29k emojis, 5 Mo) + smslareunion (12k SMS, CC BY 4.0) pour apprendre `slt/bjr/stp/mdr/ptdr/tkt/pk/koi/bcp`.

- Filtre adapté : kind `sms`/`chat`/`tweet` → seuil 10 chars / 3 mots / alnum 0.15 vs 200 chars / 30 mots / 0.30 prose. Bonus abréviations dans `quality_score` ×1.15 si marqueurs RS présents.
- Implémentation : `iter_hf_docs` détecte `sms|88mil|comere → sms`, `chat|irc|getalp → chat`, `tweet|polititweet → tweet`, licence `cc-by-4.0` pour Ortolang.
- Mélange : 85% FineWeb2-HQ + 5% Common Corpus + 5% 88milSMS + 5% smslareunion (soit ~800 Mo RS pour 16 Go 6 Md → 5%). SFT 20% RS pour comprendre mais répondre formel.
- Impact temps : négligeable, 5 Mo → 100 Mo après tokenisation, build +2-3 min.
- Tests : 36 OK (33 + 3 SMS).
- Docs : `docs/27-DATASETS-ABREVIATIONS-RS-FR.md` et `docs/17-CAHIER-SOURCES-FR.md` section N.

Commande build avec RS :

```bash
# Build principal 4 Go + 100 Mo SMS
bash scripts/build_fr_optimized.sh --max-mb 4000 --vocab 24576 --seq-len 2048 --french-only --min-quality-score 0.6
python -m blackcode.data build --src data/fr_seed --hf comere/cmr-88milsms --hf-field text --french-only --max-mb 100 --out data/sms_fr --vocab 24576 --seq-len 2048
# Fusionner (cat manifest)
```
