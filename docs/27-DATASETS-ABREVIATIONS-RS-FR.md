# Datasets de discussions avec abréviations — réseaux sociaux français

**Date : 18 octobre 2026. Recherche web_search effectuée, fiches Ortolang/CNRTL consultées. Aucun de ces corpus n'a été téléchargé dans ce dépôt.**

## Pourquoi ces données pour BlackCode FR-260M ?

Le modèle doit parler **uniquement en français** mais comprendre le français **réel** des réseaux : SMS, tchat IRC, tweets, forums. Ces corpus contiennent les abréviations typiques :

- `slt`, `bjr`, `cc`, `stp`, `svp`, `dsl`, `tkt`, `mdr`, `ptdr`, `lol`, `jpp`, `pk`, `pq`, `koi`, `keske`, `c`, `ct`, `t`, `t'es`, `f`, `g`, `chui`, `chu`, `où`, `ou`, `bcp`, `bcq`, `b1`, `a+`, `a2m1`, `biz`, `bises`, etc.
- Exemple 88milSMS : `f = fais/fera/faisais (tu f koi ?)` [Panckhurst et al. 2020]

Objectif : que le modèle comprenne ces formes sans les produire systématiquement en réponse formelle (à régler en SFT).

## Corpus repérés et vérifiés

### 1. 88milSMS — corpus de SMS authentiques français

- **Fiche officielle** : https://hdl.handle.net/11403/comere/cmr-88milsms et http://88milsms.huma-num.fr/ et https://repository.ortolang.fr/api/content/comere/v3.3/cmr-88milsms.html
- **Taille** : 88 000 SMS bruts anonymisés (2011, Montpellier), **88 522 messages finalisés, 422 participants, 29 563 emoticons/emojis**, 5 Mo (ISLRN 024-713-187-947-8) [fiche CoMeRe v3.3]
- **Licence** : **CC BY 4.0 International** sur Ortolang (depuis 2016) [fiche v3.3 : http://creativecommons.org/licenses/by/4.0/] — compatible commercial avec attribution. Version ELRA-W0082 est NC non standard, mais version CoMeRe est BY 4.0.
- **Anonymisation** : semi-automatique avec CIL juriste, questionnaire sociolinguistique, 780 téléchargements 48 pays 2014-2019.
- **Qualité** : orthographe non standard, abréviations, fautes, emojis. 100 SMS annotés + 1000 transcodés en français standard disponibles.
- **Intérêt** : meilleure source française pour abréviations SMS, style informel réel.
- **Risques** : PII déjà anonymisé, mais vérifier anonymisation ; contenu sensible possible (insultes, etc.) ; pas de mineurs ciblés (adultes Montpellier).
- **Verdict** : **Candidate pré-entraînement / SFT** en proportion limitée (5-10% du mélange), pour apprendre abréviations. Garder brut + version transcodée si possible.

### 2. CoMeRe — Communication Médiée par les Réseaux (noyau)

- **Fiche** : https://hdl.handle.net/11403/comere et https://www.ortolang.fr/market/corpora/comere v3.3
- **Taille** : **14 sous-corpus, 75M tokens** (v3.3, 2014-2016), **74 325 542 mots** (Ortolang), mono-lingue français, dialogue, TEI UTF-8, 2001-2014
- **Licence** : **CC BY 4.0 International** pour le dépôt global (sauf indication contraire par sous-corpus)
- **Genres** : SMS, Tweets, Email, forums, textchat IRC, Wiki discussions, weblog, multimodal
- **Verdict** : **Méta-corpus**, à utiliser pour découvrir sous-corpus, pas comme un seul dataset. Chaque sous-corpus a sa propre licence.

### 3. cmr-smslareunion — SMS Réunion

- **Fiche** : https://hdl.handle.net/11403/comere/cmr-smslareunion
- **Taille** : **12 622 SMS finalisés, 884 participants, 21 694 SMS bruts collectés avril-juin 2008, 1 744 users**, SMS réunionnais avec créolismes
- **Licence** : **CC BY 4.0** [fiche : http://creativecommons.org/licenses/by/4.0/]
- **Intérêt** : variation régionale française, abréviations + créole
- **Risques** : PII anonymisé, mais vérifier
- **Verdict** : **Candidate** en petite proportion, complémentaire 88milSMS

### 4. cmr-getalp_org — tchat IRC français

- **Fiche** : https://hdl.handle.net/11403/comere/cmr-getalp_org
- **Taille** : **19 997 messages, 86 964 mots, 195 participants, 8 225 événements**, canal EpikNet IRC, 2004-02 à 2004-04, 4M messages dans 105 canaux hétérogènes
- **Licence** : **CC BY-NC-SA 4.0** [fiche getalp_org]
- **Intérêt** : tchat synchrone avec abréviations, commandes IRC, style très informel
- **Risques** : NC-SA → **écarté par défaut commercial**, possible recherche non commerciale
- **Verdict** : **En attente / évaluation seulement** si usage non commercial, sinon écarté

### 5. cmr-polititweets / cmr-intermittent — Tweets français

- **Fiche** : partie de CoMeRe, tweets politiques et intermittents
- **Taille** : 34 000 tweets (selon article CoMeRe 2014 : 4 genres dont tweets 34k)
- **Licence** : à vérifier par sous-corpus, souvent CC BY 4.0 mais tweets soumis aux conditions Twitter/X (actuellement non redistribuables en texte complet selon ToS)
- **Risques** : ToS Twitter/X interdit redistribution texte complet, PII, contenu sensible, mineurs possible
- **Verdict** : **En attente**, vérifier fiche exacte et ToS actuel, préférer SMS/tchat qui sont plus stables juridiquement

### 6. Autres SMS français repérés

- **1000 SMS avec labels spatiaux** (CIRAD Dataverse doi:10.18167/DVN1/0ZGJRC) : extrait 88milSMS, 1000 SMS, **CC BY-NC-SA 4.0** → NC = écarté commercial
- **30k French SMS traduit (Belgique, Fairon et al. 2006)** : 30k SMS belges avec traduction standard, CD-Rom 2006, licence à vérifier, contient profil 1 736 auteurs
- **Texto4Science / sms4science** : 7 000 SMS québécois, 10 mois collecte, 90k tokens, abréviations québécoises
- **SUMM-RE** (LINAGORA) : conversations réunion français, CC BY-SA 4.0, 15h+ — pas abréviations RS mais dialogue oral formel

## Tableau récapitulatif

| ID | Dataset | Taille | Licence | Abréviations | PII/Mineurs | Verdict |
|---|---|---|---|---|---|---|
| RS-01 | 88milSMS | 88k SMS, 422 users, 29k emojis, 5 Mo | CC BY 4.0 (CoMeRe) | **oui fort** (f, koi, etc.) | anonymisé, à vérifier | **Candidate 5-10%** |
| RS-02 | CoMeRe global | 75M tokens, 14 sous-corpus | CC BY 4.0 global | oui | selon sous-corpus | méta, pas direct |
| RS-03 | smslareunion | 12k SMS, 884 users | CC BY 4.0 | oui + créolismes | anonymisé | candidate petite prop |
| RS-04 | getalp_org IRC | 20k messages, 87k mots | CC BY-NC-SA 4.0 | oui fort | NC = écarté commercial | éval seulement si non commercial |
| RS-05 | polititweets | 34k tweets | à vérifier + ToS X | oui | ToS X interdit | en attente |
| RS-06 | 1000 SMS spatial | 1000 SMS | CC BY-NC-SA 4.0 | oui | NC | écarté commercial |

## Intégration dans pipeline BlackCode

### Filtrage qualité adapté aux abréviations

Le filtre actuel `quality_reasons` rejette `prose trop courte` si <30 mots, mais SMS avec abréviations font souvent <20 mots. Pour RS :

- Ne pas appliquer `prose trop courte` si kind=`sms` ou `chat` et chars>50
- Garder `alnum_ratio` bas toléré pour SMS (beaucoup d'emojis/abréviations)
- Conserver `trop d'urls` et `boilerplate` (spam)
- Ajouter lexique abréviations pour score qualité : si contient `mdr|ptdr|slt|bjr|stp|tkt|pk|koi` → bonus français informel

Implémentation proposée dans `blackcode/data.py` : nouveau kind `sms`/`chat` avec seuils adaptés.

### Mélange recommandé

- **Pré-entraînement** : 85% FineWeb2-HQ filtré + 5% Common Corpus FR filtré licence + 5% 88milSMS + 5% smslareunion + 0% NC
- **SFT** : 50% instructions formelles FR + 30% dialogues Claire AN (Licence Ouverte) + 20% SMS/tchat avec abréviations (88milSMS) pour apprendre à comprendre mais répondre formel (via consigne système "réponds en français standard, pas en SMS")

### Normalisation vs conservation

- **Conserver brut** pour pré-entraînement : le modèle doit voir `slt`, `bjr`, `mdr`
- **Transcodé** pour SFT : utiliser les 1000 SMS transcodés en français standard fournis avec 88milSMS pour apprendre mapping abréviation → standard
- Évaluer : tâche "traduis SMS abrégé en français standard" (éval seulement)

## Protocole audit

```bash
# Audit 88milSMS
python scripts/audit_fr_source.py --dataset comere/cmr-88milsms --limit 1000 --samples-out data/audits/88milsms.jsonl

# Build petit lot SMS
python -m blackcode.data build --src data/fr_seed --hf comere/cmr-88milsms --hf-field text --french-only --max-mb 100 --out data/sms_fr --vocab 24576 --seq-len 2048
python scripts/eval_french_purity.py --corpus-dir data/sms_fr --limit 50
```

Vérifier manuellement 20 extraits : PII, insultes, mineurs, abréviations.

## Sources

- 88milSMS : https://hdl.handle.net/11403/comere/cmr-88milsms (CC BY 4.0), http://88milsms.huma-num.fr/, https://www.sketchengine.eu/88milsms-french-text-message-corpus/ (CC-BY 4.0)
- CoMeRe : https://hdl.handle.net/11403/comere (CC BY 4.0), 75M tokens, 14 sous-corpus, https://www.ortolang.fr/market/corpora/comere
- smslareunion : https://hdl.handle.net/11403/comere/cmr-smslareunion (CC BY 4.0, 12 622 SMS)
- getalp_org : https://hdl.handle.net/11403/comere/cmr-getalp_org (CC BY-NC-SA 4.0, 19 997 messages)
- Tweets : CoMeRe article 2014 : 34k tweets, 44k SMS, 2,3k emails, 2,7k forums
- 1000 SMS spatial : https://dataverse.cirad.fr/dataset.xhtml?persistentId=doi:10.18167/DVN1/0ZGJRC (CC BY-NC-SA 4.0)
- Conventions : https://repository.ortolang.fr/api/content/tcof/12/TCOFConventions2017.pdf

