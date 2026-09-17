# 07 — Évaluation : benchmark, partie prenante synthétique, runner de lot, notation

Ce document couvre les phases 2 à 4 de `roadmap.md` : un jeu de données annoté
de cahiers des charges, une partie prenante synthétique qui répond à la place de
l'humain, un runner déterministe qui enchaîne des exécutions complètes du graphe
sans intervention, et un **scorer** qui confronte le résultat à la vérité
terrain.

La séparation est volontaire et structure tout le reste :

```text
run_benchmark.py  ->  predictions.json   (ce que le système a trouvé)
                            +
                      ground_truth.json  (ce qu'il aurait dû trouver)
                            |
                            v
run_scoring.py    ->  scores.json / scores.csv / scores.md
```

Le runner ne note rien, le scorer n'exécute rien. Une notation coûte quelques
secondes et aucun appel LLM : on peut donc rejouer un scorer amélioré sur un run
de deux heures déjà terminé, ce qui serait impossible si les deux étaient soudés.

---

## Trois harnais, trois niveaux

| | `evals/run_evals.py` | `evals/run_benchmark.py` | `evals/run_scoring.py` |
|---|---|---|---|
| Portée | un agent isolé (`run_gap_finder`, `run_critic`) | le graphe compilé, de bout en bout | un run déjà écrit sur disque |
| LangGraph | contourné | réel (checkpointer, interrupt, routage) | absent |
| Humain | absent (pas de boucle Q/R) | simulé | — |
| Données | `evals/datasets/*.jsonl` | `evals/datasets/benchmark/` | `evals/results/<run_id>/` + `ground_truth.json` |
| Usage | itérer vite sur un prompt | mesurer le système | noter la mesure |

Les deux premiers partagent `evals/harness.py`.

---

## Le benchmark

```
evals/datasets/benchmark/
    manifest.yaml                   # dataset_version + liste ordonnée des cas
    cdc_001_smartstock/
        initial_cdc.md              # le CDC vague donné au graphe
        source_docs/                # corpus RAG propre à ce cas (peut être vide)
        ground_truth.json           # annotations expertes
        stakeholder.yaml            # profil de la partie prenante synthétique
    ...
```

Dix cas, chacun ciblant un mode de défaillance distinct :

| Cas | Ce qu'il éprouve |
|---|---|
| `cdc_001_smartstock` | contradictions denses, aucun document — tout passe par la question |
| `cdc_002_ticketing` | cas RAG de référence : 11 lacunes sur 14 sont dans les annexes |
| `cdc_003_ecommerce` | CDC d'une page, entièrement en adjectifs |
| `cdc_004_rh_onboarding` | modèle de données incomplet, données personnelles |
| `cdc_005_iot_maintenance` | contrat d'interface : volumétrie, latence, formats |
| `cdc_006_portail_client` | sécurité et habilitations ; le CDC contredit la politique RSSI |
| `cdc_007_logistique_livraison` | 9 contradictions internes — test de rappel du critic |
| `cdc_008_facturation` | règles de gestion et cas limites chiffrés en annexe |
| `cdc_009_reservation_salles` | **témoin négatif** : CDC bien écrit, 3 lacunes seulement |
| `cdc_010_migration_erp` | lacunes qui ne « sonnent » pas vagues, périmètre ambigu |

Le témoin négatif compte autant que les autres : un système qui trouve autant de
lacunes sur `cdc_009` que sur `cdc_003` sur-signale, et c'est là que la précision
mesurée en phase 4 pèsera le plus.

### `ground_truth.json`

Modèles dans `evals/dataset.py`. Validé au chargement — une catégorie inventée,
un `gap_ref` orphelin ou un document cité qui n'existe pas font échouer
`load_benchmark()`, pas le run de deux heures.

```jsonc
{
  "case_id": "cdc_001_smartstock",     // doit égaler le nom du dossier
  "title": "...", "dataset_version": "v1", "annotator": "...",
  "notes": "pourquoi ce cas existe",

  "gaps": [{
    "id": "GT-GAP-001",
    "section_id": "functional",        // doit exister dans config/sections.yaml
    "category": "business_rule",       // littéraux de src/state.py
    "severity": "blocking",
    "description": "...",
    "keywords": ["seuil", "alerte"],   // sert au matching (voir plus bas)
    "resolvable_by": "rag",            // rag | human | neither
    "expected_evidence": {             // obligatoire si resolvable_by == "rag"
      "document": "annexe1.md",
      "page": null,
      "quote": "extrait littéral, vérifié par les tests"
    }
  }],

  "contradictions": [{
    "id": "GT-CON-001",
    "statement_a": "Budget estimé : 150 000 €",
    "statement_b": "Le budget ne devra pas dépasser 80 000 €",
    "section_ids": ["constraints"], "severity": "blocking",
    "keywords": ["budget"]
  }],

  "expected_answers": [{
    "gap_ref": "GT-GAP-001",           // -> gaps[].id
    "answer": "ce que répondrait une partie prenante parfaitement informée",
    "keywords": ["role", "modifier"]
  }]
}
```

**Règle structurante** : une lacune `resolvable_by: "rag"` **ne peut pas** avoir
d'`expected_answer`. Sinon l'oracle la refermerait alors même que la récupération
a échoué, et l'échec de récupération disparaîtrait derrière une réponse humaine
qui n'aurait jamais dû être nécessaire. Le loader refuse les deux à la fois.

**Les `keywords` ne sont pas décoratifs.** Les identifiants de lacunes produits
à l'exécution sont des hachages de contenu : impossible de les annoter à
l'avance. Le rapprochement entre une question posée et une lacune annotée se
fait donc par contenu, sur ces mots-clés (sous-chaînes, accents repliés). Des
mots-clés trop génériques font matcher n'importe quoi ; trop rares, plus rien.
Deux ou trois termes distinctifs par entrée.

### Ajouter un onzième cas

1. `evals/datasets/benchmark/cdc_011_xxx/` avec les quatre fichiers.
2. Ajouter l'entrée dans `manifest.yaml` et **incrémenter `dataset_version`** :
   il est inscrit dans chaque manifeste de run, et comparer deux runs sur des
   versions de jeu différentes n'a aucun sens.
3. `uv run pytest tests/test_benchmark_harness.py` — la validation du schéma,
   des sections, des `gap_ref` et des citations littérales tourne sur tout le jeu.

---

## La partie prenante synthétique

`human_input_node` bloque sur `interrupt({"questions": [{gap_id, text}, ...]})`
et reprend sur `Command(resume={gap_id: {"text": ..., "skip": ...}})`. Un
simulateur est simplement ce qui transforme le premier en second.

```python
class Simulator(Protocol):
    mode: str
    def answer(self, questions: list[dict], state: CDCState) -> AnswerBatch: ...
```

Renvoyer `skip=True` est exactement ce que fait le bouton « Je ne sais pas » de
l'interface : `integrate_answers_node` construit alors une hypothèse. Le chemin
« hypothèse » est donc éprouvé sans code particulier, et une réponse
contradictoire remonte au critic telle quelle.

### Trois modes

| `--mode` | Implémentation | LLM | Ce qu'on mesure |
|---|---|---|---|
| `oracle` *(défaut)* | `OracleSimulator` | non | le système intègre-t-il correctement une information **correcte** ? (roadmap §15 mode A) |
| `stakeholder` | `StakeholderSimulator`, comportement du profil | oui | interaction réaliste telle qu'annotée par cas |
| `realistic` | idem, comportement forcé | oui | tenue sous interaction dégradée (roadmap §15 mode B) |

**`oracle`** rapproche chaque question de la lacune annotée la plus proche et
rejoue son `expected_answer`. Aucune correspondance au-dessus du seuil → `skip`
avec `reason: "no_ground_truth"` : deviner ici créditerait le système d'une
information que le benchmark n'a jamais promise. Entièrement déterministe.

**`stakeholder`** interroge un LLM contraint au profil. Le prompt (`prompt_id:
"simulator.answer"`, versionné dans `src/prompts.py`) impose : *répondre
uniquement à partir des connaissances listées, sinon `knows=false`*. C'est la
contrainte critique du roadmap §14 — un simulateur libre d'inventer refermerait
des lacunes que les documents ne peuvent pas fermer, et gonflerait toutes les
métriques en aval.

**`realistic`** force en plus `style: vague`, un taux de « je ne sais pas » d'au
moins 20 %, et l'injection des `contradictions` du profil. C'est le mode qui
malmène réellement les mécanismes d'hypothèse et de détection d'incohérence.
Reproductible à `--seed` donné.

### `stakeholder.yaml`

```yaml
name: "Responsable logistique"
role: "Business Owner"

knowledge:          # SEULE source autorisée pour le simulateur LLM
  - "Seuls les responsables magasin peuvent modifier les quantités en stock."
unknown:            # sujets à refuser même s'ils semblent devinables
  - "SLA exact de réponse des fournisseurs"

behavior:
  style: precise            # precise | vague
  unknown_rate: 0.0         # probabilité de se dérober (RNG graine)
  allow_contradictions: false

contradictions:     # injectées seulement si allow_contradictions
  - "Finalement tous les employés peuvent modifier le stock."
```

Ce que le profil **ne sait pas** est aussi important que ce qu'il sait : sur un
cas RAG, tout ce qui figure dans `source_docs/` doit être dans `unknown`, sans
quoi le simulateur masque un échec de récupération.

---

## Le runner de lot

```bash
uv run python evals/run_benchmark.py                                # 10 cas, mode oracle
uv run python evals/run_benchmark.py --cases cdc_003_ecommerce
uv run python evals/run_benchmark.py --mode realistic --seed 7
uv run python evals/run_benchmark.py --provider anthropic --max-turns 8 --cache
uv run python evals/run_benchmark.py --resume                        # reprend le dernier run
```

Options : `--cases`, `--mode`, `--seed`, `--provider`, `--model`,
`--temperature`, `--top-k`, `--max-turns`, `--max-questions-per-batch`,
`--cache`, `--score`, `--quiet`, `--run-id`, `--out`, `--resume`, `--force`,
`--keep-checkpoints`.

Pour chaque cas, le runner envoie exactement ce que `src/app.py::start_run`
envoie (`initial_cdc_text`, `loop_settings`, `section_statuses`), consomme le
flux jusqu'à l'interrupt, fait répondre le simulateur, reprend, et recommence
jusqu'à `graph.get_state(config).next == ()`. Un cas qui échoue est enregistré
avec sa trace (`status: "error"`) et le lot continue.

### Suivre un run en direct

Un cas à froid passe plusieurs minutes entre deux événements visibles, donc le
runner se raconte au fil de l'eau plutôt que d'imprimer une ligne une fois le cas
fini : un nœud de graphe par ligne (tour, nœud, section courante, durée) suivi
des compteurs qui bougent — lacunes trouvées, lacunes encore ouvertes, questions
posées, sections complètes — et, à chaque tour de questions, la question posée
face à la réponse simulée avec son `reason`.

```text
    t 1 gap_filler         [problem]       1.2s  13 lacune(s), 13 ouverte(s), 0 question(s), 0/4 section(s)
    — tour de questions 1 : 3 question(s)
      Q Vous avez indiqué que le site doit améliorer l'expérience utilisateur…
      R [matched] Cible : +15 % de chiffre d'affaires en ligne sur douze mois…
    t 1 integrate_answers                  0.0s  13 lacune(s), 10 ouverte(s), 3 question(s), 0/4 section(s)
```

C'est là qu'on voit la boucle *fonctionner* : le nombre de lacunes ouvertes
descend à mesure que les réponses sont intégrées, et un `reason` à répétition
(`no_ground_truth`, `hedged`) explique tout de suite un cas qui n'avance pas.

Cette trace va sur **stderr** ; `stdout` ne porte que la ligne de résultat par
cas. `2>/dev/null` donne donc le résumé seul, et `1>/dev/null` la trace seule.
`--quiet` la coupe et rend le format d'origine, une ligne par cas.

`run_scoring.py` narre de même une ligne de score par cas pendant la notation —
utile surtout avec `--judge llm`, qui dépense un appel par question posée.

### Isolation par cas

Trois choses sont globales au processus et se marcheraient dessus d'un cas à
l'autre : `rag.source_dir` (ingéré sans argument par `ingest_node`),
`rag.persist_dir` + le `lru_cache` de `get_collection`, et `quarto.output_dir`
(où `synthesizer` et `final_validator` écrivent).

`evals/harness.py::isolate` installe un `Settings` de remplacement via
`src/config.py::set_settings_override` et vide les caches qui en dérivent
(`rag.get_collection`, `llm.get_llm`, `llm._get_structured_llm`). Hors du
harnais, l'override est nul et `load_settings()` lit `config/settings.yaml`
comme avant : rien ne change pour l'application Streamlit.

### Artefacts

```
evals/results/<run_id>/
    manifest.json                   # reproductibilité (roadmap §17) — écrit AVANT la boucle
    summary.csv                     # une ligne par cas — réécrit après CHAQUE cas
    report.md                       # synthèse lisible — réécrit après CHAQUE cas
    scores.json / .csv / .md        # notation (phase 4) — écrits par run_scoring.py
    <case_id>/
        predictions.json            # lacunes, contexte, questions, provenance, journal de décisions
        transcript.jsonl            # un objet par tour d'interrupt : questions + réponses simulées
        telemetry.json              # telemetry.summary() du cas, sommé sur les segments
        status.json                 # running | done | error | interrupted
        steps.jsonl                 # un objet par tour de graphe : nœud, durée, sync_s
        state.json                  # instantané lisible du CDCState, réécrit à chaque tour
        segments.json               # temps et télémétrie de chaque processus ayant traité le cas
        simulator.json              # état du simulateur (RNG, contradictions consommées)
        checkpoint/                 # état LangGraph, purgé quand le cas est terminé
        artifacts/                  # cdc_final.qmd, qa_report.md, decision_log.jsonl
```

Tout est écrit au fil de l'eau, jamais à la fin : le checkpoint après **chaque
nœud**, le transcript après chaque tour de questions, `summary.csv` et `report.md`
après chaque cas. Chaque écriture passe par `write_atomic` (fichier temporaire +
`os.replace`), donc une interruption ne laisse jamais un artefact tronqué.
`predictions.json` est le seul marqueur de fin — pas l'existence du répertoire,
créé avant que le cas ne démarre.

`manifest.json` enregistre le commit git (et s'il était sale), la version du jeu
de données, le fournisseur / modèle / température, le modèle d'embeddings, la
taille de chunk, `top_k`, l'intégralité de `PROMPT_VERSIONS`, l'empreinte de
`sections.yaml` et `settings.yaml`, les réglages de boucle, le mode et la graine
du simulateur. Deux runs ne sont comparables que si ces champs le permettent.

`summary.csv` est descriptif : tours, lacunes par sévérité, répartition
RAG / humain / hypothèse, questions posées, « je ne sais pas », sections
complètes, temps, appels LLM, cache. La répartition RAG / humain / hypothèse est
la matière première de la « réduction d'intervention humaine » du roadmap §26 —
le ratio lui-même est calculé par le scorer.

`predictions.json` embarque aussi le **journal de décisions** du cas. Ce n'est
pas une commodité : l'ordre de classement des extraits RAG ne survit nulle part
ailleurs. Une récupération *rejetée* ne laisse aucun `ContextItem` derrière elle,
seulement une entrée `rag_rejected` avec ses `evidence_ids` — sans elle, le
Recall@K ne serait mesuré que là où la récupération a marché. Les incohérences
transverses du validateur final n'existent, de même, que dans les `details` de
son entrée `final_check`.

`transcript.jsonl` porte, pour chaque réponse, un `reason` (`matched`,
`no_ground_truth`, `unknown`, `hedged`, `contradiction`, `simulator_error`) et le
`matched_gap_ref` : un résultat surprenant se remonte au simulateur ou à l'agent
sans ambiguïté.

`evals/results/` est ignoré par git ; les jeux de données, eux, sont versionnés.

### Reprendre un run interrompu

Un lot complet, c'est dix cas de plusieurs minutes : un Ctrl-C, une coupure
réseau ou une machine qui redémarre ne doit pas tout coûter.

```bash
uv run python evals/run_benchmark.py --resume            # le run le plus récent sous --out
uv run python evals/run_benchmark.py --resume bench_20260805_084151
```

Deux granularités se combinent :

- **Par cas.** Un cas dont le `predictions.json` existe est ignoré et sa ligne de
  `summary.csv` est reconstruite depuis le disque (`load_case_row` réutilise
  `summarize`, donc la ligne est identique à celle d'un run d'une traite). Par
  convention, un cas terminé en `status: "error"` compte comme fait : il est
  rapporté tel quel, pas rejoué.
- **Par tour de graphe.** Le cas en cours, lui, repart de son dernier nœud
  terminé. `evals/checkpoints.py` rend le checkpointer LangGraph durable **sans
  dépendance supplémentaire** : `InMemorySaver` garde tout dans trois dicts, et
  `PersistentDict` (livré avec `langgraph-checkpoint`) est un `defaultdict` qui se
  sérialise atomiquement sur `sync()`. Le `thread_id` est devenu déterministe
  (`bench-<case_id>`) pour qu'un second processus s'adresse au même fil.

Ce qu'il faut savoir en touchant à ça :

- `--resume` refuse de mélanger deux configurations (jeu de données, mode et
  graine du simulateur, fournisseur, modèle) — sans quoi `summary.csv` ne voudrait
  plus rien dire. `--force` passe outre. Le `manifest.json` d'origine est conservé
  et gagne un tableau `resumed_at`.
- Sans `--resume`, réutiliser un `--run-id` **repart de zéro** : `CaseRecorder`
  purge l'état du cas, sinon on reprendrait un ancien checkpoint sans l'avoir
  demandé.
- Le simulateur `stakeholder` / `realistic` est à état (RNG, contradictions
  consommées) : il est sauvegardé après chaque tour, sinon une reprise rejouerait
  des tirages déjà dépensés. L'oracle, lui, est une fonction pure.
- `telemetry.py` est global au processus et remis à zéro par cas ; les segments
  sont donc sommés (`merge_telemetry`) pour que `llm_calls` / `llm_s` / `retries`
  / `cache_hits` restent justes après une reprise. La colonne `segments` de
  `summary.csv` vaut 1 pour un cas d'une traite.
- Le colonne `sync_s` de `steps.jsonl` mesure le coût de la persistance : tout
  l'historique de checkpoints est re-sérialisé à chaque tour, donc c'est le
  chiffre à regarder si un run devient lent.

---

## Coût et reproductibilité

Dix exécutions complètes ne sont pas gratuites : un run à froid passe plus de
cinq minutes en chauffe LLM avant la première question, et le mode `realistic`
ajoute un appel de simulateur par question. En pratique :

- `--cases <un_cas>` pour itérer sur le harnais ;
- `--cache` (`CDC_LLM_CACHE=1`) pour rejouer un run en quelques secondes — les
  identifiants étant des hachages de contenu (`src/ids.py::stable_id`), les
  prompts se répètent à l'identique ;
- `oracle` par défaut, qui ne consomme aucun appel LLM côté simulateur ;
- `--resume` après une interruption, plutôt que de relancer le lot.

Deux runs `oracle` consécutifs avec le cache produisent les mêmes identifiants
de lacunes.

---

## La notation (phase 4)

```bash
uv run python evals/run_scoring.py                          # le run le plus récent
uv run python evals/run_scoring.py --run bench_20260805_130607
uv run python evals/run_scoring.py --cases cdc_003_ecommerce
uv run python evals/run_scoring.py --judge llm              # + juge LLM des questions
uv run python evals/run_benchmark.py --cases cdc_003_ecommerce --score   # enchaîné
```

Le calcul vit dans `evals/scoring.py` (fonctions pures sur des dictionnaires),
les entrées / sorties et le rapport dans `evals/run_scoring.py`. Le scorer
**n'écrit jamais** dans ce qu'il lit : `predictions.json`, les transcripts et les
checkpoints sont des entrées, et relancer la notation deux fois produit le même
fichier.

### L'appariement, et pourquoi il est par contenu

Les identifiants de lacunes sont des hachages de contenu calculés pendant le run
(`src/ids.py::stable_id`) : une annotation ne peut pas les nommer à l'avance. Une
lacune prédite est appariée à une lacune annotée quand une part suffisante des
`keywords` de l'annotation se retrouve dans sa description, sa catégorie, ses
sections et sa question — seuil **0,5**, via `keyword_score` de
`evals/simulator.py`, la fonction que l'oracle utilise déjà pour savoir à quelle
lacune il répond. Les deux ne peuvent donc pas diverger.

L'appariement est **un pour un** et glouton (meilleur score d'abord, égalités
départagées par les identifiants) : deux lacunes prédites ne peuvent pas réclamer
la même annotation, et un rejeu apparie à l'identique. Appartenir à la bonne
section ajoute un bonus de classement, mais n'est jamais un filtre : une lacune
bien décrite mais rangée dans la mauvaise section reste une détection.

### Ce qui est mesuré

| Famille | Contenu | Roadmap |
|---|---|---|
| `gaps` | P / R / F1 micro, par catégorie et par sévérité, matrice de confusion des sévérités, **blocking-gap recall**, accord de catégorie | §7, §8 |
| `contradictions` | P / R / F1, rappel des contradictions critiques | §9 |
| `retrieval` | Recall@1/3/5, MRR, séparation « problème de récupération » / « problème de raisonnement » | §10 |
| `questions` | six dimensions notées 0 / 1 / 2, questions par lacune résolue | §11 |
| `effort` | réduction d'intervention humaine, questions / tours / temps par cas | §26 |
| `completeness` | couverture des lacunes annotées, score de qualité initial → final | §12 |

### Les choix qui font les chiffres

Quatre décisions méthodologiques comptent plus que le code :

**La précision est stricte.** Une lacune détectée qui n'apparie aucune annotation
compte comme faux positif, même si c'est une vraie lacune que l'annotateur n'a
pas écrite. La précision affichée est donc une **borne inférieure**, et
`scores.md` liste intégralement ces prédictions non appariées : on les relit
avant de croire un chiffre bas, plutôt que d'abaisser le seuil jusqu'à ce qu'il
soit joli.

**P / R / F1 par classe est strict, `found` ne l'est pas.** Une lacune appariée
mais mal étiquetée est comptée comme manquée pour la classe annotée *et* comme
faux positif pour la classe choisie — sinon la précision par classe mélangerait
deux populations. La colonne `found` ignore l'étiquette et ne demande que « la
lacune a-t-elle été remontée ? ». Le **blocking-gap recall**, métrique de sûreté
du §8, est délibérément de ce second type : ne jamais mentionner une lacune
bloquante est le danger, sous-estimer sa sévérité est un défaut plus doux, que la
matrice de confusion rapporte à part.

**Toutes les contradictions ne sont pas comparables.** `ground_truth.json`
annote des contradictions **internes au CDC** (`statement_a` et `statement_b` en
sont deux citations). Le critic, lui, détecte « cette nouvelle réponse contredit
celle d'il y a trois tours » — une population dont le benchmark ne dit rien. Une
trouvaille du critic peut donc *apparier* une annotation (elle compte comme vrai
positif), mais une trouvaille non appariée n'est **pas** un faux positif : la
compter ainsi afficherait une précision quasi nulle sur tout cas où la partie
prenante synthétique se contredit, ce que le mode `realistic` provoque
exprès. Ces trouvailles sont reportées sous `answer_level`. L'attribution se lit
dans le journal de décisions (`gap_detected` = lecture du CDC, `contradiction_found`
= critic, `final_check` = passe finale).

**Le RAG a deux dénominateurs.** `recall_at_k` porte sur les récupérations
réellement tentées : c'est le chiffre qui dit si l'index et les embeddings
fonctionnent. `recall_at_k_overall` porte sur toutes les lacunes annotées
`resolvable_by: "rag"`, une lacune jamais détectée comptant comme un échec :
c'est ce que l'utilisateur constate, document non lu. Un extrait est pertinent
quand son `chunk_id` commence par le document attendu (et contient `::p{page}::`
si l'annotation fixe une page), d'où le format d'identifiant de
`src/rag.py::_chunk_id`. Le bloc `sufficiency_judgment` tranche ensuite le §10 :
l'extrait attendu est-il remonté puis accepté, remonté puis rejeté par le
correcteur (**problème de raisonnement**), ou jamais remonté (**problème de
récupération**) ?

### Qualité des questions : déterministe d'abord

Les six dimensions du §11 sont notées par heuristiques : recouvrement de mots-clés
avec la lacune (`addresses_gap`), présence d'une citation ou d'un chiffre
(`specific`), longueur et unicité du point d'interrogation (`understandable`),
échos du texte du CDC (`has_context`), similarité de Jaccard avec les questions
déjà posées (`not_duplicate`), et — lu directement dans le transcript — la partie
prenante a-t-elle pu répondre (`answerable`). Rien de tout cela ne coûte un appel
LLM, et deux notations du même run donnent le même chiffre.

`--judge llm` ajoute un juge LLM sur la même grille, sous le `prompt_id`
`judge.question_quality`. Il est rapporté **à côté** des heuristiques, sous
`llm_judge`, jamais fondu dedans : le roadmap §23 sépare le jugement de l'IA de
la validation déterministe, et un modèle qui note le système partageant son
propre modèle n'est pas une preuve autonome. Une panne du juge coûte un verdict,
pas la notation.

`has_context` mérite une note : `ingest` n'étiquette que les morceaux qu'il sait
rattacher à une section, si bien que le titre et le chapeau du CDC — souvent le
texte le plus citable d'un CDC d'une page — n'en portent aucune. Le pool de
comparaison inclut donc toujours ces morceaux non étiquetés, et retombe sur le
CDC entier quand la section visée n'a rien à citer.

### §12 : un proxy, et il est présenté comme tel

Le roadmap §12 demande qu'un expert humain note le CDC initial et le CDC final
sur huit dimensions. Le benchmark ne porte volontairement aucun document final
de référence (note de la phase 2 : ce serait la prose d'un annotateur, pas une
vérité terrain). Ce qui est calculé est donc un proxy déterministe :

- `gt_gap_coverage` — le chiffre honnête de bout en bout : parmi les lacunes
  qu'un humain a annotées, combien le système a-t-il à la fois **trouvées et
  refermées** (par RAG, réponse, hypothèse ou déduplication) ?
- `quality_score_initial` → `final` — `src/quality.py::score_section` appliqué
  deux fois par section : une fois en comptant toutes les lacunes détectées comme
  ouvertes, une fois en ne comptant que celles encore ouvertes à la fin. L'écart
  mesure le poids de défauts retiré, pas le jugement d'un humain sur le document.

### `scores.json` et la phase 6

`scores.json` recopie l'identité du run (run_id, version du jeu de données,
commit git, fournisseur / modèle, mode et graine du simulateur, `PROMPT_VERSIONS`)
à côté de `scorer_version` et du seuil d'appariement. Deux fichiers se comparent
donc sans retourner chercher leurs manifests — c'est le point d'accroche des
rapports de régression de la phase 6. `scorer_version` se bump dès qu'un même run
produirait des chiffres différents.

---

## Le système de référence (« baseline »)

Le benchmark dit à quel point le graphe s'en sort. Il ne dit pas si **la
machinerie vaut ce qu'elle coûte** : un F1 de lacunes à 0,55 est bon ou mauvais
selon ce qu'un unique appel LLM aurait obtenu sur les mêmes CDC. `evals/baseline.py`
est cet unique appel, branché sur le même jeu de données, la même partie prenante
synthétique et le même scorer.

```text
run_baseline.py   ->  evals/results/base_<ts>/   (même arborescence qu'un run de benchmark)
run_benchmark.py  ->  evals/results/bench_<ts>/
        |                      |
        +------ run_scoring.py (inchangé) ------+
                               |
                               v
                       compare_runs.py  ->  comparison.md
```

### Ce que fait la baseline

1. **Une passe.** Le CDC entier et la liste des sections partent dans un seul
   `call_structured`, qui renvoie toutes les lacunes qu'il voit, chacune avec sa
   question déjà rédigée. Pas de sections, pas de tours, pas d'orchestrateur.
2. **Une récupération mesurée mais non exploitée.** Une requête top-k par lacune,
   journalisée avec ses rangs — Recall@K est donc mesuré sur le même index que
   pour le graphe — mais la lacune part quand même en question. Décider qu'un
   extrait *répond* à une lacune, c'est l'appel de notation de `gap_filler` ; une
   baseline qui le ferait aussi mesurerait un composant du graphe au lieu de lui
   servir de plancher. Le prix de ne pas le faire est justement le chiffre que
   cela rend visible : chaque question que le RAG aurait pu éviter est posée.
3. **Un seul lot de questions.** Toutes les lacunes ouvertes d'un coup, sans
   déduplication ni budget.
4. **Une intégration naïve.** Une réponse referme sa lacune, un « je ne sais pas »
   devient une hypothèse. Rien n'est confronté à rien : aucune contradiction ne
   peut être trouvée après la première passe.

`--rag-closes-gaps` active l'autre option naïve — faire confiance au score de
similarité au-dessus d'un plancher. Sur ce jeu de données ce n'est pas une vraie
alternative, et le run le montre : les scores se tiennent dans une bande étroite
(~0,55–0,65 sur le cas ecommerce), donc n'importe quel plancher referme toutes
les lacunes ou aucune. C'est en soi un résultat : la similarité seule ne porte
aucun signal sur « cet extrait répond-il à cette lacune ? ».

### Ce qu'elle partage avec le graphe, volontairement

`call_structured`, la configuration fournisseur/modèle, l'index RAG du cas, et
les mêmes enregistrements `Gap` / `ContextItem` / `DecisionLogEntry` avec les
mêmes ids à hash de contenu. La comparaison isole donc **l'architecture**, pas le
modèle ni la plomberie. Son prompt est enregistré sous `baseline.oneshot` dans
`src/prompts.py` : c'est un plancher, résister à l'envie d'en améliorer la
formulation pour la rendre compétitive.

Une seule différence de forme mérite d'être connue : la baseline ne découpe pas
le CDC par section, elle le dépose en un bloc non étiqueté. `scoring._InitialCdc`
lit alors le document entier comme pool de contexte pour `has_context`, lecture
légèrement **généreuse** — le biais joue contre le système testé, pas pour lui.

```bash
uv run python evals/run_baseline.py --score          # les dix cas, puis notation
uv run python evals/run_baseline.py --cases cdc_003_ecommerce --cache
uv run python evals/run_baseline.py --no-rag         # sans récupération du tout
```

Elle est très bon marché comparée au benchmark — un appel LLM par cas plus ceux
de la partie prenante, contre des centaines pour le graphe — donc pas de
checkpoint ni de reprise en cours de cas ; `--resume` ne fait que sauter les cas
déjà écrits.

### Comparer deux runs notés

```bash
uv run python evals/compare_runs.py base_20260914_101500 bench_20260914_120000
```

Lit les deux `scores.json` et écrit `comparison.md` à côté du second. Rien n'y est
recalculé : un écart n'est jamais que la différence de deux nombres que le scorer
a déjà produits, ce qui garde la comparaison honnête quand le scorer change — on
renote les deux runs et l'écart bouge des deux côtés à la fois.

Le premier run est la **référence** (la baseline), le second le **candidat** (le
système testé) : un écart positif veut donc dire que le candidat est devant, sauf
sur les métriques marquées `↓` où moins vaut mieux. La ligne de verdict compte
les victoires sur quatre métriques seulement (F1 des lacunes, rappel des lacunes
bloquantes, couverture de la vérité terrain, qualité des questions) plutôt que de
moyenner des pourcentages avec des notes sur 2, ce qui produirait un chiffre
précis et vide de sens.

L'outil **refuse** une paire non comparable — versions de jeu de données, modes
de simulateur, modèles ou versions de scorer différents — parce qu'un écart entre
ces deux-là ne mesurerait rien. `--force` l'affiche quand même, avec les
divergences listées au-dessus du tableau.
