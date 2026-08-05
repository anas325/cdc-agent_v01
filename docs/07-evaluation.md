# 07 — Évaluation : benchmark, partie prenante synthétique, runner de lot

Ce document couvre les phases 2 et 3 de `roadmap.md` : un jeu de données annoté
de cahiers des charges, une partie prenante synthétique qui répond à la place de
l'humain, et un runner déterministe qui enchaîne des exécutions complètes du
graphe sans intervention.

Ce que ce dispositif produit : des **prédictions** et des **statistiques
descriptives**. Ce qu'il ne produit pas encore : precision / recall / F1,
blocking-gap recall, Recall@K du RAG, score de qualité des questions — c'est la
phase 4, qui consommera `predictions.json` et `ground_truth.json` sans que le
jeu de données ait à changer.

---

## Deux harnais, deux niveaux

| | `evals/run_evals.py` | `evals/run_benchmark.py` |
|---|---|---|
| Portée | un agent isolé (`run_gap_finder`, `run_critic`) | le graphe compilé, de bout en bout |
| LangGraph | contourné | réel (checkpointer, interrupt, routage) |
| Humain | absent (pas de boucle Q/R) | simulé |
| Données | `evals/datasets/*.jsonl` | `evals/datasets/benchmark/` |
| Usage | itérer vite sur un prompt | mesurer le système |

Les deux partagent `evals/harness.py`.

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
`--cache`, `--run-id`, `--out`, `--resume`, `--force`, `--keep-checkpoints`.

Pour chaque cas, le runner envoie exactement ce que `src/app.py::start_run`
envoie (`initial_cdc_text`, `loop_settings`, `section_statuses`), consomme le
flux jusqu'à l'interrupt, fait répondre le simulateur, reprend, et recommence
jusqu'à `graph.get_state(config).next == ()`. Un cas qui échoue est enregistré
avec sa trace (`status: "error"`) et le lot continue.

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
    <case_id>/
        predictions.json            # lacunes, contexte, questions, provenance — écrit EN DERNIER
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
le ratio lui-même sera calculé en phase 4.

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

## Ce que la phase 4 branchera dessus

Un scorer lisant `predictions.json` + `ground_truth.json`, sans toucher au jeu de
données : precision / recall / F1 par catégorie de lacune, matrice de confusion
des sévérités et **blocking-gap recall**, precision / recall des contradictions,
Recall@K et MRR du RAG (via `expected_evidence` face aux `evidence` des
`ContextItem` de source `rag`), qualité des questions, et réduction
d'intervention humaine à partir des colonnes de `summary.csv`.
