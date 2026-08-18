"""Prompts du gap-finder — v2 (calibrés sur deux CDC Centrale Danone acceptés).

Problème traité
---------------
La notion de « bloquant » de l'agent dérive de celle du métier. Deux documents de
référence effectivement acceptés (SFD Fiabilisation des Stocks, CDC Digitalisation
alimentation ferme Lait Plus) ne contiennent NI prix, NI planning, NI plan de recette,
NI stack technique, NI hébergement, NI modèle de données, NI contrat d'API, NI la
moindre exigence non fonctionnelle chiffrée. Ils contiennent en revanche de vraies
ambiguïtés bloquantes à l'intérieur des exigences qu'ils engagent. Le discriminant
n'est donc pas « cette rubrique est-elle présente ? » mais « le texte engagé
admet-il deux implémentations ? ».

Évolutions vs v1
----------------
1. SEVERITY_RUBRIC   — un test de décision explicite plutôt qu'un simple enum.
2. SCOPE_GUARD       — rubriques habituellement et légitimement absentes : pas de lacune.
3. GAP_QUALITY_RULES — forme obligatoire : citation + deux lectures + une question fermée.
4. FEWSHOT_SECTION   — 5 exemples positifs + 6 négatifs, tirés des deux documents.
5. FEWSHOT_FRESH     — 5 exemples : résolu / partiel / renvoi / hypothèse contradictoire /
                       baisse de sévérité.
6. SELF_CHECK        — un filtre final avant émission, plus un plafond du nombre de lacunes.

Les libellés de sévérité employés ici sont ceux de `GapSeverity` (src/state.py) :
"blocking" / "important" / "nice_to_have". `FEWSHOT_ENABLED` permet d'A/B tester les
few-shots dans evals/run_benchmark.py sans toucher aux appelants.
"""

from __future__ import annotations

from src.context_utils import (
    format_all_sections_context,
    format_open_gaps,
    get_section,
)
from src.state import CDCState

FEWSHOT_ENABLED = True

GAP_TAXONOMY = [
    "functional_ambiguity",
    "nfr",
    "data_model",
    "business_rule",
    "edge_case",
    "integration",
    "acceptance_criteria",
    "contradiction",
    "scope",
]


# --------------------------------------------------------------------------------------
# Blocs partagés
# --------------------------------------------------------------------------------------

SEVERITY_RUBRIC = """RÈGLE DE SÉVÉRITÉ (test de décision, à appliquer littéralement) :
Question de référence : « un développeur qui lit UNIQUEMENT ce CDC peut-il coder cette
fonctionnalité et avoir raison ? »
- "blocking" : il existe au moins DEUX lectures du texte menant à DEUX implémentations
  incompatibles (formule de calcul, clé d'enregistrement, périmètre d'agrégation, source de
  vérité d'une donnée). Le choix ne peut pas être tranché par l'équipe technique seule, et
  une erreur de choix est silencieuse (le système fonctionne mais affiche un faux chiffre).
- "important" : l'implémentation est possible avec une hypothèse raisonnable, mais cette
  hypothèse change un comportement visible par l'utilisateur (règle de conflit, automatique
  vs manuel, droits d'accès) et doit être confirmée par le métier.
- "nice_to_have" : imprécision de vocabulaire, redondance ou incohérence de libellé qui
  n'empêche ni le chiffrage ni le développement.
Ne classe JAMAIS en "blocking" l'absence d'une rubrique entière (cf. HORS PÉRIMÈTRE)."""


SCOPE_GUARD = """HORS PÉRIMÈTRE — n'ouvre PAS de lacune sur ces sujets, sauf si le document
prétend explicitement les traiter et se contredit alors lui-même :
- Prix, budget, conditions commerciales (typiquement renvoyés au service achats).
- Planning, jalons, plan de recette/tests, critères d'acceptation contractuels.
- Stack technique, hébergement, infrastructure : « une plateforme web au Cloud » suffit,
  c'est la réponse du fournisseur, pas une lacune du CDC.
- Exigences non fonctionnelles chiffrées (volumétrie, temps de réponse, utilisateurs
  simultanés, durée de rétention) lorsque l'échelle décrite est manifestement faible
  (une ferme, trois engins, quelques centaines d'articles).
- Détail d'IHM / maquettage lorsque le document renvoie à une capture d'écran ou à l'existant.
- Modèle de données, contrats d'API, format des messages échangés : relèvent de la phase de
  conception, SAUF si le CDC engage déjà une règle de calcul dont le résultat en dépend.
Principe : une rubrique absente n'est pas une lacune. Une règle promise puis laissée
indéterminée en est une."""


GAP_QUALITY_RULES = """FORME OBLIGATOIRE DE CHAQUE LACUNE (champ description) :
1. Cite entre guillemets le passage exact du document qui pose problème (une phrase maximum).
2. Énonce les DEUX lectures possibles, chacune avec sa conséquence concrète sur le code ou
   sur le chiffre affiché.
3. Termine par UNE question fermée, répondable en une phrase par un responsable métier.
Interdits : reformuler un critère de la checklist en question (« les rôles ne sont pas
détaillés »), demander « davantage de précisions », produire une lacune sans citation,
ou fusionner plusieurs problèmes dans une seule lacune."""


SELF_CHECK = """AVANT DE RÉPONDRE, filtre ta propre liste. Pour chaque lacune candidate :
(a) puis-je citer le passage exact ? sinon → supprime-la ;
(b) puis-je formuler deux lectures qui produisent deux implémentations différentes ?
    sinon → sévérité "nice_to_have" au maximum, ou suppression ;
(c) la question est-elle répondable en une phrase par un métier ? sinon → reformule ;
(d) le sujet figure-t-il dans HORS PÉRIMÈTRE ? si oui → supprime-la.
Maximum 5 lacunes par section, triées par sévérité décroissante. Deux lacunes citées et
tranchantes valent mieux que huit lacunes génériques."""


# --------------------------------------------------------------------------------------
# Few-shots — mode section
# --------------------------------------------------------------------------------------

FEWSHOT_SECTION = """EXEMPLES DE LACUNES À REMONTER (issues de CDC réels du même contexte
industriel) :

[1] Terme employé dans une formule mais jamais défini — sévérité : blocking
Extrait : « Mettre une colonne "Total Stock PLTF" qui est égale à la somme des 4 colonnes :
Stock PLTF + Encours + Bloqué + 40N ».
Lacune : « "40N" n'est défini nulle part dans le document et n'apparaît dans aucune autre
liste de colonnes ; la section annonce 5 colonnes mais la somme n'en agrège que 4.
Lecture A : 40N désigne la colonne "Quarantaine" créée juste au-dessus, le total agrège alors
les 5 états. Lecture B : 40N est un emplacement de stock distinct non listé, et la colonne
Quarantaine reste exclue du total. Les deux produisent un total différent sans erreur visible.
Question : "40N" correspond-il à l'emplacement de la colonne Quarantaine, ou à un cinquième
état à récupérer séparément ? »

[2] Deux règles d'affectation qui se contredisent dans le même paragraphe — blocking
Extrait : « Le stock du dépôt 103 (voie express) doit être afficher dans le stock usine
d'El-Jadida (UJ6). Le dépôt 103 est considéré comme une extension du dépôt UJ6. » puis
« le dépôt 103 doit être regrouper avec UJ5 et UJ6 ».
Lacune : « Lecture A : 103 est agrégé au seul UJ6, les encours d'El-Jadida ne remontent que
sur UHT. Lecture B : 103 est agrégé au couple UJ5+UJ6, ce qui change le périmètre de la
couverture et crée un risque de double comptage si UJ6 le contient déjà.
Question : le stock du dépôt 103 doit-il être compté une seule fois sur UJ6, ou réparti /
dupliqué sur UJ5 et UJ6 ? »
(catégorie "contradiction", section_ids = toutes les sections citant le dépôt 103)

[3] Granularité incohérente entre deux sections — blocking, incohérence transversale
Extraits : « Les refus seront saisis par lot par l'opérateur de chargement » (§ suivi des
stocks) et « Saisie des refus date J-1 par ration pour chaque lot » (§ processus utilisateurs).
Lacune : « La clé d'enregistrement du refus diffère entre les deux sections : par lot, ou par
ration à l'intérieur d'un lot. Elle détermine le modèle de données et le calcul du KPI "taux
de refus par lot". Question : un refus est-il pesé une fois par lot et par jour, ou une fois
par ration ? »
(section_ids doit lister LES DEUX sections)

[4] Comportement engagé mais règle de résolution absente — important
Extrait : « Mode hors ligne : en cas de perte de connexion avec la balance, le système doit
sauvegarder toutes les données et permettre à l'utilisateur de saisir ou valider manuellement
les quantités d'ingrédients. »
Lacune : « Le comportement à la reconnexion n'est pas défini. Lecture A : la saisie manuelle
fait foi et écrase les pesées bufferisées. Lecture B : les pesées de la balance font foi et
la saisie manuelle devient un écart à justifier. Question : à la reconnexion, quelle valeur
est retenue comme officielle, et l'écart doit-il être tracé ? »

[5] Automatique vs manuel dans la même exigence — important
Extraits : « Ecran spécifique pour l'opérateur de chargement permettra l'enregistrement
automatique des refus par lot » et « Les refus seront saisis par lot par l'opérateur ... avec
détection automatique du poids ».
Lacune : « Lecture A : l'écran pré-remplit le poids détecté et l'opérateur valide. Lecture B :
l'opérateur saisit la valeur, le poids détecté servant de contrôle. Le parcours et le droit de
correction diffèrent. Question : l'opérateur peut-il modifier le poids détecté par la balance,
et si oui la modification doit-elle être justifiée ? »

EXEMPLES À NE PAS REMONTER (pièges fréquents — ces documents ont été acceptés tels quels) :

[N1] « Le prix doit être communiqué après approbation de l'offre par le service des achats. »
→ renvoi explicite et assumé. Aucune lacune, aucune sévérité.
[N2] Aucune exigence de temps de réponse, de volumétrie ni d'utilisateurs simultanés dans tout
le document. → hors périmètre à cette échelle. Aucune lacune (et surtout pas "blocking").
[N3] « Alerte en cas d'écart de dosage par ingrédient dépassant le seuil de 2 % (seuil
modifiable). » → seuil chiffré, caractère paramétrable indiqué, profil habilité défini
ailleurs. Critère satisfait : ne demande pas de « préciser les seuils ».
[N4] « L'application doit être reliée à une plateforme web au Cloud. » → choix d'architecture
laissé au fournisseur. Pas de lacune, sauf contrainte de localisation des données évoquée
ailleurs dans le document.
[N5] La mise en page d'un écran n'est décrite que par une capture d'écran annotée. →
acceptable. À ne remonter que si la capture est la seule source d'une règle de calcul.
[N6] « DINAMICA GENERALE DG400 / PERIN FARM SCALE 800. Si ces modèles ne répondent pas aux
exigences, le fournisseur devra proposer un modèle compatible. » → matériel nommé + porte de
sortie explicite. Exigence complète."""


# --------------------------------------------------------------------------------------
# Few-shots — mode fresh
# --------------------------------------------------------------------------------------

FEWSHOT_FRESH = """EXEMPLES D'ÉVALUATION D'UN NOUVEL ÉLÉMENT :

[R1] RÉSOUT COMPLÈTEMENT
Lacune : granularité de saisie des refus (par lot ou par ration ?).
Nouvel élément (USER_ANSWER) : « Le refus est pesé une seule fois par lot et par jour ; la
ration ne sert qu'à calculer la quantité attendue. »
→ Les deux lectures sont tranchées, la clé d'enregistrement est explicite.
resolved_gap_ids += [gap_id]. Aucun gap de suivi.

[R2] RÉSOUT PARTIELLEMENT → gap de suivi
Lacune : « 40N » non défini dans la formule du Total Stock PLTF.
Nouvel élément (USER_ANSWER) : « 40N c'est le stock en 40 Normal. »
→ Reformulation du libellé, pas une définition : l'emplacement correspondant et son
articulation avec « Bloqué (Normal 1) » et « Stock PLTF (Normal 2) » restent indéterminés,
le total reste non calculable. NE PAS résoudre. Créer un gap avec
follow_up_of_gap_id = <id d'origine>, sévérité conservée ("blocking"), description citant la
réponse reçue et la question qui subsiste.

[R3] RENVOI À PLUS TARD
Nouvel élément (USER_ANSWER) : « On verra ça avec le fournisseur retenu. »
→ N'apporte aucune information nouvelle. NE PAS résoudre, et NE PAS créer de gap de suivi :
ce serait un doublon de la lacune d'origine, qui reste simplement ouverte.

[R4] ASSUMPTION QUI CONTREDIT LE CONTEXTE
Nouvel élément (ASSUMPTION) : « Par défaut, l'application est maître du stock d'ingrédients
et pousse les mouvements vers l'ERP M3. »
Contexte existant : « Synchronisation avec l'ERP M3 pour garantir la cohérence des données et
éviter les doublons » et « Enregistrement manuel et validation des entrées d'ingrédients ».
→ L'hypothèse fixe une source de vérité que le document n'a jamais tranchée et qui contredit
le circuit de réception des entrées. Créer un gap de catégorie "contradiction", sévérité
"blocking", section_ids listant toutes les sections concernées. NE PAS marquer résolu.

[R5] RÉSOUT ET FAIT BAISSER LA SÉVÉRITÉ
Une réponse peut lever l'incompatibilité sans tout préciser. Si les deux implémentations
possibles convergent désormais, résous la lacune ; si un détail secondaire reste ouvert,
crée un gap de suivi en "nice_to_have" plutôt que de laisser la lacune d'origine ouverte."""


# --------------------------------------------------------------------------------------
# Constructeurs de prompt
# --------------------------------------------------------------------------------------

def build_prompt_section_mode(state: CDCState, section_id: str) -> str:
    section = get_section(state, section_id)
    hints = "\n".join(f"- {h}" for h in section.completion_hints) or "(aucun critère spécifique)"
    fewshots = f"\n{FEWSHOT_SECTION}\n" if FEWSHOT_ENABLED else ""
    return f"""Tu es un analyste qui audite un cahier des charges (CDC) pour en détecter les lacunes
et ambiguïtés, avant qu'il ne soit transmis à une équipe de développement.

CONTEXTE COMPLET (toutes sections, car les incohérences peuvent être transversales) :
{format_all_sections_context(state)}

GAPS DÉJÀ OUVERTS SUR CETTE SECTION (ne pas dupliquer) :
{format_open_gaps(state["gaps"], section_id)}

Concentre-toi maintenant sur la section : "{section.title}" (id={section.id})
Description de la section : {section.description}

Critères de complétude à vérifier un par un (checklist) :
{hints}
Un critère non couvert ne devient une lacune que si son absence change le code à écrire.

{SEVERITY_RUBRIC}

{SCOPE_GUARD}

{GAP_QUALITY_RULES}
{fewshots}
Analyse cette section de façon SYSTÉMATIQUE en passant par CHAQUE catégorie de lacune suivante,
une par une, et identifie les problèmes concrets (pas de généralités) :
{", ".join(GAP_TAXONOMY)}

Pour la catégorie "contradiction", compare explicitement le contenu de cette section avec les
AUTRES sections et signale toute incohérence transversale (severity au moins "important",
section_ids doit alors lister TOUTES les sections concernées).

Ne remonte QUE des lacunes concrètes et actionnables, citant les éléments ambigus du texte.
N'invente pas de lacune si le contexte répond déjà clairement au critère.

{SELF_CHECK}
"""


def build_prompt_fresh_mode(state: CDCState, fresh_item_ids: list[str]) -> str:
    items = [it for it in state["context_items"] if it.id in fresh_item_ids]
    items_text = "\n".join(
        f"- id={it.id} (source={it.source}, linked_gap_id={it.linked_gap_id}, sections={it.section_ids}): {it.content}"
        for it in items
    )
    gaps_text = "\n".join(
        f"- id={g.id} status={g.status} severity={g.severity}: {g.description}"
        for g in state["gaps"]
        if g.id in {it.linked_gap_id for it in items if it.linked_gap_id}
    )
    fewshots = f"\n{FEWSHOT_FRESH}\n" if FEWSHOT_ENABLED else ""
    return f"""Tu es un analyste qui audite un cahier des charges (CDC). De nouvelles informations
viennent d'être ajoutées au contexte (réponses utilisateur ou résultats RAG). Tu dois vérifier
si elles résolvent réellement les lacunes auxquelles elles répondent.

CONTEXTE COMPLET (toutes sections) :
{format_all_sections_context(state)}

NOUVEAUX ÉLÉMENTS À ÉVALUER :
{items_text}

LACUNES ORIGINALES CONCERNÉES :
{gaps_text}

CRITÈRE DE RÉSOLUTION : une lacune est résolue si, et seulement si, les deux lectures
concurrentes qu'elle décrivait ne sont plus possibles — c'est-à-dire si un développeur peut
désormais coder sans choisir à la place du métier. Une réponse qui reformule, qui renvoie à
plus tard ou qui ne répond qu'à une des deux lectures ne résout pas.

{SEVERITY_RUBRIC}

{GAP_QUALITY_RULES}
{fewshots}
Pour chaque nouvel élément lié à une lacune (linked_gap_id) :
- Si la réponse résout complètement la lacune → ajoute son gap_id à resolved_gap_ids.
- Si la réponse est partielle, ambiguë, ou introduit une nouvelle incohérence avec le reste du
  contexte → NE PAS la marquer comme résolue, et crée un nouveau gap de suivi (follow_up_of_gap_id
  = id de la lacune originale) décrivant précisément ce qui manque encore.
- Si la réponse n'apporte aucune information nouvelle (renvoi à plus tard, « à voir avec le
  fournisseur ») → ne rien résoudre et ne créer AUCUN gap de suivi : la lacune reste ouverte.
- Si l'élément est une "ASSUMPTION" (hypothèse par défaut faute de réponse), vérifie aussi
  qu'elle ne contredit pas une autre section déjà validée ; si oui, crée un gap "contradiction".

Vérifie aussi si ces nouveaux éléments révèlent une incohérence transversale avec une section
déjà marquée comme complète ailleurs dans le contexte (catégorie "contradiction").

Avant de répondre : pour chaque gap de suivi créé, vérifie qu'il cite la réponse reçue et
qu'il pose une question différente de celle de la lacune d'origine. Sinon, ne le crée pas."""
