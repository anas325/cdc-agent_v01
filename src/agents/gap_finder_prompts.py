"""Prompts du gap-finder — v3.

Problème traité
---------------
La notion de « bloquant » de l'agent dérive de celle du métier. Des CDC effectivement
acceptés ne contiennent NI prix, NI planning, NI plan de recette, NI stack technique,
NI hébergement, NI modèle de données, NI contrat d'API, NI la moindre exigence non
fonctionnelle chiffrée. Ils contiennent en revanche de vraies ambiguïtés bloquantes à
l'intérieur des exigences qu'ils engagent. Le discriminant n'est donc pas « cette
rubrique est-elle présente ? » mais « le texte engagé admet-il deux implémentations ? ».

Évolutions vs v2
----------------
1. GROUNDING_TEST   — la « double lecture » n'est retenue que si les DEUX lectures
                      s'appuient sur du texte réellement présent. En v2, la forme
                      imposée (citation + deux lectures) fonctionnait comme un
                      générateur : deux lectures s'inventent pour n'importe quelle
                      phrase, donc le filtre ne filtrait rien.
2. CATEGORY_GUIDE   — un test décisif par catégorie. En v2 les few-shots ne
                      démontraient que business_rule/contradiction, et le modèle
                      recopiait cette distribution de labels quel que soit le fond.
3. Few-shots négatifs reformulés en FORMES de piège, sans citer le passage fautif :
                      nommer un extrait pour dire « ne le remonte pas » l'amorce au
                      lieu de l'écarter.
4. Plafond ramené de 5 à 3, avec l'attendu explicite « 0 à 2 sur une section saine ».
5. Matériel des few-shots renouvelé — il ne doit recouper AUCUN cas de
   `evals/datasets/gap_finder.jsonl`, sous peine de rendre l'éval inutilisable
   (le modèle y récite la réponse au lieu de la trouver).

Les libellés de sévérité employés ici sont ceux de `GapSeverity` (src/state.py) :
"blocking" / "important" / "nice_to_have". `FEWSHOT_ENABLED` permet d'A/B tester les
few-shots dans evals/run_benchmark.py sans toucher aux appelants.

**Contrainte de maintenance** : tout exemple ajouté ici doit porter sur un extrait
absent du jeu d'évaluation. Un few-shot qui cite un cas annoté transforme la mesure
de généralisation en mesure de mémorisation.
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


GROUNDING_TEST = """TEST D'ANCRAGE (le plus important — applique-le avant tout le reste) :
Deux lectures s'inventent pour n'importe quelle phrase. Ce n'est donc PAS parce que tu
sais formuler une lecture A et une lecture B qu'il y a une lacune.
Une double lecture ne compte QUE si elle est ancrée dans un engagement réellement pris :
- soit le document dit deux choses qui tirent dans des sens opposés (deux passages),
- soit le document engage une règle, un calcul ou un indicateur dont le résultat change
  selon la lecture retenue (un passage qui promet un résultat sans donner le moyen),
- soit un critère de complétude de la section (la checklist qui t'est fournie) est resté
  sans réponse : la section s'est engagée à le traiter, son absence n'est donc pas un
  silence mais une promesse non tenue. C'est le seul cas où une rubrique manquante est
  une lacune, et il l'emporte alors sur HORS PÉRIMÈTRE.
Contre-test à appliquer systématiquement : « pour rendre ma lecture B plausible, ai-je dû
ajouter une hypothèse que le document n'évoque nulle part ? » Si oui, ta lecture B est une
invention de ta part : SUPPRIME la lacune. Le silence du document sur un sujet qu'il n'a
jamais engagé n'est pas une ambiguïté — c'est un sujet hors périmètre.
Exemples de lectures NON ancrées, à supprimer : imaginer un canal de notification (mail,
SMS) quand le document ne parle que d'affichage ; imaginer un profil habilité alors que le
document ne restreint personne ; imaginer un état de cycle de vie supplémentaire alors que
le document en énumère la liste."""


SCOPE_GUARD = """HORS PÉRIMÈTRE — n'ouvre PAS de lacune sur ces sujets, sauf si le document
prétend explicitement les traiter et se contredit alors lui-même :
- Prix, budget, conditions commerciales (typiquement renvoyés au service achats).
- Planning, jalons, plan de recette/tests, critères d'acceptation contractuels.
- Stack technique, hébergement, infrastructure : le choix est la réponse du fournisseur,
  pas une lacune du CDC.
- Exigences non fonctionnelles chiffrées (volumétrie, temps de réponse, utilisateurs
  simultanés, durée de rétention) lorsque l'échelle décrite est manifestement faible
  (une ferme, trois engins, quelques centaines d'articles).
- Détail d'IHM / maquettage lorsque le document renvoie à une capture d'écran ou à l'existant.
- Modèle de données, contrats d'API, format des messages échangés : relèvent de la phase de
  conception, SAUF si le CDC engage déjà une règle de calcul dont le résultat en dépend.
Principe : une rubrique absente n'est pas une lacune. Une règle promise puis laissée
indéterminée en est une."""


CATEGORY_GUIDE = """CHOIX DE LA CATÉGORIE — prends celle dont le test décisif est vrai, pas
la plus proche de l'exemple que tu viens de lire. Une lacune bien décrite mais mal
catégorisée est comptée comme manquée.
- "functional_ambiguity" : un TERME ou une expression du texte admet deux sens (« taux de
  référence », « améliorer », « validé ») et rien ne le définit ailleurs.
- "scope" : le PÉRIMÈTRE ou la cible sont indéterminés — ce qui est remplacé ou non, qui
  sont les utilisateurs visés, ce que « réussir » veut dire de façon mesurable.
- "data_model" : la CLÉ D'ENREGISTREMENT ou la granularité d'une donnée est indéterminée
  alors qu'un indicateur ou un calcul en dépend (par élément ou par regroupement ?).
- "business_rule" : le COMPORTEMENT attendu dans un cas prévu par le texte n'est pas
  tranché (automatique ou manuel, bloquant ou informatif, qui a le dernier mot).
- "integration" : le SENS D'ÉCHANGE ou la source de vérité entre ce système et un autre
  système nommé dans le document n'est pas fixé.
- "edge_case" : un cas limite explicitement évoqué par le texte n'a pas de comportement
  associé (échec, doublon, valeur absente).
- "nfr" : une exigence non fonctionnelle est ÉNONCÉE en adjectif sans chiffre alors que le
  document s'engage dessus. Pas d'exigence énoncée → pas de lacune "nfr".
- "acceptance_criteria" : le document promet une recette ou un critère de réussite et n'en
  donne pas. Pas de promesse → pas de lacune.
- "contradiction" : deux passages du document s'excluent. section_ids doit alors lister
  TOUTES les sections concernées.
Les catégories citées dans les exemples ci-dessous ne sont pas un ordre de préférence :
toutes les valeurs de la taxonomie sont également recevables."""


GAP_QUALITY_RULES = """FORME OBLIGATOIRE DE CHAQUE LACUNE (champ description) :
1. COMMENCE par le passage exact du document, entre guillemets (une phrase maximum).
2. Énonce les DEUX lectures, chacune avec sa conséquence concrète sur le code ou sur le
   chiffre affiché, et chacune rattachable à du texte présent (cf. TEST D'ANCRAGE).
3. Termine par UNE question fermée, répondable en une phrase par un responsable métier.
Reste compact : 4 à 6 lignes au total, pas de paragraphes aérés.
Interdits : reformuler un critère de la checklist en question (« les rôles ne sont pas
détaillés »), demander « davantage de précisions », produire une lacune sans citation,
ou fusionner plusieurs problèmes dans une seule lacune."""


SELF_CHECK = """AVANT DE RÉPONDRE, filtre ta propre liste. Pour chaque lacune candidate :
(a) puis-je citer le passage exact ? sinon → supprime-la ;
(b) mes deux lectures passent-elles le TEST D'ANCRAGE, ou ai-je ajouté une hypothèse
    absente du document ? si ajoutée → supprime-la ;
(c) la catégorie choisie est-elle celle dont le test décisif est vrai (cf. CHOIX DE LA
    CATÉGORIE) ? sinon → corrige-la ;
(d) la question est-elle répondable en une phrase par un métier ? sinon → reformule ;
(e) la réponse se trouve-t-elle déjà ailleurs dans le CONTEXTE COMPLET, dans une AUTRE
    section que celle que j'analyse ? relis-le avant de conclure — les rôles, les droits,
    les unités et les définitions sont souvent portés par une section voisine. Si oui →
    supprime-la ;
(f) le sujet figure-t-il dans HORS PÉRIMÈTRE, sans être par ailleurs promis par la
    checklist de la section ? si oui → supprime-la.
Maximum 3 lacunes par section, triées par sévérité décroissante. Sur une section saine,
l'attendu est 0 à 2 lacunes, et 0 est une réponse parfaitement valide : une liste vide vaut
mieux qu'une lacune inventée. Deux lacunes citées et tranchantes valent mieux que huit
lacunes génériques."""


# --------------------------------------------------------------------------------------
# Few-shots — mode section
# --------------------------------------------------------------------------------------
# Matériel volontairement disjoint du jeu d'évaluation (collecte de lait cru).
# Ne pas y réintroduire d'extrait figurant dans evals/datasets/gap_finder.jsonl.

FEWSHOT_SECTION = """EXEMPLES DE LACUNES À REMONTER (contexte industriel comparable ;
noter que chacune relève d'une catégorie différente) :

[1] catégorie "functional_ambiguity", sévérité "blocking" — terme employé dans un calcul
mais jamais défini.
Extrait : « La quantité facturée au producteur est le litrage corrigé selon le taux de
référence. »
Lacune : « "taux de référence" n'est défini nulle part et le document ne dit pas où il est
stocké. Lecture A : c'est une constante unique paramétrée pour toute la laiterie, la
correction est alors la même pour tous. Lecture B : il est porté par le contrat de chaque
producteur, la correction devient un calcul par ligne. Les deux produisent une facture
différente sans erreur visible. Question : le taux de référence est-il une constante unique
ou une valeur portée par le contrat de chaque producteur ? »

[2] catégorie "scope", sévérité "blocking" — périmètre de remplacement indéterminé.
Extrait : « Le module remplace le suivi Excel actuel des tournées. »
Lacune : « Lecture A : le remplacement est total, le fichier est retiré à la mise en
service et toutes ses colonnes doivent être reprises, y compris celles que ce document ne
décrit pas. Lecture B : le remplacement est partiel, le fichier reste en usage pour les cas
non couverts et seules les colonnes décrites ici sont à développer. Le périmètre à chiffrer
change du simple au double. Question : le fichier Excel est-il retiré à la mise en service,
ou reste-t-il en usage pour les cas non couverts ? »

[3] catégorie "data_model", sévérité "blocking" — granularité d'enregistrement dont dépend
un indicateur promis par le document.
Extraits : « Un indicateur affiche le taux de non-conformité par tournée » et « Les
prélèvements sont analysés par citerne ».
Lacune : « La non-conformité est constatée par citerne mais restituée par tournée, et le
document ne dit pas comment passer de l'une à l'autre. Lecture A : une tournée remplit une
seule citerne, le taux est un rapport direct. Lecture B : une tournée peut en remplir
plusieurs, il faut alors une règle de répartition entre producteurs collectés. Question :
une tournée correspond-elle toujours à une citerne unique, et sinon comment la
non-conformité est-elle imputée ? »

[4] catégorie "integration", sévérité "blocking" — sens d'échange non fixé entre deux
systèmes tous deux nommés par le document.
Extraits : « Les producteurs et leurs contrats sont repris depuis le référentiel Agri » et
« L'utilisateur peut créer un producteur depuis l'écran de saisie ».
Lacune : « Lecture A : Agri reste maître, la création locale est une fiche provisoire à
réconcilier, et il faut gérer les doublons au retour. Lecture B : l'application devient
maître après reprise, Agri n'est alimenté qu'une fois. Question : après la mise en service,
quel système fait foi sur la fiche producteur ? »

[5] catégorie "business_rule", sévérité "important" — comportement attendu dans un cas que
le texte prévoit lui-même, mais laisse indéterminé.
Extrait : « Le chauffeur valide la pesée ; en cas d'écart avec le bon de livraison, l'écart
est signalé. »
Lacune : « Lecture A : le signalement est bloquant, le chauffeur ne peut pas valider tant
que l'écart n'est pas justifié. Lecture B : le signalement est informatif, la validation
passe et l'écart est tracé pour contrôle a posteriori. Le parcours terrain diffère.
Question : un écart signalé empêche-t-il la validation de la pesée, ou est-il seulement
enregistré ? »

[6] catégorie "contradiction", sévérité "blocking" — deux sections s'excluent.
Extraits : § planning « Une tournée est clôturée par le chauffeur en fin de journée » et
§ supervision « La clôture d'une tournée est prononcée par le planificateur après
contrôle ».
Lacune : « Deux acteurs différents détiennent la même transition d'état. Lecture A : le
chauffeur clôture, le contrôle du planificateur est postérieur et ne bloque rien. Lecture B :
le chauffeur ne fait que déclarer la fin, l'état clôturé n'est atteint qu'après contrôle,
ce qui ajoute un état intermédiaire au modèle. Question : qui prononce la clôture d'une
tournée, et existe-t-il un état intermédiaire entre "déclarée terminée" et "clôturée" ? »
(section_ids doit lister LES DEUX sections)

PIÈGES FRÉQUENTS — FORMES DE FAUSSES LACUNES À NE PAS REMONTER (décrites par leur forme,
à reconnaître dans le texte que tu analyses) :

[N1] Le document confie explicitement un sujet à un autre processus ou à un tiers nommé
(achats, comité, phase ultérieure, fournisseur retenu), avec un point de passage énoncé.
→ renvoi assumé, pas une lacune. Ne demande pas de « préciser » ce qui est délégué.
[N2] Une rubrique entière est absente du document, qui ne l'a jamais promise.
→ silence, pas ambiguïté. Aucune lacune, et surtout pas "blocking".
[N3] Un seuil est déjà donné en chiffre et signalé comme paramétrable, et les profils
habilités sont définis ailleurs dans le document. → critère satisfait. N'ouvre pas de
lacune pour redemander le seuil, son unité, ou qui a le droit de le modifier.
[N4] Un choix d'architecture, d'hébergement ou de stack est laissé ouvert.
→ réponse du fournisseur. Pas de lacune, sauf contrainte de localisation des données
évoquée ailleurs dans le document.
[N5] Un écran n'est décrit que par une capture ou par renvoi à l'existant.
→ acceptable. À ne remonter que si cette capture est la seule source d'une règle de calcul.
[N6] Un matériel ou un produit est nommé, avec une clause de repli explicite en cas
d'incompatibilité. → exigence complète, y compris quand les critères de compatibilité ne
sont pas détaillés : la clause de repli est la réponse.
[N7] Le document énumère une liste (états, rôles, colonnes, canaux) et tu songes à demander
si un élément supplémentaire manque. → l'énumération fait foi. Une liste close n'est pas
une liste incomplète.
[N8] Ta lecture B n'existe que parce que tu as ajouté un mécanisme dont le document ne
parle jamais (un canal d'envoi, un droit, un état, un système tiers).
→ invention. Supprime-la, quelle que soit sa vraisemblance métier."""


# --------------------------------------------------------------------------------------
# Few-shots — mode fresh
# --------------------------------------------------------------------------------------

FEWSHOT_FRESH = """EXEMPLES D'ÉVALUATION D'UN NOUVEL ÉLÉMENT :

[R1] RÉSOUT COMPLÈTEMENT
Lacune : qui prononce la clôture d'une tournée (chauffeur ou planificateur) ?
Nouvel élément (USER_ANSWER) : « La clôture est prononcée par le planificateur après
contrôle ; le chauffeur ne fait que déclarer la fin de tournée, ce qui met la tournée en
état "à contrôler". »
→ Les deux lectures sont tranchées et l'état intermédiaire est nommé.
resolved_gap_ids += [gap_id]. Aucun gap de suivi.

[R2] RÉSOUT PARTIELLEMENT → gap de suivi
Lacune : « taux de référence » non défini dans le calcul de la quantité facturée.
Nouvel élément (USER_ANSWER) : « C'est le taux de référence contractuel. »
→ Reformulation du libellé, pas une définition : on ignore toujours s'il est unique ou
porté par chaque contrat, et où il est stocké ; la facture reste non calculable.
NE PAS résoudre. Créer un gap avec follow_up_of_gap_id = <id d'origine>, sévérité conservée
("blocking"), description citant la réponse reçue et la question qui subsiste.

[R3] RENVOI À PLUS TARD
Nouvel élément (USER_ANSWER) : « Ce point sera arbitré en comité de pilotage. »
→ N'apporte aucune information nouvelle. NE PAS résoudre, et NE PAS créer de gap de suivi :
ce serait un doublon de la lacune d'origine, qui reste simplement ouverte.

[R4] ASSUMPTION QUI CONTREDIT LE CONTEXTE
Nouvel élément (ASSUMPTION) : « Par défaut, une tournée ne remplit qu'une seule citerne. »
Contexte existant : « Une tournée dessert plusieurs producteurs sur une demi-journée » et
« Les prélèvements sont analysés par citerne ».
→ L'hypothèse fixe une cardinalité que le document contredit ailleurs et efface la règle de
répartition nécessaire au calcul. Créer un gap de catégorie "contradiction", sévérité
"blocking", section_ids listant toutes les sections concernées. NE PAS marquer résolu.

[R5] RÉSOUT ET FAIT BAISSER LA SÉVÉRITÉ
Une réponse peut lever l'incompatibilité sans tout préciser. Si les deux implémentations
possibles convergent désormais, résous la lacune ; si un détail secondaire reste ouvert
(un libellé, un tri d'affichage), crée un gap de suivi en "nice_to_have" plutôt que de
laisser la lacune d'origine ouverte."""


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
Ces critères sont l'engagement de la section : un critère laissé sans réponse par le
contexte est une lacune ancrée, même si le sujet paraît relever du HORS PÉRIMÈTRE. Vérifie
d'abord qu'aucune AUTRE section du contexte complet n'y répond déjà.

{GROUNDING_TEST}

{SEVERITY_RUBRIC}

{SCOPE_GUARD}

{CATEGORY_GUIDE}

{GAP_QUALITY_RULES}
{fewshots}
Passe en revue CHAQUE catégorie de la taxonomie suivante, une par une, et n'en retiens que
celles pour lesquelles tu trouves un problème concret et ancré (pas de généralités) :
{", ".join(GAP_TAXONOMY)}
Beaucoup de catégories ne donneront rien sur une section donnée : c'est le cas normal, ne
force pas une lacune pour remplir une case.

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

{GROUNDING_TEST}

{SEVERITY_RUBRIC}

{CATEGORY_GUIDE}

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

Avant de répondre : pour chaque gap de suivi créé, vérifie qu'il cite la réponse reçue, qu'il
passe le TEST D'ANCRAGE et qu'il pose une question différente de celle de la lacune d'origine.
Sinon, ne le crée pas."""
