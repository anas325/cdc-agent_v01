# Rapport QA — Cahier des charges

## Lacunes résolues (3)
- [important] Le problème n’est pas formulé avec des métriques précises ; seul l’objectif de réduction du délai moyen est mesurable, mais le problème lui‑même reste qualitatif.
- [important] Les parties prenantes (demandeur, agent IT, admin) ne sont pas explicitement mentionnées dans la section problème, bien qu’elles apparaissent dans la section 2.
- [important] Need real measurements for ticket tracking, duplication, and visibility percentages to validate performance objectives.

## Hypothèses retenues faute de réponse (2)
- [important] Le problème est décrit sans mesures précises : "pertes de suivi, doublons, aucune visibilité" ne sont pas quantifiés, ce qui empêche de vérifier l’atteinte des objectifs de réduction de délai et de traçabilité.
- [important] La valeur fournie dans l’ASSUMPTION (20 % non suivis, 15 % doublons, 70 % visibles) n’est pas une mesure validée ; il reste donc à collecter des données réelles pour quantifier les pertes de suivi, les doublons et la visibilité afin de pouvoir mesurer l’atteinte des objectifs de réduction de délai et de traçabilité.

## Lacunes reportées (limite de tours atteinte) (3)
- [important] The new element claims that the annexes specify the roles of demandeur, agent IT, and admin, thereby clarifying scope and addressing a gap. However, the existing annexes section (id=ctx_acf1694c) contains only a glossary and contact information, with no role definitions, contradicting the claim.
- [important] The assumption states that only 70% of tickets are visible in real time to the requester, whereas the project objective (section 1.2) and the functional requirement F04 (notifications) both assert that all tickets should provide real‑time visibility to the requester.
- [important] The new assumption 20 %/15 %/70 % contradicts the earlier assumption 30 %/10 %/60 % regarding the percentage of tickets not followed correctly, duplicated, and visible in real time.

## Éléments de contexte non intégrés au document (2)
- (assumption) ASSUMPTION: En l'absence de mesures précises, on suppose que 30 % des tickets créés ne sont pas suivis correctement, 10 % génèrent des doublons, et 60 % sont visibles en temps réel par le demandeur, afin de pouvoir définir les indicateurs de performance et ajuster les objectifs de réduction de délai et de traçabilité.
- (user_answer) 20 %/15 %/70 %

## Incohérences détectées lors de la relecture finale (1)
- La fonctionnalité F02 décrit un statut « Non assigné » pour les tickets sans agent disponible, mais le modèle de données et la liste des statuts dans la table `tickets` ne comprennent pas ce statut (les statuts possibles sont open, in_progress, awaiting_requester, resolved, closed).
