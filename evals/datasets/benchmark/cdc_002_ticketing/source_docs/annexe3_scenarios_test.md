# Annexe 3 — Scénarios de test & parcours utilisateur
## Projet : Outil interne de suivi des demandes IT

Ce document traduit les règles de gestion (Annexe 1) et le contrat d'API (Annexe 2) en scénarios de test concrets, prêts à être utilisés pour la recette.

---

## 1. Parcours utilisateur de référence (happy path)

```
1. Demandeur se connecte
2. Demandeur clique "Nouveau ticket"
3. Demandeur remplit titre + description + catégorie "Bug"
4. Demandeur soumet → ticket créé en statut "Ouvert"
5. Système assigne automatiquement à l'agent disponible le moins chargé
6. Agent reçoit email de notification
7. Agent change le statut en "En cours"
8. Demandeur reçoit email de notification
9. Agent résout le problème, passe le statut en "Résolu" + commentaire
10. Demandeur reçoit email, ne répond pas
11. (3 jours plus tard) Ticket auto-clôturé
```

---

## 2. Scénarios de test — Création de ticket (F01)

| ID | Scénario | Donnée d'entrée | Résultat attendu |
|---|---|---|---|
| T01 | Création valide | title="Écran cassé" (11 car.), description valide, category="materiel" | 201, ticket créé, statut "open" |
| T02 | Titre trop court | title="abc" (3 car.) | 400 `TICKET_TITLE_TOO_SHORT` |
| T03 | Catégorie manquante | category=null | 400 `TICKET_INVALID_CATEGORY` |
| T04 | Priorité non fournie | priority=undefined | 201, priority="medium" par défaut |
| T05 | Fichier joint 12 Mo | attachment 12MB | 400 `FILE_TOO_LARGE` |
| T06 | Fichier .exe joint | attachment type=.exe | 400 `FILE_TYPE_NOT_ALLOWED` |
| T07 | 4 fichiers joints | 4 attachments | 400 `TICKET_TOO_MANY_ATTACHMENTS` |
| T08 | Doublon en 5 min | 2e ticket même titre, même demandeur, +2min | 201 avec warning non bloquant |

---

## 3. Scénarios de test — Attribution (F02)

| ID | Scénario | Contexte | Résultat attendu |
|---|---|---|---|
| T09 | Attribution round-robin | 3 agents dispo, agent A a le moins de tickets ouverts | Ticket assigné à agent A |
| T10 | Aucun agent disponible | 0 agent actif dans catégorie "acces" | status="open", assigned_agent_id=null |
| T11 | Alerte après 4h ouvrées | Ticket non-assigné depuis 4h ouvrées (créé vendredi 17h) | Alerte envoyée lundi 13h (pas vendredi 21h) |
| T12 | Priorité haute sans agent | priority="high", 0 agent dispo | Alerte immédiate (pas d'attente 4h) |
| T13 | Agent désactivé pendant traitement | Ticket assigné à agent X, X désactivé | Ticket repasse "Non assigné" + nouvelle attribution déclenchée |

---

## 4. Scénarios de test — Transitions de statut (F03)

| ID | Scénario | Transition tentée | Résultat attendu |
|---|---|---|---|
| T14 | Transition valide | Ouvert → En cours | 200, statut mis à jour |
| T15 | Transition invalide (saut d'étape) | Ouvert → Clôturé (par agent) | 409 `INVALID_STATUS_TRANSITION` |
| T16 | Demandeur tente de changer le statut | Demandeur → PATCH status | 403 `FORBIDDEN_STATUS_CHANGE` |
| T17 | Réouverture d'un ticket clôturé | Clôturé → (tentative réouverture) | Refusé, message d'invitation à créer un nouveau ticket |
| T18 | Auto-résolution après 7 jours | "En attente demandeur" sans réponse 7j calendaires | Statut passe à "Résolu" + mention système |
| T19 | Réouverture dans les 3 jours | "Résolu" → demandeur rouvre à J+2 | Statut "En cours", ré-assigné au même agent si actif |
| T20 | Auto-clôture après 3 jours | "Résolu" sans action à J+3 | Statut "Clôturé" automatiquement |

---

## 5. Scénarios de test — Permissions (Sécurité)

| ID | Scénario | Utilisateur | Résultat attendu |
|---|---|---|---|
| T21 | Demandeur voit ses tickets uniquement | Demandeur A liste les tickets | Ne voit pas les tickets du demandeur B |
| T22 | Agent voit sa catégorie | Agent catégorie "bug" liste les tickets | Voit tickets "bug" assignés + non-assignés, pas "materiel" |
| T23 | Admin réassigne | Admin PATCH /assign | 200, réassignation effectuée + entrée historique |
| T24 | Non-admin tente réassignation | Agent PATCH /assign | 403 `FORBIDDEN` |
| T25 | Auto-assignation interdite | agent_id == requester_id | 400 `SELF_ASSIGNMENT_FORBIDDEN` |
| T26 | Suppression = soft delete | Admin supprime un ticket | `deleted_at` renseigné, ticket absent des listes mais présent en DB |

---

## 6. Scénarios de test — Tableau de bord (F05)

| ID | Scénario | Contexte | Résultat attendu |
|---|---|---|---|
| T27 | Période sans données | from/to sur période vide | `empty_period: true`, UI affiche "Aucune donnée" |
| T28 | Ticket créé + clôturé même jour | 1 ticket, cycle complet en 1 jour | Compté dans "créés" ET "clôturés" du jour |
| T29 | Calcul temps moyen exclut l'attente | Ticket avec 2j en "En attente demandeur" sur 5j total | `avg_resolution_time` = 3j (pas 5j) |
| T30 | Accès non-admin | Demandeur accède au dashboard | Vue limitée à ses propres tickets |

---

## 7. Checklist de recette finale

- [ ] Tous les scénarios T01–T30 passent
- [ ] Aucune notification en double envoyée (test anti-spam < 1min)
- [ ] Les délais (4h / 3j / 7j) respectent bien les heures/jours ouvrés, pas calendaires
- [ ] Aucun accès croisé entre demandeurs (test avec 2 comptes distincts)
- [ ] Formats de date cohérents (ISO 8601) sur toutes les réponses API
