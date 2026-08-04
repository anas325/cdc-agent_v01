# Annexe 1 — Règles de gestion détaillées & cas limites
## Projet : Outil interne de suivi des demandes IT

Ce document complète le CDC en levant les zones d'ambiguïté sur le comportement attendu, notamment pour les cas limites et les cas d'erreur.

---

## 1. Création de ticket (F01)

| Cas | Comportement attendu |
|---|---|
| Titre vide ou < 5 caractères | Rejet, message : "Le titre doit contenir au moins 5 caractères" |
| Description vide | Rejet, message : "La description est obligatoire" |
| Catégorie non sélectionnée | Rejet (RG01) |
| Priorité non sélectionnée | Valeur par défaut = "Moyenne" (pas de rejet) |
| Fichier joint > 10 Mo | Rejet, message : "Fichier trop volumineux (max 10 Mo)" |
| Fichier joint type non autorisé | Autorisés : .pdf, .png, .jpg, .docx, .xlsx, .log, .txt. Tout autre type → rejet |
| Plusieurs fichiers joints | Max 3 fichiers par ticket |
| Demandeur crée un 2e ticket identique (même titre) en < 5 min | Avertissement non bloquant : "Un ticket similaire existe déjà (#ID). Continuer quand même ?" |

---

## 2. Attribution automatique (F02)

| Cas | Comportement attendu |
|---|---|
| Aucun agent dans la catégorie | Ticket reste "Non assigné", alerte admin après 4h ouvrées (pas 4h calendaires) |
| Plusieurs agents disponibles | Attribution au round-robin (agent avec le moins de tickets ouverts) |
| Agent assigné devient inactif/absent (compte désactivé) | Ticket repasse automatiquement en "Non assigné" et déclenche une nouvelle attribution |
| Admin réassigne manuellement un ticket déjà assigné | Autorisé à tout moment, sans confirmation, mais génère une entrée dans l'historique du ticket |
| Ticket "Haute priorité" sans agent disponible | Alerte immédiate (pas d'attente de 4h) |

---

## 3. Statuts et cycle de vie (F03)

### Transitions autorisées
```
Ouvert ──────────────► En cours
En cours ────────────► En attente demandeur
En cours ────────────► Résolu
En attente demandeur ─► En cours (dès réponse du demandeur)
En attente demandeur ─► Résolu (si pas de réponse après 7 jours → auto-résolu)
Résolu ───────────────► Clôturé (auto après 3 jours OU manuel par demandeur)
Résolu ───────────────► En cours (si demandeur rouvre dans les 3 jours)
Clôturé ──────────────► (aucune transition — ticket verrouillé en lecture seule)
```

| Cas | Comportement attendu |
|---|---|
| Demandeur essaie de rouvrir un ticket "Clôturé" | Refusé. Message : "Ce ticket est clôturé, créez un nouveau ticket en le référençant" |
| Ticket "En attente demandeur" sans réponse après 7 jours calendaires | Passe automatiquement en "Résolu" (avec mention système "Auto-résolu — absence de réponse") |
| Ticket "Résolu" rouvert par le demandeur | Repasse en "En cours", ré-assigné au même agent si toujours actif, sinon nouvelle attribution |
| Agent tente de clôturer directement un ticket "Ouvert" (sans passer par "Résolu") | Interdit. L'agent doit d'abord passer par "Résolu" |
| Weekend/jours fériés dans le calcul des délais (4h, 3 jours, 7 jours) | Tous les délais sont calculés en **jours/heures ouvrés** (Lun-Ven, 9h-18h), pas calendaires |

---

## 4. Notifications (F04)

| Cas | Comportement attendu |
|---|---|
| Changement de statut multiple en < 1 min (ex: script/erreur) | Notifications groupées, un seul email envoyé après 1 min d'inactivité (anti-spam) |
| Email du demandeur invalide/bounce | Le ticket reste actif, une alerte est loguée pour l'admin mais ne bloque pas le workflow |
| Agent assigné = demandeur (auto-assignation) | Cas interdit — un utilisateur ne peut pas être agent sur son propre ticket |
| Notification en dehors des heures ouvrées | Envoyée immédiatement quand même (email n'est pas bloqué, contrairement aux délais de calcul) |

---

## 5. Permissions et sécurité

| Rôle | Peut créer | Peut voir | Peut modifier statut | Peut réassigner | Peut supprimer |
|---|---|---|---|---|---|
| Demandeur | Ses propres tickets | Ses propres tickets uniquement | Non (sauf réouverture) | Non | Non |
| Agent IT | Oui | Tickets assignés + non-assignés de sa catégorie | Oui (sur ses tickets) | Non | Non |
| Admin | Oui | Tous les tickets | Oui (tous) | Oui | Oui (soft delete uniquement) |

**RG-SEC-01** : un demandeur ne peut jamais voir les tickets d'un autre demandeur, même dans la même catégorie.
**RG-SEC-02** : la suppression est toujours un "soft delete" (champ `deleted_at`), jamais une suppression physique — pour conserver l'historique/audit.

---

## 6. Tableau de bord (F05)

| Cas | Comportement attendu |
|---|---|
| Aucun ticket sur la période sélectionnée | Affiche "Aucune donnée pour cette période" (pas de graphique vide/cassé) |
| Ticket créé et clôturé le même jour | Compté dans les deux statistiques (créés ET clôturés) de ce jour |
| Temps moyen de résolution — méthode de calcul | `date_clôture - date_création`, en excluant le temps passé en "En attente demandeur" |
| Utilisateur non-admin accède au tableau de bord | Vue limitée à ses propres tickets uniquement (pas de stats globales) |
