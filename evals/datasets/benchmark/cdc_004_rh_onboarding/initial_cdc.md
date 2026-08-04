# Cahier des Charges
## Projet : Portail d'intégration des nouveaux collaborateurs

---

# 1. Problématique et contexte

## Contexte

Le groupe recrute environ 300 personnes par an sur quatre entités juridiques.
L'intégration d'un nouveau collaborateur repose aujourd'hui sur une série d'emails
échangés entre les RH, le manager, l'IT et les services généraux.

Il arrive régulièrement qu'un collaborateur arrive sans poste de travail, sans badge,
ou sans accès aux applications métier.

## Objectif

Fluidifier le parcours d'intégration et supprimer les oublis d'équipement.

## Parties prenantes

- Service RH
- Managers
- Service IT
- Services généraux

---

# 2. Spécifications fonctionnelles

## F1 — Création d'un dossier d'intégration

Les RH créent un dossier dès la signature du contrat.

Le dossier contient les informations du collaborateur.

## F2 — Génération des tâches

Le système génère automatiquement les tâches à réaliser avant l'arrivée.

Chaque tâche est affectée à un service.

Les tâches doivent être terminées à temps.

## F3 — Suivi

Le manager suit l'avancement du dossier.

Une relance est envoyée si nécessaire.

## F4 — Documents

Le collaborateur dépose ses documents administratifs sur le portail.

Les documents sont vérifiés puis archivés.

## Cas particuliers

Si une arrivée est annulée, le dossier est fermé.

Si la date d'arrivée change, les tâches sont recalculées.

---

# 3. Spécifications techniques

## Architecture

Application web hébergée sur le cloud du groupe.

Backend Java, frontend Angular, base de données PostgreSQL.

## Intégrations

Le portail récupère les collaborateurs depuis le SIRH.

Il envoie les demandes d'équipement à l'outil de ticketing IT.

## Données

Un dossier d'intégration contient :

- le nom du collaborateur
- sa date d'arrivée
- son manager
- la liste des tâches

## Sécurité

Les données RH sont confidentielles.

L'accès est restreint aux personnes autorisées.

## Performance

Le portail doit rester fluide.

---

# 4. Contraintes et périmètre

## Budget

Budget alloué : 90 000 €.

## Planning

Mise en service souhaitée avant la rentrée de septembre.

## Hors périmètre

La gestion de la paie.

Le suivi de la période d'essai.
