# Cahier des Charges
## Projet : Refonte du moteur de facturation abonnements

---

# 1. Problématique et contexte

## Contexte

L'offre de services est passée en trois ans de 2 à 14 formules d'abonnement.
Le moteur de facturation actuel, développé en interne, ne sait pas gérer les
changements de formule en cours de mois et impose un retraitement manuel sur
environ 400 factures par mois.

## Objectif

Automatiser complètement le cycle de facturation mensuel des 22 000 abonnés.

## Parties prenantes

- Comptabilité
- Service client
- Direction financière
- Équipe produit

---

# 2. Spécifications fonctionnelles

## F1 — Facturation périodique

Le système génère les factures des abonnés chaque mois.

## F2 — Changement de formule

Un abonné peut changer de formule.

La facture est ajustée en conséquence.

## F3 — Remises

Des remises commerciales peuvent être appliquées.

## F4 — Recouvrement

Les factures impayées font l'objet d'une relance.

## F5 — Avoirs

Le service client peut émettre un avoir.

## Cas particuliers

Un abonné qui résilie est facturé jusqu'à la fin de la période.

---

# 3. Spécifications techniques

## Architecture

Service de facturation isolé, appelé par le back-office.

## Intégrations

Le système récupère les abonnements depuis le référentiel client.

Il exporte les écritures vers la comptabilité.

## Données

Une facture comporte un numéro, un montant et une date.

## Performance

La facturation mensuelle doit s'exécuter dans la nuit.

## Sécurité

Les factures sont des documents sensibles.

---

# 4. Contraintes et périmètre

## Budget

Budget : 140 000 €.

## Planning

Bascule prévue en début d'exercice.

## Contraintes réglementaires

La facturation doit être conforme à la réglementation en vigueur.

## Hors périmètre

Le recouvrement judiciaire.

La facturation des clients grands comptes.
