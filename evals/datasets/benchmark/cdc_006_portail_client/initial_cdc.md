# Cahier des Charges
## Projet : Portail client distributeurs

---

# 1. Problématique et contexte

## Contexte

Les 400 distributeurs partenaires passent leurs commandes par téléphone et par email
auprès de l'administration des ventes. Chaque commande est ressaisie manuellement
dans l'ERP.

Les distributeurs n'ont aucune visibilité sur leurs encours, leurs livraisons
ni leurs factures.

## Objectif

Permettre aux distributeurs de commander et de consulter leurs documents en autonomie.

## Parties prenantes

- Distributeurs partenaires
- Administration des ventes
- Comptabilité
- Direction commerciale

---

# 2. Spécifications fonctionnelles

## F1 — Accès au portail

Les distributeurs se connectent au portail avec leur compte.

Chaque distributeur voit ses propres données.

## F2 — Passage de commande

Le distributeur constitue sa commande depuis le catalogue.

La commande est ensuite transmise à l'ERP.

## F3 — Suivi des livraisons

Le distributeur consulte l'état de ses livraisons.

## F4 — Documents

Le distributeur télécharge ses factures et ses bons de livraison.

## F5 — Administration

L'administration des ventes gère les comptes distributeurs.

## Cas particuliers

Un distributeur bloqué pour impayé ne peut plus commander.

---

# 3. Spécifications techniques

## Architecture

Application web exposée sur Internet.

## Intégrations

Le portail échange avec l'ERP pour les commandes, les livraisons et les factures.

## Sécurité

Le portail est sécurisé.

Les mots de passe sont chiffrés.

Les données des distributeurs sont protégées.

## Performance

Le portail doit supporter la charge.

## Données

Un distributeur possède un code client et un nom.

---

# 4. Contraintes et périmètre

## Budget

Enveloppe de 250 000 €.

## Planning

Ouverture aux 40 premiers distributeurs en fin d'année.

## Hors périmètre

Le paiement en ligne.

L'application mobile.
