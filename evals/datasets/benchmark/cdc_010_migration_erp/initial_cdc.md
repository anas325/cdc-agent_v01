# Cahier des Charges
## Projet : Migration du module Achats vers le nouvel ERP

---

# 1. Problématique et contexte

## Contexte

Le groupe migre progressivement vers un nouvel ERP. Le module Achats de l'ancien
système, en service depuis onze ans, doit être repris.

Il gère les demandes d'achat, les commandes fournisseurs, les réceptions et le
rapprochement avec les factures, pour 7 sites et environ 180 utilisateurs.

## Objectif

Basculer les achats sur le nouvel ERP sans interrompre l'activité.

## Parties prenantes

- Direction des achats
- Acheteurs des 7 sites
- Comptabilité fournisseurs
- Équipe ERP

---

# 2. Spécifications fonctionnelles

## F1 — Reprise des processus achats

Les processus existants sont repris dans le nouvel ERP.

Les écarts de fonctionnement sont traités au cas par cas.

## F2 — Reprise des données

Les données de l'ancien système sont reprises.

L'historique est conservé.

## F3 — Circuit de validation

Les demandes d'achat suivent un circuit de validation.

Le circuit reste identique à l'existant.

## F4 — Interfaces

Les interfaces existantes sont adaptées.

## Cas particuliers

Les commandes en cours au moment de la bascule sont traitées.

---

# 3. Spécifications techniques

## Architecture

Le nouvel ERP est un progiciel du marché, hébergé par l'éditeur.

## Intégrations

Les interfaces avec la comptabilité, le référentiel fournisseurs et le portail
fournisseurs doivent continuer à fonctionner.

## Données

Les données à reprendre concernent les fournisseurs, les articles, les commandes
et les réceptions.

## Performance

Les performances doivent être au moins équivalentes à l'existant.

## Sécurité

Les habilitations sont reprises.

---

# 4. Contraintes et périmètre

## Budget

Enveloppe globale : 480 000 €.

## Planning

Bascule prévue au 1er juillet.

La bascule ne pourra pas avoir lieu pendant la période de clôture annuelle.

Le projet démarre en mai.

## Périmètre

Sont concernés les 7 sites.

La première phase couvre les sites pilotes.

Le module Achats est repris intégralement.

Certaines fonctionnalités marginales pourront être abandonnées.

## Hors périmètre

Le module Stocks.

La refonte des processus achats.
