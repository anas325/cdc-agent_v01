# Cahier des Charges
## Projet : Suivi des livraisons du dernier kilomètre

---

# 1. Problématique et contexte

## Contexte

La filiale logistique livre 3 000 colis par jour en zone urbaine avec 45 véhicules.

Le suivi repose sur des appels téléphoniques entre le dispatcheur et les chauffeurs.
Le service client ne peut pas répondre aux demandes de position d'un colis.

## Objectif

Donner au service client et au destinataire une visibilité en temps réel sur la livraison.

Le projet doit couvrir l'ensemble des 45 véhicules dès la mise en service.

Le déploiement se fera progressivement, en commençant par 5 véhicules.

## Parties prenantes

- Dispatcheurs
- Chauffeurs
- Service client
- Destinataires

---

# 2. Spécifications fonctionnelles

## F1 — Application chauffeur

Le chauffeur consulte sa tournée et déclare chaque livraison.

L'application fonctionne sans connexion réseau.

Toutes les actions du chauffeur sont enregistrées immédiatement sur le serveur central.

## F2 — Suivi temps réel

La position des véhicules est remontée toutes les 30 secondes.

Pour préserver la batterie, la position est remontée toutes les 10 minutes.

## F3 — Notification destinataire

Le destinataire reçoit un SMS avec un créneau de livraison.

Le créneau est d'une heure.

Un créneau de deux heures est communiqué au destinataire la veille.

## F4 — Preuve de livraison

Le chauffeur recueille une signature sur l'écran.

En cas d'absence, le colis est déposé chez un commerçant partenaire.

## Cas particuliers

Si le destinataire est absent et qu'aucun commerçant n'est disponible, le colis revient à l'agence.

---

# 3. Spécifications techniques

## Architecture

Application mobile Android pour les chauffeurs.

Backend hébergé sur le cloud public.

Les données de géolocalisation ne quittent jamais le datacenter de la filiale.

## Intégrations

Le système reçoit les tournées du système de planification.

Il envoie les statuts de livraison au système de suivi client.

## Performance

Le tableau dispatch doit afficher la position des véhicules en moins de 2 secondes.

Le rafraîchissement du tableau dispatch est effectué toutes les 5 minutes.

## Données

Une livraison comporte un numéro de colis, une adresse et un statut.

## Disponibilité

Le service doit être disponible 24h/24.

Une interruption de service est prévue chaque nuit entre 1h et 4h pour les traitements par lots.

---

# 4. Contraintes et périmètre

## Budget

Budget d'investissement : 320 000 €.

Le coût total du projet ne doit pas dépasser 200 000 €.

## Planning

Mise en service : septembre.

## Contraintes réglementaires

Les données de géolocalisation des chauffeurs sont soumises au RGPD.

Les positions sont conservées trois ans pour analyse.

## Hors périmètre

La facturation des transporteurs.
