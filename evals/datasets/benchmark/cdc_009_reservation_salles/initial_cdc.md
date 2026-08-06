# Cahier des Charges
## Projet : Réservation des salles de réunion

---

# 1. Problématique et contexte

## Contexte

Le siège compte 24 salles de réunion réparties sur 3 étages, utilisées par
environ 600 collaborateurs. Les réservations se font aujourd'hui sur un fichier
partagé, ce qui provoque en moyenne 12 conflits de réservation par semaine
constatés par les services généraux.

## Objectif

Supprimer les conflits de réservation et donner une visibilité immédiate sur la
disponibilité des salles.

Indicateur de succès : moins d'un conflit de réservation par semaine, trois mois
après la mise en service.

## Parties prenantes

| Rôle | Description |
|---|---|
| Collaborateur | Réserve une salle pour lui-même ou son équipe |
| Assistant(e) de direction | Réserve au nom d'un tiers, dispose d'un droit prioritaire sur les salles de direction |
| Services généraux | Gèrent le référentiel des salles et arbitrent les litiges |

---

# 2. Spécifications fonctionnelles

## F1 — Consultation des disponibilités

Le collaborateur consulte les salles disponibles sur un créneau donné, filtrées
par étage, capacité et équipement (visioconférence, tableau, écran).

**RG01** : une salle n'est proposée que si elle est libre sur la totalité du
créneau demandé.

*Critère d'acceptation* : une salle réservée n'apparaît jamais dans les
résultats d'une recherche portant sur un créneau qui la chevauche.

## F2 — Réservation

Le collaborateur réserve une salle disponible en indiquant l'objet, l'heure de
début, l'heure de fin et le nombre de participants.

**RG02** : la durée d'une réservation est comprise entre 15 minutes et 4 heures.

**RG03** : une réservation ne peut pas être créée plus de 60 jours à l'avance.

**RG04** : le nombre de participants ne peut pas dépasser la capacité de la salle.

*Critère d'acceptation* : deux réservations concurrentes sur la même salle et le
même créneau se soldent par une réservation acceptée et une refusée avec le
message « Cette salle vient d'être réservée ».

## F3 — Annulation et modification

L'organisateur ou un(e) assistant(e) de direction annule ou modifie une
réservation.

**RG05** : une réservation ne peut plus être modifiée après son heure de début.

**RG06** : une réservation non confirmée dans les 10 minutes suivant son heure de
début est automatiquement libérée (« no-show »).

*Critère d'acceptation* : une réservation non confirmée à H+10 redevient
disponible à la réservation.

## F4 — Référentiel des salles

Les services généraux créent, modifient et désactivent les salles.

**RG07** : la désactivation d'une salle ne supprime jamais les réservations
passées ; les réservations futures sont annulées et leurs organisateurs notifiés.

*Critère d'acceptation* : après désactivation d'une salle, ses organisateurs
futurs reçoivent une notification et l'historique reste consultable.

## Cas particuliers

| Cas | Comportement attendu |
|---|---|
| Réservation à cheval sur deux jours | Refusée : une réservation est bornée à une journée |
| Salle en travaux | Désactivée temporairement par les services généraux, invisible à la recherche |
| Collaborateur ayant quitté l'entreprise | Ses réservations futures sont annulées et notifiées aux services généraux |
| Jour férié | Réservation autorisée, aucune règle particulière |

---

# 3. Spécifications techniques

## Architecture

Application web hébergée sur le cloud privé du groupe, derrière le reverse proxy
du siège. Backend Python FastAPI, frontend React, base PostgreSQL.

## Authentification et habilitations

Authentification par le SSO d'entreprise (Azure AD / OIDC), aucun mot de passe
géré par l'application.

| Rôle | Droits |
|---|---|
| Collaborateur | Consulter, réserver, annuler ses propres réservations |
| Assistant(e) | Consulter, réserver et annuler au nom d'un tiers |
| Services généraux | Tout, plus la gestion du référentiel des salles |

## Intégrations

Synchronisation des réservations vers les agendas Exchange par API Graph, dans le
sens application → agenda uniquement, à chaque création, modification ou
annulation.

Référentiel des collaborateurs lu depuis Azure AD, synchronisation quotidienne à 4h.

## Modèle de données

**Salle** : identifiant, nom, étage, capacité, liste d'équipements, statut
(active / désactivée).

**Réservation** : identifiant, salle, organisateur, créneau (début, fin), objet,
nombre de participants, statut (confirmée / en attente / annulée / libérée),
horodatages de création et de modification.

## Exigences non fonctionnelles

| Exigence | Cible |
|---|---|
| Temps de réponse d'une recherche | < 500 ms au 95e centile |
| Utilisateurs simultanés | 80 en pointe (9h-10h) |
| Disponibilité | 99,5 % en heures ouvrées (7h-20h) |
| Sauvegarde | quotidienne, RPO 24h, RTO 4h |
| Navigateurs | Chrome et Edge, deux dernières versions |
| Traçabilité | création, modification et annulation journalisées 12 mois |

---

# 4. Contraintes et périmètre

## Budget

Budget validé : 60 000 €, hors licences Azure existantes.

## Planning

Livraison le 15 mars, recette du 15 au 31 mars, mise en service le 1er avril.

## Hors périmètre

- La réservation d'équipements mobiles (vidéoprojecteurs, visio portables)
- La commande de restauration liée à une réunion
- Les salles des sites régionaux (siège uniquement en v1)
- Une application mobile native

## Contraintes réglementaires

Les données traitées se limitent à l'identité professionnelle du réservant ;
aucune donnée personnelle sensible. Les journaux sont purgés au bout de 12 mois.
