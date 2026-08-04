# Cahier des Charges
## Projet : Maintenance prédictive des lignes de conditionnement

---

# 1. Problématique et contexte

## Contexte

Le site de production exploite 12 lignes de conditionnement.
Les arrêts non planifiés représentent environ 6 % du temps d'ouverture.

Les capteurs installés sur les lignes remontent déjà des mesures vers la plateforme
industrielle, mais personne ne les exploite pour anticiper les pannes.

## Objectif

Réduire les arrêts non planifiés en alertant l'équipe maintenance avant la panne.

## Parties prenantes

- Équipe maintenance
- Responsable de production
- Direction industrielle
- Équipe IT industrielle

---

# 2. Spécifications fonctionnelles

## F1 — Collecte des mesures

Le système récupère les mesures des capteurs.

Les mesures sont historisées.

## F2 — Détection d'anomalie

Un modèle analyse les mesures et détecte les comportements anormaux.

Une alerte est créée quand une anomalie est détectée.

## F3 — Gestion des alertes

L'équipe maintenance consulte les alertes et les traite.

Une alerte traitée est clôturée.

## F4 — Tableau de bord

Un tableau de bord présente l'état des lignes.

Il est mis à jour en temps réel.

## Cas particuliers

Si un capteur ne répond plus, le système le signale.

---

# 3. Spécifications techniques

## Architecture

Le système est déployé sur le cloud du groupe.

Les données transitent par la plateforme industrielle existante.

## Intégrations

Le système consomme l'API de la plateforme industrielle.

Il pousse les interventions vers la GMAO.

## Données

Une mesure comporte un identifiant de capteur, une valeur et un horodatage.

## Performance

Le système doit traiter les mesures rapidement.

Le tableau de bord doit être réactif.

## Sécurité

Les accès sont contrôlés.

---

# 4. Contraintes et périmètre

## Budget

Budget prévisionnel : 180 000 € sur deux ans.

## Planning

Pilote sur deux lignes au premier semestre, généralisation ensuite.

## Hors périmètre

Le remplacement des capteurs existants.

La maintenance curative.
