# Cahier des Charges
## Projet : SmartStock

---

# 1. Problématique et contexte

## Contexte

La société RetailPlus possède 18 magasins de distribution.
Aujourd'hui les stocks sont gérés manuellement dans Excel.
Les ruptures de stock sont fréquentes et les commandes sont souvent effectuées trop tard.

## Objectif

Réduire les ruptures de stock de 25% dans les six premiers mois après le déploiement.

Réduire le temps nécessaire pour préparer une commande fournisseur de 40 minutes à moins de 10 minutes.

## Parties prenantes

- Responsable logistique
- Responsable magasin
- Direction
- Employés de magasin

## Périmètre métier

Le système doit permettre :

- consulter les stocks
- recevoir des alertes
- créer des commandes fournisseurs
- consulter l'historique

---

# 2. Spécifications fonctionnelles

## F1 — Authentification

Les utilisateurs doivent pouvoir se connecter.

Les administrateurs disposent de plus de droits.

### Règle métier

Un utilisateur ne peut modifier que les informations autorisées.

---

## F2 — Gestion des stocks

Les employés peuvent modifier le stock.

Le système doit empêcher les erreurs.

Les mouvements doivent être enregistrés.

---

## F3 — Alertes

Lorsque le stock devient faible, une notification est envoyée.

Le seuil dépend du produit.

---

## F4 — Rapports

Le système génère des rapports.

Les rapports doivent être rapides.

---

## Cas particuliers

Si Internet est indisponible, le système continue de fonctionner.

Les conflits de modification seront gérés automatiquement.

---

# 3. Spécifications techniques

## Architecture

Application Web.

Backend en Python FastAPI.

Frontend React.

Base de données PostgreSQL.

Le système sera déployé sur AWS.

Le système sera installé uniquement sur les serveurs internes.

---

## Intégrations

Connexion à l'ERP.

Synchronisation avec SAP.

Les données sont échangées régulièrement.

---

## Sécurité

Authentification JWT.

Toutes les données sont sécurisées.

Les mots de passe sont chiffrés.

---

## Performance

Le système doit être rapide.

Le temps de réponse doit être inférieur à 5 secondes.

Toutes les recherches doivent répondre en moins de 200 ms.

Le système pourra évoluer facilement.

---

## Disponibilité

Le système devra être disponible 99%.

Le système devra être toujours disponible.

---

## Données

Chaque produit possède :

- id
- nom
- prix
- quantité

Les fournisseurs seront stockés dans une autre base.

---

# 4. Contraintes et périmètre

## Budget

Budget estimé :
150 000 €

Le budget ne devra pas dépasser 80 000 €.

---

## Planning

Livraison prévue :
Décembre 2026.

Le projet devra être terminé dans un délai de 3 mois.

---

## Hors périmètre

Application mobile.

Gestion financière.

---

## Contraintes réglementaires

Respect du RGPD.

Les données pourront être conservées sans limite de durée.