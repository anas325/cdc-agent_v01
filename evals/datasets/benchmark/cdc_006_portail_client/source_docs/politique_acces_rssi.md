# Politique d'accès et de sécurité applicative — RSSI

Document opposable à tout projet exposant un service à des utilisateurs externes
au groupe. Toute dérogation doit être validée par le comité sécurité.

---

## 1. Authentification des utilisateurs externes

- Les partenaires externes s'authentifient via la fédération **Keycloak partenaires**
  (OIDC), jamais par un annuaire interne.
- **Aucune application ne gère de mot de passe en propre.** La politique de mot de
  passe (12 caractères minimum, rotation annuelle, blocage après 5 échecs pendant
  15 minutes) est appliquée par Keycloak.
- La double authentification (OTP par email ou application) est **obligatoire pour
  tout compte pouvant engager une dépense** — passage de commande inclus.
- Durée de session : 30 minutes d'inactivité, 8 heures maximum.

---

## 2. Modèle d'habilitation

Trois rôles standard sont imposés pour un portail partenaire :

| Rôle | Périmètre |
|---|---|
| `partenaire.lecteur` | consultation des commandes, livraisons et documents de son entité |
| `partenaire.acheteur` | idem + création et validation de commande |
| `partenaire.admin` | idem + gestion des utilisateurs de sa propre entité |

**RG-ACC-01** : le cloisonnement par entité est vérifié côté serveur à chaque
requête. Un identifiant d'entité transmis par le client n'est jamais considéré
comme fiable.

**RG-ACC-02** : la gestion des comptes partenaires est déléguée au
`partenaire.admin` de chaque entité. L'administration interne ne crée que le
premier compte administrateur.

---

## 3. Traçabilité

Sont journalisés et conservés **13 mois** : les connexions et échecs de connexion,
les changements d'habilitation, les téléchargements de document et toute action
engageant une commande. Les journaux sont horodatés et non modifiables.

---

## 4. Exposition et chiffrement

- TLS 1.2 minimum, HSTS activé, pas de contenu mixte.
- Les documents (factures, bons de livraison) sont servis via des URL signées
  valables **15 minutes**, jamais via une URL devinable ou permanente.
- Les données personnelles des contacts partenaires sont chiffrées au repos.

---

## 5. Disponibilité et charge

- Portail partenaire : disponibilité cible **99,5 %** en heures ouvrées, sauvegarde
  quotidienne, RPO 24h, RTO 4h.
- Charge de référence pour un portail de 400 partenaires : **120 utilisateurs
  simultanés**, pic à 300 en fin de mois (clôture de facturation).
- Temps de réponse attendu : moins de **1 seconde** au 95e centile pour une page
  de consultation.

---

## 6. Conservation et suppression

- Un compte partenaire inactif depuis 18 mois est désactivé automatiquement.
- La suppression d'un compte est logique (`disabled_at`) : les journaux et les
  documents restent accessibles à l'administration interne pour les besoins
  d'audit et de contentieux.
