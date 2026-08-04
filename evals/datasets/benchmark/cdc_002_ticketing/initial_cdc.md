# Cahier des Charges Technique
## Projet : Outil interne de suivi des demandes IT (Ticketing léger)

---

## 1. Présentation du projet

### 1.1 Contexte
L'équipe IT interne gère actuellement les demandes (bugs, accès, matériel) via des emails et un fichier Excel partagé. Cela génère des pertes de suivi, des doublons, et aucune visibilité sur les délais de traitement.

### 1.2 Objectifs
- Centraliser toutes les demandes IT dans un outil unique
- Réduire le délai moyen de traitement de 3 jours à 1 jour
- Donner une visibilité en temps réel aux demandeurs sur le statut de leur ticket

### 1.3 Périmètre
**Inclus :**
- Création, suivi et clôture de tickets
- Notifications par email
- Tableau de bord basique (nombre de tickets par statut/priorité)

**Exclus (hors périmètre v1) :**
- Intégration avec Active Directory
- Application mobile
- Chat en temps réel

### 1.4 Enjeux et gains attendus
- Gain de temps estimé : ~5h/semaine pour l'équipe IT
- Meilleure traçabilité pour les audits internes

---

## 2. Acteurs et parties prenantes

| Rôle | Description | Exemple |
|---|---|---|
| Demandeur | Créé un ticket, suit son avancement | Tout employé |
| Agent IT | Traite les tickets assignés | Équipe support IT |
| Admin | Gère les utilisateurs, catégories, priorités | Responsable IT |

---

## 3. Spécifications fonctionnelles

### F01 — Création d'un ticket
- Le demandeur remplit : titre, description, catégorie (Bug / Accès / Matériel / Autre), priorité (Basse/Moyenne/Haute)
- Un fichier joint (max 10 Mo) peut être ajouté
- **Règle de gestion RG01** : un ticket ne peut pas être créé sans catégorie sélectionnée

### F02 — Attribution automatique
- Chaque catégorie est liée à une file d'agents par défaut
- **RG02** : si aucun agent n'est disponible dans la catégorie, le ticket reste "Non assigné" et une alerte est envoyée à l'admin après 4h

### F03 — Suivi de statut
- Statuts possibles : `Ouvert` → `En cours` → `En attente demandeur` → `Résolu` → `Clôturé`
- **RG03** : un ticket passé en "Résolu" est auto-clôturé après 3 jours sans réponse du demandeur

### F04 — Notifications
- Email envoyé au demandeur à chaque changement de statut
- Email envoyé à l'agent lors d'une nouvelle assignation

### F05 — Tableau de bord
- Vue agrégée : nombre de tickets par statut, par priorité, temps moyen de résolution
- Filtrable par période et par catégorie

*(Cas d'usage détaillé, exemple)*
> **En tant que** demandeur, **je veux** être notifié quand mon ticket change de statut, **afin de** ne pas avoir à relancer l'IT par email.

---

## 4. Spécifications techniques

### 4.1 Architecture
```
[Frontend React] <--> [API REST (Node/Express)] <--> [PostgreSQL]
                              |
                        [Service Email (SMTP)]
```

### 4.2 Stack technique
- Frontend : React + Tailwind
- Backend : Node.js / Express
- Base de données : PostgreSQL
- Auth : JWT (session 8h)

### 4.3 Modèle de données (extrait)

**Table `tickets`**
| Champ | Type | Contrainte |
|---|---|---|
| id | UUID | PK |
| title | varchar(200) | NOT NULL |
| description | text | NOT NULL |
| category | enum | NOT NULL |
| priority | enum | default 'medium' |
| status | enum | default 'open' |
| requester_id | UUID | FK -> users |
| assigned_agent_id | UUID | FK -> users, nullable |
| created_at | timestamp | default now() |
| updated_at | timestamp | |

### 4.4 API (extrait)
```
POST   /api/tickets              → créer un ticket
GET    /api/tickets?status=open  → lister les tickets
PATCH  /api/tickets/:id/status   → changer le statut
GET    /api/dashboard/stats      → stats agrégées
```

### 4.5 Intégrations
- SMTP interne existant pour les emails (pas de nouveau service à provisionner)

---

## 5. Exigences non-fonctionnelles

| Exigence | Cible |
|---|---|
| Temps de réponse API | < 300ms pour 95% des requêtes |
| Disponibilité | 99% (heures ouvrées) |
| Utilisateurs simultanés | 50 max (usage interne) |
| Navigateurs supportés | Chrome, Edge (2 dernières versions) |
| Sécurité | HTTPS obligatoire, mots de passe hashés (bcrypt) |

---

## 6. Contraintes

- Doit être déployable sur l'infra interne existante (Docker + serveur on-premise)
- Pas de budget pour un service tiers payant (ex: Zendesk)
- Doit être livré avant la fin du trimestre

---

## 7. Livrables

- Code source (repo Git)
- Documentation d'installation (README)
- Jeu de données de test
- Recette fonctionnelle (checklist des F01 à F05 validées)

### Critères d'acceptation (exemples)
- [ ] Un ticket créé apparaît immédiatement dans le tableau de bord
- [ ] Un email est reçu dans les 30 secondes suivant un changement de statut
- [ ] Un ticket "Résolu" sans réponse est bien auto-clôturé après 3 jours

---

## 8. Planning (indicatif)

| Lot | Contenu | Durée estimée |
|---|---|---|
| Lot 1 | Modèle de données + API tickets (F01-F03) | 1 semaine |
| Lot 2 | Notifications email (F04) | 3 jours |
| Lot 3 | Tableau de bord (F05) | 4 jours |
| Lot 4 | Tests + recette | 3 jours |

---

## 9. Annexes

**Glossaire**
- *Ticket* : demande unitaire créée par un utilisateur
- *Agent* : membre de l'équipe IT traitant les tickets
- *SLA* : Service Level Agreement, engagement de délai

**Contact**
- Référent métier : [nom]
- Référent technique : [nom]
