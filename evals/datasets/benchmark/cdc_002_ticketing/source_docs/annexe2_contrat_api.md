# Annexe 2 — Contrat d'API détaillé
## Projet : Outil interne de suivi des demandes IT

Ce document précise les schémas de requêtes/réponses et les codes d'erreur pour chaque endpoint, afin d'éliminer toute ambiguïté d'implémentation côté back/front.

---

## Conventions générales

- Format : JSON uniquement (`Content-Type: application/json`)
- Auth : header `Authorization: Bearer <JWT>` sur tous les endpoints sauf `/auth/login`
- Dates : format ISO 8601 (`2026-07-09T14:30:00Z`)
- Pagination : `?page=1&limit=20` (défaut `limit=20`, max `limit=100`)

### Format d'erreur standard
```json
{
  "error": {
    "code": "TICKET_INVALID_CATEGORY",
    "message": "La catégorie sélectionnée n'existe pas",
    "field": "category"
  }
}
```

### Codes HTTP utilisés
| Code | Usage |
|---|---|
| 200 | Succès (GET, PATCH) |
| 201 | Ressource créée (POST) |
| 400 | Requête invalide (validation) |
| 401 | Non authentifié |
| 403 | Authentifié mais non autorisé (permissions) |
| 404 | Ressource introuvable |
| 409 | Conflit (ex: transition de statut invalide) |
| 500 | Erreur serveur |

---

## 1. `POST /api/tickets`
Crée un nouveau ticket.

**Requête**
```json
{
  "title": "Écran externe ne s'allume plus",
  "description": "Le moniteur Dell ne s'allume plus depuis ce matin",
  "category": "materiel",
  "priority": "medium",
  "attachments": ["file_uuid_1"]
}
```

**Contraintes de validation**
| Champ | Type | Requis | Contrainte |
|---|---|---|---|
| title | string | oui | 5-200 caractères |
| description | string | oui | 10-5000 caractères |
| category | enum | oui | `bug`, `acces`, `materiel`, `autre` |
| priority | enum | non | `low`, `medium`, `high` — défaut `medium` |
| attachments | array[uuid] | non | max 3, fichiers déjà uploadés via `/api/files` |

**Réponse 201**
```json
{
  "id": "a1b2c3d4-...",
  "title": "Écran externe ne s'allume plus",
  "status": "open",
  "category": "materiel",
  "priority": "medium",
  "requester_id": "u-123",
  "assigned_agent_id": null,
  "created_at": "2026-07-09T14:30:00Z"
}
```

**Erreurs possibles**
- `400 TICKET_TITLE_TOO_SHORT`
- `400 TICKET_INVALID_CATEGORY`
- `400 TICKET_ATTACHMENT_TOO_LARGE`
- `400 TICKET_TOO_MANY_ATTACHMENTS`

---

## 2. `GET /api/tickets`
Liste les tickets (filtrée selon le rôle — voir Annexe 1, section 5).

**Query params**
| Param | Type | Exemple |
|---|---|---|
| status | enum | `?status=open` |
| category | enum | `?category=bug` |
| priority | enum | `?priority=high` |
| assigned_to | uuid | `?assigned_to=u-45` |
| page / limit | int | `?page=2&limit=20` |

**Réponse 200**
```json
{
  "data": [ { "id": "...", "title": "...", "status": "open", "...": "..." } ],
  "pagination": { "page": 1, "limit": 20, "total": 143, "total_pages": 8 }
}
```

---

## 3. `PATCH /api/tickets/:id/status`
Change le statut d'un ticket. Valide la transition selon le diagramme d'état (Annexe 1, section 3).

**Requête**
```json
{ "status": "resolved", "comment": "Câble remplacé, testé OK" }
```

**Réponse 200**
```json
{ "id": "a1b2c3d4-...", "status": "resolved", "updated_at": "2026-07-09T16:00:00Z" }
```

**Erreurs possibles**
- `409 INVALID_STATUS_TRANSITION` — ex: tentative de passer directement de "Ouvert" à "Clôturé"
- `403 FORBIDDEN_STATUS_CHANGE` — un demandeur tente de changer le statut (hors réouverture)
- `404 TICKET_NOT_FOUND`

---

## 4. `PATCH /api/tickets/:id/assign`
Réassigne un ticket (admin uniquement).

**Requête**
```json
{ "agent_id": "u-45" }
```

**Erreurs possibles**
- `403 FORBIDDEN` — utilisateur non-admin
- `400 AGENT_NOT_IN_CATEGORY` — l'agent choisi n'appartient pas à la file de la catégorie du ticket
- `400 SELF_ASSIGNMENT_FORBIDDEN` — agent_id == requester_id du ticket

---

## 5. `GET /api/dashboard/stats`
Retourne les statistiques agrégées.

**Query params**
| Param | Exemple |
|---|---|
| from / to | `?from=2026-06-01&to=2026-06-30` |
| category | `?category=bug` (optionnel) |

**Réponse 200**
```json
{
  "total_tickets": 87,
  "by_status": { "open": 12, "in_progress": 8, "waiting": 3, "resolved": 20, "closed": 44 },
  "by_priority": { "low": 30, "medium": 45, "high": 12 },
  "avg_resolution_time_hours": 26.4,
  "empty_period": false
}
```

**Cas particulier** : si aucune donnée sur la période → `"empty_period": true` et les autres champs à `0`/`{}` (le frontend affiche le message "Aucune donnée" plutôt qu'un graphique cassé — voir Annexe 1 section 6).

---

## 6. `POST /api/files`
Upload d'une pièce jointe (appelé avant la création/mise à jour d'un ticket).

**Requête** : `multipart/form-data`, champ `file`

**Réponse 201**
```json
{ "file_id": "file_uuid_1", "filename": "photo_ecran.jpg", "size_bytes": 240000, "mime_type": "image/jpeg" }
```

**Erreurs possibles**
- `400 FILE_TOO_LARGE` (> 10 Mo)
- `400 FILE_TYPE_NOT_ALLOWED` (hors liste : pdf, png, jpg, docx, xlsx, log, txt)
