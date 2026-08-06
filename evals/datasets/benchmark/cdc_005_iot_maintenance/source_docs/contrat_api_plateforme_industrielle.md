# Contrat d'interface — Plateforme industrielle (IIoT-Core v3)

Document de référence maintenu par l'équipe IT industrielle. Il fait foi pour
toute application consommant les mesures capteurs du site.

---

## 1. Accès et authentification

- Protocole : API REST sur HTTPS, `https://iiot-core.interne/api/v3`
- Authentification : OAuth2 client credentials, jeton valable 60 minutes
- Chaque application dispose d'un `client_id` dédié par environnement
- Quota : 600 requêtes par minute et par `client_id`, réponse `429` au-delà

---

## 2. Récupération des mesures

### `GET /measures`

Paramètres :

| Paramètre | Obligatoire | Description |
|---|---|---|
| `line_id` | oui | identifiant de ligne (`L01` à `L12`) |
| `from`, `to` | oui | bornes ISO 8601, fenêtre maximale de 24 heures |
| `sensor_type` | non | `vibration`, `temperature`, `pression`, `courant` |
| `limit` | non | défaut 1000, maximum 5000 |

Réponse :

```json
{
  "measures": [
    {
      "sensor_id": "L03-VIB-02",
      "line_id": "L03",
      "sensor_type": "vibration",
      "value": 4.72,
      "unit": "mm/s",
      "quality": "good",
      "recorded_at": "2026-03-11T08:15:00Z"
    }
  ],
  "next_cursor": null
}
```

Le champ `quality` vaut `good`, `uncertain` ou `bad`. **Une mesure `bad` ne doit
jamais être utilisée pour un calcul d'anomalie** : elle indique un défaut de
chaîne d'acquisition, pas un comportement machine.

---

## 3. Flux temps réel

### `wss://iiot-core.interne/api/v3/stream`

- Abonnement par ligne, message JSON identique au format `measures`
- Fréquence d'émission : une mesure par capteur toutes les **10 secondes**
- Le flux ne rejoue pas l'historique : en cas de coupure, rattraper par
  `GET /measures` sur la fenêtre manquante

---

## 4. Détection de capteur muet

Un capteur est considéré **muet** par la plateforme après **3 cycles
d'émission manqués**, soit 30 secondes. Son état passe alors à `stale` et
l'endpoint `GET /sensors/{id}/health` renvoie :

```json
{ "sensor_id": "L03-VIB-02", "state": "stale", "last_seen": "2026-03-11T08:14:30Z" }
```

La plateforme n'envoie aucune notification : c'est à l'application consommatrice
de surveiller l'état.

---

## 5. Volumétrie de référence

- 12 lignes, environ 40 capteurs par ligne, soit ~480 capteurs
- Une mesure par capteur toutes les 10 secondes, soit ~4,1 millions de mesures par jour
- Rétention côté plateforme : **90 jours**, au-delà les mesures sont purgées

---

## 6. Écriture vers la GMAO

La GMAO n'est pas accessible directement. Toute demande d'intervention passe par
la file `gmao.work-orders` (AMQP), message :

```json
{
  "external_ref": "ALERT-2026-0412",
  "line_id": "L03",
  "priority": "haute",
  "description": "Vibration anormale palier moteur",
  "requested_at": "2026-03-11T08:20:00Z"
}
```

Acquittement asynchrone sur `gmao.work-orders.ack` avec le numéro d'ordre GMAO.
Aucune mise à jour ultérieure n'est poussée : l'état de l'intervention se
consulte dans la GMAO.

---

## 7. Engagements de service

- Disponibilité de la plateforme : 99,5 % en heures de production (5h-23h)
- Latence `GET /measures` : moins de 800 ms au 95e centile
- Fenêtre de maintenance hebdomadaire : dimanche 2h-4h, flux interrompu
