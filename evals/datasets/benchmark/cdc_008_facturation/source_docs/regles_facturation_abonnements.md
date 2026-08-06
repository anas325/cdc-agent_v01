# Règles de gestion — Facturation des abonnements

Référentiel métier maintenu par la direction financière. Il fait foi pour tout
calcul de facture, d'avoir ou de relance.

---

## 1. Cycle de facturation

- La facture est émise le **1er du mois** pour le mois en cours (facturation à échoir).
- Date d'échéance : **15 jours** après émission.
- La numérotation est **séquentielle, continue et sans trou** sur l'exercice, au
  format `FA-{exercice}-{séquence sur 6 chiffres}`. Une facture émise ne peut
  jamais être supprimée ni modifiée — seule l'émission d'un avoir la corrige.

---

## 2. Changement de formule en cours de période

**RG-FAC-01** : le changement prend effet au **premier jour du mois suivant** si
la formule est moins chère (rétrogradation), et **immédiatement** si elle est
plus chère (montée en gamme).

**RG-FAC-02** : en cas de montée en gamme, la différence est facturée au
**prorata du nombre de jours restants**, calculée sur une base de 30 jours,
arrondie au centime supérieur.

**RG-FAC-03** : deux changements de formule au maximum par abonné et par mois.
Le troisième est refusé avec le message « Limite de changements atteinte pour ce mois ».

---

## 3. Remises

| Type | Règle |
|---|---|
| Remise commerciale ponctuelle | En pourcentage, plafonnée à 30 %, validée par un responsable au-delà de 15 % |
| Remise fidélité | 5 % automatique à partir de 24 mois d'ancienneté continue |
| Remise de parrainage | Un mois offert, non cumulable avec la remise commerciale |

**RG-REM-01** : les remises ne se cumulent jamais au-delà de **35 %** du montant HT.
En cas de dépassement, seule la remise la plus favorable est appliquée.

**RG-REM-02** : une remise n'est jamais appliquée rétroactivement sur une facture
déjà émise ; elle prend effet à la facture suivante.

---

## 4. Résiliation

- La résiliation prend effet à la **fin de la période facturée** : aucun
  remboursement au prorata.
- Une résiliation demandée **dans les 14 jours** suivant la souscription initiale
  ouvre droit à un **remboursement intégral** (droit de rétractation).
- Un abonné résilié reste facturable pour toute consommation antérieure non réglée.

---

## 5. Recouvrement

| Échéance dépassée de | Action |
|---|---|
| 3 jours | Relance automatique par email |
| 10 jours | Deuxième relance par email + SMS |
| 21 jours | Mise en demeure, suspension du service |
| 45 jours | Transfert au recouvrement (hors périmètre applicatif) |

**RG-REC-01** : la suspension du service n'entraîne **pas** l'arrêt de la
facturation tant que l'abonnement n'est pas résilié.
**RG-REC-02** : aucune relance n'est envoyée pour un solde inférieur à **5 € TTC**.

---

## 6. Avoirs

- Un avoir porte toujours sur **une facture précise** et ne peut jamais dépasser
  son montant.
- Numérotation séparée `AV-{exercice}-{séquence}`, également continue.
- Un avoir supérieur à **200 € TTC** requiert la validation d'un responsable
  comptable.
- Les avoirs partiels sont autorisés ; le cumul des avoirs sur une facture ne
  peut pas dépasser son montant TTC.

---

## 7. Obligations légales

- Mentions obligatoires : identité et TVA intracommunautaire des deux parties,
  numéro et date, désignation, montant HT, taux et montant de TVA, montant TTC,
  date d'échéance, mention des pénalités de retard.
- Conservation des factures : **10 ans**.
- Les écritures comptables sont exportées quotidiennement au format FEC.
