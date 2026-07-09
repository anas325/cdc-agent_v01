# Politique de sécurité et d'architecture — Direction IT

## Authentification
Toute nouvelle application interne ou orientée client doit utiliser le SSO
d'entreprise (Azure AD / OIDC). Aucune gestion de mot de passe custom n'est
autorisée sauf dérogation validée par la RSSI.

## Hébergement
Les applications sont hébergées sur le cloud privé du groupe (OVHcloud,
région Gravelines). L'hébergement chez un tiers public (AWS, GCP, Azure)
nécessite une validation du comité sécurité.

## Disponibilité
Les applications critiques pour le business (paiement, commande) doivent
viser une disponibilité de 99.9% (SLA), avec sauvegarde quotidienne des
données et un plan de reprise d'activité (RPO 24h, RTO 4h).

## Volumétrie de référence
Pour le e-commerce, le pic de trafic de référence est celui du Black Friday :
environ 5000 utilisateurs simultanés, 50 commandes/minute en pointe.

## Intégrations standard
- Paiement : PSP interne "PayDanone" (API REST, contrat déjà signé).
- Emailing transactionnel : Sendgrid (compte groupe existant).
- ERP : SAP, échange via API REST interne (module OrderSync).
