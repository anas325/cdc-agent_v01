"""Notifications Telegram — un util minimal pour prévenir en dehors du terminal.

Un run de benchmark dure des heures sans surveillance ; ce module permet d'en
recevoir les étapes importantes sur son téléphone. Deux règles guident sa forme :

- **silencieux si non configuré** : sans `TELEGRAM_BOT_TOKEN` /
  `TELEGRAM_CHAT_ID` dans `src/.env`, `send_message` ne fait rien et renvoie
  False, si bien qu'un appelant n'a pas à se demander s'il peut appeler ;
- **jamais bloquant** : une panne réseau ou un refus de l'API est journalisé,
  jamais propagé — une notification perdue ne doit pas faire échouer le travail
  qu'elle ne fait que décrire.

Volontairement sur `urllib` (stdlib) : un POST JSON ne justifie pas une
dépendance de plus.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"

# L'API rejette les messages au-delà de 4096 caractères : on tronque plutôt que
# de perdre l'update entier.
MAX_LEN = 4096


def is_configured() -> bool:
    """Vrai si le token du bot et le chat id sont tous deux dans l'environnement."""
    return bool(os.getenv(TOKEN_ENV) and os.getenv(CHAT_ENV))


def send_message(
    text: str,
    *,
    chat_id: str | None = None,
    silent: bool = False,
    timeout: float = 10.0,
) -> bool:
    """Envoie `text` sur Telegram ; renvoie True si l'API a accepté le message.

    `chat_id` surcharge la destination par défaut (`TELEGRAM_CHAT_ID`) et
    `silent=True` demande une notification sans son, pour les updates de routine.
    """
    token = os.getenv(TOKEN_ENV)
    chat_id = chat_id or os.getenv(CHAT_ENV)
    if not token or not chat_id:
        logger.debug("Telegram non configuré (%s / %s) — message ignoré", TOKEN_ENV, CHAT_ENV)
        return False

    if len(text) > MAX_LEN:
        text = text[: MAX_LEN - 1] + "…"

    payload = json.dumps(
        {"chat_id": chat_id, "text": text, "disable_notification": silent}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE}/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        # On journalise le corps de la réponse ("chat not found", token révoqué…)
        # et surtout pas l'exception nue : son `url` contient le token du bot.
        body = exc.read().decode("utf-8", "replace")[:200]
        logger.warning("Telegram a refusé le message (HTTP %s) : %s", exc.code, body)
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Telegram injoignable : %s", exc)
    return False




if __name__ == "__main__":
    # Test manuel : `uv run python -m src.utils.notify [texte]`. C'est le seul
    # moyen de vérifier les credentials — l'envoi étant best-effort, un token
    # invalide est invisible depuis les appelants.
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    message = " ".join(sys.argv[1:]) or "Test depuis src/utils/notify.py ✅"
    if not is_configured():
        print(f"Non configuré : renseignez {TOKEN_ENV} et {CHAT_ENV} dans src/.env")
        raise SystemExit(1)
    ok = send_message(message)
    print("envoyé" if ok else "échec — voir le log ci-dessus")
    raise SystemExit(0 if ok else 1)
