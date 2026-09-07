"""Style commun aux figures du rapport.

Autonome : n'importe rien du code du projet. Palette sobre, lisible en
impression noir et blanc comme à l'écran.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_RACINE = Path(__file__).resolve().parents[2]
FIG_DIR = _RACINE / "report" / "figures"
DPI = 200

# Palette : bleu institutionnel, ambre, vert, rouge, gris.
BLEU = "#1f4e79"
BLEU_CLAIR = "#7fa8cd"
AMBRE = "#c8860d"
VERT = "#2e7d4f"
ROUGE = "#b03a2e"
GRIS = "#8a8f98"
GRIS_CLAIR = "#d7dade"

PALETTE = [BLEU, AMBRE, VERT, ROUGE, BLEU_CLAIR, GRIS]


def set_output_dir(path: str | Path) -> None:
    """Rediriger les figures (option --out des scripts)."""
    global FIG_DIR
    FIG_DIR = Path(path).resolve()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#4a4f57",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRIS_CLAIR,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"écrit : {path}")
    return path
