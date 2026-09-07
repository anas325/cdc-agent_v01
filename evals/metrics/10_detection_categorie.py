"""Figure 10 — Détection des lacunes par catégorie.

Source : `scores.json` du dernier run de benchmark (ou de celui passé en
`--run`), bloc `totals.gaps.by_category` — ou le bloc du cas choisi avec
`--case`. `detection_recall` ignore l'étiquette de catégorie et ne demande que
« la lacune a-t-elle été remontée ? » ; `recall` strict exige en plus la bonne
étiquette.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _style import BLEU, BLEU_CLAIR, GRIS, apply_style, save

import _data


def charger() -> dict[str, tuple[int, float, float, float]]:
    """catégorie : (annotées, detection_recall, recall strict, précision)."""
    run, _ = _data.cli("scores")
    par_categorie = run.bloc("gaps")["by_category"]
    return {
        cat: (m["annotated"], m["detection_recall"], m["recall"], m["precision"])
        for cat, m in sorted(par_categorie.items())
    }


def main() -> None:
    donnees = charger()
    apply_style()
    cats = list(donnees)
    x = np.arange(len(cats))
    largeur = 0.38

    det = [donnees[c][1] * 100 for c in cats]
    strict = [donnees[c][2] * 100 for c in cats]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    b1 = ax.bar(x - largeur / 2, det, largeur, label="Lacune remontée (found)", color=BLEU)
    b2 = ax.bar(x + largeur / 2, strict, largeur, label="Rappel strict (bonne catégorie)", color=BLEU_CLAIR)

    for barres in (b1, b2):
        for b in barres:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 2, f"{h:.0f}", ha="center",
                    fontsize=7.5, color="#4a4f57")

    etiquettes = [f"{c}\n(n={donnees[c][0]})" for c in cats]
    ax.set_xticks(x)
    ax.set_xticklabels(etiquettes, fontsize=7.6)
    ax.set_ylabel("Taux (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Détection des lacunes par catégorie — écart entre « remontée » et « bien étiquetée »")
    ax.legend(fontsize=8, loc="upper right", ncols=2)
    ax.grid(axis="x", visible=False)
    ax.text(0.0, -0.34, "n = nombre de lacunes annotées par un expert dans cette catégorie",
            transform=ax.transAxes, fontsize=7.5, color=GRIS)

    save(fig, "10_detection_categorie.png")


if __name__ == "__main__":
    main()
