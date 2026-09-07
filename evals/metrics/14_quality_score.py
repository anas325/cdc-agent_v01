"""Figure 14 — Évolution du score de qualité des sections.

Source : `scores.json` du dernier run de benchmark (ou de celui passé en
`--run`), bloc `completeness` de chaque cas scoré (`--case` pour n'en garder
qu'un). `src/quality.py::score_section` est appliqué deux fois par section :
une fois en comptant toutes les lacunes détectées comme ouvertes (état
initial), une fois en ne comptant que celles encore ouvertes à la fin. L'écart
mesure le poids de défauts retiré, pas le jugement d'un expert sur le document
final.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _style import BLEU, GRIS, VERT, apply_style, save

import _data


def charger() -> list[tuple[str, str, float, float, float]]:
    """(libellé long, libellé court, score initial, score final, couverture %)."""
    run, _ = _data.cli("scores")
    mode = run.simulator_mode
    cas = []
    for bloc in run.cases():
        completude = bloc["completeness"]
        court = "_".join(bloc["case_id"].split("_")[:2])
        cas.append((
            f"{bloc['case_id']}\n(simulateur « {mode} »)",
            court,
            completude["quality_score_initial"],
            completude["quality_score_final"],
            completude["gt_gap_coverage"] * 100,
        ))
    return cas


def main() -> None:
    cas = charger()
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    x = np.arange(len(cas))
    w = 0.32
    ini = [c[2] for c in cas]
    fin = [c[3] for c in cas]

    b1 = ax1.bar(x - w / 2, ini, w, label="CDC initial", color=GRIS)
    b2 = ax1.bar(x + w / 2, fin, w, label="CDC affiné", color=VERT)
    for barres in (b1, b2):
        for b in barres:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                     f"{b.get_height():.1f}", ha="center", fontsize=8.5,
                     fontweight="bold", color="#4a4f57")
    for xi, (_, _, a, b, _) in zip(x, cas):
        ax1.annotate("", xy=(xi + w / 2, b + 8), xytext=(xi - w / 2, a + 8),
                     arrowprops={"arrowstyle": "->", "color": "#b03a2e", "lw": 1.3})
        ax1.text(xi, max(a, b) + 11, f"+{b - a:.1f}", ha="center", fontsize=9,
                 fontweight="bold", color="#b03a2e")

    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in cas], fontsize=8)
    ax1.set_ylim(0, 122)
    ax1.set_ylabel("Score de qualité des sections (0–100)")
    if len(cas) == 1:
        # Sans voisin, matplotlib serre l'axe sur la seule paire de barres et
        # elles occupent toute la figure : on garde la largeur d'un run à deux cas.
        ax1.set_xlim(-0.9, 0.9)
    ax1.set_title("Poids de défauts retiré du document", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="x", visible=False)

    cov = [c[4] for c in cas]
    ax2.bar(x, cov, 0.42, color=BLEU)
    for xi, v in zip(x, cov):
        ax2.text(xi, v + 1.8, f"{v:.1f} %", ha="center", fontsize=9,
                 fontweight="bold", color="#4a4f57")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[1] for c in cas], fontsize=8.5)
    if len(cas) == 1:
        ax2.set_xlim(-0.9, 0.9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Couverture (%)")
    ax2.set_title("Lacunes annotées trouvées\net refermées", fontsize=10)
    ax2.grid(axis="x", visible=False)

    save(fig, "14_quality_score.png")


if __name__ == "__main__":
    main()
