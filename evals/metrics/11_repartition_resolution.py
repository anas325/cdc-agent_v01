"""Figure 11 — Provenance des informations produites et effort humain épargné.

Source : `scores.json` du dernier run de benchmark (ou de celui passé en
`--run`), bloc `totals.effort` — ou le bloc du cas choisi avec `--case`.
`resolved_by_rag`, `resolved_by_human` et `assumed` comptent les ContextItem
ajoutés au CDC initial ; `human_intervention_reduction` donne la part des
lacunes fermées sans solliciter l'auteur.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from _style import AMBRE, BLEU, GRIS_CLAIR, VERT, apply_style, save

import _data


def charger() -> tuple[list[tuple[str, int, str]], float]:
    run, _ = _data.cli("scores")
    effort = run.bloc("effort")
    provenance = [
        ("Recherche documentaire\n(RAG)", effort["resolved_by_rag"], VERT),
        ("Réponse de\nl'auteur", effort["resolved_by_human"], BLEU),
        ("Hypothèse proposée\npar le système", effort["assumed"], AMBRE),
    ]
    return provenance, effort["human_intervention_reduction"] * 100


def main() -> None:
    provenance, hir = charger()
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    valeurs = [v for _, v, _ in provenance]
    total = sum(valeurs)
    ax1.pie(
        valeurs,
        labels=[f"{n}\n{v} ({v / total:.0%})" for n, v, _ in provenance],
        colors=[c for _, _, c in provenance],
        startangle=90, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.8},
        textprops={"fontsize": 8.4},
        labeldistance=1.16,
    )
    ax1.set_title(f"Provenance des {total} informations ajoutées au CDC", fontsize=10)

    ax2.barh([0], [hir], height=0.42, color=VERT)
    ax2.barh([0], [100 - hir], left=[hir], height=0.42, color=GRIS_CLAIR)
    ax2.text(hir / 2, 0, f"{hir:.1f} %", ha="center", va="center",
             color="white", fontweight="bold", fontsize=11)
    ax2.text(hir + (100 - hir) / 2, 0, "sollicitation\nde l'auteur", ha="center",
             va="center", color="#4a4f57", fontsize=8.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("Part des lacunes fermées sans l'auteur (%)")
    ax2.set_title("Réduction de l'intervention humaine", fontsize=10)
    ax2.grid(axis="y", visible=False)
    for cote in ("left", "right", "top"):
        ax2.spines[cote].set_visible(False)

    save(fig, "11_repartition_resolution.png")


if __name__ == "__main__":
    main()
