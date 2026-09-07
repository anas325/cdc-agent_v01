"""Figure 13 — Qualité des questions posées à l'auteur.

Source : `scores.json` du dernier run de benchmark (ou de celui passé en
`--run`), bloc `totals.questions` — ou le bloc du cas choisi avec `--case`.
Chaque dimension est notée 0, 1 ou 2 par des heuristiques déterministes —
aucun appel LLM (le juge LLM optionnel de `run_scoring.py` reste à côté).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _style import AMBRE, BLEU, ROUGE, VERT, apply_style, save

import _data

# Glose française des dimensions produites par evals/scoring.py.
GLOSE = {
    "understandable": "compréhensible",
    "not_duplicate": "non redondante",
    "has_context": "cite le CDC",
    "addresses_gap": "vise la lacune",
    "specific": "chiffrée ou citée",
    "answerable": "a reçu une réponse",
}


def charger() -> tuple[list[tuple[str, float]], float, int]:
    run, _ = _data.cli("scores")
    questions = run.bloc("questions")
    dimensions = [
        (f"{nom}\n({GLOSE.get(nom, nom)})", note)
        for nom, note in sorted(questions["by_dimension"].items(), key=lambda kv: -kv[1])
    ]
    return dimensions, questions["mean_score"], questions["asked"]


def couleur(v: float) -> str:
    if v >= 1.7:
        return VERT
    if v >= 1.2:
        return BLEU
    if v >= 0.8:
        return AMBRE
    return ROUGE


def main() -> None:
    dimensions, moyenne, posees = charger()
    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))

    noms = [n for n, _ in dimensions]
    vals = [v for _, v in dimensions]
    y = np.arange(len(noms))[::-1]

    ax.barh(y, vals, height=0.58, color=[couleur(v) for v in vals])
    for yi, v in zip(y, vals):
        ax.text(v + 0.04, yi, f"{v:.2f}", va="center", fontsize=9,
                fontweight="bold", color="#4a4f57")

    ax.axvline(moyenne, color="#4a4f57", linestyle="--", linewidth=1.1)
    ax.text(moyenne + 0.03, len(noms) - 0.55, f"moyenne globale {moyenne:.2f}",
            fontsize=8, color="#4a4f57")

    ax.set_yticks(y)
    ax.set_yticklabels(noms, fontsize=8.2)
    ax.set_xlim(0, 2.15)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("Note moyenne sur 2")
    ax.set_title(f"Qualité des {posees} questions posées — "
                 f"{len(dimensions)} dimensions déterministes")
    ax.grid(axis="y", visible=False)

    save(fig, "13_qualite_questions.png")


if __name__ == "__main__":
    main()
