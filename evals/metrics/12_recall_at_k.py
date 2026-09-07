"""Figure 12 — Performance de la récupération documentaire (RAG).

Source : `scores.json` du dernier run de benchmark (ou de celui passé en
`--run`), bloc `totals.retrieval` — ou le bloc du cas choisi avec `--case`.
Deux dénominateurs : les récupérations réellement tentées (`attempted`) et
l'ensemble des lacunes annotées `resolvable_by: "rag"` (`annotated`), une
lacune jamais détectée comptant comme un échec. Le diagnostic de droite vient
de `sufficiency_judgment` et du détail `per_gap` de chaque cas.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _style import AMBRE, BLEU, ROUGE, VERT, apply_style, save

import _data


def charger() -> dict:
    run, _ = _data.cli("scores")
    reperage = run.bloc("retrieval")

    def taux(bloc: dict) -> dict[str, float]:
        return {f"Recall@{k.lstrip('@')}": v * 100
                for k, v in sorted(bloc.items(), key=lambda kv: int(kv[0].lstrip("@")))}

    jugement = reperage["sufficiency_judgment"]
    # Une lacune tentée mais dont aucun extrait pertinent n'est remonté : le
    # rang est nul. Le détail per_gap n'existe qu'au niveau des cas.
    jamais = sum(
        1
        for cas in run.cases()
        for gap in cas["retrieval"]["per_gap"]
        if gap["attempted"] and gap["rank"] is None
    )
    return {
        "tentees": taux(reperage["recall_at_k"]),
        "global": taux(reperage["recall_at_k_overall"]),
        "mrr": (reperage["mrr"], reperage["mrr_overall"]),
        "attempted": reperage["attempted"],
        "annotated": reperage["annotated"],
        "diagnostic": [
            ("Extrait récupéré\net accepté", jugement["correct_accept"], VERT),
            ("Récupéré mais rejeté\n→ problème de raisonnement", jugement["wrong_reject"], AMBRE),
            ("Jamais récupéré\n→ problème de récupération", jamais, ROUGE),
        ],
    }


def main() -> None:
    d = charger()
    tentees, glob, mrr = d["tentees"], d["global"], d["mrr"]
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    ks = list(tentees)
    x = np.arange(len(ks))
    w = 0.38
    b1 = ax1.bar(x - w / 2, [tentees[k] for k in ks], w,
                 label=f"Sur récupérations tentées (n={d['attempted']})", color=BLEU)
    b2 = ax1.bar(x + w / 2, [glob[k] for k in ks], w,
                 label=f"Sur toutes les lacunes RAG annotées (n={d['annotated']})", color="#7fa8cd")
    for barres in (b1, b2):
        for b in barres:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                     f"{b.get_height():.1f}", ha="center", fontsize=7.8, color="#4a4f57")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ks)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Rappel (%)")
    ax1.set_title(f"Rappel documentaire — MRR {mrr[0]:.2f} / {mrr[1]:.2f}", fontsize=10)
    ax1.legend(fontsize=7.6, loc="upper left")
    ax1.grid(axis="x", visible=False)

    noms = [n for n, _, _ in d["diagnostic"]]
    vals = [v for _, v, _ in d["diagnostic"]]
    cols = [c for _, _, c in d["diagnostic"]]
    y = np.arange(len(noms))[::-1]
    ax2.barh(y, vals, height=0.55, color=cols)
    for yi, v in zip(y, vals):
        ax2.text(v + 0.12, yi, str(v), va="center", fontsize=9, fontweight="bold",
                 color="#4a4f57")
    ax2.set_yticks(y)
    ax2.set_yticklabels(noms, fontsize=8)
    ax2.set_xlim(0, max(5, max(vals) * 1.25))
    ax2.set_xlabel("Nombre de lacunes")
    ax2.set_title("Récupération ou raisonnement ?", fontsize=10)
    ax2.grid(axis="y", visible=False)

    save(fig, "12_recall_at_k.png")


if __name__ == "__main__":
    main()
