"""Figure 15 — Où passe le temps de calcul, par nœud du graphe.

Source : les `telemetry.json` du dernier run de benchmark (ou de celui passé en
`--run`), bloc `by_node`, fusionnés sur les cas du run — `--case` pour n'en
garder qu'un. Le nœud `human_input` est déjà exclu par `src/telemetry.py` : sa
durée est le temps de réflexion humain (ici celui du simulateur), pas du temps
machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _style import AMBRE, BLEU, GRIS, apply_style, save

import _data

# En dessous, un nœud ne pèse rien à l'échelle de la figure : il reste dans le
# graphique mais est mentionné en note.
SEUIL_NEGLIGEABLE_S = 0.05


def charger() -> tuple[list[tuple[str, int, float, int, float]], float, str]:
    """(nœud, exécutions, total s, appels LLM, part du calcul), total, étendue."""
    run, _ = _data.cli("telemetry")
    tele = run.telemetrie()
    noeuds = [
        (nom, slot["count"], slot["total_s"], slot.get("llm_calls", 0), slot["share"])
        for nom, slot in sorted(tele["by_node"].items(), key=lambda kv: -kv[1]["total_s"])
    ]
    cas = run.cas_avec_telemetrie()
    etendue = run.case_id or (cas[0] if len(cas) == 1 else f"{len(cas)} cas")
    return noeuds, tele["compute_s"], etendue


def main() -> None:
    noeuds, total_calcul, etendue = charger()
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.3))

    visibles = [n for n in noeuds if n[2] > 0]
    noms = [n[0] for n in visibles]
    totaux = [n[2] for n in visibles]
    llm = [n[3] for n in visibles]
    parts = [n[4] for n in visibles]

    y = np.arange(len(noms))[::-1]
    domine = noms[0] if noms else None
    couleurs = [AMBRE if n == domine else BLEU if a else GRIS
                for n, a in zip(noms, llm)]
    ax.barh(y, totaux, height=0.58, color=couleurs)

    for yi, t, p, c in zip(y, totaux, parts, llm):
        ax.text(t + max(totaux) * 0.012, yi, f"{t:.1f} s  ·  {p:.0%}  ·  {c} appel(s) LLM",
                va="center", fontsize=8, color="#4a4f57")

    ax.set_yticks(y)
    ax.set_yticklabels(noms, fontsize=9)
    ax.set_xlim(0, max(totaux) * 1.45)
    ax.set_xlabel("Temps cumulé (secondes)")
    ax.set_title(f"Répartition du temps de calcul par nœud — {etendue}, "
                 f"{_data.fr(total_calcul)} s au total")
    ax.grid(axis="y", visible=False)

    negligeables = [n for n in noeuds if n[2] < SEUIL_NEGLIGEABLE_S]
    if negligeables:
        suite = ("aucun appel LLM sur ce run."
                 if all(n[3] == 0 for n in negligeables) else "aiguillage et écritures d'état.")
        sujet = (f"Le nœud {negligeables[0][0]} pèse" if len(negligeables) == 1
                 else f"Les nœuds {', '.join(n[0] for n in negligeables)} pèsent")
        ax.text(0.0, -0.19, f"{sujet} moins de {int(SEUIL_NEGLIGEABLE_S * 1000)} ms "
                            f"cumulées : {suite}",
                transform=ax.transAxes, fontsize=7.6, color=GRIS)

    save(fig, "15_temps_par_noeud.png")


if __name__ == "__main__":
    main()
