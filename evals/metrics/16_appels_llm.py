"""Figure 16 — Coût des appels LLM par schéma de sortie.

Source : les `telemetry.json` du dernier run de benchmark (ou de celui passé en
`--run`), clé `calls` — le détail par appel écrit par `evals/run_benchmark.py`,
fusionné sur les cas du run (`--case` pour n'en garder qu'un). Chaque appel
passe par `call_structured` et rend un modèle Pydantic ; le schéma est donc
l'unité d'analyse naturelle.

Les appels servis par le cache disque (`CDC_LLM_CACHE=1`) sont écartés par
défaut : leur durée mesure le disque, pas le modèle. `--include-cache` les
réintègre.

Les runs antérieurs à l'enregistrement du détail par appel n'ont que des
agrégats par schéma, d'où l'échappatoire `--calls-csv` : le CSV « Appels LLM »
exporté par l'onglet « Sous le capot » de l'interface Streamlit a les mêmes
colonnes.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from _style import AMBRE, BLEU, PALETTE, VERT, apply_style, save

import _data

# Couleurs stables pour les schémas historiques ; les autres prennent la palette.
COULEURS = {
    "GapFinderOutput": AMBRE,
    "RagGrade": VERT,
    "QuestionDraft": BLEU,
    "DedupVerdict": "#7fa8cd",
}


def _depuis_csv(path: Path) -> list[tuple[str, float, int]]:
    """(schéma, durée s, tentatives) depuis un export « Appels LLM »."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        lignes = list(csv.reader(fh))
    entete, corps = lignes[0], lignes[1:]
    i_schema, i_duree, i_essais = 1, 2, 4
    for i, nom in enumerate(entete):
        if nom.startswith("Schéma"):
            i_schema = i
        elif nom.startswith("Durée"):
            i_duree = i
        elif nom.startswith("Tentatives"):
            i_essais = i
    return [(r[i_schema], float(r[i_duree]), int(r[i_essais])) for r in corps if r]


def charger() -> list[tuple[str, float, int]]:
    def ajouts(parser) -> None:
        parser.add_argument("--calls-csv", default=None,
                            help="export CSV « Appels LLM » à utiliser à la place "
                                 "du détail par appel du run")
        parser.add_argument("--include-cache", action="store_true",
                            help="compter aussi les appels servis par le cache disque")

    externe = any(a.startswith("--calls-csv") for a in sys.argv)
    besoin = "any" if externe else "telemetry"
    run, args = _data.cli(besoin, ajouts=ajouts)

    if args.calls_csv:
        chemin = Path(args.calls_csv)
        print(f"source : {chemin}", file=sys.stderr)
        return _depuis_csv(chemin)

    appels = run.telemetrie().get("calls")
    if not appels:
        raise SystemExit(
            f"{run.dir.name} ne contient pas le détail par appel "
            "(telemetry.json, cle \"calls\") : relancer le benchmark, ou passer "
            "--calls-csv sur un export de l'onglet « Sous le capot »."
        )

    retenus = [a for a in appels if args.include_cache or not a.get("cache_hit")]
    ecartes = len(appels) - len(retenus)
    if ecartes:
        print(f"{ecartes} appel(s) servis par le cache écartés "
              f"(--include-cache pour les garder)", file=sys.stderr)
    if not retenus:
        raise SystemExit("tous les appels de ce run viennent du cache : "
                         "utiliser --include-cache, ou un run à froid.")
    return [(a["schema"], a["duration_s"], a.get("attempts", 1)) for a in retenus]


def main() -> None:
    appels = charger()
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    par_schema: dict[str, list[float]] = defaultdict(list)
    for schema, duree, _ in appels:
        par_schema[schema].append(duree)

    schemas = sorted(par_schema, key=lambda s: -sum(par_schema[s]))
    couleurs = {s: COULEURS.get(s, PALETTE[i % len(PALETTE)])
                for i, s in enumerate(schemas)}
    totaux = [sum(par_schema[s]) for s in schemas]
    nb = [len(par_schema[s]) for s in schemas]
    y = np.arange(len(schemas))[::-1]

    ax1.barh(y, totaux, height=0.56, color=[couleurs[s] for s in schemas])
    for yi, t, n in zip(y, totaux, nb):
        dedans = t > 0.68 * max(totaux)
        ax1.text(t - max(totaux) * 0.016 if dedans else t + max(totaux) * 0.013, yi,
                 f"{t:.0f} s ({n} appels)",
                 va="center", ha="right" if dedans else "left", fontsize=8,
                 color="white" if dedans else "#4a4f57",
                 fontweight="bold" if dedans else "normal")
    ax1.set_yticks(y)
    ax1.set_yticklabels(schemas, fontsize=8.5)
    ax1.set_xlim(0, max(totaux) * 1.32)
    ax1.set_xlabel("Temps cumulé (s)")
    ax1.set_title("Temps LLM cumulé par schéma de sortie", fontsize=10)
    ax1.grid(axis="y", visible=False)

    positions = [par_schema[s] for s in schemas]
    bp = ax2.boxplot(positions[::-1], orientation="horizontal", patch_artist=True, widths=0.55,
                     medianprops={"color": "#2a2e34", "linewidth": 1.4},
                     flierprops={"marker": "o", "markersize": 3.5,
                                 "markerfacecolor": "#b03a2e",
                                 "markeredgecolor": "none"})
    for patch, s in zip(bp["boxes"], schemas[::-1]):
        patch.set_facecolor(couleurs[s])
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    ax2.set_yticklabels(schemas[::-1], fontsize=8.5)
    ax2.set_xlabel("Durée d'un appel (s)")
    ax2.set_title("Dispersion des durées par appel", fontsize=10)
    ax2.grid(axis="y", visible=False)

    save(fig, "16_appels_llm.png")


if __name__ == "__main__":
    main()
