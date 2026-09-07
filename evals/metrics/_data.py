"""Accès aux artefacts d'un run de benchmark, pour les figures du rapport.

Autonome : ne dépend que de la bibliothèque standard, comme `_style`. Les
scripts de figures n'embarquent plus de constantes recopiées à la main — ils
lisent le dernier run présent dans `evals/results/`, ou celui passé en
paramètre (`--run bench_20260806_100245`, ou un chemin de dossier).

Deux familles d'artefacts sont exposées :

- `scores.json`, écrit par `evals/run_scoring.py` — bloc `totals` (agrégé sur
  les cas scorés) ou bloc d'un cas précis avec `--case` ;
- `telemetry.json`, écrit par `evals/run_benchmark.py` dans chaque dossier de
  cas — fusionné ici sur les cas retenus, de la même façon que
  `run_benchmark.merge_telemetry` recolle les segments d'un cas repris.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import _style

RACINE = Path(__file__).resolve().parents[2]
RESULTS_DIR = RACINE / "evals" / "results"


def fr(valeur: float, decimales: int = 1) -> str:
    """Nombre formaté à la française (virgule décimale)."""
    return f"{valeur:.{decimales}f}".replace(".", ",")


def _lire_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _fusionner_telemetrie(resumes: list[dict]) -> dict:
    """Somme les `telemetry.summary()` de plusieurs cas.

    Les parts et les moyennes sont des ratios : elles sont recalculées, pas
    moyennées. Même logique que `run_benchmark.merge_telemetry`, réécrite ici
    pour que les figures restent indépendantes du code du projet.
    """
    resumes = [r for r in resumes if r]
    if not resumes:
        return {}
    if len(resumes) == 1:
        return resumes[0]

    scalaires = ("compute_s", "wait_s", "llm_s", "node_count", "llm_count",
                 "retry_count", "failure_count", "cache_hit_count")
    fusion: dict = {clef: sum(r.get(clef, 0) for r in resumes) for clef in scalaires}

    for groupe in ("by_node", "by_schema"):
        agg: dict[str, dict] = {}
        for resume in resumes:
            for nom, slot in (resume.get(groupe) or {}).items():
                cur = agg.setdefault(nom, {"count": 0, "total_s": 0.0, "max_s": 0.0})
                cur["count"] += slot.get("count", 0)
                cur["total_s"] += slot.get("total_s", 0.0)
                cur["max_s"] = max(cur["max_s"], slot.get("max_s", 0.0))
                for extra in ("llm_calls", "retried", "failed"):
                    if extra in slot:
                        cur[extra] = cur.get(extra, 0) + slot[extra]
        fusion[groupe] = agg

    for groupe, total in (("by_node", fusion["compute_s"]), ("by_schema", fusion["llm_s"])):
        for slot in fusion[groupe].values():
            slot["mean_s"] = slot["total_s"] / slot["count"] if slot["count"] else 0.0
            slot["share"] = slot["total_s"] / total if total else 0.0

    appels: list[dict] = []
    for resume in resumes:
        appels.extend(resume.get("calls") or [])
    if appels:
        fusion["calls"] = appels

    fusion["llm_share_of_compute"] = (
        fusion["llm_s"] / fusion["compute_s"] if fusion["compute_s"] else 0.0
    )
    return fusion


def _a_des_scores(dossier: Path) -> bool:
    return (dossier / "scores.json").exists()


def _a_de_la_telemetrie(dossier: Path) -> bool:
    return any(
        (_lire_json(cas / "telemetry.json") or {}).get("by_node")
        for cas in sorted(dossier.iterdir())
        if cas.is_dir()
    )


BESOINS = {
    "scores": _a_des_scores,
    "telemetry": _a_de_la_telemetrie,
    "any": lambda dossier: True,  # figure alimentée par une source externe
}
_MANQUE = {
    "scores": "scores.json — lancer d'abord evals/run_scoring.py",
    "telemetry": "telemetry.json dans au moins un dossier de cas",
    "any": "rien",
}


def _resoudre(arg: str | None, besoin: str) -> Path | None:
    """Le dossier de run demandé — None quand la figure se passe d'un run."""
    utilisable = BESOINS[besoin]
    if besoin == "any" and not arg:
        return None
    if arg:
        dossier = Path(arg)
        if not dossier.is_dir():
            dossier = RESULTS_DIR / arg
        if not dossier.is_dir():
            raise SystemExit(f"run introuvable : {arg}")
        if not utilisable(dossier):
            raise SystemExit(f"{dossier.name} : il manque {_MANQUE[besoin]}")
        return dossier

    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"aucun run : {RESULTS_DIR} n'existe pas")
    candidats = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and utilisable(d)]
    if not candidats:
        raise SystemExit(f"aucun run exploitable dans {RESULTS_DIR} "
                         f"(besoin : {_MANQUE[besoin]})")
    return max(candidats, key=lambda d: d.stat().st_mtime)


@dataclass
class Run:
    """Un dossier de run, plus le cas éventuellement sélectionné."""

    dir: Path
    manifest: dict
    scores: dict | None
    case_id: str | None

    @property
    def run_id(self) -> str:
        return self.manifest.get("run_id") or self.dir.name

    @property
    def simulator_mode(self) -> str:
        return (self.manifest.get("simulator_mode")
                or (self.scores or {}).get("simulator_mode") or "?")

    @property
    def model(self) -> str:
        return ((self.manifest.get("llm") or {}).get("model")
                or ((self.scores or {}).get("llm") or {}).get("model") or "?")

    # --- scores.json ---------------------------------------------------

    def cases(self) -> list[dict]:
        """Blocs de cas scorés, filtrés par --case le cas échéant."""
        tous = (self.scores or {}).get("cases") or []
        if self.case_id:
            gardes = [c for c in tous if c["case_id"] == self.case_id]
            if not gardes:
                raise SystemExit(f"cas non scoré dans ce run : {self.case_id}")
            return gardes
        if not tous:
            raise SystemExit(f"aucun cas scoré dans {self.dir / 'scores.json'}")
        return tous

    def bloc(self, nom: str) -> dict:
        """Un bloc de métriques : `totals[nom]`, ou celui du cas sélectionné."""
        if self.case_id:
            return self.cases()[0][nom]
        totaux = (self.scores or {}).get("totals") or {}
        if nom not in totaux:
            raise SystemExit(f"bloc « {nom} » absent de {self.dir / 'scores.json'}")
        return totaux[nom]

    def libelle_cas(self) -> str:
        """« cdc_002_ticketing » ou « 2 cas », pour les sous-titres."""
        noms = [c["case_id"] for c in self.cases()]
        return noms[0] if len(noms) == 1 else f"{len(noms)} cas"

    # --- telemetry.json ------------------------------------------------

    def telemetrie(self) -> dict:
        dossiers = [d for d in sorted(self.dir.iterdir()) if d.is_dir()]
        if self.case_id:
            dossiers = [d for d in dossiers if d.name == self.case_id]
            if not dossiers:
                raise SystemExit(f"cas absent du run : {self.case_id}")
        fusion = _fusionner_telemetrie([_lire_json(d / "telemetry.json") for d in dossiers])
        if not fusion.get("by_node"):
            raise SystemExit(f"aucune télémétrie exploitable dans {self.dir}")
        return fusion

    def cas_avec_telemetrie(self) -> list[str]:
        return [d.name for d in sorted(self.dir.iterdir())
                if d.is_dir() and (_lire_json(d / "telemetry.json") or {}).get("by_node")]


def cli(besoin: str = "scores", *, ajouts=None) -> tuple[Run, argparse.Namespace]:
    """Analyse `--run` / `--case` / `--out` et charge le run demandé."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None,
                        help="run_id sous evals/results/ ou chemin de dossier "
                             "(défaut : le dernier run exploitable)")
    parser.add_argument("--case", default=None,
                        help="restreindre à un cas (défaut : tout le run)")
    parser.add_argument("--out", default=None,
                        help=f"dossier de sortie des PNG (défaut : {_style.FIG_DIR})")
    if ajouts is not None:
        ajouts(parser)
    args = parser.parse_args()

    if args.out:
        _style.set_output_dir(args.out)

    dossier = _resoudre(args.run, besoin)
    if dossier is None:
        return Run(dir=RESULTS_DIR, manifest={}, scores=None, case_id=args.case), args

    run = Run(
        dir=dossier,
        manifest=_lire_json(dossier / "manifest.json") or {},
        scores=_lire_json(dossier / "scores.json"),
        case_id=args.case,
    )
    etendue = run.case_id or (run.libelle_cas() if run.scores else "run complet")
    print(f"run : {dossier.name} · {etendue} · modèle {run.model}"
          f" · simulateur {run.simulator_mode}", file=sys.stderr)
    return run, args
