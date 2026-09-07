"""Génère toutes les figures de `evals/metrics/` en une passe.

Chaque script de figure reste exécutable seul ; celui-ci les enchaîne avec les
mêmes options et rend un bilan. Les options communes (`--run`, `--case`,
`--out`) sont transmises à tous, les options spécifiques (`--calls-csv`,
`--include-cache`) seulement aux scripts qui les acceptent.

    uv run --with matplotlib python evals/metrics/run_all.py
    uv run --with matplotlib python evals/metrics/run_all.py --run bench_20260806_100245
    uv run --with matplotlib python evals/metrics/run_all.py --only 15 16 --out /tmp/figs

Une figure qui échoue n'interrompt pas les autres : la figure 16 réclame le
détail par appel LLM, absent des runs antérieurs à son enregistrement, et cela
ne doit pas priver le rapport des six autres. Le code de sortie vaut le nombre
d'échecs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent

COMMUNES = ("--run", "--case", "--out")
SPECIFIQUES = ("--calls-csv", "--include-cache")


def figures() -> list[Path]:
    """Les scripts numérotés, dans l'ordre des figures du rapport."""
    return sorted(p for p in DOSSIER.glob("[0-9]*_*.py"))


def commande(script: Path, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(script)]
    source = script.read_text(encoding="utf-8")
    for option in COMMUNES + SPECIFIQUES:
        valeur = getattr(args, option.lstrip("-").replace("-", "_"))
        if not valeur:
            continue
        if option in SPECIFIQUES and option not in source:
            continue  # ce script ne connaît pas l'option
        cmd.append(option)
        if valeur is not True:  # les drapeaux n'ont pas de valeur
            cmd.append(str(valeur))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None,
                        help="run_id sous evals/results/ ou chemin de dossier "
                             "(défaut : le dernier run exploitable)")
    parser.add_argument("--case", default=None, help="restreindre à un cas")
    parser.add_argument("--out", default=None, help="dossier de sortie des PNG")
    parser.add_argument("--calls-csv", default=None,
                        help="export CSV « Appels LLM » pour la figure 16")
    parser.add_argument("--include-cache", action="store_true",
                        help="figure 16 : compter aussi les appels servis par le cache")
    parser.add_argument("--only", nargs="+", metavar="N", default=None,
                        help="ne générer que ces figures (numéros, ex. --only 15 16)")
    args = parser.parse_args()

    scripts = figures()
    if args.only:
        voulus = {n.zfill(2) for n in args.only}
        scripts = [s for s in scripts if s.name.split("_")[0] in voulus]
        inconnus = voulus - {s.name.split("_")[0] for s in scripts}
        if inconnus:
            raise SystemExit(f"figure(s) inconnue(s) : {', '.join(sorted(inconnus))}")
    if not scripts:
        raise SystemExit(f"aucun script de figure dans {DOSSIER}")

    echecs: list[tuple[str, int]] = []
    for script in scripts:
        print(f"\n=== {script.name} " + "=" * max(0, 60 - len(script.name)), flush=True)
        code = subprocess.run(commande(script, args)).returncode
        if code != 0:
            echecs.append((script.name, code))

    print(f"\n{len(scripts) - len(echecs)}/{len(scripts)} figure(s) générée(s).")
    for nom, code in echecs:
        print(f"  échec : {nom} (code {code})", file=sys.stderr)
    return len(echecs)


if __name__ == "__main__":
    sys.exit(main())
