"""Lipsync regression gate.

Loads an in-repo baseline scorecard, runs the matrix against the installed
lipsync backends on a corpus, diffs the two, and exits 0 on pass, 1 on
regression. Callable from CI and from the dev's shell with the same flags.

When no lipsync backends are installed in the environment (as in GitHub
Actions free runners) every backend row rendered zero clips, so the
candidate scorecard has only None metrics. The diff then reports zero
regressed pairs and the gate exits 0 with a clear "no opinion" message -
same local-config -> cloud-burst -> skip pattern used across the pipeline.

Threshold: 5% relative drop by default, overridable with
LIPSYNC_REGRESSION_THRESHOLD or --threshold. Relative so a metric moving
from 0.01 to 0.02 does not trip the gate while a 10 -> 2 drop does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Make ``filmmaker`` importable when this script is invoked directly from
# the repo root (e.g. CI runs `python scripts/lipsync_regression_check.py`).
_ROOT_STR = str(_repo_root())
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)


def _load_baseline(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_corpus(path: Optional[Path]) -> List[Tuple[str, Optional[str]]]:
    if not path or not path.exists():
        return []
    out: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append((rec["reference"], rec.get("audio")))
    return out


def _candidate_from_backends(
    backend_names: List[str],
    corpus: List[Tuple[str, Optional[str]]],
    output_root: Path,
    project_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run the matrix against the installed backends. Returns the dict form
    of the scorecard; an empty corpus yields a scorecard with zero clips
    per backend (every metric None), which the gate treats as 'no opinion'.
    """
    from filmmaker import lipsync_eval as le  # local import to keep CLI light

    output_root.mkdir(parents=True, exist_ok=True)

    def _noop(_ref, _audio, _out):
        return None

    drivers: Dict[str, Any] = {}
    if project_dir and project_dir.is_dir():
        from filmmaker.project import Project
        project = Project.load(str(project_dir.parent), project_dir.name)
        for n in backend_names:
            drivers[n] = le.studiolite_backend_driver(project, n)
    else:
        for n in backend_names:
            drivers[n] = _noop

    card = le.run_matrix(drivers, corpus, str(output_root))
    return le.scorecard_to_dict(card)


def _rendered_pairs(scorecard: Dict[str, Any]) -> int:
    """Count (backend, metric) pairs that produced a numeric value."""
    count = 0
    for row in scorecard.get("backends") or []:
        for v in (row.get("metrics") or {}).values():
            if v is not None:
                count += 1
    return count


def _format_regressions(diff_dict: Dict[str, Any]) -> str:
    rows: List[str] = []
    for d in diff_dict.get("deltas") or []:
        if not d.get("regressed"):
            continue
        pct = d.get("relative_delta")
        rows.append(
            f"  {d.get('backend')}/{d.get('metric')}: "
            f"baseline={d.get('baseline')} candidate={d.get('candidate')} "
            f"delta={d.get('delta')} relative={pct}"
        )
    return "\n".join(rows)


def _threshold_from_env_and_arg(arg_value: Optional[float]) -> float:
    if arg_value is not None:
        return float(arg_value)
    env = os.environ.get("LIPSYNC_REGRESSION_THRESHOLD")
    if env:
        try:
            return float(env)
        except ValueError:
            print(
                f"warning: LIPSYNC_REGRESSION_THRESHOLD={env!r} is not a "
                "number; falling back to default",
                file=sys.stderr,
            )
    from filmmaker import lipsync_eval as le
    return le.DEFAULT_REGRESSION_THRESHOLD


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lipsync scorecard regression gate.",
    )
    parser.add_argument(
        "--baseline",
        default=str(_repo_root() / "tests" / "data" / "lipsync_baseline.json"),
        help="Path to the baseline scorecard JSON.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="JSONL corpus of {reference, audio?} pairs. Empty / missing "
             "means a no-op candidate run - the gate then just checks the "
             "baseline parses and exits 0.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Working directory for candidate renders. Defaults to a "
             "temp dir under the repo.",
    )
    parser.add_argument(
        "--backend",
        default="musetalk,latentsync,echomimicv3",
        help="Comma-separated backends to drive.",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Studio project root used to look up backend configs. When "
             "omitted the backends fall back to noop drivers.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Relative drop that counts as a regression (0.05 = 5%%). "
             "Overrides LIPSYNC_REGRESSION_THRESHOLD.",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Skip the matrix run; read the candidate scorecard from this "
             "JSON file instead. Useful for local diffs.",
    )
    parser.add_argument(
        "--write-candidate",
        default=None,
        help="If set, write the computed candidate scorecard to this path.",
    )
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"error: baseline not found: {baseline_path}", file=sys.stderr)
        return 2

    try:
        baseline = _load_baseline(baseline_path)
    except json.JSONDecodeError as exc:
        print(f"error: baseline is not valid JSON: {exc}", file=sys.stderr)
        return 2

    threshold = _threshold_from_env_and_arg(args.threshold)

    if args.candidate:
        with open(args.candidate, "r", encoding="utf-8") as f:
            candidate = json.load(f)
    else:
        corpus = _load_corpus(Path(args.corpus) if args.corpus else None)
        backend_names = [n.strip() for n in args.backend.split(",") if n.strip()]
        out_dir = Path(args.out) if args.out else _repo_root() / ".lipsync_gate_tmp"
        candidate = _candidate_from_backends(
            backend_names, corpus, out_dir,
            Path(args.project_dir) if args.project_dir else None,
        )

    if args.write_candidate:
        with open(args.write_candidate, "w", encoding="utf-8") as f:
            json.dump(candidate, f, indent=2, sort_keys=True)

    # Short-circuit: if neither side has a numeric opinion, the gate has
    # nothing to compare. Report it clearly and exit 0 (skip).
    base_pairs = _rendered_pairs(baseline)
    cand_pairs = _rendered_pairs(candidate)
    if base_pairs == 0 or cand_pairs == 0:
        print(
            "lipsync models unavailable or baseline is a placeholder "
            f"(baseline opinion-pairs={base_pairs}, candidate={cand_pairs}); "
            "skipping gate."
        )
        return 0

    from filmmaker import lipsync_eval as le
    diff = le.compare_scorecards(baseline, candidate, threshold=threshold)
    diff_dict = le.diff_to_dict(diff)

    if not diff.regressed_pairs:
        print(
            f"lipsync regression gate PASS "
            f"(threshold={threshold}, pairs_compared={len(diff.deltas)})"
        )
        return 0

    print(
        f"lipsync regression gate FAIL - {len(diff.regressed_pairs)} "
        f"regressed pair(s) at threshold={threshold}:"
    )
    print(_format_regressions(diff_dict))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
