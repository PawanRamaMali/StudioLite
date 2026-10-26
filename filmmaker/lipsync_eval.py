"""Local lipsync evaluation protocol.

Why this exists: deep-research round 3 confirmed that no third-party
2026 paper benchmarks EchoMimic V3 against LatentSync, MuseTalk, Hallo3,
Sonic, or FLOAT on SyncNet / LMD / FID / FVD at the same time. Each
model's paper excludes the others. For StudioLite to pick a lipsync
default on quality grounds (not just infrastructure / license) a user
has to run the comparison themselves.

Shape:
  - A ``MetricFn`` reads a (reference_video, candidate_video, audio)
    triple and returns a float in [0, inf) or None when it has no
    verdict. Lower-is-better and higher-is-better are declared in the
    MetricSpec; aggregation respects the direction.
  - Lightweight structural metrics (frame-count delta, duration drift)
    are on by default so a developer without SyncNet weights still sees
    useful output in CI.
  - Heavy metrics (SyncNet, LMD, FID, FVD) are opt-in via adapter
    functions keyed off env vars - same pattern as vbench2's official
    CLI shell-out. The adapter never guesses binaries or weight paths;
    missing env => the scorer silently returns None.
  - ``run_matrix`` drives StudioLite's lipsync backends across a corpus
    of (video, audio) pairs, invoking each backend's run() entrypoint
    and scoring the output. Returns a MatrixScorecard that ranks
    backends per-metric.

This module does not call any backend's model code directly - every
backend is a user-owned subprocess install described by a config.json
dropped under <output_dir>/lipsync-<name>/.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("studiolite.filmmaker.lipsync_eval")


MetricFn = Callable[[str, str, Optional[str]], Optional[float]]


@dataclass(frozen=True)
class MetricSpec:
    """A lipsync metric. ``higher_is_better=True`` means a bigger raw
    value is a better result (e.g. SSIM, SyncNet confidence); the
    aggregator uses this to rank backends."""
    name: str
    higher_is_better: bool
    fn: MetricFn
    description: str = ""


_METRICS: List[MetricSpec] = []


def register_metric(spec: MetricSpec) -> None:
    for i, existing in enumerate(_METRICS):
        if existing.name == spec.name:
            _METRICS[i] = spec
            return
    _METRICS.append(spec)


def registered_metrics() -> List[MetricSpec]:
    return list(_METRICS)


def clear_metrics() -> None:
    """Drop every registered metric. Tests use this between cases; the
    production path calls ``_register_default_metrics`` once at import."""
    _METRICS.clear()


# ---------------------------------------------------------------------------
# Shapes

@dataclass
class ClipMetric:
    metric: str
    value: Optional[float]
    higher_is_better: bool


@dataclass
class ClipResult:
    reference_video: str
    candidate_video: str
    audio: Optional[str]
    metrics: List[ClipMetric] = field(default_factory=list)


@dataclass
class BackendRow:
    """One backend's aggregate across all corpus pairs."""
    backend: str
    clip_count: int
    rendered_count: int                  # how many clips the backend produced
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    per_clip: List[ClipResult] = field(default_factory=list)


@dataclass
class MatrixScorecard:
    corpus_count: int
    backends: List[BackendRow]
    ranking: Dict[str, List[str]] = field(default_factory=dict)
    """Per-metric backend ranking, best-first. Backends with no verdict on
    a given metric are omitted from that metric's ranking (not sorted to
    the bottom), so a scorer-coverage change does not look like a loss."""


# ---------------------------------------------------------------------------
# Default lightweight metrics (ffprobe only; no weights required).

def _ffprobe_duration(path: str) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return None


def _ffprobe_frame_count(path: str) -> Optional[int]:
    if not path or not os.path.exists(path):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=False, timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int((proc.stdout or "").strip())
    except ValueError:
        return None


def _duration_drift_metric(ref_video: str, cand_video: str,
                            _audio: Optional[str]) -> Optional[float]:
    """Absolute drift between the reference and candidate durations, in
    seconds. Lower is better. A backend that shortens or lengthens the
    clip is almost always doing something wrong - at best it is a
    pipeline bug, at worst a mouth-only backend ate the trailing frames.
    """
    a = _ffprobe_duration(ref_video)
    b = _ffprobe_duration(cand_video)
    if a is None or b is None:
        return None
    return abs(a - b)


def _frame_count_delta_metric(ref_video: str, cand_video: str,
                                _audio: Optional[str]) -> Optional[float]:
    """Absolute delta in frame counts. Lower is better. Combined with
    the duration drift this catches dropped frames vs stretched time."""
    a = _ffprobe_frame_count(ref_video)
    b = _ffprobe_frame_count(cand_video)
    if a is None or b is None:
        return None
    return float(abs(a - b))


# ---------------------------------------------------------------------------
# Opt-in heavy metrics - shell out to a user-owned install.
#
# The research gave us the four canonical metrics (SyncNet, LMD, FID,
# FVD) and the direct consequence of no public comparison existing is
# that StudioLite has to drive the eval locally. The adapters below
# invoke whichever binaries the user has on PATH or in env, and emit a
# score. Missing binaries => None, never a crash.


_SYNCNET_BIN = "LIPSYNC_EVAL_SYNCNET_BIN"      # path to a syncnet CLI
_LMD_BIN = "LIPSYNC_EVAL_LMD_BIN"              # landmark distance CLI
_FID_BIN = "LIPSYNC_EVAL_FID_BIN"              # FID CLI (expects frames)
_FVD_BIN = "LIPSYNC_EVAL_FVD_BIN"              # FVD CLI


def _run_cli_score(bin_env: str, args: List[str]) -> Optional[float]:
    """Invoke a user-owned CLI described by ``bin_env`` and read its
    stdout as either a bare float or a JSON object with ``score``."""
    bin_path = os.environ.get(bin_env)
    if not bin_path:
        return None
    if not os.path.exists(bin_path):
        logger.debug("binary at %s=%s missing", bin_env, bin_path)
        return None
    try:
        proc = subprocess.run(
            [bin_path, *args], capture_output=True, text=True,
            check=False, timeout=600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("lipsync-eval binary %s failed: %s", bin_env, exc)
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    # Prefer JSON {"score": ...}; fall back to a bare float.
    try:
        data = json.loads(out)
        if isinstance(data, dict) and "score" in data:
            return float(data["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        return float(out.split()[0])
    except (ValueError, IndexError):
        return None


def _syncnet_metric(_ref: str, candidate: str,
                     audio: Optional[str]) -> Optional[float]:
    """SyncNet confidence score (higher is better). The adapter shells
    out to the binary at $LIPSYNC_EVAL_SYNCNET_BIN with --video and
    --audio; the user owns the install and the weights."""
    if not audio:
        return None
    return _run_cli_score(_SYNCNET_BIN,
                           ["--video", candidate, "--audio", audio])


def _lmd_metric(reference: str, candidate: str,
                 _audio: Optional[str]) -> Optional[float]:
    """Landmark distance (lower is better). Compares per-frame mouth
    landmarks between the reference and the candidate."""
    return _run_cli_score(_LMD_BIN,
                           ["--ref", reference, "--cand", candidate])


def _fid_metric(reference: str, candidate: str,
                 _audio: Optional[str]) -> Optional[float]:
    """FID (lower is better). The user's CLI is expected to handle its
    own frame extraction and Inception setup."""
    return _run_cli_score(_FID_BIN,
                           ["--ref", reference, "--cand", candidate])


def _fvd_metric(reference: str, candidate: str,
                 _audio: Optional[str]) -> Optional[float]:
    """FVD (lower is better). The user's CLI is expected to handle its
    own I3D setup."""
    return _run_cli_score(_FVD_BIN,
                           ["--ref", reference, "--cand", candidate])


def _register_default_metrics() -> None:
    register_metric(MetricSpec(
        name="duration_drift_sec",
        higher_is_better=False,
        fn=_duration_drift_metric,
        description="Absolute duration drift between reference and candidate.",
    ))
    register_metric(MetricSpec(
        name="frame_count_delta",
        higher_is_better=False,
        fn=_frame_count_delta_metric,
        description="Absolute delta in frame counts.",
    ))
    register_metric(MetricSpec(
        name="syncnet_confidence",
        higher_is_better=True,
        fn=_syncnet_metric,
        description=f"SyncNet confidence; requires {_SYNCNET_BIN}.",
    ))
    register_metric(MetricSpec(
        name="lmd",
        higher_is_better=False,
        fn=_lmd_metric,
        description=f"Mouth landmark distance; requires {_LMD_BIN}.",
    ))
    register_metric(MetricSpec(
        name="fid",
        higher_is_better=False,
        fn=_fid_metric,
        description=f"Frechet Inception Distance; requires {_FID_BIN}.",
    ))
    register_metric(MetricSpec(
        name="fvd",
        higher_is_better=False,
        fn=_fvd_metric,
        description=f"Frechet Video Distance; requires {_FVD_BIN}.",
    ))


_register_default_metrics()


# ---------------------------------------------------------------------------
# Core scoring entry points

def score_pair(reference_video: str, candidate_video: str,
                audio: Optional[str] = None) -> ClipResult:
    """Score a single (reference, candidate, audio) triple with every
    registered metric. A metric that raises is treated as None."""
    rows: List[ClipMetric] = []
    for spec in _METRICS:
        try:
            value = spec.fn(reference_video, candidate_video, audio)
        except Exception as exc:
            logger.warning("metric %s raised: %s", spec.name, exc)
            value = None
        rows.append(ClipMetric(metric=spec.name, value=value,
                                higher_is_better=spec.higher_is_better))
    return ClipResult(reference_video=reference_video,
                       candidate_video=candidate_video,
                       audio=audio, metrics=rows)


# ---------------------------------------------------------------------------
# Head-to-head matrix.
#
# Drives a set of backends across a corpus of (reference, audio) pairs,
# collecting one ClipResult per (backend, pair) and aggregating.


BackendDriver = Callable[[str, Optional[str], str],
                          Optional[Dict[str, Any]]]
# A driver receives (reference_video, audio, output_path) and renders
# the candidate at output_path. Returns the backend-specific details
# dict (any shape), or None when the backend failed to render.


def run_matrix(backends: Dict[str, BackendDriver],
                corpus: List[Tuple[str, Optional[str]]],
                output_root: str) -> MatrixScorecard:
    """Run every backend against every (reference, audio) pair and
    score the results.

    ``backends`` is {name: driver}; ``corpus`` is a list of (video,
    audio) tuples with audio optional. Each backend writes its
    candidate to ``<output_root>/<backend>/<i>.mp4``; a backend that
    returns None is marked ``rendered=False`` and left out of ranking
    for that pair.
    """
    os.makedirs(output_root, exist_ok=True)
    rows: List[BackendRow] = []
    for name, driver in backends.items():
        backend_dir = os.path.join(output_root, name)
        os.makedirs(backend_dir, exist_ok=True)
        per_clip: List[ClipResult] = []
        rendered = 0
        for i, (ref, audio) in enumerate(corpus):
            cand = os.path.join(backend_dir, f"{i:03d}.mp4")
            try:
                driver(ref, audio, cand)
            except Exception as exc:
                logger.warning("driver %s raised on pair %d: %s",
                                name, i, exc)
            if os.path.exists(cand):
                rendered += 1
                per_clip.append(score_pair(ref, cand, audio))
            else:
                per_clip.append(ClipResult(reference_video=ref,
                                            candidate_video=cand,
                                            audio=audio, metrics=[]))
        rows.append(BackendRow(backend=name, clip_count=len(corpus),
                                rendered_count=rendered,
                                metrics=_aggregate_metrics(per_clip),
                                per_clip=per_clip))

    ranking = _rank_backends(rows)
    return MatrixScorecard(corpus_count=len(corpus),
                            backends=rows, ranking=ranking)


def _aggregate_metrics(per_clip: List[ClipResult]
                        ) -> Dict[str, Optional[float]]:
    """Mean per-metric score across clips, skipping None. A metric with
    zero opinions aggregates to None - never to 0.0 - so an uninstalled
    SyncNet does not fake a tie."""
    out: Dict[str, Optional[float]] = {}
    for spec in _METRICS:
        values = [m.value for clip in per_clip for m in clip.metrics
                  if m.metric == spec.name and m.value is not None]
        out[spec.name] = (sum(values) / len(values)) if values else None
    return out


def _rank_backends(rows: List[BackendRow]) -> Dict[str, List[str]]:
    """For each metric emit a backend ranking best-first. Backends with
    None on a metric are excluded from that metric's ranking so a
    missing scorer does not sort them to the bottom."""
    out: Dict[str, List[str]] = {}
    for spec in _METRICS:
        with_opinion = [(r.backend, r.metrics.get(spec.name))
                        for r in rows if r.metrics.get(spec.name) is not None]
        with_opinion.sort(key=lambda b: b[1],
                           reverse=spec.higher_is_better)
        out[spec.name] = [name for name, _ in with_opinion]
    return out


# ---------------------------------------------------------------------------
# Convenience: drivers wired to StudioLite's registered lipsync backends.

def studiolite_backend_driver(project: "Any", backend_name: str
                                ) -> BackendDriver:
    """Return a BackendDriver that invokes one of StudioLite's three
    lipsync backends (musetalk, latentsync, echomimicv3) via the
    existing filmmaker.lipsync module.

    The driver assembles the minimal voice_lines manifest the backend
    expects and reuses the backend's run() method. Backends that
    report themselves unavailable (check_installed() False) write
    nothing to disk; the matrix then marks their row rendered=False.
    """
    from . import lipsync as _lipsync
    backend = _lipsync.get_backend(backend_name)

    def driver(reference_video: str, audio: Optional[str],
                output_path: str) -> Optional[Dict[str, Any]]:
        status = backend.check_installed(project)
        if not status.available:
            return None
        voice_lines: List[Dict[str, Any]] = []
        if audio:
            # The backends expect a per-line manifest with synthesised
            # wav paths. For eval we fake a single-line manifest - the
            # backend uses it only to find audio for lip motion.
            voice_lines = [{
                "scene_id": "eval", "shot_id": "001",
                "speaker": "EVAL",
                "text": "", "wav": _project_rel(project, audio),
                "synthesized": True, "error": None,
            }]
        try:
            return backend.run(project, reference_video, voice_lines,
                                output_path)
        except Exception as exc:
            logger.warning("backend %s raised: %s", backend_name, exc)
            return None

    return driver


def _project_rel(project: "Any", abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, project.dir).replace(os.sep, "/")
    except (ValueError, AttributeError):
        return abs_path


# ---------------------------------------------------------------------------
# Serialisation + CLI

def scorecard_to_dict(card: MatrixScorecard) -> Dict[str, Any]:
    return asdict(card)


# ---------------------------------------------------------------------------
# Scorecard diff for regression gating.
#
# The CI gate (scripts/lipsync_regression_check.py) runs the matrix against
# an in-repo baseline and uses this helper to decide pass/fail. Mirrors the
# shape of vbench2.compare_scorecards but operates per-(backend, metric)
# since lipsync is a backend-vs-backend bakeoff, not a single aggregate
# score. Thresholds are relative so a metric that moves from 0.01 to 0.02
# on noise does not trip the gate while a 10 -> 2 drop does.

DEFAULT_REGRESSION_THRESHOLD = 0.05


@dataclass
class MetricDelta:
    backend: str
    metric: str
    higher_is_better: bool
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]        # candidate - baseline, in the raw metric units
    relative_delta: Optional[float]
    regressed: bool               # relative drop (respecting direction) > threshold


@dataclass
class ScorecardDiff:
    threshold: float
    deltas: List[MetricDelta]
    regressed_pairs: List[str]    # "backend/metric" labels for the gate's exit code


def _coerce_scorecard(card: Any) -> Dict[str, Any]:
    """Accept either a live MatrixScorecard or its dict form (as produced
    by scorecard_to_dict / persisted to disk)."""
    if isinstance(card, MatrixScorecard):
        return scorecard_to_dict(card)
    if isinstance(card, dict):
        return card
    raise TypeError(f"not a scorecard: {type(card).__name__}")


def _metric_direction(card: Dict[str, Any], metric: str) -> Optional[bool]:
    """Pull higher_is_better for a metric out of a scorecard dict. Falls
    back to the live registry if the card does not carry it (older cards)."""
    for row in card.get("backends") or []:
        for clip in row.get("per_clip") or []:
            for m in clip.get("metrics") or []:
                if m.get("metric") == metric and "higher_is_better" in m:
                    return bool(m["higher_is_better"])
    for spec in _METRICS:
        if spec.name == metric:
            return spec.higher_is_better
    return None


def _relative_change(base: float, cand: float) -> float:
    """Candidate relative to baseline. ``(cand - base) / |base|`` with a
    small-baseline guard so a baseline near zero does not blow up the ratio.
    """
    denom = abs(base)
    if denom < 1e-9:
        # When baseline is effectively zero, fall back to absolute delta
        # scaled to the threshold's unit. Anything non-zero in the candidate
        # counts as a 100% change; both sides exactly zero is 0.
        if abs(cand) < 1e-9:
            return 0.0
        return 1.0 if cand > base else -1.0
    return (cand - base) / denom


def compare_scorecards(baseline: Any, candidate: Any, *,
                        threshold: float = DEFAULT_REGRESSION_THRESHOLD
                        ) -> ScorecardDiff:
    """Compare a candidate scorecard against a baseline, per (backend, metric).

    ``threshold`` is the minimum relative drop (respecting the metric's
    direction) that counts as a regression. 0.05 = 5%.

    A pair is regressed when both sides have a numeric opinion and the
    candidate is worse than the baseline by more than the threshold.
    "Worse" means lower for higher-is-better metrics, higher for
    lower-is-better. Pairs with None on either side are reported but never
    flagged - a scorer-coverage change is not a quality regression.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    base = _coerce_scorecard(baseline)
    cand = _coerce_scorecard(candidate)
    cand_by_backend = {r.get("backend"): r for r in cand.get("backends") or []}

    deltas: List[MetricDelta] = []
    regressed: List[str] = []
    for brow in base.get("backends") or []:
        name = brow.get("backend") or ""
        base_metrics = brow.get("metrics") or {}
        cand_metrics = (cand_by_backend.get(name) or {}).get("metrics") or {}
        for metric, b_val in base_metrics.items():
            c_val = cand_metrics.get(metric)
            direction = _metric_direction(base, metric)
            if direction is None:
                direction = _metric_direction(cand, metric)
            higher_is_better = bool(direction) if direction is not None else True

            if b_val is None or c_val is None:
                deltas.append(MetricDelta(
                    backend=name, metric=metric,
                    higher_is_better=higher_is_better,
                    baseline=b_val, candidate=c_val,
                    delta=None, relative_delta=None, regressed=False,
                ))
                continue
            try:
                bf = float(b_val)
                cf = float(c_val)
            except (TypeError, ValueError):
                deltas.append(MetricDelta(
                    backend=name, metric=metric,
                    higher_is_better=higher_is_better,
                    baseline=b_val, candidate=c_val,
                    delta=None, relative_delta=None, regressed=False,
                ))
                continue
            raw_delta = cf - bf
            rel = _relative_change(bf, cf)
            # Regression test respects direction. For higher-is-better a
            # negative relative change is bad; for lower-is-better a
            # positive one is bad.
            is_regression = (
                (higher_is_better and rel < -threshold)
                or (not higher_is_better and rel > threshold)
            )
            if is_regression:
                regressed.append(f"{name}/{metric}")
            deltas.append(MetricDelta(
                backend=name, metric=metric,
                higher_is_better=higher_is_better,
                baseline=bf, candidate=cf,
                delta=raw_delta, relative_delta=rel,
                regressed=is_regression,
            ))

    return ScorecardDiff(threshold=threshold, deltas=deltas,
                          regressed_pairs=regressed)


def diff_to_dict(diff: ScorecardDiff) -> Dict[str, Any]:
    return asdict(diff)


def _cli_run(argv: List[str]) -> int:
    """CLI entry point for the matrix runner.

    Usage:
      python -m filmmaker.lipsync_eval run --corpus <jsonl> --out <dir>
      [--backend musetalk,latentsync,echomimicv3] [--project-dir <dir>]

    The corpus JSONL has one record per line: {"reference": "<video>",
    "audio": "<audio or null>"}. --project-dir points at a StudioLite
    project root for backend configs; a non-existent dir runs with
    None drivers that just score the reference against itself (useful
    for smoke-testing the harness without any backend installed).
    """
    corpus_path = None
    out_dir = None
    backends_arg = "musetalk,latentsync,echomimicv3"
    project_dir = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--corpus" and i + 1 < len(argv):
            corpus_path = argv[i + 1]; i += 2; continue
        if tok == "--out" and i + 1 < len(argv):
            out_dir = argv[i + 1]; i += 2; continue
        if tok == "--backend" and i + 1 < len(argv):
            backends_arg = argv[i + 1]; i += 2; continue
        if tok == "--project-dir" and i + 1 < len(argv):
            project_dir = argv[i + 1]; i += 2; continue
        print(f"unknown argument: {tok}")
        return 2
    if not corpus_path or not out_dir:
        print("usage: python -m filmmaker.lipsync_eval run --corpus <jsonl> "
              "--out <dir> [--backend a,b,c] [--project-dir <dir>]")
        return 2
    corpus: List[Tuple[str, Optional[str]]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            corpus.append((rec["reference"], rec.get("audio")))
    names = [n.strip() for n in backends_arg.split(",") if n.strip()]

    def _noop_driver(_ref, _audio, _out):
        return None

    backends: Dict[str, BackendDriver] = {}
    if project_dir and os.path.isdir(project_dir):
        from .project import Project
        project = Project.load(os.path.dirname(project_dir),
                                 os.path.basename(project_dir))
        for n in names:
            backends[n] = studiolite_backend_driver(project, n)
    else:
        for n in names:
            backends[n] = _noop_driver

    card = run_matrix(backends, corpus, out_dir)
    print(json.dumps(scorecard_to_dict(card), indent=2))
    return 0


def _main(argv: List[str]) -> int:
    if not argv or argv[0] != "run":
        print("usage: python -m filmmaker.lipsync_eval run ...")
        return 2
    return _cli_run(argv[1:])


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(_main(_sys.argv[1:]))
