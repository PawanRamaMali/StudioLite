"""VBench-2.0 regression harness.

VBench-2.0 (arxiv 2503.21755) scores a video generator along five intrinsic
dimensions - Human Fidelity, Controllability, Creativity, Physics, and
Commonsense - and 18 sub-capabilities. Surface metrics (per-frame FID,
dynamic degree) correlate only 0.41/-0.05 with physics adherence on
WorldModelBench, so this harness replaces them for regression work.

Design:
  - A Scorer is a small callable that reads a clip and returns a float in
    [0, 1] or None when it has no verdict. Each sub-capability has one or
    more Scorers; the dimension score is their mean, skipping None.
  - The default scorers are deliberately lightweight (ffprobe + ffmpeg
    sampling) so a developer without GPU weights can still run the
    harness in CI. Each one documents what it approximates and what the
    "real" VBench-2.0 scorer would do.
  - The real VBench-2.0 code is a heavy external install (VLM weights,
    diffusion priors, grader models). A caller can register their own
    scorers via :func:`register_scorer`, point them at any VLM they like,
    and the harness will pick them up on the next run.
  - CLI: ``python -m filmmaker.vbench2 score <clip-dir>`` prints a
    JSON scorecard. The same module exposes :func:`score_dir` for use
    inside tests and the continuity stage.

This is a measurement harness, not a model. It does not generate video.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger("studiolite.filmmaker.vbench2")


# The five dimensions and their sub-capabilities, taken verbatim from
# VBench-2.0 Table 1 (arxiv 2503.21755). The ordering here is the one the
# paper uses so a reader can map scorecard rows back to the paper.
DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "human_fidelity": (
        "human_anatomy",
        "human_identity",
        "human_clothes",
        "dynamic_attribute",
    ),
    "controllability": (
        "composition",
        "camera_motion",
        "diversity",
        "complex_landscape",
        "complex_plot",
    ),
    "creativity": (
        "composition_creativity",
        "creative_anomaly",
    ),
    "physics": (
        "mechanics",
        "material",
        "thermotics",
        "multi_object_interaction",
    ),
    "commonsense": (
        "instance_preservation",
        "motion_rationality",
        "multi_view_consistency",
    ),
}


# Score is bounded to [0, 1] where 1 = passes the sub-capability without
# issue and 0 = fails it entirely. None means "scorer has no opinion" and
# is excluded from aggregation - critical so a missing-weights scorer does
# not drag the dimension score to zero.
ScorerFn = Callable[[str], Optional[float]]


@dataclass
class ScorerSpec:
    name: str
    dimension: str
    capability: str
    fn: ScorerFn
    description: str = ""


# A scorer is registered under (dimension, capability). Multiple scorers
# per capability are averaged. External callers register their own via
# register_scorer(); the defaults below are structural placeholders.
_SCORERS: List[ScorerSpec] = []


def register_scorer(spec: ScorerSpec) -> None:
    """Register a scorer. Later calls replace an earlier one with the same
    (dimension, capability, name). Callers rebuilding the harness in-place
    (e.g. plugging in a real VLM) can rely on this to idempotently swap in
    a stronger scorer without duplicating entries."""
    if spec.dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension: {spec.dimension}")
    if spec.capability not in DIMENSIONS[spec.dimension]:
        raise ValueError(
            f"unknown capability {spec.capability!r} for dimension {spec.dimension!r}"
        )
    for i, existing in enumerate(_SCORERS):
        if (existing.name == spec.name
                and existing.dimension == spec.dimension
                and existing.capability == spec.capability):
            _SCORERS[i] = spec
            return
    _SCORERS.append(spec)


def registered_scorers() -> List[ScorerSpec]:
    """Snapshot of registered scorers. Tests read this to assert the right
    scorers are wired for a given test."""
    return list(_SCORERS)


# ---------------------------------------------------------------------------
# Scorecard shape

@dataclass
class CapabilityScore:
    dimension: str
    capability: str
    score: Optional[float]           # None when no scorer had an opinion
    scorer_names: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionScore:
    dimension: str
    score: Optional[float]           # mean of its CapabilityScores (ignoring None)
    capabilities: List[CapabilityScore] = field(default_factory=list)


@dataclass
class ClipScorecard:
    clip_path: str
    dimensions: List[DimensionScore]
    overall: Optional[float]         # mean of its DimensionScores (ignoring None)


@dataclass
class DirScorecard:
    clip_count: int
    per_clip: List[ClipScorecard]
    aggregate: List[DimensionScore]  # mean across clips, per dimension
    overall: Optional[float]


# ---------------------------------------------------------------------------
# Scoring

def score_clip(clip_path: str) -> ClipScorecard:
    """Score a single clip across all registered scorers.

    Clip does not need to be readable - every default scorer handles a
    missing or malformed file by returning None rather than raising, so
    the harness still produces a scorecard for a broken input.
    """
    results: Dict[Tuple[str, str], List[Tuple[str, Optional[float]]]] = {}
    for spec in _SCORERS:
        try:
            value = spec.fn(clip_path)
        except Exception as exc:
            logger.warning("scorer %s raised on %s: %s", spec.name, clip_path, exc)
            value = None
        results.setdefault((spec.dimension, spec.capability), []).append(
            (spec.name, value)
        )

    dim_rows: List[DimensionScore] = []
    for dim, caps in DIMENSIONS.items():
        cap_rows: List[CapabilityScore] = []
        for cap in caps:
            entries = results.get((dim, cap), [])
            opinions = [v for _, v in entries if v is not None]
            cap_score = sum(opinions) / len(opinions) if opinions else None
            cap_rows.append(CapabilityScore(
                dimension=dim, capability=cap, score=cap_score,
                scorer_names=[n for n, _ in entries],
            ))
        dim_opinions = [c.score for c in cap_rows if c.score is not None]
        dim_score = sum(dim_opinions) / len(dim_opinions) if dim_opinions else None
        dim_rows.append(DimensionScore(dimension=dim, score=dim_score,
                                        capabilities=cap_rows))

    overall_opinions = [d.score for d in dim_rows if d.score is not None]
    overall = (sum(overall_opinions) / len(overall_opinions)
               if overall_opinions else None)
    return ClipScorecard(clip_path=clip_path, dimensions=dim_rows, overall=overall)


def score_dir(clip_dir: str, suffix: str = ".mp4") -> DirScorecard:
    """Score every clip in a directory and aggregate per dimension.

    Non-recursive on purpose - callers organizing by project should pass
    the specific motion/upscale/final directory they care about.
    """
    if not os.path.isdir(clip_dir):
        raise FileNotFoundError(clip_dir)
    clips = sorted(
        os.path.join(clip_dir, name)
        for name in os.listdir(clip_dir)
        if name.lower().endswith(suffix.lower())
    )

    per_clip = [score_clip(c) for c in clips]

    # Aggregate: for each dimension, average the dimension score across
    # clips. Clips with None on a dimension are skipped for that dimension.
    aggregate: List[DimensionScore] = []
    for dim in DIMENSIONS:
        dim_opinions: List[float] = []
        for card in per_clip:
            for row in card.dimensions:
                if row.dimension == dim and row.score is not None:
                    dim_opinions.append(row.score)
        agg_score = (sum(dim_opinions) / len(dim_opinions)
                     if dim_opinions else None)
        aggregate.append(DimensionScore(dimension=dim, score=agg_score))

    overall_opinions = [d.score for d in aggregate if d.score is not None]
    overall = (sum(overall_opinions) / len(overall_opinions)
               if overall_opinions else None)
    return DirScorecard(clip_count=len(per_clip), per_clip=per_clip,
                         aggregate=aggregate, overall=overall)


def scorecard_to_dict(card: Any) -> Dict[str, Any]:
    """Serialize any scorecard dataclass to a plain dict - stable JSON shape
    so a human can diff two runs."""
    return asdict(card)


# ---------------------------------------------------------------------------
# Default lightweight scorers.
#
# None of these replace the real VBench-2.0 scorers. They are coarse
# structural checks that catch the obvious failure modes (zero-byte clip,
# wrong resolution, hard duplicate frames) so the harness is useful in CI
# before a user installs the real VLM-based scorers.

def _ffprobe_json(clip_path: str) -> Optional[Dict[str, Any]]:
    """Return ffprobe's JSON view of a clip, or None on any error."""
    if not clip_path or not os.path.exists(clip_path):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", clip_path],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("ffprobe failed on %s: %s", clip_path, exc)
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _video_stream(info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for stream in (info or {}).get("streams") or []:
        if stream.get("codec_type") == "video":
            return stream
    return None


def _playable_scorer(clip_path: str) -> Optional[float]:
    """Instance preservation (commonsense) proxy: does the clip play at all?

    The real VBench-2.0 scorer tracks whether every named instance stays
    present across the clip. We can only check whether anything is there
    at all - a trivial but necessary floor. 1.0 if the clip opens and has
    a video stream, 0.0 if it is unreadable or has no frames.
    """
    info = _ffprobe_json(clip_path)
    stream = _video_stream(info)
    if stream is None:
        return 0.0
    try:
        frames = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    if frames > 0:
        return 1.0
    # Some demuxers omit nb_frames; fall back to a positive duration from
    # either the stream or the container format section.
    for candidate in (stream.get("duration"),
                      ((info or {}).get("format") or {}).get("duration")):
        try:
            if candidate is not None and float(candidate) > 0:
                return 1.0
        except (TypeError, ValueError):
            continue
    return 0.0


def _non_static_scorer(clip_path: str) -> Optional[float]:
    """Motion rationality (commonsense) proxy: is the clip not a stuck frame?

    The real scorer uses optical flow and semantic motion coherence. Here
    we sample two frames one second apart and compare their PSNR via
    ffmpeg - a stuck clip registers as near-infinite PSNR. Returns None
    when sampling fails rather than scoring 0.0, so a missing-ffmpeg
    environment degrades to "no opinion" instead of a bad verdict.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostats",
                "-i", clip_path, "-filter_complex",
                "[0:v]select='between(t,0,0.001)',setpts=N[a];"
                "[0:v]select='between(t,1.0,1.001)',setpts=N[b];"
                "[a][b]psnr", "-f", "null", "-",
            ],
            capture_output=True, text=True, check=False, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    out = (proc.stderr or "")
    # ffmpeg psnr line looks like: PSNR y:... u:... v:... average:12.34 min:...
    for line in out.splitlines():
        if "average:" in line:
            try:
                token = line.split("average:")[1].split()[0]
                psnr = float(token)
            except (IndexError, ValueError):
                continue
            # PSNR > 40 dB on two neighbouring frames = essentially static
            # (matches ffmpeg's documented stale-frame threshold). Map
            # linearly: 20 dB and below -> 1.0, 50 dB and above -> 0.0.
            if psnr >= 50:
                return 0.0
            if psnr <= 20:
                return 1.0
            return 1.0 - (psnr - 20.0) / 30.0
    return None


def _resolution_scorer(clip_path: str) -> Optional[float]:
    """Composition (controllability) proxy: is the clip at least 720p?

    The real scorer checks framing, rule-of-thirds adherence, and prompt
    composition. Here we only penalize sub-720p output since it is a
    reliable signal of a degraded pipeline run. 1.0 for >=1280x720,
    0.5 for 640x360-1280x720, 0.0 below that.
    """
    stream = _video_stream(_ffprobe_json(clip_path))
    if stream is None:
        return None
    try:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width == 0 or height == 0:
        return None
    if width >= 1280 and height >= 720:
        return 1.0
    if width >= 640 and height >= 360:
        return 0.5
    return 0.0


def _register_default_scorers() -> None:
    register_scorer(ScorerSpec(
        name="ffprobe.playable",
        dimension="commonsense", capability="instance_preservation",
        fn=_playable_scorer,
        description="Clip opens and has at least one frame.",
    ))
    register_scorer(ScorerSpec(
        name="ffmpeg.psnr-neighbour",
        dimension="commonsense", capability="motion_rationality",
        fn=_non_static_scorer,
        description="Two frames one second apart are not near-identical (PSNR < 50 dB).",
    ))
    register_scorer(ScorerSpec(
        name="ffprobe.resolution",
        dimension="controllability", capability="composition",
        fn=_resolution_scorer,
        description="Clip is at least 720p.",
    ))


_register_default_scorers()


# ---------------------------------------------------------------------------
# CLI entry point

def _main(argv: List[str]) -> int:
    if len(argv) < 2 or argv[0] != "score":
        print("usage: python -m filmmaker.vbench2 score <clip-dir> [--suffix .mp4]",
              flush=True)
        return 2
    clip_dir = argv[1]
    suffix = ".mp4"
    i = 2
    while i < len(argv):
        if argv[i] == "--suffix" and i + 1 < len(argv):
            suffix = argv[i + 1]
            i += 2
            continue
        print(f"unknown argument: {argv[i]}", flush=True)
        return 2
    card = score_dir(clip_dir, suffix=suffix)
    print(json.dumps(scorecard_to_dict(card), indent=2))
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(_main(_sys.argv[1:]))
