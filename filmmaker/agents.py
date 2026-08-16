"""
All T1 stage agents in one place.

Each agent is a plain function `run(project) -> dict` that reads any needed
upstream artifacts, does its work, and returns the artifact dict to persist.
The orchestrator calls them; they don't call each other.

Adding a new stage → write another `run_*` here and register it in
`filmmaker/registry.py`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from . import llm
from .project import Project

logger = logging.getLogger("studiolite.filmmaker")


# ---------------------------------------------------------------------------
# 1. Producer — brief → 3 loglines → choose one
# ---------------------------------------------------------------------------

_PRODUCER_SYSTEM = """\
You are a seasoned indie-film Producer developing a short film.
Given the user's one-paragraph brief plus target runtime and visual style,
respond with EXACTLY THREE distinct loglines the film could pursue.
Each logline is one sentence, 20-35 words, in the classic format:
"When [inciting incident], a [protagonist] must [goal] before [stakes]."

Also return a suggested title per logline (short, evocative) and a tone tag
(one of: drama, thriller, comedy, sci-fi, horror, romance, mystery, action, documentary, experimental).

Return JSON of the shape:
{
  "loglines": [
    {"title": "...", "tone": "...", "logline": "..."},
    {"title": "...", "tone": "...", "logline": "..."},
    {"title": "...", "tone": "...", "logline": "..."}
  ],
  "recommended_index": 0
}
`recommended_index` is your own pick (0-based). The user can override.
"""


def run_producer(project: Project) -> Dict[str, Any]:
    meta = project.meta
    cfg = meta.config
    user_msg = (
        f"Brief:\n{meta.brief}\n\n"
        f"Target runtime: {cfg.target_minutes:.1f} minutes\n"
        f"Visual style: {cfg.style}\n"
    )
    raw = _chat(project, "producer", _PRODUCER_SYSTEM, user_msg, want_json=True, temperature=0.9)
    data = llm.parse_json(raw)
    loglines = data.get("loglines") or []
    if not isinstance(loglines, list) or len(loglines) == 0:
        raise llm.LLMError("Producer returned no loglines.")
    # Normalize
    loglines = [{
        "title": str(l.get("title", "Untitled")).strip(),
        "tone": str(l.get("tone", "drama")).strip().lower(),
        "logline": str(l.get("logline", "")).strip(),
    } for l in loglines if isinstance(l, dict)]
    idx = int(data.get("recommended_index", 0))
    idx = max(0, min(len(loglines) - 1, idx))
    return {
        "loglines": loglines,
        "chosen_index": idx,       # user can PUT-edit this
        "recommended_index": idx,
    }


# ---------------------------------------------------------------------------
# 2. Screenwriter — chosen logline → screenplay
# ---------------------------------------------------------------------------

_SCREENWRITER_SYSTEM = """\
You are a professional screenwriter drafting a SHORT film in Fountain-style
plaintext screenplay format. Follow these conventions strictly:

- Scene headings on their own line: `INT. LOCATION - TIME` or `EXT. LOCATION - TIME`.
- Action lines: present tense, visual, one short paragraph.
- Character cue on its own line in ALL CAPS, then the dialogue line below it.
- Parentheticals in (parens) on their own line under the cue, used sparingly.
- Use a blank line between blocks.

Constraints:
- Total runtime target: about ONE screen page per minute.
- Cast: 2-4 named speaking characters. Introduce each in ALL CAPS on first
  appearance in an action line.
- Make it filmable with the given visual style; avoid long monologues.
- No preface, no epilogue, no formatting tricks — just the screenplay.
"""


def run_screenwriter(project: Project) -> Dict[str, Any]:
    prod = project.read_artifact("producer") or {}
    loglines = prod.get("loglines") or []
    idx = int(prod.get("chosen_index", 0))
    if not (0 <= idx < len(loglines)):
        raise ValueError("Producer artifact has no valid chosen logline.")
    chosen = loglines[idx]
    cfg = project.meta.config

    user_msg = (
        f"Title: {chosen.get('title')}\n"
        f"Tone: {chosen.get('tone')}\n"
        f"Logline: {chosen.get('logline')}\n"
        f"Target runtime: {cfg.target_minutes:.1f} minutes\n"
        f"Visual style: {cfg.style}\n\n"
        "Write the screenplay now."
    )
    fountain = _chat(project, "screenwriter", _SCREENWRITER_SYSTEM, user_msg,
                     want_json=False, temperature=0.7, max_tokens=6000)
    fountain = fountain.strip()
    if not fountain:
        raise llm.LLMError("Screenwriter returned an empty draft.")
    return {
        "fountain": fountain,
        "title": chosen.get("title", "Untitled Film"),
        "revisions": 0,
    }


# ---------------------------------------------------------------------------
# 3. Story Editor — critique + revise
# ---------------------------------------------------------------------------

_STORY_EDITOR_SYSTEM = """\
You are a story editor doing ONE revision pass on a short-film draft.

Return JSON:
{
  "notes": "5-8 bullet critique focused on structure, character motivation, and shootability.",
  "revised_fountain": "the full revised screenplay in the same Fountain-style format"
}

Rules:
- The revised draft must still be filmable with 2-4 characters and roughly the same runtime.
- Preserve scene headings unless a scene truly does not earn its place.
- Do NOT add stage directions the visual generator can't render (e.g., "her eyes fill with 20 years of regret").
"""


def run_story_editor(project: Project) -> Dict[str, Any]:
    draft = project.read_artifact("screenwriter") or {}
    fountain = draft.get("fountain", "")
    if not fountain.strip():
        raise ValueError("Screenwriter artifact is empty.")
    user_msg = f"Original draft:\n\n{fountain}"
    raw = _chat(project, "story_editor", _STORY_EDITOR_SYSTEM, user_msg,
                want_json=True, temperature=0.4, max_tokens=6000)
    data = llm.parse_json(raw)
    notes = str(data.get("notes", "")).strip()
    revised = str(data.get("revised_fountain", "")).strip()
    if not revised:
        raise llm.LLMError("Story editor returned no revised draft.")
    return {
        "notes": notes,
        "revised_fountain": revised,
    }


# ---------------------------------------------------------------------------
# 4. Script Breakdown — screenplay → structured scene DB
# ---------------------------------------------------------------------------

_BREAKDOWN_SYSTEM = """\
You are a 1st AD doing a script breakdown. Convert the screenplay into a
structured scene list.

For each scene, extract:
- id: `s01`, `s02`, ...
- heading: the raw slugline (`INT./EXT. LOCATION - TIME`)
- summary: one sentence, present tense, plain English
- location: short slug like "diner-kitchen"
- interior: true/false
- time_of_day: "day"|"night"|"dawn"|"dusk"|"unspecified"
- characters: list of character names appearing / speaking
- mood: one short phrase ("tense stillness", "hopeful bustle", etc.)
- estimated_seconds: your best guess of screen time, integer

Return JSON: {"scenes": [ ... ]}. No commentary.
"""


def run_breakdown(project: Project) -> Dict[str, Any]:
    editor = project.read_artifact("story_editor") or {}
    screenplay = editor.get("revised_fountain") or ""
    if not screenplay.strip():
        # Fall back to the screenwriter draft if the story editor was skipped.
        draft = project.read_artifact("screenwriter") or {}
        screenplay = draft.get("fountain", "")
    if not screenplay.strip():
        raise ValueError("No screenplay to break down.")

    user_msg = f"Screenplay:\n\n{screenplay}"
    raw = _chat(project, "breakdown", _BREAKDOWN_SYSTEM, user_msg,
                want_json=True, temperature=0.2, max_tokens=6000)
    data = llm.parse_json(raw)
    scenes = data.get("scenes") or []
    scenes = [_normalize_scene(i, s) for i, s in enumerate(scenes) if isinstance(s, dict)]
    if not scenes:
        raise llm.LLMError("Breakdown returned no scenes.")
    return {"scenes": scenes}


def _normalize_scene(i: int, s: Dict[str, Any]) -> Dict[str, Any]:
    sid = str(s.get("id") or f"s{i+1:02d}")
    chars = s.get("characters") or []
    if not isinstance(chars, list):
        chars = [str(chars)]
    return {
        "id": sid,
        "heading": str(s.get("heading", "")).strip(),
        "summary": str(s.get("summary", "")).strip(),
        "location": str(s.get("location", "unknown")).strip(),
        "interior": bool(s.get("interior", True)),
        "time_of_day": str(s.get("time_of_day", "unspecified")).lower(),
        "characters": [str(c).strip() for c in chars if str(c).strip()],
        "mood": str(s.get("mood", "")).strip(),
        "estimated_seconds": int(s.get("estimated_seconds", 12) or 12),
    }


# ---------------------------------------------------------------------------
# 5. Storyboard — scene → shots
# ---------------------------------------------------------------------------

_STORYBOARD_SYSTEM = """\
You are a storyboard artist. For the given SCENE, break it into 3-8 discrete
shots. Return JSON:

{
  "shots": [
    {
      "id": "sh1",
      "description": "one-sentence visual description of the moment",
      "subject": "who or what is on frame",
      "action": "what happens in this shot",
      "dialogue": "the exact spoken line as written in the screenplay, or empty string"
    },
    ...
  ]
}

Every shot must be renderable as a single still image (T1) and later a short
motion clip (T2). Keep descriptions concrete and visual.

DIALOGUE RULES — non-negotiable:
- The `screenplay` field of the input contains the source of truth.
- Every spoken line in that screenplay (the text on the line under a
  character cue in ALL CAPS) MUST land verbatim in the `dialogue` field
  of exactly one shot. Do not paraphrase, do not shorten, do not merge
  lines across characters.
- If a shot has no dialogue, use an empty string. Non-speaking shots are
  fine — but every screenplay line must appear somewhere.
- Never invent dialogue that isn't in the screenplay.
"""


def run_storyboard(project: Project) -> Dict[str, Any]:
    breakdown = project.read_artifact("breakdown") or {}
    editor = project.read_artifact("story_editor") or {}
    screenwriter = project.read_artifact("screenwriter") or {}
    scenes = breakdown.get("scenes") or []
    if not scenes:
        raise ValueError("Breakdown artifact has no scenes.")

    full_screenplay = (editor.get("revised_fountain")
                       or screenwriter.get("fountain")
                       or "")
    scene_scripts = _split_screenplay_by_scene(full_screenplay, scenes)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for scene in scenes:
        sid = scene["id"]
        user_msg = json.dumps({
            "scene": scene,
            "screenplay": scene_scripts.get(sid) or full_screenplay,
        }, ensure_ascii=False, indent=2)
        raw = _chat(project, "storyboard", _STORYBOARD_SYSTEM, user_msg,
                    want_json=True, temperature=0.4, max_tokens=3000)
        data = llm.parse_json(raw)
        shots = data.get("shots") or []
        norm: List[Dict[str, Any]] = []
        for i, s in enumerate(shots):
            if not isinstance(s, dict):
                continue
            norm.append({
                "id": str(s.get("id") or f"sh{i+1}"),
                "description": str(s.get("description", "")).strip(),
                "subject": str(s.get("subject", "")).strip(),
                "action": str(s.get("action", "")).strip(),
                "dialogue": str(s.get("dialogue", "")).strip(),
            })
        if not norm:
            # Never leave a scene with zero shots — synthesize one.
            norm = [{
                "id": "sh1",
                "description": scene.get("summary", ""),
                "subject": ", ".join(scene.get("characters", [])) or "the scene",
                "action": scene.get("summary", ""),
                "dialogue": "",
            }]

        # Warn when the scene's screenplay clearly has speaking characters
        # but the storyboard emitted zero dialogue. That's the failure mode
        # #61 was filed for — surface it instead of silently dropping the
        # dialogue on the floor.
        script = scene_scripts.get(sid) or ""
        expected_speech = _screenplay_has_dialogue(script) if script else bool(scene.get("characters"))
        got_dialogue = any(x["dialogue"] for x in norm)
        if expected_speech and not got_dialogue:
            project.append_event({
                "type": "storyboard_dialogue_missing",
                "scene_id": sid,
                "message": ("Screenplay has spoken lines but the storyboard "
                            "emitted none. Voice Actor will have nothing to say."),
            })

        out[sid] = norm
    return {"scenes": out}


def _split_screenplay_by_scene(full: str, scenes: List[Dict[str, Any]]) -> Dict[str, str]:
    """Split the Fountain screenplay into per-scene slices by walking scene
    headings (INT./EXT. lines) in order and matching them to breakdown scenes
    by index. Not slug-aware — scene N of the screenplay maps to scenes[N] of
    the breakdown, which is how the breakdown is generated in the first place.
    """
    if not full or not scenes:
        return {}
    import re as _re
    heading = _re.compile(r"^(?:INT\.|EXT\.|INT/EXT\.|EXT/INT\.)", _re.M)
    starts = [m.start() for m in heading.finditer(full)]
    if not starts:
        return {}
    slices = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(full)
        slices.append(full[start:end].strip())
    out: Dict[str, str] = {}
    for i, scene in enumerate(scenes):
        if i < len(slices):
            out[scene["id"]] = slices[i]
    return out


def _screenplay_has_dialogue(script: str) -> bool:
    """A Fountain dialogue block is an ALL-CAPS character cue on its own line
    followed by a non-blank line. Detect that pattern without importing a real
    Fountain parser — false positives (props in caps) are cheap; a false
    negative just means we skip the warning, which is fine."""
    import re as _re
    return bool(_re.search(
        r"(?m)^[A-Z][A-Z .'\-]{1,40}\s*(?:\([^)]*\))?\s*\n\s*\S",
        script,
    ))


# ---------------------------------------------------------------------------
# 6. Cinematographer — per-shot camera / lens / duration
# ---------------------------------------------------------------------------

_CINEMATOGRAPHER_SYSTEM = """\
You are the Director of Photography. For each shot in the given scene, assign
concrete production values so the shot generator has enough to work with.

Return JSON:
{
  "shots": [
    {
      "id": "sh1",
      "framing": "WS" | "MS" | "MCU" | "CU" | "ECU" | "OTS" | "POV",
      "lens_mm": 24 | 35 | 50 | 85 | 135,
      "camera_move": "static" | "pan" | "dolly" | "handheld" | "crane" | "whip",
      "lighting": "one-line description of key light + mood",
      "palette": "3-5 color words separated by commas",
      "duration_sec": integer 2..10
    }, ...
  ]
}
Match the scene mood and the film's visual style.
"""


def run_cinematographer(project: Project) -> Dict[str, Any]:
    breakdown = project.read_artifact("breakdown") or {}
    storyboard = project.read_artifact("storyboard") or {}
    scenes = breakdown.get("scenes") or []
    shots_by_scene = storyboard.get("scenes") or {}
    if not scenes:
        raise ValueError("Breakdown missing.")

    style = project.meta.config.style
    out: Dict[str, Dict[str, Any]] = {}  # scene_id -> {shot_id -> dp_data}
    for scene in scenes:
        sid = scene["id"]
        shots = shots_by_scene.get(sid) or []
        if not shots:
            continue
        user_msg = json.dumps({
            "style": style,
            "scene": scene,
            "shots": shots,
        }, ensure_ascii=False, indent=2)
        raw = _chat(project, "cinematographer", _CINEMATOGRAPHER_SYSTEM, user_msg,
                    want_json=True, temperature=0.4, max_tokens=2500)
        data = llm.parse_json(raw)
        by_id: Dict[str, Any] = {}
        for item in data.get("shots") or []:
            if not isinstance(item, dict):
                continue
            shot_id = str(item.get("id") or "")
            if not shot_id:
                continue
            by_id[shot_id] = {
                "framing":    str(item.get("framing", "MS")).upper(),
                "lens_mm":    int(item.get("lens_mm", 35) or 35),
                "camera_move":_normalize_camera_move(item.get("camera_move")),
                "lighting":   str(item.get("lighting", "natural key light")).strip(),
                "palette":    str(item.get("palette", "neutral")).strip(),
                "duration_sec": max(1, min(15, int(item.get("duration_sec", 4) or 4))),
            }
        # Fill defaults for any missing shots.
        for sh in shots:
            by_id.setdefault(sh["id"], {
                "framing": "MS", "lens_mm": 35, "camera_move": "static",
                "lighting": "natural key light", "palette": "neutral", "duration_sec": 4,
            })
        out[sid] = by_id
    return {"scenes": out}


_ALLOWED_CAMERA_MOVES = {"static", "pan", "dolly", "handheld", "crane", "whip"}


def _normalize_camera_move(raw: Any) -> str:
    """Snap the LLM's camera_move to the DP-schema enum. Small models like to
    return descriptive phrases ("pan right to follow the character") instead
    of the plain enum; we accept the intent as long as an allowed token is in
    the phrase, otherwise fall back to static."""
    s = str(raw or "static").strip().lower()
    if s in _ALLOWED_CAMERA_MOVES:
        return s
    for tok in s.replace(",", " ").replace("-", " ").replace("/", " ").split():
        if tok in _ALLOWED_CAMERA_MOVES:
            return tok
    return "static"


# ---------------------------------------------------------------------------
# 7. Shot Generator — SDXL keyframe per shot (T1 stub, no motion)
# ---------------------------------------------------------------------------

# SDXL Turbo handles ~4 prompts per batch cleanly on 12 GB VRAM. Bigger cards
# can push higher, but the wall-clock win flattens past ~4 anyway.
_SDXL_BATCH_SIZE = 4

_QUALITY_STEPS: Dict[str, int] = {
    "draft":    12,
    "standard": 30,
    "high":     50,
    "ultra":    75,
}


def _quality_to_steps(q: str) -> int:
    return _QUALITY_STEPS.get((q or "standard").lower(), _QUALITY_STEPS["standard"])


def run_shots(project: Project) -> Dict[str, Any]:
    """T1: render one SDXL keyframe per shot. Never blocks the pipeline on a
    missing SDXL — if imagegen isn't importable or CUDA is absent, the stage
    still succeeds and produces placeholder text-card PNGs so the Editor has
    something to slice together. T2 upgrades to real video."""
    breakdown = project.read_artifact("breakdown") or {}
    storyboard = project.read_artifact("storyboard") or {}
    cinema = project.read_artifact("cinematographer") or {}
    scenes = breakdown.get("scenes") or []
    shots_by_scene = storyboard.get("scenes") or {}
    dp_by_scene = cinema.get("scenes") or {}
    style = project.meta.config.style
    steps = _quality_to_steps(project.meta.config.quality)

    render_one = _load_sdxl_renderer()
    render_batch = _load_sdxl_batch_renderer()

    # First pass: build the flat work list so we can batch across scenes.
    jobs: List[Dict[str, Any]] = []
    for scene in scenes:
        sid = scene["id"]
        shots = shots_by_scene.get(sid, [])
        dp = dp_by_scene.get(sid, {})
        scene_dir = os.path.join(project.shots_dir, sid)
        os.makedirs(scene_dir, exist_ok=True)
        for shot in shots:
            shot_id = shot["id"]
            dp_shot = dp.get(shot_id, {})
            jobs.append({
                "scene_id": sid,
                "shot_id": shot_id,
                "prompt": _shot_prompt(style, scene, shot, dp_shot),
                "png_path": os.path.join(scene_dir, f"{shot_id}.png"),
                "duration_sec": int(dp_shot.get("duration_sec", 4) or 4),
                "dialogue": shot.get("dialogue", ""),
                "rendered": False,
                "error": None,
            })

    total = len(jobs)
    project.append_event({"type": "shots_progress", "done": 0, "total": total})

    # Batch through the work list. On any batch failure, fall back to per-shot
    # for just that batch — a single OOM shouldn't discard the whole render.
    i = 0
    done = 0
    while i < total:
        chunk = jobs[i:i + _SDXL_BATCH_SIZE]
        used_batch = False
        if render_batch is not None and len(chunk) > 1:
            try:
                render_batch(
                    [j["prompt"] for j in chunk],
                    [j["png_path"] for j in chunk],
                    steps=steps,
                )
                for j in chunk:
                    j["rendered"] = True
                used_batch = True
            except Exception as e:
                logger.warning(
                    "SDXL batch of %d failed (%s); dropping to per-shot for this chunk",
                    len(chunk), e,
                )

        if not used_batch:
            for j in chunk:
                try:
                    if render_one is not None:
                        render_one(j["prompt"], j["png_path"], steps=steps)
                        j["rendered"] = True
                    else:
                        _placeholder_png(j["png_path"], f"{j['scene_id']}/{j['shot_id']}")
                except Exception as e:
                    logger.exception(
                        "SDXL render failed for %s/%s; using placeholder",
                        j["scene_id"], j["shot_id"],
                    )
                    j["error"] = str(e)
                    _placeholder_png(j["png_path"], f"{j['scene_id']}/{j['shot_id']}")

        done += len(chunk)
        i += _SDXL_BATCH_SIZE
        project.append_event({"type": "shots_progress", "done": done, "total": total})
        logger.info("shots progress: %d / %d", done, total)

    result_shots = [{
        "scene_id":     j["scene_id"],
        "shot_id":      j["shot_id"],
        "prompt":       j["prompt"],
        "path":         os.path.relpath(j["png_path"], project.dir).replace(os.sep, "/"),
        "duration_sec": j["duration_sec"],
        "dialogue":     j["dialogue"],
        "rendered":     j["rendered"],
        "error":        j["error"],
    } for j in jobs]
    return {"shots": result_shots}


def _shot_prompt(style: str, scene: Dict[str, Any], shot: Dict[str, Any], dp: Dict[str, Any]) -> str:
    """Build a single-line SDXL prompt from the DP + storyboard artifacts."""
    style_prefix = "cinematic photorealistic still, film grain" if style == "photoreal" \
                   else "stylized illustrated frame, painterly, cinematic"
    parts = [
        style_prefix,
        f"{dp.get('framing', 'MS')} shot",
        f"{dp.get('lens_mm', 35)}mm lens",
        shot.get("description") or shot.get("action") or "",
        f"location: {scene.get('location', '')}",
        f"time of day: {scene.get('time_of_day', '')}",
        f"mood: {scene.get('mood', '')}",
        f"lighting: {dp.get('lighting', '')}",
        f"palette: {dp.get('palette', '')}",
    ]
    return ", ".join([p for p in parts if p]).strip()


def _load_sdxl_renderer():
    """Return a callable `(prompt, out_path, *, steps=None) -> None`, or None if unavailable."""
    try:
        from reelforge import rf_generate_image
    except Exception as e:  # pragma: no cover
        logger.info("SDXL not available (%s); shot generator will use placeholders.", e)
        return None

    def _render(prompt: str, out_path: str, *, steps: Optional[int] = None) -> None:
        # rf_generate_image writes to `.mp/<uuid>.png` and returns that path —
        # not a PIL.Image, despite the plausible-sounding name.
        src = rf_generate_image(
            prompt, provider="sdxl_turbo", aspect_ratio="16:9", steps=steps,
        )
        shutil.move(src, out_path)
    return _render


def _load_sdxl_batch_renderer():
    """Return a callable `(prompts, out_paths, *, steps=None) -> None` that
    runs the SDXL pipeline on a whole batch in one forward pass, or None if
    the underlying pipeline isn't importable. Falls back cleanly (raises) on
    per-batch OOM so the caller can drop to per-shot mode."""
    try:
        from reelforge import rf_load_sdxl_pipeline
        import reelforge as _rf
    except Exception as e:  # pragma: no cover
        logger.info("SDXL batch renderer unavailable (%s)", e)
        return None

    def _render_batch(prompts: List[str], out_paths: List[str], *,
                      steps: Optional[int] = None) -> None:
        if len(prompts) != len(out_paths):
            raise ValueError("prompts / out_paths length mismatch")
        rf_load_sdxl_pipeline(None)
        pipe = getattr(_rf, "_sdxl_pipe", None)
        if pipe is None:
            raise RuntimeError("SDXL pipeline did not load")

        # Match the wrapper's default aspect ratio (16:9) and style-preset knobs
        # so batched output matches single-shot output.
        ar = _rf.ASPECT_RATIOS.get("16:9", {"image_gen_width": 1024, "image_gen_height": 576})
        preset = _rf.IMAGE_STYLE_PRESETS.get("photorealistic", {"positive": "", "negative": ""})
        enriched = [f"{p}, {preset['positive']}" for p in prompts]
        negatives = [preset["negative"]] * len(prompts)
        result = pipe(
            prompt=enriched,
            negative_prompt=negatives,
            num_inference_steps=steps or 30,
            guidance_scale=7.5,
            width=ar["image_gen_width"],
            height=ar["image_gen_height"],
        )
        images = result.images
        if len(images) != len(out_paths):
            raise RuntimeError(f"SDXL returned {len(images)} images for {len(out_paths)} prompts")
        for img, out in zip(images, out_paths):
            img.save(out, "PNG")
    return _render_batch


def _placeholder_png(path: str, label: str) -> None:
    """No-dep placeholder: 1280x720 dark card with a big centered label."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        # Even Pillow is missing? Write a 1-byte marker so the editor doesn't crash.
        with open(path, "wb") as f:
            f.write(b"\0")
        return
    img = Image.new("RGB", (1280, 720), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((1280 - tw) / 2, (720 - th) / 2), label, fill=(220, 220, 235), font=font)
    img.save(path, "PNG")


# ---------------------------------------------------------------------------
# 8. Editor — FFmpeg slideshow of keyframes with per-shot durations
# ---------------------------------------------------------------------------

def run_editor(project: Project) -> Dict[str, Any]:
    shots_art = project.read_artifact("shots") or {}
    cinema = project.read_artifact("cinematographer") or {}
    shots: List[Dict[str, Any]] = shots_art.get("shots") or []
    if not shots:
        raise ValueError("No rendered shots to edit.")

    dp_by_scene: Dict[str, Dict[str, Any]] = cinema.get("scenes") or {}

    clips_dir = os.path.join(project.artifacts_dir, "editor_clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths: List[str] = []
    for i, s in enumerate(shots):
        img_abs = os.path.join(project.dir, s["path"])
        # Any missing image? Skip rather than fail — the placeholder writer
        # already covers the normal missing case.
        if not os.path.exists(img_abs):
            continue
        dur = max(1, int(s.get("duration_sec", 4) or 4))
        scene_id = s.get("scene_id", "")
        shot_id = s.get("shot_id", "")
        camera_move = ((dp_by_scene.get(scene_id) or {})
                       .get(shot_id) or {}).get("camera_move", "static")
        clip_out = os.path.join(clips_dir, f"{i:03d}_{scene_id}_{shot_id}.mp4")
        try:
            _render_kenburns_clip(img_abs, clip_out, dur, camera_move)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not on PATH — cannot assemble the final cut.") from e
        except Exception:
            # Per-shot fallback: static concat-style clip so a broken zoompan
            # graph doesn't nuke the whole render.
            logger.exception("Ken Burns render failed for %s/%s; using static clip",
                             scene_id, shot_id)
            try:
                _render_static_clip(img_abs, clip_out, dur)
            except Exception:
                logger.exception("Static fallback failed for %s/%s; skipping", scene_id, shot_id)
                continue
        clip_paths.append(clip_out)

    if not clip_paths:
        raise ValueError("No renderable shot images found.")

    concat_txt = os.path.join(project.artifacts_dir, "editor.concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")

    out_path = os.path.join(project.dir, "final.mp4")
    # All per-shot clips share codec/pix_fmt/size/fps by construction, so
    # concat with stream copy — no re-encode, no framerate rewriting.
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not on PATH — cannot assemble the final cut.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed:\n{e.stderr[-800:] if e.stderr else e}") from e

    total = sum(max(1, int(s.get("duration_sec", 4) or 4)) for s in shots)
    return {
        "output_path": os.path.relpath(out_path, project.dir).replace(os.sep, "/"),
        "duration_sec": total,
        "shot_count": len(shots),
    }


def _render_kenburns_clip(img: str, out_path: str, duration_sec: int, camera_move: str) -> None:
    """Render one still into a duration_sec clip at 1280x720 30fps with camera
    motion matched to the DP's `camera_move` intent."""
    fps = 30
    frames = duration_sec * fps
    # Feed zoompan a large source so the crop stays sharp — otherwise ffmpeg
    # scales the still to output size first and the zoom shows pixel edges.
    src = ("scale=3840:2160:force_original_aspect_ratio=increase,"
           "crop=3840:2160,setsar=1")
    kb = _kenburns_expr(camera_move, frames, fps)
    vf = f"{src},zoompan={kb}:s=1280x720:fps={fps}"
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img,
        "-vf", vf,
        "-t", str(duration_sec),
        "-fps_mode", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _kenburns_expr(camera_move: str, frames: int, fps: int) -> str:
    """Return the `z=...:x=...:y=...:d=N` clause for a zoompan filter, mapped
    from the Cinematographer's camera_move field. `on` is the current output
    frame index inside zoompan expressions."""
    move = (camera_move or "static").lower()
    span = max(frames - 1, 1)
    if move == "dolly":
        z = "'min(zoom+0.0015,1.30)'"
        x = "'iw/2-(iw/zoom/2)'"
        y = "'ih/2-(ih/zoom/2)'"
    elif move == "pan":
        z = "'1.15'"
        x = f"'(iw-iw/zoom)*on/{span}'"
        y = "'(ih-ih/zoom)/2'"
    elif move == "crane":
        z = "'1.15'"
        x = "'(iw-iw/zoom)/2'"
        y = f"'(ih-ih/zoom)*(1-on/{span})'"
    elif move == "whip":
        z = "'1.20'"
        x = f"'(iw-iw/zoom)*min(on*2/{span},1)'"
        y = "'(ih-ih/zoom)/2'"
    elif move == "handheld":
        z = "'1.12'"
        x = f"'(iw-iw/zoom)/2 + 20*sin(on/{fps}*6.28)'"
        y = f"'(ih-ih/zoom)/2 + 15*sin(on/{fps}*4.71)'"
    else:
        # static or unknown — very slow push-in reads as "digital cinema"
        z = "'min(zoom+0.0004,1.08)'"
        x = "'iw/2-(iw/zoom/2)'"
        y = "'ih/2-(ih/zoom/2)'"
    return f"z={z}:x={x}:y={y}:d={frames}"


def _render_static_clip(img: str, out_path: str, duration_sec: int) -> None:
    """Fallback: encode the still as a static 1280x720 30fps clip so it can
    still concat cleanly with the successful Ken Burns clips around it."""
    vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
          "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30")
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img,
        "-vf", vf,
        "-t", str(duration_sec),
        "-fps_mode", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 9. Voice Casting — LLM assigns a Piper voice per named character.
# ---------------------------------------------------------------------------

_PIPER_POOL: List[Dict[str, str]] = [
    {"name": "Amy",      "gender": "female", "accent": "US"},
    {"name": "Ryan",     "gender": "male",   "accent": "US"},
    {"name": "Lessac",   "gender": "male",   "accent": "US"},
    {"name": "Libritts", "gender": "neutral","accent": "US"},
    {"name": "Kathleen", "gender": "female", "accent": "US"},
    {"name": "Danny",    "gender": "male",   "accent": "US"},
    {"name": "Alan",     "gender": "male",   "accent": "UK"},
    {"name": "Alba",     "gender": "female", "accent": "UK"},
    {"name": "Jenny",    "gender": "female", "accent": "UK"},
    {"name": "Northern", "gender": "male",   "accent": "UK"},
    {"name": "Semaine",  "gender": "female", "accent": "UK"},
    {"name": "Southern", "gender": "female", "accent": "UK"},
]
_PIPER_NAMES = {v["name"] for v in _PIPER_POOL}


_VOICE_CAST_SYSTEM = """\
You cast voice actors for a short film. Given the list of characters and the
available voice pool, assign ONE voice per character. Prefer variety and
match gender / register where reasonable.

Return JSON:
{
  "cast": [
    {"character": "MARGO", "voice": "Amy",  "why": "warm mid-range for the narrator"},
    ...
  ]
}
The `voice` field MUST be one of the pool names. No commentary.
"""


def run_voice_cast(project: Project) -> Dict[str, Any]:
    breakdown = project.read_artifact("breakdown") or {}
    scenes = breakdown.get("scenes") or []
    characters: List[str] = []
    seen = set()
    for scene in scenes:
        for c in scene.get("characters", []) or []:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c); characters.append(c)
    if not characters:
        return {"cast": {}, "note": "No named characters in breakdown; nothing to cast."}

    user_msg = json.dumps({
        "characters": characters,
        "voice_pool": _PIPER_POOL,
    }, ensure_ascii=False, indent=2)
    try:
        raw = _chat(project, "voice_cast", _VOICE_CAST_SYSTEM, user_msg,
                    want_json=True, temperature=0.3, max_tokens=1500)
        data = llm.parse_json(raw)
        cast_list = data.get("cast") or []
    except Exception as e:
        logger.warning("voice_cast LLM failed (%s); falling back to round-robin", e)
        cast_list = []

    cast: Dict[str, Dict[str, str]] = {}
    fallback = [v["name"] for v in _PIPER_POOL]
    for i, char in enumerate(characters):
        item = next((c for c in cast_list
                     if isinstance(c, dict)
                     and str(c.get("character", "")).strip().upper() == char.upper()), None)
        picked = str((item or {}).get("voice", "")).strip()
        if picked not in _PIPER_NAMES:
            picked = fallback[i % len(fallback)]
        cast[char] = {"voice": picked, "why": str((item or {}).get("why", "")).strip()}
    return {"cast": cast}


# ---------------------------------------------------------------------------
# 10. Voice Actor — Piper synthesizes each dialogue line.
# ---------------------------------------------------------------------------

def run_voice_actor(project: Project) -> Dict[str, Any]:
    storyboard = project.read_artifact("storyboard") or {}
    breakdown  = project.read_artifact("breakdown") or {}
    cast_art   = project.read_artifact("voice_cast") or {}
    scenes: List[Dict[str, Any]] = breakdown.get("scenes") or []
    shots_by_scene: Dict[str, List[Dict[str, Any]]] = storyboard.get("scenes") or {}
    cast: Dict[str, Dict[str, str]] = cast_art.get("cast") or {}

    voice_dir = os.path.join(project.artifacts_dir, "voice")
    os.makedirs(voice_dir, exist_ok=True)

    # PiperTTS handles voice model download on first use; if piper-tts isn't
    # importable, fall back to silent placeholders so the pipeline still runs.
    tts_cache: Dict[str, Any] = {}
    try:
        from mpv2.classes.PiperTts import PiperTTS
    except Exception as e:  # pragma: no cover
        logger.warning("Piper unavailable (%s); voice_actor will emit silent placeholders", e)
        PiperTTS = None  # type: ignore

    def get_tts(voice_name: str):
        if PiperTTS is None:
            return None
        if voice_name not in tts_cache:
            try:
                tts_cache[voice_name] = PiperTTS(voice_name=voice_name)
            except Exception as e:
                logger.warning("Piper voice %s failed to init (%s)", voice_name, e)
                tts_cache[voice_name] = None
        return tts_cache[voice_name]

    lines: List[Dict[str, Any]] = []
    for scene in scenes:
        sid = scene["id"]
        for shot in shots_by_scene.get(sid, []) or []:
            dialogue = str(shot.get("dialogue", "") or "").strip()
            if not dialogue:
                continue
            # Best-effort: try to pull the speaking character from the shot's
            # subject; otherwise pick the first character named in the scene.
            speaker = _guess_speaker(shot, scene, cast)
            voice_name = (cast.get(speaker, {}) or {}).get("voice") or _PIPER_POOL[0]["name"]

            wav_rel = f"voice/{sid}_{shot['id']}.wav"
            wav_abs = os.path.join(project.artifacts_dir, "voice", f"{sid}_{shot['id']}.wav")
            ok = False
            err: Optional[str] = None
            tts = get_tts(voice_name)
            if tts is not None:
                try:
                    tts.synthesize(dialogue, wav_abs)
                    ok = True
                except Exception as e:
                    logger.warning("Piper synth failed for %s/%s (%s); writing silence",
                                   sid, shot["id"], e)
                    err = str(e)
            if not ok:
                _write_silent_wav(wav_abs, 1.0)
            lines.append({
                "scene_id": sid,
                "shot_id":  shot["id"],
                "speaker":  speaker,
                "voice":    voice_name,
                "text":     dialogue,
                "wav":      wav_rel,
                "duration_sec": _wav_duration_seconds(wav_abs),
                "synthesized": ok,
                "error": err,
            })
    return {"lines": lines}


def _guess_speaker(shot: Dict[str, Any], scene: Dict[str, Any],
                   cast: Dict[str, Dict[str, str]]) -> str:
    """Storyboard shots have `subject` but not a strict speaker field. Best-guess:
    match a cast character whose name appears in subject/action, else fall back
    to the scene's first character, else 'NARRATOR'."""
    haystack = f"{shot.get('subject', '')} {shot.get('action', '')}".upper()
    for char in cast.keys():
        if char and char.upper() in haystack:
            return char
    chars = scene.get("characters") or []
    if chars:
        return str(chars[0])
    return "NARRATOR"


def _write_silent_wav(path: str, duration_sec: float, rate: int = 22050) -> None:
    import wave, struct
    n = max(1, int(duration_sec * rate))
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def _wav_duration_seconds(path: str) -> float:
    try:
        import wave
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 22050)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 11. Composer — MusicGen scores the whole cut.
# ---------------------------------------------------------------------------

_COMPOSER_SYSTEM = """\
You are a film composer. Given the film's tone, brief, and total duration in
seconds, write ONE short text prompt suitable for MusicGen. Focus on:
mood, instrumentation, tempo (bpm), and a hint at dynamics.

Return JSON: {"prompt": "..."} — one sentence, under 40 words. No commentary.
"""


def run_composer(project: Project) -> Dict[str, Any]:
    editor_art = project.read_artifact("editor") or {}
    duration = int(editor_art.get("duration_sec", 0) or 0)
    if duration <= 0:
        raise ValueError("Editor artifact missing a valid duration_sec.")

    prod = project.read_artifact("producer") or {}
    loglines = prod.get("loglines") or []
    idx = int(prod.get("chosen_index", 0))
    chosen = loglines[idx] if 0 <= idx < len(loglines) else {}
    user_msg = json.dumps({
        "brief":    project.meta.brief[:400],
        "tone":     chosen.get("tone", "drama"),
        "title":    chosen.get("title", ""),
        "duration_sec": duration,
    }, ensure_ascii=False, indent=2)

    prompt = ""
    try:
        raw = _chat(project, "composer", _COMPOSER_SYSTEM, user_msg,
                    want_json=True, temperature=0.5, max_tokens=400)
        prompt = str((llm.parse_json(raw) or {}).get("prompt", "")).strip()
    except Exception as e:
        logger.warning("composer LLM failed (%s); using generic cue", e)

    if not prompt:
        prompt = f"cinematic underscore, subtle strings and piano, {chosen.get('tone','drama')} mood, 80 bpm"

    score_abs = os.path.join(project.artifacts_dir, "score.wav")
    ok = False
    err: Optional[str] = None
    try:
        _render_musicgen(prompt, score_abs, duration_sec=duration)
        ok = True
    except Exception as e:
        logger.warning("MusicGen unavailable/failed (%s); score will be silent", e)
        err = str(e)
        _write_silent_wav(score_abs, float(duration))
    return {
        "prompt": prompt,
        "path":   os.path.relpath(score_abs, project.dir).replace(os.sep, "/"),
        "duration_sec": duration,
        "rendered": ok,
        "error": err,
    }


def _render_musicgen(prompt: str, out_path: str, *, duration_sec: int) -> None:
    """Render a single MusicGen track. Uses facebook/musicgen-small — first
    call pulls ~2 GB from HF Hub; subsequent calls hit the local cache."""
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import torch as _torch

    model_id = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # MusicGen small is capped at ~30 s per generate call. Chain multiple
    # generations back-to-back to cover the full film length.
    sr = model.config.audio_encoder.sampling_rate
    tokens_per_sec = 50  # MusicGen encodec frame rate
    chunk_len_sec = 30
    chunks: list = []
    remaining = duration_sec
    while remaining > 0:
        this = min(chunk_len_sec, remaining)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with _torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=this * tokens_per_sec)
        arr = audio[0, 0].cpu().numpy().astype(np.float32)
        chunks.append(arr)
        remaining -= this
    combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    # Normalize to -3 dBFS to leave headroom for the mixer.
    peak = float(np.max(np.abs(combined))) or 1.0
    combined = (combined / peak) * 0.7
    sf.write(out_path, combined, sr)


# ---------------------------------------------------------------------------
# 12. Mixer — ffmpeg mux: silent cut + dialogue at shot offsets + score with
#     sidechain ducking under speech.
# ---------------------------------------------------------------------------

def run_mixer(project: Project) -> Dict[str, Any]:
    editor_art   = project.read_artifact("editor") or {}
    composer_art = project.read_artifact("composer") or {}
    voice_art    = project.read_artifact("voice_actor") or {}
    shots_art    = project.read_artifact("shots") or {}

    silent_rel = editor_art.get("output_path")
    if not silent_rel:
        raise ValueError("Editor artifact missing output_path.")
    silent_abs = os.path.join(project.dir, silent_rel)
    if not os.path.exists(silent_abs):
        raise ValueError(f"Silent cut missing at {silent_abs}")

    # Cumulative shot offsets (ms) — dialogue for shot N lands at sum(durations 0..N-1).
    offsets_ms: Dict[str, int] = {}
    cursor_ms = 0
    for s in shots_art.get("shots") or []:
        key = f"{s['scene_id']}::{s['shot_id']}"
        offsets_ms[key] = cursor_ms
        cursor_ms += max(1, int(s.get("duration_sec", 4) or 4)) * 1000

    # Dialogue lines with their absolute offset within the cut.
    dialogue: List[Dict[str, Any]] = []
    for line in voice_art.get("lines") or []:
        key = f"{line['scene_id']}::{line['shot_id']}"
        wav_abs = os.path.join(project.artifacts_dir, "voice",
                               f"{line['scene_id']}_{line['shot_id']}.wav")
        if not os.path.exists(wav_abs):
            continue
        dialogue.append({"wav": wav_abs, "delay_ms": offsets_ms.get(key, 0)})

    score_rel = composer_art.get("path")
    score_abs = os.path.join(project.dir, score_rel) if score_rel else None
    have_score = bool(score_abs and os.path.exists(score_abs))
    have_speech = bool(dialogue)

    out_abs = os.path.join(project.dir, "final_mixed.mp4")
    _run_mixer_ffmpeg(
        silent_video=silent_abs,
        dialogue=dialogue,
        score=score_abs if have_score else None,
        out_path=out_abs,
    )
    return {
        "output_path":  os.path.relpath(out_abs, project.dir).replace(os.sep, "/"),
        "has_dialogue": have_speech,
        "has_score":    have_score,
        "dialogue_lines": len(dialogue),
    }


def _run_mixer_ffmpeg(*, silent_video: str, dialogue: List[Dict[str, Any]],
                      score: Optional[str], out_path: str) -> None:
    """Build and run the ffmpeg mux. Handles four cases: dialogue+score,
    dialogue only, score only, and neither (in which case we just remux
    the silent cut through the same encoder settings for consistency)."""
    inputs: List[str] = ["-i", silent_video]
    if score:
        inputs += ["-i", score]
    for d in dialogue:
        inputs += ["-i", d["wav"]]

    filter_parts: List[str] = []
    # Input indices — 0 = video; 1 = score (if any); N = dialogue lines after.
    score_idx = 1 if score else None
    dlg_start = (2 if score else 1)

    dlg_labels: List[str] = []
    for i, d in enumerate(dialogue):
        idx = dlg_start + i
        lbl = f"d{i}"
        delay = int(d["delay_ms"])
        filter_parts.append(
            f"[{idx}:a]adelay={delay}|{delay},aformat=sample_rates=44100:channel_layouts=stereo[{lbl}]"
        )
        dlg_labels.append(f"[{lbl}]")

    if dlg_labels:
        # Merge all dialogue tracks into a single speech bus. When there's
        # also a score to duck under, we need the speech bus in two places
        # (sidechain input AND the final amix), so asplit=2 duplicates it —
        # ffmpeg filter labels are single-use.
        mix_expr = (f"{''.join(dlg_labels)}amix=inputs={len(dlg_labels)}:normalize=0"
                    if len(dlg_labels) > 1 else dlg_labels[0])
        if score is not None:
            filter_parts.append(f"{mix_expr},asplit=2[speech][speech_side]"
                                if len(dlg_labels) > 1
                                else f"{mix_expr}asplit=2[speech][speech_side]")
        else:
            filter_parts.append(f"{mix_expr}anull[speech]"
                                if len(dlg_labels) == 1
                                else f"{mix_expr}[speech]")

    if score is not None:
        filter_parts.append(
            f"[{score_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.45[music_raw]"
        )
        if dlg_labels:
            filter_parts.append(
                "[music_raw][speech_side]sidechaincompress="
                "threshold=0.04:ratio=8:attack=5:release=250[music_ducked]"
            )
            filter_parts.append("[speech][music_ducked]amix=inputs=2:normalize=0[audio]")
        else:
            filter_parts.append("[music_raw]anull[audio]")
    elif dlg_labels:
        filter_parts.append("[speech]anull[audio]")

    cmd = ["ffmpeg", "-y"] + inputs
    if filter_parts:
        cmd += ["-filter_complex", ";".join(filter_parts),
                "-map", "0:v", "-map", "[audio]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path]
    else:
        # Neither dialogue nor score — remux silent cut with an aac silent track
        # so downstream consumers still see a well-formed audio stream.
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mixer ffmpeg failed:\n{(e.stderr or '')[-800:]}") from e


# ---------------------------------------------------------------------------
# shared LLM plumbing
# ---------------------------------------------------------------------------

def _chat(project: Project, stage_key: str, system: str, user: str, *,
          want_json: bool, temperature: float, max_tokens: int = 4096) -> str:
    cfg = project.meta.config
    per = cfg.per_stage.get(stage_key, {}) if cfg.per_stage else {}
    backend = per.get("llm_backend") or cfg.llm_backend
    model = per.get("llm_model") or cfg.llm_model
    host = per.get("llm_host") or cfg.llm_host
    return llm.chat(
        system=system, user=user,
        backend=backend, model=model, host=host,
        temperature=temperature, max_tokens=max_tokens,
        want_json=want_json,
    )
