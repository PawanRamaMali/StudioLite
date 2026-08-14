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
      "dialogue": "any spoken line, or empty string"
    },
    ...
  ]
}

Every shot must be renderable as a single still image (T1) and later a short
motion clip (T2). Keep descriptions concrete and visual.
"""


def run_storyboard(project: Project) -> Dict[str, Any]:
    breakdown = project.read_artifact("breakdown") or {}
    scenes = breakdown.get("scenes") or []
    if not scenes:
        raise ValueError("Breakdown artifact has no scenes.")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for scene in scenes:
        user_msg = json.dumps(scene, ensure_ascii=False, indent=2)
        raw = _chat(project, "storyboard", _STORYBOARD_SYSTEM, user_msg,
                    want_json=True, temperature=0.5, max_tokens=2500)
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
        out[scene["id"]] = norm
    return {"scenes": out}


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
                "camera_move":str(item.get("camera_move", "static")).lower(),
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


# ---------------------------------------------------------------------------
# 7. Shot Generator — SDXL keyframe per shot (T1 stub, no motion)
# ---------------------------------------------------------------------------

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

    render = _load_sdxl_renderer()
    result_shots: List[Dict[str, Any]] = []

    for scene in scenes:
        sid = scene["id"]
        shots = shots_by_scene.get(sid, [])
        dp = dp_by_scene.get(sid, {})
        scene_dir = os.path.join(project.shots_dir, sid)
        os.makedirs(scene_dir, exist_ok=True)
        for shot in shots:
            shot_id = shot["id"]
            dp_shot = dp.get(shot_id, {})
            prompt = _shot_prompt(style, scene, shot, dp_shot)
            png_path = os.path.join(scene_dir, f"{shot_id}.png")
            rendered = False
            error: Optional[str] = None
            try:
                if render is not None:
                    render(prompt, png_path)
                    rendered = True
                else:
                    _placeholder_png(png_path, f"{sid}/{shot_id}")
            except Exception as e:
                logger.exception("SDXL render failed for %s/%s; using placeholder", sid, shot_id)
                error = str(e)
                _placeholder_png(png_path, f"{sid}/{shot_id}")
            result_shots.append({
                "scene_id": sid,
                "shot_id": shot_id,
                "prompt": prompt,
                "path": os.path.relpath(png_path, project.dir).replace(os.sep, "/"),
                "duration_sec": int(dp_shot.get("duration_sec", 4) or 4),
                "dialogue": shot.get("dialogue", ""),
                "rendered": rendered,
                "error": error,
            })
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
    """Return a callable `(prompt, out_path) -> None`, or None if unavailable."""
    try:
        from reelforge import rf_generate_image
    except Exception as e:  # pragma: no cover
        logger.info("SDXL not available (%s); shot generator will use placeholders.", e)
        return None

    def _render(prompt: str, out_path: str) -> None:
        img = rf_generate_image(prompt, provider="sdxl_turbo", aspect_ratio="16:9")
        # rf_generate_image returns a PIL.Image
        img.save(out_path, "PNG")
    return _render


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
    shots: List[Dict[str, Any]] = shots_art.get("shots") or []
    if not shots:
        raise ValueError("No rendered shots to edit.")
    concat_lines: List[str] = []
    for s in shots:
        rel = s["path"]
        # Any missing image? Skip rather than fail — the placeholder writer
        # already covers the normal missing case.
        img_abs = os.path.join(project.dir, rel)
        if not os.path.exists(img_abs):
            continue
        dur = max(1, int(s.get("duration_sec", 4) or 4))
        concat_lines.append(f"file '{img_abs.replace(os.sep, '/')}'")
        concat_lines.append(f"duration {dur}")
    if not concat_lines:
        raise ValueError("No renderable shot images found.")
    # ffmpeg concat demuxer needs the last file repeated (no `duration` on it).
    last_rel = None
    for s in reversed(shots):
        img_abs = os.path.join(project.dir, s["path"])
        if os.path.exists(img_abs):
            last_rel = img_abs
            break
    if last_rel:
        concat_lines.append(f"file '{last_rel.replace(os.sep, '/')}'")

    concat_txt = os.path.join(project.dir, "artifacts", "editor.concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines))

    out_path = os.path.join(project.dir, "final.mp4")
    # The concat demuxer emits variable per-frame timing from the `duration`
    # entries above. `fps=30` inside the filter graph converts that to a
    # constant 30 fps stream, which QuickTime / Chrome MP4 playback expects.
    # We deliberately avoid `-r 30 -vsync vfr` — modern ffmpeg (>=7) rejects
    # that combo and the old `-vsync` flag is deprecated in favour of `-fps_mode`.
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-fps_mode", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
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
