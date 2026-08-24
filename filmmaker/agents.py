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
    raw = _chat(project, "producer", _PRODUCER_SYSTEM, user_msg,
                want_json=True, temperature=0.9, max_tokens=2000)
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
plaintext screenplay format. Follow these conventions STRICTLY — the
downstream pipeline parses your output and silently drops any dialogue whose
character cue is not in ALL CAPS on its own line.

- Scene headings on their own line: `INT. LOCATION - TIME` or `EXT. LOCATION - TIME`.
- Action lines: present tense, visual, one short paragraph.
- Character cue on its own line in ALL CAPS. NEVER Title Case. Example:
      MARGO
      Where is he?
  Not:
      Margo
      Where is he?
- Parentheticals in (parens) on their own line under the cue, used sparingly:
      MARGO
      (quietly)
      Where is he?
  Do NOT put dialogue on the same line as the parenthetical.
- Blank line between blocks.

Constraints:
- Total runtime target: about ONE screen page per minute.
- **Scene count scales with runtime**. Under 2 minutes: 1-2 scenes is fine.
  2 to 3 minutes: at least 3 scenes. 4 minutes or longer: at least 4 scenes
  with distinct headings (location or time change), so the film has real
  structure instead of one long conversation in a single room.
- Cast: 2-4 named speaking characters. Introduce each in ALL CAPS on first
  appearance in an action line, followed by a short physical description in
  parentheses on the SAME line — age range, wardrobe, one distinctive
  feature. Downstream stages parse those descriptors to keep the character
  looking consistent across shots. Example:
      MARGO (60s, silver bob, apron dusted with flour) wipes the counter.
- Make it filmable with the given visual style; avoid long monologues.
- No preface, no epilogue, no formatting tricks. Just the screenplay.
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

SCENE COUNT — the film has a stated target runtime (in `target_minutes`
below). If the screenplay reads like one long single-location scene, break
it into MULTIPLE scenes at any natural beat — location change, time jump,
mood shift, new character arriving. Aim for one scene per 60 to 90 seconds
of runtime. Under 2 minutes may be one or two scenes; a 5 minute film
needs at least 4 scenes.

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

    target_minutes = project.meta.config.target_minutes
    user_msg = (f"target_minutes: {target_minutes}\n\n"
                f"Screenplay:\n\n{screenplay}")
    raw = _chat(project, "breakdown", _BREAKDOWN_SYSTEM, user_msg,
                want_json=True, temperature=0.2, max_tokens=6000)
    data = llm.parse_json(raw)
    scenes = data.get("scenes") or []
    scenes = [_normalize_scene(i, s) for i, s in enumerate(scenes) if isinstance(s, dict)]
    if not scenes:
        raise llm.LLMError("Breakdown returned no scenes.")

    # Enforce target runtime. The LLM under-estimates for long films (a 10
    # minute brief comes back as 4 scenes x 60s = 240s). Rescale proportionally
    # so the sum matches the requested runtime within a few percent.
    target_sec = int(target_minutes * 60)
    current = sum(s["estimated_seconds"] for s in scenes) or 1
    if target_sec >= 30 and abs(current - target_sec) / target_sec > 0.15:
        scale = target_sec / current
        for s in scenes:
            s["estimated_seconds"] = max(15, int(round(s["estimated_seconds"] * scale)))
        # Fix the last one to swallow rounding drift.
        drift = target_sec - sum(s["estimated_seconds"] for s in scenes)
        if scenes and abs(drift) <= 30:
            scenes[-1]["estimated_seconds"] = max(15, scenes[-1]["estimated_seconds"] + drift)

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
You are a storyboard artist. For the given SCENE, break it into discrete
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

SHOT COUNT — pace shots to the scene's `estimated_seconds`. Aim for one
shot per 6 to 10 seconds. Minimum 3 shots for any scene. The soft
maximum scales with scene length: roughly `estimated_seconds / 6`, but
never more than 20 for one scene. A 20 second scene should have 3 or 4
shots. A 60 second scene: 8 to 10. A 120 second scene: 15 to 20. Do
NOT pack more shots than the scene runtime warrants.

Every shot must be renderable as a single still image (T1) and later a short
motion clip (T2). Keep descriptions concrete and visual. When you name a
character in `subject` or `action`, use the SAME name the screenplay uses
in its cues (so downstream stages can match dialogue to the right shot).

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
                    want_json=True, temperature=0.4, max_tokens=5000)
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
      "duration_sec": integer 6..18
    }, ...
  ]
}

DURATION — the sum of `duration_sec` across all shots in a scene should
roughly equal the scene's `estimated_seconds`. So if the scene runs 90
seconds and has 12 shots, each shot averages about 7 to 8 seconds. Do
NOT default every shot to 6. Vary duration to match the beat — quick
reaction cutaways can be 6 seconds, dialogue and establishing shots
should run 10 to 15 seconds so they can breathe. Match the scene mood
and the film's visual style. Prefer fewer, longer shots over many short
ones — held frames read as intentional; rapid cutting reads as churn.
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
                    want_json=True, temperature=0.4, max_tokens=8000)
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
                "duration_sec": max(6, min(20, int(item.get("duration_sec", 8) or 8))),
            }
        # Fill defaults for any missing shots.
        for sh in shots:
            by_id.setdefault(sh["id"], {
                "framing": "MS", "lens_mm": 35, "camera_move": "static",
                "lighting": "natural key light", "palette": "neutral", "duration_sec": 8,
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
    variant = project.meta.config.sdxl_variant

    render_one = _load_sdxl_renderer(variant=variant)
    # Batch renderer is Turbo-only for now (SDXL base uses CPU offload; batching
    # forces the whole pipeline back to GPU and pushes us into OOM territory).
    render_batch = _load_sdxl_batch_renderer() if variant == "turbo" else None

    # Parse character bios from the screenplay so we can bake them into
    # every shot prompt where the character appears. Same physical descriptors
    # show up in every prompt for MARGO, so SDXL is nudged toward the same
    # look even without IP-Adapter as a backstop.
    editor = project.read_artifact("story_editor") or {}
    screenwriter = project.read_artifact("screenwriter") or {}
    full_screenplay = (editor.get("revised_fountain")
                       or screenwriter.get("fountain") or "")
    character_bios = _extract_character_bios(full_screenplay)

    # Load character portraits (from the character_portraits stage) as PIL
    # images keyed by uppercased name. Each shot whose subject/action names
    # one of these characters will pass the portrait as an IP-Adapter
    # reference so the same face lands in every appearance.
    portraits_art = project.read_artifact("character_portraits") or {}
    portraits_pil: Dict[str, Any] = {}
    try:
        from PIL import Image as _PILImage
        for name, meta in (portraits_art.get("portraits") or {}).items():
            p = meta.get("path")
            if not p:
                continue
            abs_p = os.path.join(project.dir, p)
            if not os.path.exists(abs_p):
                continue
            try:
                portraits_pil[name.upper()] = _PILImage.open(abs_p).convert("RGB")
            except Exception as e:
                logger.warning("Could not load portrait for %s: %s", name, e)
    except Exception:
        pass

    def portrait_for_shot(shot: Dict[str, Any]) -> Any:
        """Return a PIL portrait for the primary character in this shot, or
        None if no named character matches."""
        if not portraits_pil:
            return None
        hay = f"{shot.get('subject','')} {shot.get('action','')}".upper()
        for name_up, img in portraits_pil.items():
            for tok in name_up.split():
                if len(tok) >= 3 and tok in hay:
                    return img
        return None

    # Voice actor's per-shot dialogue durations. If a shot has spoken lines,
    # we extend its screen time so the line fits (plus half a second of pad)
    # and no two lines end up overlapping in the mixer.
    voice_art = project.read_artifact("voice_actor") or {}
    dialogue_dur_by_key: Dict[str, float] = {}
    for line in voice_art.get("lines", []) or []:
        key = f"{line['scene_id']}::{line['shot_id']}"
        dialogue_dur_by_key[key] = max(
            dialogue_dur_by_key.get(key, 0.0),
            float(line.get("duration_sec", 0) or 0),
        )

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
            base_dur = int(dp_shot.get("duration_sec", 4) or 4)
            dlg_dur = dialogue_dur_by_key.get(f"{sid}::{shot_id}", 0.0)
            # Enough room for the wav to play out with pad + a hair extra
            # so the next shot's line never starts before this one finishes.
            fitted = int(dlg_dur + 0.75) if dlg_dur > 0 else 0
            duration_sec = max(base_dur, fitted, 3)
            jobs.append({
                "scene_id": sid,
                "shot_id": shot_id,
                "prompt": _shot_prompt(style, scene, shot, dp_shot, character_bios),
                "png_path": os.path.join(scene_dir, f"{shot_id}.png"),
                "duration_sec": duration_sec,
                "dialogue": shot.get("dialogue", ""),
                "ref_image": portrait_for_shot(shot),
                "rendered": False,
                "error": None,
            })

    total = len(jobs)
    project.append_event({"type": "shots_progress", "done": 0, "total": total})

    # Batch through the work list. On any batch failure, fall back to per-shot
    # for just that batch — a single OOM shouldn't discard the whole render.
    # We split by whether the chunk has any character references — mixed
    # batches technically work (diffusers accepts per-slot None) but keep it
    # simple: character-heavy chunks tend to run smoothly grouped.
    i = 0
    done = 0
    while i < total:
        chunk = jobs[i:i + _SDXL_BATCH_SIZE]
        used_batch = False
        chunk_has_refs = any(j["ref_image"] is not None for j in chunk)
        if render_batch is not None and len(chunk) > 1:
            try:
                render_batch(
                    [j["prompt"] for j in chunk],
                    [j["png_path"] for j in chunk],
                    steps=steps,
                    ref_images=([j["ref_image"] for j in chunk] if chunk_has_refs else None),
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
                        render_one(j["prompt"], j["png_path"], steps=steps,
                                   ref_image=j["ref_image"])
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


def _shot_prompt(style: str, scene: Dict[str, Any], shot: Dict[str, Any],
                 dp: Dict[str, Any], character_bios: Optional[Dict[str, str]] = None) -> str:
    """Build a single-line SDXL prompt from the DP + storyboard artifacts.
    If character_bios is provided, physical descriptors for any character
    named in this shot's subject/action get baked into the prompt so the
    same person renders with the same look across scenes."""
    style_prefix = "cinematic photorealistic still, film grain" if style == "photoreal" \
                   else "stylized illustrated frame, painterly, cinematic"

    parts = [
        style_prefix,
        f"{dp.get('framing', 'MS')} shot",
        f"{dp.get('lens_mm', 35)}mm lens",
    ]

    # Character descriptors take priority so they land inside the CLIP 77-token
    # window before palette/lighting boilerplate gets truncated.
    if character_bios:
        haystack = f"{shot.get('subject','')} {shot.get('action','')} {shot.get('description','')}".upper()
        for name, bio in character_bios.items():
            for tok in name.upper().split():
                if len(tok) >= 3 and tok in haystack:
                    parts.append(f"{name}: {bio}")
                    break

    parts += [
        shot.get("description") or shot.get("action") or "",
        f"location: {scene.get('location', '')}",
        f"time of day: {scene.get('time_of_day', '')}",
        f"mood: {scene.get('mood', '')}",
        f"lighting: {dp.get('lighting', '')}",
        f"palette: {dp.get('palette', '')}",
    ]
    return ", ".join([p for p in parts if p]).strip()


_NAME_BLOCKLIST = {
    "THE", "A", "AN", "THIS", "THAT", "IT", "HE", "SHE", "THEY", "WE",
    "SOME", "EVERY", "MY", "OUR", "HIS", "HER", "THEIR", "YOUR", "THEIR",
    "SUDDENLY", "MEANWHILE", "OUTSIDE", "INSIDE", "LATER", "THEN", "NOW",
    "THERE", "HERE", "WHERE", "WHEN", "WHILE", "AFTER", "BEFORE", "AS",
}
_CUE_PAREN_TERMS = {
    "V.O.", "O.S.", "CONT'D", "CONTD", "MORE", "BEAT", "PAUSE",
    "OFF SCREEN", "OFFSCREEN", "VOICE OVER", "VOICEOVER",
}


def _extract_character_bios(script: str) -> Dict[str, str]:
    """Walk action lines for character INTRODUCTIONS of the form
    `NAME (age, wardrobe, feature) does something...` or the comma variant
    `NAME, description[, more description].`. Skips dialogue blocks so we
    don't grab spoken lines that happen to start with a Title-Case word,
    and skips cue-style annotations like `(V.O.)` or `(to himself)`."""
    import re as _re
    bios: Dict[str, str] = {}
    lines = script.splitlines()
    # Intros often land mid-paragraph ("The passenger is JENNY, mid-30s, ..."),
    # so we scan each action line with finditer instead of anchoring to `^`.
    paren_intro = _re.compile(r"\b([A-Z][A-Za-z]{2,}(?: [A-Z][A-Za-z]{2,}){0,2})\s*\(([^)]{5,120})\)\s+[a-z]")
    comma_intro = _re.compile(r"\b([A-Z][A-Za-z]{2,}(?: [A-Z][A-Za-z]{2,}){0,2}),\s*([^\n]{5,300}?)(?=[.\n]|$)")

    def _accept(name: str, desc: str) -> Optional[tuple]:
        clean = _clean_name(name)
        if not clean:
            return None
        head = clean.upper().split()[0] if clean.split() else ""
        if head in _NAME_BLOCKLIST:
            return None
        d = desc.strip()
        if d.upper() in _CUE_PAREN_TERMS:
            return None
        if d.lower().startswith(("to ", "at ", "on ", "off ", "for ", "into ",
                                 "with a ", "under ", "over ", "toward ",
                                 "in a ", "at the ")):
            return None
        if not _looks_like_desc(d):
            return None
        return (clean, d)

    # Walk with a small state machine so we ONLY look at action lines. A cue
    # (character line) puts us into dialogue mode; blank line exits it.
    in_dialogue = False
    for raw in lines:
        line = raw.strip()
        if not line:
            in_dialogue = False
            continue
        if not in_dialogue and _looks_like_cue(raw) is not None:
            in_dialogue = True
            continue
        if in_dialogue:
            continue

        for m in paren_intro.finditer(line):
            picked = _accept(m.group(1), m.group(2))
            if picked and picked[0] not in bios:
                bios[picked[0]] = picked[1]
        for m in comma_intro.finditer(line):
            picked = _accept(m.group(1), m.group(2))
            if picked and picked[0] not in bios:
                bios[picked[0]] = picked[1]
    return bios


def _clean_name(raw: str) -> str:
    # Drop trailing verbs the regex sometimes swallows. Keep the leading
    # capitalised words only.
    parts = raw.strip().split()
    kept: List[str] = []
    for w in parts:
        if not w:
            break
        if w[0].isupper() or (kept and w in {"of", "the"}):
            kept.append(w)
        else:
            break
    return " ".join(kept[:3]).strip()


_DESC_AGE = None
_DESC_WORDS = None


def _looks_like_desc(text: str) -> bool:
    """A real physical description carries an age marker (e.g. `mid-30s`,
    `60s`, `40 years old`) or a specific wardrobe / feature word matched at
    word boundaries. Substring matching would false-positive `hair` inside
    `chair` and `s ` inside `Let's`, which is how earlier passes leaked
    dialogue text into the bio dict."""
    import re as _re
    global _DESC_AGE, _DESC_WORDS
    if _DESC_AGE is None:
        _DESC_AGE = _re.compile(
            r"\b(?:mid[-\s]?)?(?:early|late)?[-\s]?"
            r"(?:\d{1,2}s|\d{1,2}[-\s]?year[-\s]?old|"
            r"teen|teens|twenties|thirties|forties|fifties|sixties|seventies|eighties)\b",
            _re.I,
        )
        _DESC_WORDS = _re.compile(
            r"\b(?:hair|beard|eyes?|jacket|coat|shirt|dress|apron|boots|"
            r"glasses|hat|scarf|uniform|wearing|clad|bald|tall|short|thin|"
            r"stout|young|elderly|grizzled|weathered|middle[-\s]?aged|"
            r"grey|gray|silver|blond|blonde|brunette|redhead|freckled|"
            r"tattoo|scar|goatee|moustache|mustache|stubble|pale|dark|"
            r"skinny|slim|broad|muscular|heavyset|petite|slender|lanky)\b",
            _re.I,
        )
    if _DESC_AGE.search(text):
        return True
    if _DESC_WORDS.search(text):
        return True
    return False


_IP_ADAPTER_STATE: str = "unknown"   # unknown | loaded | disabled


def _ensure_ip_adapter_loaded() -> bool:
    """Try to load IP-Adapter on reelforge's cached SDXL pipeline. Returns
    True if the adapter is currently loaded and usable; False otherwise.

    We can't tell in advance whether the pipeline architecture is compatible
    (SDXL Turbo, for instance, is distilled and rejects IP-Adapter at
    inference time with a matmul shape error). Callers should be ready to
    catch a RuntimeError from the actual pipeline call and invoke
    _disable_ip_adapter() to fall back gracefully."""
    global _IP_ADAPTER_STATE
    if _IP_ADAPTER_STATE in ("loaded", "disabled"):
        return _IP_ADAPTER_STATE == "loaded"
    try:
        from reelforge import rf_load_sdxl_pipeline
        import reelforge as _rf
        rf_load_sdxl_pipeline(None)
        pipe = getattr(_rf, "_sdxl_pipe", None)
        if pipe is None:
            _IP_ADAPTER_STATE = "disabled"
            return False
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
        )
        pipe.set_ip_adapter_scale(0.55)
        _IP_ADAPTER_STATE = "loaded"
        logger.info("IP-Adapter loaded on SDXL pipeline (scale=0.55)")
        return True
    except Exception as e:
        _IP_ADAPTER_STATE = "disabled"
        logger.warning("Failed to load IP-Adapter (%s); shots will render without face conditioning", e)
        return False


def _disable_ip_adapter() -> None:
    """Unload IP-Adapter after an incompatibility error at inference time.
    Marks the module state so we don't try to reload it this session."""
    global _IP_ADAPTER_STATE
    try:
        import reelforge as _rf
        pipe = getattr(_rf, "_sdxl_pipe", None)
        if pipe is not None and hasattr(pipe, "unload_ip_adapter"):
            pipe.unload_ip_adapter()
    except Exception:
        pass
    _IP_ADAPTER_STATE = "disabled"
    logger.warning("IP-Adapter disabled for the rest of the session")


def _hf_token() -> Optional[str]:
    """Return the user's HuggingFace token from `HF_TOKEN` or the standard
    `HUGGING_FACE_HUB_TOKEN` env var. None means anonymous — subject to the
    aggressive rate limit that stalls fresh model pulls after 1-3 GB."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


_SDXL_BASE_PIPE = None  # module-level cache so we load SDXL base once per session


def _load_sdxl_base_pipeline():
    """Load `stabilityai/stable-diffusion-xl-base-1.0` on demand. Returns the
    diffusers pipeline or None if unavailable. Uses fp16 + CPU offload so it
    fits alongside Ollama and Wan on a 12GB card."""
    global _SDXL_BASE_PIPE
    if _SDXL_BASE_PIPE is not None:
        return _SDXL_BASE_PIPE
    try:
        import torch as _torch
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=_torch.float16,
            variant="fp16",
            token=_hf_token(),
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        _SDXL_BASE_PIPE = pipe
        logger.info("SDXL 1.0 base pipeline loaded (fp16 + CPU offload)")
        return pipe
    except Exception as e:
        logger.warning("SDXL base unavailable (%s); falling back to reelforge Turbo path", e)
        return None


def _load_sdxl_renderer(variant: str = "turbo"):
    """Return a callable `(prompt, out_path, *, steps=None, ref_image=None) -> None`,
    or None if unavailable. `variant` selects the SDXL flavour:
    - "turbo": reelforge SDXL Turbo (fast, distilled — faces are flat)
    - "base":  SDXL 1.0 base at 30+ steps (slow, much higher fidelity)"""
    if variant == "base":
        pipe = _load_sdxl_base_pipeline()
        if pipe is None:
            variant = "turbo"  # fall through
        else:
            def _render_base(prompt: str, out_path: str, *,
                             steps: Optional[int] = None, ref_image=None) -> None:
                import torch as _torch
                # SDXL base without a refiner needs 30+ steps to look good.
                # Bake in a generic quality suffix so callers don't have to.
                enriched = (prompt + ", cinematic, sharp focus, high detail, film still")
                negative = ("low quality, distorted, deformed hands, extra fingers, "
                            "blurry, mangled face, watermark, text")
                with _torch.no_grad():
                    img = pipe(
                        prompt=enriched, negative_prompt=negative,
                        num_inference_steps=max(30, steps or 30),
                        guidance_scale=7.5, width=1024, height=576,
                    ).images[0]
                img.save(out_path, "PNG")
            return _render_base

    # Turbo path via reelforge.
    try:
        from reelforge import rf_generate_image
        import reelforge as _rf
    except Exception as e:  # pragma: no cover
        logger.info("SDXL not available (%s); shot generator will use placeholders.", e)
        return None

    def _render(prompt: str, out_path: str, *, steps: Optional[int] = None,
                ref_image=None) -> None:
        # rf_generate_image writes to `.mp/<uuid>.png` and returns that path.
        use_ref = ref_image is not None and _ensure_ip_adapter_loaded()
        try:
            if use_ref:
                src = rf_generate_image(
                    prompt, provider="sdxl_turbo", aspect_ratio="16:9",
                    steps=steps, ip_adapter_image=ref_image,
                )
            else:
                src = rf_generate_image(
                    prompt, provider="sdxl_turbo", aspect_ratio="16:9", steps=steps,
                )
        except RuntimeError as e:
            # SDXL Turbo distilled attention rejects the IP-Adapter projection
            # with `mat1 and mat2 shapes cannot be multiplied`. Disable
            # IP-Adapter for the rest of the session and retry once without.
            if use_ref and "shapes cannot be multiplied" in str(e):
                _disable_ip_adapter()
                src = rf_generate_image(
                    prompt, provider="sdxl_turbo", aspect_ratio="16:9", steps=steps,
                )
            else:
                raise
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
                      steps: Optional[int] = None,
                      ref_images: Optional[List[Any]] = None) -> None:
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

        kwargs: Dict[str, Any] = {
            "prompt": enriched,
            "negative_prompt": negatives,
            "num_inference_steps": steps or 30,
            "guidance_scale": 7.5,
            "width": ar["image_gen_width"],
            "height": ar["image_gen_height"],
        }
        # If any shot in the batch has a reference image, load IP-Adapter and
        # pass a list matching the batch size. Diffusers accepts None per slot
        # to mean "no reference for this prompt".
        use_refs = (ref_images is not None
                    and any(r is not None for r in ref_images)
                    and _ensure_ip_adapter_loaded())
        if use_refs:
            if len(ref_images) != len(prompts):
                raise ValueError("ref_images length must match prompts length")
            kwargs["ip_adapter_image"] = ref_images

        try:
            result = pipe(**kwargs)
        except RuntimeError as e:
            # Turbo's distilled attention rejects the IP-Adapter projection.
            # Disable IP-Adapter and retry without references.
            if use_refs and "shapes cannot be multiplied" in str(e):
                _disable_ip_adapter()
                kwargs.pop("ip_adapter_image", None)
                result = pipe(**kwargs)
            else:
                raise

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
    motion_art = project.read_artifact("motion_shots") or {}
    shots: List[Dict[str, Any]] = shots_art.get("shots") or []
    if not shots:
        raise ValueError("No rendered shots to edit.")

    dp_by_scene: Dict[str, Dict[str, Any]] = cinema.get("scenes") or {}

    # If motion_shots ran, prefer those mp4s over Ken Burns on the stills.
    # Build a lookup: (scene_id, shot_id) -> abs mp4 path.
    motion_by_key: Dict[str, str] = {}
    for m in motion_art.get("motion_shots", []) or []:
        p = m.get("path")
        if not p:
            continue
        mp4_abs = os.path.join(project.dir, p)
        if os.path.exists(mp4_abs):
            motion_by_key[f"{m['scene_id']}::{m['shot_id']}"] = mp4_abs

    clips_dir = os.path.join(project.artifacts_dir, "editor_clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths: List[str] = []
    motion_used = 0
    kb_used = 0
    for i, s in enumerate(shots):
        img_abs = os.path.join(project.dir, s["path"])
        if not os.path.exists(img_abs):
            continue
        dur = max(1, int(s.get("duration_sec", 4) or 4))
        scene_id = s.get("scene_id", "")
        shot_id = s.get("shot_id", "")
        camera_move = ((dp_by_scene.get(scene_id) or {})
                       .get(shot_id) or {}).get("camera_move", "static")
        clip_out = os.path.join(clips_dir, f"{i:03d}_{scene_id}_{shot_id}.mp4")

        motion_src = motion_by_key.get(f"{scene_id}::{shot_id}")
        if motion_src is not None:
            # Re-encode the Wan clip to our canonical 1280x720 30fps H.264
            # container so the concat below can stream-copy without seams.
            try:
                _standardize_motion_clip(motion_src, clip_out, dur)
                clip_paths.append(clip_out)
                motion_used += 1
                continue
            except Exception:
                logger.exception("Motion clip normalize failed for %s/%s; using Ken Burns",
                                 scene_id, shot_id)
        # Ken Burns fallback path.
        try:
            _render_kenburns_clip(img_abs, clip_out, dur, camera_move)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not on PATH — cannot assemble the final cut.") from e
        except Exception:
            logger.exception("Ken Burns render failed for %s/%s; using static clip",
                             scene_id, shot_id)
            try:
                _render_static_clip(img_abs, clip_out, dur)
            except Exception:
                logger.exception("Static fallback failed for %s/%s; skipping", scene_id, shot_id)
                continue
        clip_paths.append(clip_out)
        kb_used += 1

    if not clip_paths:
        raise ValueError("No renderable shot images found.")

    concat_txt = os.path.join(project.artifacts_dir, "editor.concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")

    # Editor writes to silent_cut.mp4 rather than final.mp4 so it doesn't
    # collide with the titles stage output (which IS final.mp4). Every stage
    # from here on reads editor.output_path explicitly.
    out_path = os.path.join(project.dir, "silent_cut.mp4")
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
        "motion_clips_used": motion_used,
        "kenburns_clips_used": kb_used,
    }


def _standardize_motion_clip(src: str, dst: str, duration_sec: int) -> None:
    """Re-encode a Wan I2V mp4 to the editor's canonical 1280x720 30fps H.264
    (yuv420p) container so all clips concat with stream copy. Duration is
    padded or trimmed to `duration_sec`."""
    fps = 30
    vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
          "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30")
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", src,
        "-t", str(duration_sec),
        "-vf", vf,
        "-fps_mode", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


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
    """Synthesize dialogue with Piper. The Storyboard's `dialogue` field is
    unreliable — the LLM drops lines, misattributes speakers, and reorders
    them. So we ignore that field and parse the screenplay directly, keeping
    the true speaker on every line and distributing lines across shots by
    reading order."""
    storyboard   = project.read_artifact("storyboard") or {}
    breakdown    = project.read_artifact("breakdown") or {}
    cast_art     = project.read_artifact("voice_cast") or {}
    editor       = project.read_artifact("story_editor") or {}
    screenwriter = project.read_artifact("screenwriter") or {}

    scenes: List[Dict[str, Any]] = breakdown.get("scenes") or []
    shots_by_scene: Dict[str, List[Dict[str, Any]]] = storyboard.get("scenes") or {}
    cast: Dict[str, Dict[str, str]] = cast_art.get("cast") or {}

    full_screenplay = (editor.get("revised_fountain")
                       or screenwriter.get("fountain")
                       or "")
    per_scene_script = _split_screenplay_by_scene(full_screenplay, scenes)

    os.makedirs(os.path.join(project.artifacts_dir, "voice"), exist_ok=True)

    voice_backend = (project.meta.config.voice_backend or "piper").lower()

    # XTTS-v2 path — one shared model, per-character speaker embeddings via
    # reference clips or the built-in speaker library. Falls through to
    # Piper on any load failure.
    xtts_wrap = _load_xtts_if_requested(voice_backend, project)

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

    def voice_for(speaker: str) -> str:
        # Breakdown often captures a fuller character name than the screenplay
        # cue ("Jacob Lucas" vs "JACOB"). Match exact first, then substring,
        # then first-word. Case-insensitive throughout.
        sp = speaker.upper().strip()
        cast_items = [(k, v) for k, v in cast.items() if k]
        for k, v in cast_items:
            if k.upper().strip() == sp:
                return v.get("voice") or _PIPER_POOL[0]["name"]
        for k, v in cast_items:
            ku = k.upper().strip()
            if sp and (sp in ku or ku in sp):
                return v.get("voice") or _PIPER_POOL[0]["name"]
        sp_first = sp.split()[0] if sp else ""
        for k, v in cast_items:
            ku_first = k.upper().strip().split()[0] if k.strip() else ""
            if sp_first and sp_first == ku_first:
                return v.get("voice") or _PIPER_POOL[0]["name"]
        return _PIPER_POOL[0]["name"]

    lines: List[Dict[str, Any]] = []
    for scene in scenes:
        sid = scene["id"]
        shots = shots_by_scene.get(sid, []) or []
        if not shots:
            continue
        script = per_scene_script.get(sid) or ""
        dialogue = _parse_screenplay_dialogue(script)
        if not dialogue:
            continue

        # Distribute lines across shots. First pass: try to place each line
        # on a shot whose subject/action text mentions the speaker (rough
        # proximity match, in reading order). Second pass: for any line not
        # placed, fall back to even index distribution. This keeps the right
        # voice under the right face when the storyboard bothers to describe
        # who's on screen, without breaking on generic shots.
        assignments: List[Optional[Dict[str, Any]]] = [None] * len(shots)

        def shot_mentions(idx: int, speaker: str) -> bool:
            hay = f"{shots[idx].get('subject','')} {shots[idx].get('action','')}".upper()
            for tok in speaker.upper().split():
                if len(tok) >= 3 and tok in hay:
                    return True
            return False

        next_search_idx = 0
        unplaced: List[int] = []
        for i, line in enumerate(dialogue):
            speaker = line["speaker"]
            placed = False
            # Search forward from where the last line landed. Prefer earliest
            # unassigned shot that mentions the speaker; keeps reading order.
            for k in range(next_search_idx, len(shots)):
                if assignments[k] is None and shot_mentions(k, speaker):
                    assignments[k] = line
                    next_search_idx = k + 1
                    placed = True
                    break
            if not placed:
                unplaced.append(i)

        # Second pass: fill unplaced lines by proportional index, skipping
        # already-taken slots.
        for i in unplaced:
            slot = i * len(shots) // max(1, len(dialogue))
            while slot < len(shots) and assignments[slot] is not None:
                slot += 1
            if slot >= len(shots):
                slot = 0
                while slot < len(shots) and assignments[slot] is not None:
                    slot += 1
            if slot < len(shots):
                assignments[slot] = dialogue[i]

        for slot_idx, line in enumerate(assignments):
            if line is None:
                continue
            shot = shots[slot_idx]
            speaker = line["speaker"]
            voice_name = voice_for(speaker)

            wav_rel = f"voice/{sid}_{shot['id']}.wav"
            wav_abs = os.path.join(project.artifacts_dir, "voice",
                                   f"{sid}_{shot['id']}.wav")
            ok = False
            err: Optional[str] = None
            # XTTS first if requested — cloned voice with more prosody.
            if xtts_wrap is not None:
                try:
                    xtts_wrap.synthesize(line["text"], wav_abs, speaker=speaker)
                    _pad_wav_with_silence(wav_abs, head_sec=0.25, tail_sec=0.45)
                    ok = True
                except Exception as e:
                    logger.warning("XTTS synth failed for %s/%s (%s); falling back to Piper",
                                   sid, shot["id"], e)
                    err = f"xtts: {e}"

            tts = get_tts(voice_name) if not ok else None
            if not ok and tts is not None:
                try:
                    tts.synthesize(line["text"], wav_abs)
                    # Piper voices land tight against the file boundaries and
                    # feel rushed inside a short shot. Bake head+tail silence
                    # in so the line has breathing room when mixed.
                    _pad_wav_with_silence(wav_abs, head_sec=0.25, tail_sec=0.45)
                    ok = True
                    err = None
                except Exception as e:
                    logger.warning("Piper synth failed for %s/%s (%s); writing silence",
                                   sid, shot["id"], e)
                    err = f"{err or ''}; piper: {e}"
            if not ok:
                _write_silent_wav(wav_abs, 1.0)
            lines.append({
                "scene_id": sid,
                "shot_id":  shot["id"],
                "speaker":  speaker,
                "voice":    voice_name,
                "text":     line["text"],
                "wav":      wav_rel,
                "duration_sec": _wav_duration_seconds(wav_abs),
                "synthesized": ok,
                "error": err,
            })
    return {"lines": lines}


_CUE_STRICT = None  # populated below
_CUE_LOOSE  = None
_PAREN_LINE = None
_PAREN_THEN_TEXT = None


def _looks_like_cue(line: str) -> Optional[tuple]:
    """Match a Fountain character cue on `line`. Returns (speaker, paren) or None.

    Accepts both classical ALL-CAPS cues (SASHA, OPPOSING STREET CHEF) and
    the Title-Case cues small models keep emitting (Karl, Old Timothy).
    Requires that every WORD of the cue starts with an uppercase letter,
    which weeds out random prose lines. Cue may end with an optional
    parenthetical like `(annoyed)`.
    """
    import re as _re
    global _CUE_STRICT, _CUE_LOOSE
    if _CUE_STRICT is None:
        _CUE_STRICT = _re.compile(r"^([A-Z][A-Z0-9 .'\-]{0,40})(?:\s*\(([^)]*)\))?\s*$")
        _CUE_LOOSE  = _re.compile(r"^([A-Za-z][A-Za-z0-9 .'\-]{0,40})(?:\s*\(([^)]*)\))?\s*$")
    s = line.strip()
    if not s or len(s) > 60:
        return None
    m = _CUE_STRICT.match(s)
    if m:
        return (m.group(1).strip(), (m.group(2) or "").strip())
    m = _CUE_LOOSE.match(s)
    if not m:
        return None
    name = m.group(1).strip()
    # Every word must start uppercase (Title Case). Blocks prose sentences.
    words = [w for w in name.split() if w]
    if not words or not all(w[0].isupper() and len(w) <= 20 for w in words):
        return None
    # Cheap sanity: cue should be short (<= 5 words).
    if len(words) > 5:
        return None
    return (name, (m.group(2) or "").strip())


def _parse_screenplay_dialogue(text: str) -> List[Dict[str, str]]:
    """Return `[{speaker, text}, ...]` extracted from Fountain-style screenplay
    text.

    Real character cues must be preceded by a blank line — that's the
    Fountain convention that separates them from character INTRODUCTIONS in
    action prose (e.g. `SASHA (a young woman in a chef's apron) stands at
    one end...` reads as a cue but isn't).

    Handles three dialogue layouts we've actually seen from various LLMs:
      1. Classical: cue on one line, dialogue on the next.
      2. Cue + optional paren line + dialogue.
      3. Cue on one line, next line starts with `(paren) actual dialogue text` —
         extract the text AFTER the paren.
    """
    import re as _re
    global _PAREN_LINE, _PAREN_THEN_TEXT
    if _PAREN_LINE is None:
        _PAREN_LINE = _re.compile(r"^\s*\([^)]*\)\s*$")
        _PAREN_THEN_TEXT = _re.compile(r"^\s*\([^)]*\)\s+(.+?)\s*$")

    # Layout 4: everything on ONE line. `LUCY (to herself) Just one more.`
    # or `JESSICA Doc? I'm here.`. Preceded by a blank line, otherwise we'd
    # false-match on character intros embedded in action prose (e.g.
    # "A dim room. DOCTOR LUCY MITCHELL, mid-30s, adjusts her glasses.").
    inline_one = _re.compile(
        r"^([A-Za-z][A-Za-z0-9 .'\-]{0,40}?)"     # speaker (non-greedy)
        r"(?:\s*\(([^)]*)\))?"                     # optional parenthetical
        r"\s+"
        r"([A-Z\"'…].*)$"                     # dialogue starts with cap / quote / ellipsis
    )

    def _inline_speaker(speaker_raw: str, paren: str, dialogue: str) -> Optional[str]:
        # Speaker must pass the cue check AND be short (1-3 words) to keep
        # multi-word character intros out.
        cue = _looks_like_cue(speaker_raw)
        if cue is None:
            return None
        words = [w for w in speaker_raw.split() if w]
        if not words or len(words) > 3:
            return None
        # No paren annotation => require a speech marker in the dialogue.
        # Real spoken lines almost always carry ? / ! / ellipsis; character
        # intros ("mid-30s, in a lab coat") do not.
        if not paren and not any(m in dialogue for m in ("?", "!", "…", "...")):
            return None
        return cue[0]

    out: List[Dict[str, str]] = []
    lines = text.splitlines()
    i = 0
    prev_was_dialogue = False  # rapid back-and-forth often has no blank between exchanges
    while i < len(lines):
        raw = lines[i]
        if not raw.strip():
            prev_was_dialogue = False
            i += 1
            continue
        preceded_by_blank = (i == 0) or (not lines[i - 1].strip())
        can_be_cue = preceded_by_blank or prev_was_dialogue

        # Try inline single-line layout first — it's the most specific match.
        if can_be_cue:
            m = inline_one.match(raw.rstrip())
            if m:
                speaker = _inline_speaker(m.group(1).strip(), (m.group(2) or "").strip(), m.group(3).strip())
                if speaker:
                    out.append({"speaker": speaker, "text": m.group(3).strip()})
                    prev_was_dialogue = True
                    i += 1
                    continue

        cue_match = _looks_like_cue(raw) if can_be_cue else None
        if cue_match and i + 1 < len(lines):
            speaker, _paren = cue_match
            j = i + 1
            # Layout 3: `(paren) actual dialogue` — pull the text out and skip past.
            m_inline = _PAREN_THEN_TEXT.match(lines[j]) if j < len(lines) else None
            if m_inline:
                inline_text = m_inline.group(1).strip()
                # Keep pulling continuation lines that are neither blank nor cues.
                j += 1
                extra: List[str] = [inline_text]
                while j < len(lines) and lines[j].strip() and not _looks_like_cue(lines[j]):
                    extra.append(lines[j].strip())
                    j += 1
                out.append({"speaker": speaker, "text": " ".join(extra)})
                prev_was_dialogue = True
                i = j
                continue
            # Layout 2: optional parenthetical on its own line ("(quietly)")
            # between the cue and the dialogue.
            if j < len(lines) and _PAREN_LINE.match(lines[j]):
                j += 1
            # Layout 1: dialogue is the consecutive non-blank, non-cue lines
            # that follow.
            spoken: List[str] = []
            while j < len(lines) and lines[j].strip() and not _looks_like_cue(lines[j]):
                spoken.append(lines[j].strip())
                j += 1
            if spoken:
                out.append({"speaker": speaker, "text": " ".join(spoken)})
                prev_was_dialogue = True
            else:
                prev_was_dialogue = False
            i = j
            continue
        prev_was_dialogue = False
        i += 1
    return out


class _XTTSWrap:
    """Thin adapter over Coqui XTTS-v2. Held on-demand from
    `_load_xtts_if_requested` so the pipeline never imports TTS when the
    user asked for Piper.

    Speaker selection: prefer a per-character reference clip at
    `<project.dir>/artifacts/voice_refs/<safe_name>.wav`; fall back to one
    of the built-in library speakers picked by hashing the character name."""

    _BUILTIN_SPEAKERS = [
        "Damien Black", "Gilberto Mathias", "Ige Behringer",
        "Sofia Hellen", "Tanja Adelina", "Uta Obando",
    ]

    def __init__(self, tts, project):
        self._tts = tts
        self._project = project
        self._voice_refs_dir = os.path.join(project.artifacts_dir, "voice_refs")

    def _speaker_for(self, speaker: str) -> Dict[str, Any]:
        """Return kwargs for TTS.tts_to_file that pick the right voice."""
        ref = os.path.join(self._voice_refs_dir, f"{_safe_filename(speaker)}.wav")
        if os.path.exists(ref):
            return {"speaker_wav": ref}
        # Deterministic library speaker per character so a name always maps
        # to the same voice within a film.
        idx = abs(hash(speaker.upper())) % len(self._BUILTIN_SPEAKERS)
        return {"speaker": self._BUILTIN_SPEAKERS[idx]}

    def synthesize(self, text: str, out_path: str, *, speaker: str = "") -> None:
        kwargs = self._speaker_for(speaker)
        # language="en" locks XTTS to English output. Multi-language films
        # would need a per-line language tag; not wired yet.
        self._tts.tts_to_file(text=text, file_path=out_path, language="en", **kwargs)


def _load_xtts_if_requested(voice_backend: str, project) -> Optional[_XTTSWrap]:
    """Load Coqui XTTS-v2 when the project asks for `voice_backend="xtts"`.
    Returns None on any failure — including the common one of `coqui-tts`
    not being installed — so the caller falls back to Piper cleanly."""
    if voice_backend != "xtts":
        return None
    try:
        # transformers 5.x renamed `isin_mps_friendly` out of pytorch_utils.
        # coqui-tts 0.27.5 still imports it. Provide the shim before the
        # first `from TTS...` import so the module loads.
        import torch as _torch
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "isin_mps_friendly"):
            def _isin(elements, test_elements):
                return _torch.isin(elements, test_elements)
            _pu.isin_mps_friendly = _isin
        # XTTS-v2 model gates its download behind a CPML license prompt that
        # blocks on stdin. Setting COQUI_TOS_AGREED=1 tells the loader the
        # user has read and accepted the Coqui Public Model License terms
        # (see https://coqui.ai/cpml). Set explicitly here so background
        # runs don't hang waiting for a Y/N.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        # XTTS-v2 downloads ~2GB on first use; subsequent runs hit the local cache.
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                  progress_bar=False).to(device)
        logger.info("XTTS-v2 loaded on %s", device)
        return _XTTSWrap(tts, project)
    except ImportError as e:
        logger.warning("XTTS unavailable (%s); disabled. Install with: pip install \"coqui-tts[codec]\"", e)
        return None
    except Exception as e:
        logger.warning("XTTS load failed (%s); falling back to Piper", e)
        return None


def _pad_wav_with_silence(path: str, *, head_sec: float, tail_sec: float) -> None:
    """Prepend/append zero-amplitude samples to a WAV in place, matching its
    channels/sampwidth/framerate exactly."""
    import wave
    with wave.open(path, "rb") as r:
        ch, sw, sr = r.getnchannels(), r.getsampwidth(), r.getframerate()
        frames = r.readframes(r.getnframes())
    head = b"\x00" * (int(head_sec * sr) * ch * sw)
    tail = b"\x00" * (int(tail_sec * sr) * ch * sw)
    with wave.open(path, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr)
        w.writeframes(head + frames + tail)


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
    """Render a single MusicGen track. Tries the `medium` variant first for
    a noticeably richer arrangement, falls back to `small` on any load error
    (OOM, missing weights). First-time download is ~1.5-4 GB depending on
    variant; subsequent runs hit the local cache."""
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import torch as _torch

    # Try local caches ONLY — don't hit HF Hub live, unauthenticated pulls
    # are throttled to a crawl and can hang a run for hours waiting on a
    # 3+ GB model. Anyone who wants medium/large should pull it out of
    # band with `huggingface-cli download`.
    from pathlib import Path
    cache_root = Path(os.environ.get("HF_HOME") or
                      Path.home() / ".cache" / "huggingface" / "hub")
    def _cached(model_id: str) -> bool:
        return (cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots").exists()
    candidates = [m for m in ("facebook/musicgen-medium", "facebook/musicgen-small")
                  if _cached(m)]
    if not candidates:
        candidates = ["facebook/musicgen-small"]  # last-ditch download attempt
    for model_id in candidates:
        try:
            processor = AutoProcessor.from_pretrained(
                model_id, local_files_only=_cached(model_id), token=_hf_token())
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, local_files_only=_cached(model_id), token=_hf_token())
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info("MusicGen loaded (%s) on %s", model_id, device)
            break
        except Exception as e:
            logger.warning("MusicGen %s failed to load (%s); trying next tier", model_id, e)
            model = None  # type: ignore
    if model is None:
        raise RuntimeError("No MusicGen variant could be loaded")

    # MusicGen small is capped at ~30 s per generate call. Chain multiple
    # generations back-to-back to cover the full film length. Overlap-request
    # each subsequent chunk by `xfade_sec` and crossfade the seam so the
    # tempo/key jump at 30 s stops being audible.
    sr = model.config.audio_encoder.sampling_rate
    tokens_per_sec = 50           # MusicGen encodec frame rate
    chunk_len_sec = 30
    xfade_sec = 1.0
    chunks: list = []
    remaining = duration_sec
    while remaining > 0:
        # Ask for extra so the crossfade region is fresh audio, not the tail
        # of the last chunk faded out to nothing.
        want = min(chunk_len_sec, remaining) + (xfade_sec if chunks else 0)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with _torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=int(want * tokens_per_sec))
        arr = audio[0, 0].cpu().numpy().astype(np.float32)
        chunks.append(arr)
        remaining -= chunk_len_sec  # advance by the "playable" length, not `want`

    combined = _crossfade_concat(chunks, sr=sr, xfade_sec=xfade_sec)
    # Normalize to -3 dBFS to leave headroom for the mixer.
    peak = float(np.max(np.abs(combined))) or 1.0
    combined = (combined / peak) * 0.7
    # Trim to the requested duration so we never overshoot the final cut.
    max_samples = int(duration_sec * sr)
    if len(combined) > max_samples:
        combined = combined[:max_samples]
    sf.write(out_path, combined, sr)


def _crossfade_concat(chunks: List["np.ndarray"], *, sr: int, xfade_sec: float) -> "np.ndarray":
    """Concatenate audio chunks with an equal-power crossfade over the seam.
    Falls back to a plain concatenate for single-chunk input."""
    import numpy as np
    if len(chunks) <= 1:
        return chunks[0]
    xfade = int(xfade_sec * sr)
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        n = min(xfade, len(out), len(nxt))
        if n <= 0:
            out = np.concatenate([out, nxt])
            continue
        # Equal-power (cosine) crossfade keeps perceived loudness flat.
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        fade_out = np.cos(t)
        fade_in  = np.sin(t)
        seam = out[-n:] * fade_out + nxt[:n] * fade_in
        out = np.concatenate([out[:-n], seam, nxt[n:]])
    return out


# ---------------------------------------------------------------------------
# 12. Mixer — ffmpeg mux: silent cut + dialogue at shot offsets + score with
#     sidechain ducking under speech.
# ---------------------------------------------------------------------------

def run_mixer(project: Project) -> Dict[str, Any]:
    editor_art   = project.read_artifact("editor") or {}
    composer_art = project.read_artifact("composer") or {}
    ambient_art  = project.read_artifact("ambient") or {}
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
        dialogue.append({
            "wav": wav_abs,
            "delay_ms": offsets_ms.get(key, 0),
            "dur_ms": int(_wav_duration_seconds(wav_abs) * 1000),
        })

    # Prevent overlap. Sort by scheduled offset, then push any line back so
    # it starts no earlier than the previous line's end + a short gap. This
    # is a fallback in case the shot-duration fitting in run_shots didn't
    # give a line enough room (very long dialogue, wrong shot assignment).
    _MIN_GAP_MS = 200
    dialogue.sort(key=lambda d: d["delay_ms"])
    scheduled_end = 0
    for d in dialogue:
        if d["delay_ms"] < scheduled_end + _MIN_GAP_MS:
            d["delay_ms"] = scheduled_end + _MIN_GAP_MS
        scheduled_end = d["delay_ms"] + d["dur_ms"]

    score_rel = composer_art.get("path")
    score_abs = os.path.join(project.dir, score_rel) if score_rel else None
    have_score = bool(score_abs and os.path.exists(score_abs))
    have_speech = bool(dialogue)

    ambient_rel = ambient_art.get("path")
    ambient_abs = os.path.join(project.dir, ambient_rel) if ambient_rel else None
    have_ambient = bool(ambient_abs and os.path.exists(ambient_abs))

    out_abs = os.path.join(project.dir, "final_mixed.mp4")
    total_dur = float(editor_art.get("duration_sec", 0) or 0)
    _run_mixer_ffmpeg(
        silent_video=silent_abs,
        dialogue=dialogue,
        score=score_abs if have_score else None,
        ambient=ambient_abs if have_ambient else None,
        out_path=out_abs,
        duration_sec=total_dur if total_dur > 0 else None,
    )
    return {
        "output_path":  os.path.relpath(out_abs, project.dir).replace(os.sep, "/"),
        "has_dialogue": have_speech,
        "has_score":    have_score,
        "has_ambient":  have_ambient,
        "dialogue_lines": len(dialogue),
    }


def _run_mixer_ffmpeg(*, silent_video: str, dialogue: List[Dict[str, Any]],
                      score: Optional[str], ambient: Optional[str] = None,
                      out_path: str, duration_sec: Optional[float] = None) -> None:
    """Build and run the ffmpeg mux. Four sources: silent video, optional
    score, optional ambient bed, and 0..N dialogue wavs. Speech ducks the
    score via sidechain compression. Ambient sits at a low fixed level and
    is not ducked (it's atmospheric, not focal). Output length is pinned to
    the video's duration."""
    inputs: List[str] = ["-i", silent_video]
    input_idx = 1
    score_idx: Optional[int] = None
    if score:
        inputs += ["-i", score]
        score_idx = input_idx
        input_idx += 1
    ambient_idx: Optional[int] = None
    if ambient:
        inputs += ["-i", ambient]
        ambient_idx = input_idx
        input_idx += 1
    dlg_start = input_idx
    for d in dialogue:
        inputs += ["-i", d["wav"]]

    filter_parts: List[str] = []

    dlg_labels: List[str] = []
    for i, d in enumerate(dialogue):
        idx = dlg_start + i
        lbl = f"d{i}"
        delay = int(d["delay_ms"])
        filter_parts.append(
            f"[{idx}:a]adelay={delay}|{delay},aformat=sample_rates=44100:channel_layouts=stereo[{lbl}]"
        )
        dlg_labels.append(f"[{lbl}]")

    # Build the speech bus. When there's also a score to duck under, we need
    # it in two branches (sidechain input AND the final amix), so asplit=2
    # duplicates it — ffmpeg filter labels are single-use. And pad the
    # sidechain branch to the requested duration (or a generous ceiling) so
    # sidechaincompress doesn't stop emitting audio when the last dialogue
    # line ends — sidechaincompress output length matches the shorter of its
    # two inputs, so a short speech bus would silently truncate the whole mix.
    pad_target = duration_sec if duration_sec and duration_sec > 0 else 3600.0
    if dlg_labels:
        mix_expr = (f"{''.join(dlg_labels)}amix=inputs={len(dlg_labels)}:normalize=0"
                    if len(dlg_labels) > 1 else dlg_labels[0])
        # `mix_expr` is either a filter chain (needs `,` before next filter)
        # or a bare label like "[d0]" (next filter follows directly).
        sep = "," if len(dlg_labels) > 1 else ""
        if score is not None:
            filter_parts.append(
                f"{mix_expr}{sep}asplit=2[speech][speech_pre_pad];"
                f"[speech_pre_pad]apad=whole_dur={pad_target}[speech_side]"
            )
        else:
            filter_parts.append(f"{mix_expr}{sep}anull[speech]"
                                if not sep
                                else f"{mix_expr}{sep}anull[speech]")

    # Ambient sits low, no ducking (it's atmosphere), stereo-normalised.
    if ambient is not None:
        filter_parts.append(
            f"[{ambient_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.35[ambient_bus]"
        )

    if score is not None:
        filter_parts.append(
            f"[{score_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.45[music_raw]"
        )
        if dlg_labels:
            filter_parts.append(
                "[music_raw][speech_side]sidechaincompress="
                "threshold=0.04:ratio=8:attack=5:release=250[music_ducked]"
            )
            if ambient is not None:
                filter_parts.append(
                    "[speech][music_ducked][ambient_bus]amix=inputs=3:normalize=0:duration=longest[audio]"
                )
            else:
                filter_parts.append(
                    "[speech][music_ducked]amix=inputs=2:normalize=0:duration=longest[audio]"
                )
        else:
            # No speech to duck. If we have ambient, mix it under the music.
            if ambient is not None:
                filter_parts.append(
                    "[music_raw][ambient_bus]amix=inputs=2:normalize=0:duration=longest[audio]"
                )
            else:
                filter_parts.append("[music_raw]anull[audio]")
    elif dlg_labels:
        if ambient is not None:
            filter_parts.append(
                "[speech][ambient_bus]amix=inputs=2:normalize=0:duration=longest[audio]"
            )
        else:
            filter_parts.append("[speech]anull[audio]")
    elif ambient is not None:
        filter_parts.append("[ambient_bus]anull[audio]")

    # Pin the output length to the editor's video duration when we know it.
    # Without this, a short dialogue chain (e.g. one line early in the film)
    # makes amix produce a stream that ends before the video, and ffmpeg's
    # `-shortest` truncates the whole cut. Explicit `-t` keeps the video full
    # length and pads the audio track to match.
    duration_args = ["-t", f"{duration_sec:.3f}"] if duration_sec and duration_sec > 0 else []

    cmd = ["ffmpeg", "-y"] + inputs
    if filter_parts:
        cmd += ["-filter_complex", ";".join(filter_parts),
                "-map", "0:v", "-map", "[audio]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k"] + duration_args + [out_path]
    else:
        # Neither dialogue nor score — remux silent cut with an aac silent track
        # so downstream consumers still see a well-formed audio stream.
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k"] + duration_args + [out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mixer ffmpeg failed:\n{(e.stderr or '')[-800:]}") from e


# ---------------------------------------------------------------------------
# shared LLM plumbing
# ---------------------------------------------------------------------------

def _chat(project: Project, stage_key: str, system: str, user: str, *,
          want_json: bool, temperature: float, max_tokens: int = 4096) -> str:
    """Single blocking chat call. Retries ONCE on LLMError with a bigger
    max_tokens budget and slightly cooler temperature so the same truncation
    or parse issue doesn't recur — Ollama occasionally cuts a JSON response
    mid-array when the model runs long, and a modest retry usually clears it.
    """
    cfg = project.meta.config
    per = cfg.per_stage.get(stage_key, {}) if cfg.per_stage else {}
    backend = per.get("llm_backend") or cfg.llm_backend
    model = per.get("llm_model") or cfg.llm_model
    host = per.get("llm_host") or cfg.llm_host

    try:
        return llm.chat(
            system=system, user=user,
            backend=backend, model=model, host=host,
            temperature=temperature, max_tokens=max_tokens,
            want_json=want_json,
        )
    except llm.LLMError as e:
        logger.warning("LLM call for %s failed (%s); retrying with larger budget", stage_key, e)
        return llm.chat(
            system=system, user=user,
            backend=backend, model=model, host=host,
            temperature=max(0.1, temperature - 0.2),
            max_tokens=int(max_tokens * 1.5),
            want_json=want_json,
        )


# ---------------------------------------------------------------------------
# NEW STAGE — Character Portraits
#
# Renders one canonical portrait per named character before Shots runs. The
# portraits are the reference images IP-Adapter conditions on during shot
# render, so the same face lands in every shot that character appears in.
# ---------------------------------------------------------------------------

def run_character_portraits(project: Project) -> Dict[str, Any]:
    """Produce one 1024x1024 canonical portrait per named character. Uses
    the same SDXL wrapper as run_shots but with a portrait-tuned prompt.
    Falls back to a placeholder card if SDXL is unavailable so downstream
    stages can still key off `portraits[name] = path`."""
    voice_cast = project.read_artifact("voice_cast") or {}
    editor     = project.read_artifact("story_editor") or {}
    screenwriter = project.read_artifact("screenwriter") or {}
    cast = voice_cast.get("cast") or {}
    script = editor.get("revised_fountain") or screenwriter.get("fountain") or ""
    bios = _extract_character_bios(script)

    portraits_dir = os.path.join(project.artifacts_dir, "portraits")
    os.makedirs(portraits_dir, exist_ok=True)

    # Build the character list from the cast (voice_cast is the source of truth
    # for "named characters we care about"). Merge bios by name.
    characters: List[Dict[str, str]] = []
    seen = set()
    for name in cast.keys():
        clean = name.strip()
        if not clean or clean.upper() in seen:
            continue
        seen.add(clean.upper())
        # Bios are case-sensitive by capture; match case-insensitively.
        bio = ""
        for bio_name, bio_desc in bios.items():
            if bio_name.upper() == clean.upper() or (
                len(clean.split()) and clean.split()[0].upper() ==
                (bio_name.split()[0].upper() if bio_name.split() else "")):
                bio = bio_desc
                break
        characters.append({"name": clean, "bio": bio})

    if not characters:
        return {"portraits": {}, "note": "No named characters in voice_cast; nothing to portrait."}

    render = _load_sdxl_renderer(variant=project.meta.config.sdxl_variant)
    steps = _quality_to_steps(project.meta.config.quality)

    style = project.meta.config.style
    style_prefix = ("cinematic photorealistic headshot portrait, film grain"
                    if style == "photoreal"
                    else "stylized painterly headshot portrait, cinematic")

    portraits: Dict[str, Dict[str, Any]] = {}
    for char in characters:
        name = char["name"]
        bio = char["bio"] or "adult, contemporary casual clothing"
        prompt = (f"{style_prefix}, {name}: {bio}, "
                  f"neutral background, soft key light, 85mm lens, "
                  f"looking slightly off camera, upper body")
        safe_name = _safe_filename(name)
        png_path = os.path.join(portraits_dir, f"{safe_name}.png")
        rendered = False
        err: Optional[str] = None
        try:
            if render is not None:
                render(prompt, png_path, steps=steps)
                rendered = True
            else:
                _placeholder_png(png_path, name)
        except Exception as e:
            logger.exception("Portrait render failed for %s", name)
            err = str(e)
            _placeholder_png(png_path, name)
        portraits[name] = {
            "path": os.path.relpath(png_path, project.dir).replace(os.sep, "/"),
            "prompt": prompt,
            "rendered": rendered,
            "error": err,
        }
    return {"portraits": portraits}


def _safe_filename(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip()) or "unnamed"


# ---------------------------------------------------------------------------
# NEW STAGE — Colorist
#
# Applies a mood-driven color grade to the assembled cut. Uses ffmpeg curves
# and eq filters (no LUT files needed) with per-scene knobs derived from
# the breakdown's `mood` and `time_of_day` fields. Falls back to identity
# grade if ffmpeg fails so downstream stages still have a video.
# ---------------------------------------------------------------------------

def run_colorist(project: Project) -> Dict[str, Any]:
    """Grade the mixed cut in one ffmpeg pass. Currently applies a single
    global grade derived from the film's dominant tone. A per-scene grade
    with segmented filter graphs is a later iteration — this baseline lifts
    the flat SDXL Turbo look immediately."""
    mixer_art = project.read_artifact("mixer") or {}
    editor_art = project.read_artifact("editor") or {}
    breakdown = project.read_artifact("breakdown") or {}
    producer  = project.read_artifact("producer") or {}

    # Prefer the mixed cut (has audio) so the graded output keeps sound.
    src_rel = mixer_art.get("output_path") or editor_art.get("output_path")
    if not src_rel:
        raise ValueError("No mixer/editor output to grade.")
    src_abs = os.path.join(project.dir, src_rel)
    if not os.path.exists(src_abs):
        raise ValueError(f"Source cut missing at {src_abs}")

    # Pick a grade from the film's tone + scene moods.
    tone = ""
    idx = int(producer.get("chosen_index", 0))
    loglines = producer.get("loglines") or []
    if 0 <= idx < len(loglines):
        tone = str(loglines[idx].get("tone", "")).lower()

    scenes = breakdown.get("scenes") or []
    night_count = sum(1 for s in scenes if str(s.get("time_of_day", "")).lower() in ("night", "dusk"))
    mostly_night = night_count > len(scenes) / 2 if scenes else False

    grade = _pick_grade(tone, mostly_night)

    out_abs = os.path.join(project.dir, "final_graded.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", src_abs,
        "-vf", grade["vf"],
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        out_abs,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"colorist ffmpeg failed:\n{(e.stderr or '')[-800:]}") from e

    return {
        "output_path": os.path.relpath(out_abs, project.dir).replace(os.sep, "/"),
        "grade_name": grade["name"],
        "grade_vf": grade["vf"],
        "source": src_rel,
    }


def _pick_grade(tone: str, mostly_night: bool) -> Dict[str, str]:
    """Return a named ffmpeg -vf grade suited to the film's tone. All grades
    end with a slight contrast+saturation lift so SDXL Turbo output looks less
    flat, then apply a color cast via colorbalance/curves."""
    base_contrast = "eq=contrast=1.08:saturation=1.05"
    if tone in ("thriller", "horror", "mystery") or mostly_night:
        # Cold, teal shadows, crushed blacks
        vf = f"{base_contrast}:brightness=-0.02,curves=b='0/0 0.5/0.55 1/1':r='0/0 0.5/0.42 1/0.95'"
        return {"name": "cold_teal", "vf": vf}
    if tone in ("comedy", "romance"):
        # Warm, punchy, bright
        vf = f"{base_contrast}:brightness=0.02,curves=r='0/0 0.5/0.55 1/1':b='0/0 0.5/0.45 1/0.95'"
        return {"name": "warm_punchy", "vf": vf}
    if tone in ("action", "sci-fi"):
        # Teal-and-orange
        vf = f"{base_contrast},colorbalance=rm=0.05:gm=-0.02:bm=-0.08:rs=-0.05:gs=0.02:bs=0.08"
        return {"name": "teal_orange", "vf": vf}
    if tone in ("drama", "documentary"):
        # Muted, filmic
        vf = "eq=contrast=1.05:saturation=0.9:gamma=1.02"
        return {"name": "muted_filmic", "vf": vf}
    # Default: mild lift
    return {"name": "neutral_lift", "vf": base_contrast}


# ---------------------------------------------------------------------------
# NEW STAGE — Titles & Credits
#
# Generates a front title card (film title over the first shot's frame,
# fading in and out) and an end credits scroll (crew and cast on black).
# Prepends and appends to the graded (or mixed) cut.
# ---------------------------------------------------------------------------

def run_titles(project: Project) -> Dict[str, Any]:
    """Prepend a title card and append an end credits scroll. The result
    lands at `final.mp4` — the film's canonical export."""
    colorist_art = project.read_artifact("colorist") or {}
    mixer_art    = project.read_artifact("mixer") or {}
    editor_art   = project.read_artifact("editor") or {}
    producer     = project.read_artifact("producer") or {}
    voice_cast   = project.read_artifact("voice_cast") or {}

    src_rel = (colorist_art.get("output_path") or
               mixer_art.get("output_path") or
               editor_art.get("output_path"))
    if not src_rel:
        raise ValueError("No source cut to title.")
    src_abs = os.path.join(project.dir, src_rel)
    if not os.path.exists(src_abs):
        raise ValueError(f"Source cut missing at {src_abs}")

    # Pull the film title from the picked logline (falls back to project meta).
    idx = int(producer.get("chosen_index", 0))
    loglines = producer.get("loglines") or []
    title = ""
    tone = ""
    if 0 <= idx < len(loglines):
        title = str(loglines[idx].get("title", "")).strip()
        tone = str(loglines[idx].get("tone", "")).strip()
    if not title:
        title = project.meta.title

    # Cast lines for the end credits
    cast = voice_cast.get("cast") or {}
    cast_lines = [f"{name.strip()} .... {info.get('voice', 'Voice')}"
                  for name, info in cast.items() if name.strip()]

    titles_dir = os.path.join(project.artifacts_dir, "titles")
    os.makedirs(titles_dir, exist_ok=True)
    title_mp4 = os.path.join(titles_dir, "front_title.mp4")
    credits_mp4 = os.path.join(titles_dir, "end_credits.mp4")

    # Front title: 3s black card with the film title, fade in/out.
    _render_title_card(title_mp4, title=title, subtitle=tone.upper() if tone else "", duration=3.0)

    # End credits: 6s scroll of cast + a "made with" line.
    credits_lines = ["CAST", ""] + cast_lines + ["", "MADE WITH", "STUDIOLITE FILM STUDIO"]
    _render_credits(credits_mp4, lines=credits_lines, duration=6.0)

    concat_txt = os.path.join(titles_dir, "concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in (title_mp4, src_abs, credits_mp4):
            f.write(f"file '{p.replace(os.sep, '/')}'\n")

    out_abs = os.path.join(project.dir, "final.mp4")
    # Re-encode the concat so the front/body/end segments align on codec, fps,
    # sample rate. Copy-mode would need identical stream params across all
    # three which the title/credits synths don't guarantee.
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        out_abs,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"titles ffmpeg failed:\n{(e.stderr or '')[-800:]}") from e

    return {
        "output_path": os.path.relpath(out_abs, project.dir).replace(os.sep, "/"),
        "title": title,
        "cast_line_count": len(cast_lines),
        "source": src_rel,
    }


def _render_title_card(out_path: str, *, title: str, subtitle: str, duration: float) -> None:
    """Black 1280x720 card with the title fading in for the first second,
    holding, then fading out. Optional subtitle underneath.

    Uses drawtext's `textfile=` mechanism (no shell escaping headaches for
    apostrophes) plus an explicit `fontfile=` (Windows ffmpeg has no
    fontconfig by default). Runs with cwd = titles_dir so filter args stay
    free of colons — Windows drive letters (`C:\\`) get parsed as filter
    option separators."""
    parent = os.path.dirname(out_path)
    _ensure_titles_font(parent)
    with open(os.path.join(parent, "_title.txt"), "w", encoding="utf-8") as f:
        f.write(title)
    if subtitle:
        with open(os.path.join(parent, "_subtitle.txt"), "w", encoding="utf-8") as f:
            f.write(subtitle)

    fade_in_end = 1.0
    hold_end = max(fade_in_end + 0.5, duration - 1.0)
    vf = (
        "drawtext=textfile=_title.txt:fontfile=font.ttf:fontcolor=white:fontsize=64:"
        "x=(w-text_w)/2:y=(h-text_h)/2-40:"
        f"alpha='if(lt(t,{fade_in_end}),t/{fade_in_end}"
        f",if(lt(t,{hold_end}),1,max(0,({duration}-t)/1)))'"
    )
    if subtitle:
        vf += (
            ",drawtext=textfile=_subtitle.txt:fontfile=font.ttf:fontcolor=white:fontsize=22:"
            "x=(w-text_w)/2:y=(h-text_h)/2+40:"
            f"alpha='if(lt(t,{fade_in_end}),t/{fade_in_end}"
            f",if(lt(t,{hold_end}),0.7,max(0,({duration}-t)/1)))'"
        )
    out_name = os.path.basename(out_path)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}:r=30",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_name,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=parent)


def _render_credits(out_path: str, *, lines: List[str], duration: float) -> None:
    """Black card with lines rising slowly from the bottom over `duration`
    seconds. Same textfile+fontfile+cwd strategy as _render_title_card."""
    total_h = 720
    line_gap = 40
    stack_h = len(lines) * line_gap
    speed = (total_h + stack_h) / duration  # pixels per second
    parent = os.path.dirname(out_path)
    _ensure_titles_font(parent)

    parts: List[str] = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        with open(os.path.join(parent, f"_credit_{i:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(ln)
        parts.append(
            f"drawtext=textfile=_credit_{i:02d}.txt:fontfile=font.ttf:fontcolor=white:fontsize=24:"
            f"x=(w-text_w)/2:y=h+{i*line_gap}-t*{speed:.2f}"
        )
    vf = ",".join(parts) if parts else "null"
    out_name = os.path.basename(out_path)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}:r=30",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_name,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=parent)


def _ensure_titles_font(dir_path: str) -> None:
    """Copy a system font next to the title text files as `font.ttf` so
    drawtext can find it without fontconfig. Reuses the copy if already
    present. Tries a few common Windows fonts; falls through silently on
    non-Windows (drawtext will fall back to any system font Fontconfig can
    resolve)."""
    dst = os.path.join(dir_path, "font.ttf")
    if os.path.exists(dst):
        return
    for src in (r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\calibri.ttf"):
        if os.path.exists(src):
            try:
                shutil.copyfile(src, dst)
                return
            except Exception:
                continue


# ---------------------------------------------------------------------------
# NEW STAGE — Motion Shots (Phase B)
#
# Wan 2.1 T2V-1.3B turns each SDXL keyframe into a short motion clip. Uses
# the image-to-video pathway when the keyframe exists so composition and
# character look are preserved from the still. Falls back to a per-shot
# Ken Burns clip on OOM or missing model.
# ---------------------------------------------------------------------------

_WAN_PIPE = None
_WAN_STATE = "unknown"   # unknown | loaded | disabled


_WAN_MODE = "unknown"  # unknown | i2v | t2v — set by _load_wan_pipeline


_SVD_PIPE = None
_SVD_STATE = "unknown"   # unknown | loaded | disabled


def _load_svd_pipeline():
    """Load Stable Video Diffusion img2vid_xt. ~10 GB weights + fp16 fits
    on 12 GB with CPU offload. Takes a still image and produces ~4 seconds
    of true motion. This is the primary motion backend when a keyframe
    exists — Wan T2V is a fallback that discards composition."""
    global _SVD_PIPE, _SVD_STATE
    if _SVD_STATE == "disabled":
        return None
    if _SVD_STATE == "loaded":
        return _SVD_PIPE
    try:
        import torch as _torch
        from diffusers import StableVideoDiffusionPipeline
        # local_files_only avoids Hub round-trips for non-fp16 duplicate
        # files whose absence would stall a fresh cache anonymously. Once
        # the variant shards are on disk they're all we need.
        #
        # Prefer the smaller `stable-video-diffusion-img2vid` (14-frame)
        # over the `img2vid-xt` (25-frame) — the xt variant loads OK on 12GB
        # with cpu offload but the swap thrashing makes every 4s clip take
        # 5-10 minutes. Fall back to xt when the small variant isn't cached.
        for repo in (
            "stabilityai/stable-video-diffusion-img2vid",
            "stabilityai/stable-video-diffusion-img2vid-xt",
        ):
            try:
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    repo, torch_dtype=_torch.float16, variant="fp16",
                    local_files_only=True, token=_hf_token(),
                )
                logger.info("SVD loaded from %s", repo)
                break
            except Exception:
                pipe = None
        if pipe is None:
            _SVD_STATE = "disabled"
            logger.warning("No cached SVD variant found; motion_shots will fall back to Wan/Ken Burns")
            return None
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        _SVD_PIPE = pipe
        _SVD_STATE = "loaded"
        logger.info("SVD img2vid-xt pipeline loaded (fp16 + CPU offload)")
        return pipe
    except Exception as e:
        _SVD_STATE = "disabled"
        logger.warning("SVD unavailable (%s); motion_shots will use Wan T2V or Ken Burns", e)
        return None


def _disable_svd() -> None:
    global _SVD_PIPE, _SVD_STATE
    _SVD_PIPE = None
    _SVD_STATE = "disabled"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass


def _render_svd_i2v(pipe, image_path: str, out_path: str, *,
                    duration_sec: int) -> None:
    """Generate a motion clip from the SDXL keyframe. SVD is fixed-format:
    1024x576 output, 25 frames at 6-7 fps for a ~4-second clip. We over-
    generate then trim/tile to the shot's `duration_sec` in the Editor."""
    import torch as _torch
    from PIL import Image as _PILImage
    from diffusers.utils import export_to_video
    img = _PILImage.open(image_path).convert("RGB").resize((1024, 576))
    # motion_bucket_id controls motion intensity (127 = medium; 180 = punchier).
    # noise_aug_strength = adds a hair of image noise which helps SVD move
    # rather than stay locked to the still.
    with _torch.no_grad():
        frames = pipe(
            img,
            num_frames=25,
            fps=7,
            motion_bucket_id=150,
            noise_aug_strength=0.05,
            num_inference_steps=25,
        ).frames[0]
    export_to_video(frames, out_path, fps=7)


_ANIMATEDIFF_PIPE = None
_ANIMATEDIFF_STATE = "unknown"   # unknown | loaded | disabled


def _load_animatediff_pipeline():
    """Load AnimateDiff on top of an SD 1.5 base. Uses guoyww's motion
    adapter v1.5 (~2 GB). We DON'T stack on SDXL — the SDXL motion module
    is larger (~7 GB) and slower; SD 1.5 + AnimateDiff runs a 16-frame
    clip in ~30 seconds on a 12 GB card versus SVD's 5-10 minutes.

    Motion adapter modulates temporal attention on the pretrained UNet.
    Because it uses a different base than our SDXL Turbo shot generator,
    the output style will differ from the keyframe — that's a known limit
    of this backend. The trade-off is speed."""
    global _ANIMATEDIFF_PIPE, _ANIMATEDIFF_STATE
    if _ANIMATEDIFF_STATE == "disabled":
        return None
    if _ANIMATEDIFF_STATE == "loaded":
        return _ANIMATEDIFF_PIPE
    try:
        import torch as _torch
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=_torch.float16, token=_hf_token(),
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=_torch.float16, token=_hf_token(),
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            clip_sample=False, timestep_spacing="linspace",
            beta_schedule="linear", steps_offset=1,
        )
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        _ANIMATEDIFF_PIPE = pipe
        _ANIMATEDIFF_STATE = "loaded"
        logger.info("AnimateDiff pipeline loaded (SD 1.5 + guoyww v1.5-2)")
        return pipe
    except Exception as e:
        _ANIMATEDIFF_STATE = "disabled"
        logger.warning("AnimateDiff unavailable (%s); motion_shots will use SVD or Wan", e)
        return None


def _disable_animatediff() -> None:
    global _ANIMATEDIFF_PIPE, _ANIMATEDIFF_STATE
    _ANIMATEDIFF_PIPE = None
    _ANIMATEDIFF_STATE = "disabled"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass


def _render_animatediff_t2v(pipe, out_path: str, *, prompt: str,
                            duration_sec: int, camera_move: str) -> None:
    """Text-to-video via AnimateDiff. Doesn't consume the SDXL keyframe
    directly (init_image support requires the img2img variant which isn't
    loaded here). We lean on the same prompt that produced the keyframe
    plus a motion hint, similar to Wan T2V."""
    import torch as _torch
    from diffusers.utils import export_to_video
    motion_hint = {
        "pan":     "smooth horizontal camera pan",
        "dolly":   "slow dolly-in",
        "crane":   "camera cranes upward",
        "handheld":"subtle handheld shake",
        "whip":    "fast whip pan",
        "static":  "gentle atmospheric motion, minimal camera",
    }.get(camera_move, "gentle atmospheric motion")
    full_prompt = f"{prompt}. {motion_hint}."
    fps = 8   # AnimateDiff native
    # 32 frames = 4s of native motion. Kills the visible looping the
    # editor's tile-to-fit does when a 2s clip is stretched over a
    # 5-8s shot. If VRAM chokes we retry once at 24 frames (3s), which
    # still doubles what 16-frame clips gave us.
    num_frames = min(32, max(16, duration_sec * fps))
    try:
        with _torch.no_grad():
            out = pipe(
                prompt=full_prompt,
                negative_prompt="low quality, distorted, deformed, watermark",
                num_frames=num_frames,
                guidance_scale=7.5,
                num_inference_steps=20,
            )
    except _torch.cuda.OutOfMemoryError:
        _torch.cuda.empty_cache()
        with _torch.no_grad():
            out = pipe(
                prompt=full_prompt,
                negative_prompt="low quality, distorted, deformed, watermark",
                num_frames=24,
                guidance_scale=7.5,
                num_inference_steps=20,
            )
    frames = out.frames[0]
    export_to_video(frames, out_path, fps=fps)


def _load_wan_pipeline():
    """Load a Wan 2.1 pipeline lazily. Prefers T2V-1.3B (~7 GB at fp16) which
    fits alongside SDXL and MusicGen on a 12 GB card. The 14B I2V variant
    needs ~28 GB and segfaults loading here, so we skip it entirely. Returns
    the pipeline handle on success, None on any failure. Weights come from
    the local HF cache — nothing is downloaded from here."""
    global _WAN_PIPE, _WAN_STATE, _WAN_MODE
    if _WAN_STATE == "disabled":
        return None
    if _WAN_STATE == "loaded":
        return _WAN_PIPE
    try:
        import torch as _torch
        from diffusers import WanPipeline
        # T2V 1.3B — composition-conditions from the SDXL keyframe via the
        # prompt only (the pipeline is text-to-video). We still get real
        # motion; character consistency comes from the same descriptors
        # baked into every shot prompt.
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            torch_dtype=_torch.float16,
            token=_hf_token(),
        )
        pipe.enable_model_cpu_offload()  # keeps peak VRAM under ~7 GB
        pipe.set_progress_bar_config(disable=True)
        _WAN_PIPE = pipe
        _WAN_STATE = "loaded"
        _WAN_MODE = "t2v"
        logger.info("Wan T2V-1.3B pipeline loaded")
        return pipe
    except Exception as e:
        _WAN_STATE = "disabled"
        logger.warning("Wan pipeline unavailable (%s); motion_shots will fall back to Ken Burns", e)
        return None


def _disable_wan() -> None:
    global _WAN_PIPE, _WAN_STATE
    _WAN_PIPE = None
    _WAN_STATE = "disabled"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass
    logger.warning("Wan pipeline disabled for the rest of the session")


_MOTION_SHOTS_MAX_WAN = 40   # Wan T2V is ~260s per 3s clip. Cap when Wan is
                              # the only motion backend so long films finish.


def run_motion_shots(project: Project) -> Dict[str, Any]:
    """For each shot with a rendered SDXL keyframe, generate a short motion
    clip. Backend chosen by `motion_backend` config:
    - "auto": SVD -> AnimateDiff -> Wan -> Ken Burns (best quality first)
    - "animatediff": SD 1.5 + AnimateDiff motion module (~30s per clip)
    - "svd": SVD img2vid keyframe-conditioned (slow on 12GB, 5-10 min)
    - "wan": Wan 2.1 T2V (text only, discards keyframe)
    - "kenburns": pan on stills (no motion generation)

    The Wan-only cap of 40 shots only applies when no other motion
    backend is available."""
    shots_art = project.read_artifact("shots") or {}
    cinema    = project.read_artifact("cinematographer") or {}
    shots: List[Dict[str, Any]] = shots_art.get("shots") or []
    dp_by_scene: Dict[str, Dict[str, Any]] = cinema.get("scenes") or {}

    motion_dir = os.path.join(project.artifacts_dir, "motion")
    os.makedirs(motion_dir, exist_ok=True)

    backend_cfg = (project.meta.config.motion_backend or "auto").lower()

    # Load only the pipelines we might use, in try-order.
    svd_pipe = None
    animatediff_pipe = None
    wan_pipe = None
    if backend_cfg == "kenburns":
        pass  # nothing to load; every shot falls through
    elif backend_cfg == "animatediff":
        animatediff_pipe = _load_animatediff_pipeline()
    elif backend_cfg == "svd":
        svd_pipe = _load_svd_pipeline()
    elif backend_cfg == "wan":
        wan_pipe = _load_wan_pipeline()
    else:  # auto
        svd_pipe = _load_svd_pipeline()
        if svd_pipe is None:
            animatediff_pipe = _load_animatediff_pipeline()
        if svd_pipe is None and animatediff_pipe is None:
            if len(shots) > _MOTION_SHOTS_MAX_WAN:
                logger.warning(
                    "Skipping motion_shots — no SVD or AnimateDiff, and %d shots "
                    "is over the Wan-only cap of %d. Editor will use Ken Burns.",
                    len(shots), _MOTION_SHOTS_MAX_WAN,
                )
                return {
                    "motion_shots": [],
                    "svd_count": 0, "animatediff_count": 0,
                    "wan_count": 0, "kb_count": 0,
                    "skipped_reason": f"only Wan available and {len(shots)} > {_MOTION_SHOTS_MAX_WAN}",
                }
            wan_pipe = _load_wan_pipeline()

    used_svd = 0
    used_animatediff = 0
    used_wan = 0
    used_kb = 0
    total = len(shots)
    project.append_event({"type": "motion_progress", "done": 0, "total": total})

    result: List[Dict[str, Any]] = []
    for i, s in enumerate(shots):
        sid = s["scene_id"]; shot_id = s["shot_id"]
        keyframe_abs = os.path.join(project.dir, s["path"])
        dur = max(2, int(s.get("duration_sec", 4) or 4))
        clip_dur = min(dur, 5)
        mp4_abs = os.path.join(motion_dir, f"{sid}_{shot_id}.mp4")
        dp_shot = (dp_by_scene.get(sid) or {}).get(shot_id, {})
        move = dp_shot.get("camera_move", "static")

        rendered = False
        backend = "kenburns"
        error: Optional[str] = None

        # Try SVD first (best quality, keyframe-conditioned)
        if svd_pipe is not None and os.path.exists(keyframe_abs):
            try:
                _render_svd_i2v(svd_pipe, keyframe_abs, mp4_abs, duration_sec=clip_dur)
                rendered = True; backend = "svd"; used_svd += 1
            except Exception as e:
                logger.exception("SVD failed for %s/%s", sid, shot_id)
                error = f"svd: {e}"
                if "out of memory" in str(e).lower():
                    _disable_svd()
                    svd_pipe = None

        # AnimateDiff — fast text-to-video via SD 1.5 motion adapter
        if not rendered and animatediff_pipe is not None:
            try:
                _render_animatediff_t2v(
                    animatediff_pipe, mp4_abs,
                    prompt=s.get("prompt", "") or "",
                    duration_sec=clip_dur, camera_move=move,
                )
                rendered = True; backend = "animatediff"; used_animatediff += 1
                error = None
            except Exception as e:
                logger.exception("AnimateDiff failed for %s/%s", sid, shot_id)
                error = f"{error or ''}; animatediff: {e}"
                if "out of memory" in str(e).lower():
                    _disable_animatediff()
                    animatediff_pipe = None

        # Wan T2V (text-only, discards keyframe)
        if not rendered and wan_pipe is not None and os.path.exists(keyframe_abs):
            try:
                _render_wan_i2v(
                    wan_pipe, keyframe_abs, mp4_abs,
                    prompt=s.get("prompt", "") or "",
                    duration_sec=clip_dur, camera_move=move,
                )
                rendered = True; backend = "wan"; used_wan += 1
                error = None
            except Exception as e:
                logger.exception("Wan I2V failed for %s/%s", sid, shot_id)
                error = f"{error or ''}; wan: {e}"
                if "out of memory" in str(e).lower():
                    _disable_wan()
                    wan_pipe = None

        # Final fallback: Ken Burns on the still
        if not rendered:
            try:
                _render_kenburns_clip(keyframe_abs, mp4_abs, dur, move)
                backend = "kenburns"; used_kb += 1
            except Exception as e:
                logger.exception("Ken Burns fallback failed for %s/%s", sid, shot_id)
                error = f"{error or ''}; kb: {e}"

        result.append({
            "scene_id": sid, "shot_id": shot_id,
            "path": os.path.relpath(mp4_abs, project.dir).replace(os.sep, "/"),
            "duration_sec": dur,
            "backend": backend,
            "rendered": rendered,
            "error": error,
        })
        project.append_event({"type": "motion_progress", "done": i + 1,
                              "total": total, "backend": backend})

    return {
        "motion_shots": result,
        "svd_count": used_svd,
        "animatediff_count": used_animatediff,
        "wan_count": used_wan,
        "kb_count": used_kb,
    }


def _render_wan_i2v(pipe, image_path: str, out_path: str, *,
                    prompt: str, duration_sec: int, camera_move: str) -> None:
    """Generate a motion clip for the shot. The T2V pipeline can't take the
    keyframe as input, so we lean on the prompt (which is the same prompt
    that produced the keyframe, plus a motion hint from camera_move) to
    reproduce a similar frame."""
    import torch as _torch
    from diffusers.utils import export_to_video
    motion_hint = {
        "pan":     "smooth horizontal camera pan",
        "dolly":   "slow dolly-in, subject gets closer",
        "crane":   "camera cranes upward",
        "handheld":"subtle handheld shake",
        "whip":    "fast whip pan",
        "static":  "gentle atmospheric motion, minimal camera",
    }.get(camera_move, "gentle atmospheric motion")
    full_prompt = f"{prompt}. {motion_hint}."
    fps = 15
    num_frames = duration_sec * fps
    with _torch.no_grad():
        frames = pipe(
            prompt=full_prompt,
            height=480, width=832,
            num_frames=num_frames,
            guidance_scale=5.0,
            num_inference_steps=15,
        ).frames[0]
    export_to_video(frames, out_path, fps=fps)


# ---------------------------------------------------------------------------
# NEW STAGE — Ambient Bed (Phase C)
#
# AudioLDM 2 renders a per-scene atmosphere loop (rain, café hum, night wind).
# Layered under dialogue and score by the mixer. On any failure the stage
# writes a silent wav so the mixer has something to key off.
# ---------------------------------------------------------------------------

def run_ambient(project: Project) -> Dict[str, Any]:
    """Generate a single film-length ambient bed keyed off the film's tone
    and location moods. Uses AudioLDM 2 if importable; falls back to
    silence otherwise. Mixer layers it under dialogue and score."""
    editor_art = project.read_artifact("editor") or {}
    breakdown  = project.read_artifact("breakdown") or {}
    producer   = project.read_artifact("producer") or {}
    total_dur = int(editor_art.get("duration_sec", 0) or 0)
    if total_dur <= 0:
        raise ValueError("Editor artifact missing duration_sec")

    # Derive an ambience prompt from tone + moods.
    tone = ""
    idx = int(producer.get("chosen_index", 0))
    loglines = producer.get("loglines") or []
    if 0 <= idx < len(loglines):
        tone = str(loglines[idx].get("tone", "")).lower()
    scenes = breakdown.get("scenes") or []
    moods = " ".join(str(s.get("mood", "")) for s in scenes)[:200]
    locations = " ".join(str(s.get("location", "")).replace("-", " ") for s in scenes)[:200]
    prompt = f"{tone} film ambience, {moods}, {locations}, subtle atmospheric background".strip(", ")

    out_abs = os.path.join(project.artifacts_dir, "ambient.wav")
    ok = False; err: Optional[str] = None
    try:
        _render_audioldm(prompt, out_abs, duration_sec=total_dur)
        ok = True
    except Exception as e:
        logger.warning("AudioLDM unavailable/failed (%s); ambient will be silent", e)
        err = str(e)
        _write_silent_wav(out_abs, float(total_dur))
    return {
        "prompt": prompt,
        "path": os.path.relpath(out_abs, project.dir).replace(os.sep, "/"),
        "duration_sec": total_dur,
        "rendered": ok,
        "error": err,
    }


def _render_audioldm(prompt: str, out_path: str, *, duration_sec: int) -> None:
    """Render a single ambient clip. AudioLDM 2 tops out at ~10s per call;
    we chain multiple with a short crossfade to cover the film's length."""
    import numpy as np
    import soundfile as sf
    import torch as _torch
    from diffusers import AudioLDM2Pipeline
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", torch_dtype=_torch.float16, token=_hf_token(),
    ).to("cuda")

    # transformers 5.x removed `_update_model_kwargs_for_generation` from the
    # base model class, but AudioLDM2 still calls it. Rebind the newer
    # public equivalent from the generation mixin (or a no-op if that's also
    # missing) so the pipeline can complete a forward pass.
    lm = getattr(pipe, "language_model", None)
    if lm is not None and not hasattr(lm, "_update_model_kwargs_for_generation"):
        # Try to borrow from GenerationMixin, else stub with identity behaviour.
        try:
            from transformers.generation.utils import GenerationMixin
            lm._update_model_kwargs_for_generation = (
                GenerationMixin._update_model_kwargs_for_generation.__get__(lm)
            )
        except Exception:
            def _noop(outputs, model_kwargs, *args, **kwargs):
                return model_kwargs
            lm._update_model_kwargs_for_generation = _noop
    sr = 16000
    chunk_len_sec = 10
    remaining = duration_sec
    chunks: List = []
    while remaining > 0:
        this = min(chunk_len_sec, remaining)
        out = pipe(
            prompt=prompt,
            negative_prompt="dialogue, speech, vocals",
            num_inference_steps=25,
            audio_length_in_s=this,
        ).audios[0]
        chunks.append(np.asarray(out, dtype=np.float32))
        remaining -= this

    if len(chunks) == 1:
        combined = chunks[0]
    else:
        xfade = int(0.5 * sr)
        combined = chunks[0].copy()
        for nxt in chunks[1:]:
            n = min(xfade, len(combined), len(nxt))
            t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
            fade_out = np.cos(t); fade_in = np.sin(t)
            seam = combined[-n:] * fade_out + nxt[:n] * fade_in
            combined = np.concatenate([combined[:-n], seam, nxt[n:]])

    # Keep the ambient quiet so it doesn't step on dialogue and score.
    peak = float(np.max(np.abs(combined))) or 1.0
    combined = (combined / peak) * 0.35
    max_samples = int(duration_sec * sr)
    if len(combined) > max_samples:
        combined = combined[:max_samples]
    sf.write(out_path, combined, sr)
