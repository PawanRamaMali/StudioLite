"""Stage-runner tests for the OmniGen2 and Wan2.1-FLF2V branches.

The dispatcher-level helpers already have their own tests
(``test_shots_omnigen2.py`` and ``test_motion_wan21_flf2v.py``). This
module runs the full ``run_shots`` / ``run_motion_shots`` / new
``run_transitions`` stage with the backend forced to the roadmap value,
patching the dispatcher helpers so the stage never shells out, and
confirms:
  - the right branch is taken from the config (omnigen2 / wan21_flf2v)
  - rendered artifacts are copied into the per-shot output paths
  - a dispatcher that returns ``mode="skipped"`` falls through to the
    stage's existing non-roadmap path (SDXL / Ken Burns)
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest


class _FakeProject:
    """Just enough of ``filmmaker.project.Project`` for the three stages."""

    def __init__(self, tmp: str, *, sdxl_variant: str = "turbo",
                 motion_backend: str = "auto",
                 transitions_backend: str = "none"):
        self.dir = tmp
        self.artifacts_dir = os.path.join(tmp, "artifacts")
        self.shots_dir = os.path.join(self.artifacts_dir, "shots")
        os.makedirs(self.shots_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self.meta = SimpleNamespace(config=SimpleNamespace(
            style="stylized", quality="standard",
            sdxl_variant=sdxl_variant,
            motion_backend=motion_backend,
            transitions_backend=transitions_backend,
        ))

    def read_artifact(self, key: str):
        return self._artifacts.get(key)

    def write_artifact(self, key: str, data):
        self._artifacts[key] = data

    def append_event(self, ev: Dict[str, Any]) -> None:
        self.events.append(ev)


def _seed_two_shots(project: _FakeProject) -> None:
    project.write_artifact("breakdown", {
        "scenes": [{"id": "s01", "summary": "test scene",
                    "location": "studio", "time_of_day": "day",
                    "mood": "curious", "characters": []}],
    })
    project.write_artifact("storyboard", {
        "scenes": {"s01": [
            {"id": "sh1", "description": "shot one",
             "subject": "a robot", "action": "waves", "dialogue": ""},
            {"id": "sh2", "description": "shot two",
             "subject": "the robot", "action": "turns", "dialogue": ""},
        ]},
    })
    project.write_artifact("cinematographer", {
        "scenes": {"s01": {
            "sh1": {"framing": "MS", "lens_mm": 35,
                     "camera_move": "static", "lighting": "soft",
                     "palette": "warm", "duration_sec": 4},
            "sh2": {"framing": "CU", "lens_mm": 50,
                     "camera_move": "pan", "lighting": "hard",
                     "palette": "cool", "duration_sec": 4},
        }},
    })


# ---------------------------------------------------------------------------
# run_shots: sdxl_variant == "omnigen2"
# ---------------------------------------------------------------------------

def test_run_shots_omnigen2_branch_copies_rendered_pngs():
    """When sdxl_variant=='omnigen2' and the dispatcher returns rendered
    shots, run_shots copies each one into the shot's png_path and marks
    the job rendered without ever calling SDXL."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, sdxl_variant="omnigen2")
        _seed_two_shots(p)

        def _og2_dispatch(project, prompts, out_dir):
            os.makedirs(out_dir, exist_ok=True)
            rendered: List[Dict[str, str]] = []
            for pr in prompts:
                abs_png = os.path.join(
                    out_dir, f"{pr['scene_id']}_{pr['shot_id']}.png")
                with open(abs_png, "wb") as f:
                    # > 1024 bytes so run_shots passes its size gate
                    f.write(b"\x89PNG\r\n\x1a\n" + b"x" * 2048)
                rendered.append({"scene_id": pr["scene_id"],
                                  "shot_id": pr["shot_id"],
                                  "path": abs_png})
            return {"backend": "omnigen2", "mode": "local",
                    "shots": rendered}

        def _fail_sdxl(prompt, out_path, *, steps=None, ref_image=None):
            raise AssertionError("SDXL per-shot renderer should not be called "
                                  "when OmniGen2 handles every shot")

        def _fail_sdxl_batch(prompts, out_paths, *, steps=None,
                              ref_images=None):
            raise AssertionError("SDXL batch renderer should not be called "
                                  "when OmniGen2 handles every shot")

        with patch.object(agents, "_render_shots_omnigen2",
                           side_effect=_og2_dispatch), \
             patch.object(agents, "_load_sdxl_renderer",
                           return_value=_fail_sdxl), \
             patch.object(agents, "_load_sdxl_batch_renderer",
                           return_value=_fail_sdxl_batch):
            result = agents.run_shots(p)

        assert len(result["shots"]) == 2
        assert all(s["rendered"] for s in result["shots"])
        # Each shot's rendered PNG lives at the standard shots-dir path.
        for s in result["shots"]:
            abs_png = os.path.join(p.dir, s["path"])
            assert os.path.exists(abs_png)
            assert os.path.getsize(abs_png) > 1024
        # The dispatcher-fired event is logged.
        og2_evs = [e for e in p.events if e.get("type") == "shots_omnigen2"]
        assert og2_evs, f"expected shots_omnigen2 event, got {p.events}"
        assert og2_evs[0]["mode"] == "local"
        assert og2_evs[0]["rendered"] == 2


def test_run_shots_omnigen2_skip_falls_through_to_sdxl():
    """When the OmniGen2 dispatcher returns mode='skipped', run_shots must
    still produce rendered keyframes via the SDXL path - no shot is left
    unrendered because the roadmap backend bowed out."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, sdxl_variant="omnigen2")
        _seed_two_shots(p)

        def _og2_skip(project, prompts, out_dir):
            return {"backend": "omnigen2", "mode": "skipped",
                    "shots": [], "install_hint": "fake hint"}

        def _sdxl_single(prompt, out_path, *, steps=None, ref_image=None):
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nsdxl")

        with patch.object(agents, "_render_shots_omnigen2",
                           side_effect=_og2_skip), \
             patch.object(agents, "_load_sdxl_batch_renderer",
                           return_value=None), \
             patch.object(agents, "_load_sdxl_renderer",
                           return_value=_sdxl_single):
            result = agents.run_shots(p)

        assert len(result["shots"]) == 2
        assert all(s["rendered"] for s in result["shots"])


def test_run_shots_default_variant_does_not_dispatch_omnigen2():
    """A non-omnigen2 variant must never call the dispatcher - we don't
    want surprise cloud spend on a plain run."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, sdxl_variant="turbo")
        _seed_two_shots(p)

        def _no_dispatch(*_a, **_kw):
            raise AssertionError("OmniGen2 dispatcher must not run for "
                                  "non-omnigen2 variants")

        def _sdxl_single(prompt, out_path, *, steps=None, ref_image=None):
            with open(out_path, "wb") as f:
                f.write(b"ok")

        with patch.object(agents, "_render_shots_omnigen2",
                           side_effect=_no_dispatch), \
             patch.object(agents, "_load_sdxl_batch_renderer",
                           return_value=None), \
             patch.object(agents, "_load_sdxl_renderer",
                           return_value=_sdxl_single):
            agents.run_shots(p)


# ---------------------------------------------------------------------------
# run_motion_shots: motion_backend == "wan21_flf2v"
# ---------------------------------------------------------------------------

def _seed_shots_artifact_for_motion(project: _FakeProject) -> None:
    """Motion-shots needs a rendered shots artifact plus the keyframe
    PNGs on disk. We create a 2048-byte stub so the >100000 reuse guard
    in run_motion_shots does NOT trigger."""
    _seed_two_shots(project)
    shots_dir = os.path.join(project.shots_dir, "s01")
    os.makedirs(shots_dir, exist_ok=True)
    rendered: List[Dict[str, Any]] = []
    for shot_id in ("sh1", "sh2"):
        png = os.path.join(shots_dir, f"{shot_id}.png")
        with open(png, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"x" * 128)
        rendered.append({
            "scene_id": "s01", "shot_id": shot_id,
            "prompt": "a robot",
            "path": os.path.relpath(png, project.dir).replace(os.sep, "/"),
            "duration_sec": 4, "dialogue": "",
            "rendered": True, "error": None,
        })
    project.write_artifact("shots", {"shots": rendered})


def test_run_motion_shots_wan21_flf2v_renders_each_shot():
    """motion_backend=='wan21_flf2v' dispatches each shot to the FLF2V
    helper; a rendered mp4 shows up on disk and the backend count is 2.
    No in-process pipeline (SVD/AnimateDiff/Wan) is loaded."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, motion_backend="wan21_flf2v")
        _seed_shots_artifact_for_motion(p)

        def _fake_flf2v(project, first, last, out, *, duration_sec, prompt):
            with open(out, "wb") as f:
                f.write(b"ID3" + b"x" * 200_000)
            return {"backend": "wan21_flf2v", "mode": "local",
                    "output_path": out}

        with patch.object(agents, "_render_transition_flf2v",
                           side_effect=_fake_flf2v), \
             patch.object(agents, "_load_svd_pipeline",
                           side_effect=AssertionError("SVD must not load")), \
             patch.object(agents, "_load_animatediff_pipeline",
                           side_effect=AssertionError("AnimateDiff must not load")), \
             patch.object(agents, "_load_wan_pipeline",
                           side_effect=AssertionError("Wan must not load")), \
             patch.object(agents, "_load_wan22_pipeline",
                           side_effect=AssertionError("Wan22 must not load")):
            result = agents.run_motion_shots(p)

        assert result["wan21_flf2v_count"] == 2
        assert result["backend_counts"]["wan21_flf2v"] == 2
        assert all(s["backend"] == "wan21_flf2v" for s in result["motion_shots"])
        assert all(s["rendered"] for s in result["motion_shots"])


def test_run_motion_shots_wan21_flf2v_skip_falls_through_to_kenburns():
    """When the FLF2V helper returns mode='skipped', the shot falls
    through the stage's existing fallback chain and lands on Ken Burns.
    The run still completes - a roadmap backend going cold must never
    discard the pipeline."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, motion_backend="wan21_flf2v")
        _seed_shots_artifact_for_motion(p)

        def _skip(project, first, last, out, *, duration_sec, prompt):
            return {"backend": "wan21_flf2v", "mode": "skipped",
                    "skipped": "no cloud, no local"}

        def _kb(img, out, dur, move):
            with open(out, "wb") as f:
                f.write(b"ID3" + b"x" * 200_000)

        with patch.object(agents, "_render_transition_flf2v",
                           side_effect=_skip), \
             patch.object(agents, "_render_kenburns_clip",
                           side_effect=_kb):
            result = agents.run_motion_shots(p)

        assert result["wan21_flf2v_count"] == 0
        assert result["kb_count"] == 2
        assert all(s["backend"] == "kenburns" for s in result["motion_shots"])


# ---------------------------------------------------------------------------
# run_transitions
# ---------------------------------------------------------------------------

def _write_fake_clip(path: str, size: int = 200_000) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ID3" + b"x" * size)


def _seed_motion_artifact_for_transitions(project: _FakeProject) -> None:
    """A motion_shots artifact plus three fake mp4s on disk so there are
    two boundaries to transition across."""
    motion_dir = os.path.join(project.artifacts_dir, "motion")
    os.makedirs(motion_dir, exist_ok=True)
    shots: List[Dict[str, Any]] = []
    for i, (sid, shot_id) in enumerate(
            [("s01", "sh1"), ("s01", "sh2"), ("s01", "sh3")]):
        mp4 = os.path.join(motion_dir, f"{sid}_{shot_id}.mp4")
        _write_fake_clip(mp4)
        shots.append({
            "scene_id": sid, "shot_id": shot_id,
            "path": os.path.relpath(mp4, project.dir).replace(os.sep, "/"),
            "duration_sec": 4, "backend": "wan22",
            "rendered": True, "error": None,
        })
    project.write_artifact("motion_shots", {"motion_shots": shots})


def test_run_transitions_noop_when_backend_is_none():
    """transitions_backend == 'none' is the default: stage returns an
    empty artifact without ever extracting frames or calling FLF2V."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, transitions_backend="none")
        _seed_motion_artifact_for_transitions(p)

        with patch.object(agents, "_extract_motion_endpoint",
                           side_effect=AssertionError("must not extract")), \
             patch.object(agents, "_render_transition_flf2v",
                           side_effect=AssertionError("must not render")):
            r = agents.run_transitions(p)

        assert r["backend"] == "none"
        assert r["transitions"] == []


def test_run_transitions_wan21_flf2v_renders_each_boundary():
    """Three shots => two boundaries. Both boundaries render via the
    FLF2V helper; the artifact records relative paths under
    artifacts/transitions/."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, transitions_backend="wan21_flf2v")
        _seed_motion_artifact_for_transitions(p)

        def _fake_extract(mp4_abs, out_png_abs, *, at):
            os.makedirs(os.path.dirname(out_png_abs), exist_ok=True)
            with open(out_png_abs, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"y" * 128)
            return True

        def _fake_flf2v(project, first, last, out, *, duration_sec, prompt):
            with open(out, "wb") as f:
                f.write(b"ID3" + b"z" * 200_000)
            return {"backend": "wan21_flf2v", "mode": "local",
                    "output_path": out}

        with patch.object(agents, "_extract_motion_endpoint",
                           side_effect=_fake_extract), \
             patch.object(agents, "_render_transition_flf2v",
                           side_effect=_fake_flf2v):
            r = agents.run_transitions(p)

        assert r["backend"] == "wan21_flf2v"
        assert r["boundary_count"] == 2
        assert r["rendered_count"] == 2
        assert r["skipped_count"] == 0
        assert len(r["transitions"]) == 2
        for row in r["transitions"]:
            assert row["mode"] == "local"
            assert row["path"].startswith("artifacts/transitions/")
            abs_mp4 = os.path.join(p.dir,
                                    row["path"].replace("/", os.sep))
            assert os.path.exists(abs_mp4) and os.path.getsize(abs_mp4) > 1024
        assert r["transitions"][0]["from_key"] == "s01::sh1"
        assert r["transitions"][0]["to_key"] == "s01::sh2"
        assert r["transitions"][1]["from_key"] == "s01::sh2"
        assert r["transitions"][1]["to_key"] == "s01::sh3"


def test_run_transitions_skips_boundary_when_flf2v_skips():
    """When the FLF2V helper returns mode='skipped' for a boundary the
    stage records the error with its install hint and keeps going; the
    other boundary still renders. No clip file is written for the
    skipped slot."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, transitions_backend="wan21_flf2v")
        _seed_motion_artifact_for_transitions(p)

        def _fake_extract(mp4_abs, out_png_abs, *, at):
            os.makedirs(os.path.dirname(out_png_abs), exist_ok=True)
            with open(out_png_abs, "wb") as f:
                f.write(b"\x89PNG")
            return True

        calls = {"n": 0}

        def _flaky(project, first, last, out, *, duration_sec, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                with open(out, "wb") as f:
                    f.write(b"ID3" + b"x" * 200_000)
                return {"backend": "wan21_flf2v", "mode": "local",
                        "output_path": out}
            return {"backend": "wan21_flf2v", "mode": "skipped",
                    "skipped": "ran out of VRAM",
                    "install_hint": "install hint"}

        with patch.object(agents, "_extract_motion_endpoint",
                           side_effect=_fake_extract), \
             patch.object(agents, "_render_transition_flf2v",
                           side_effect=_flaky):
            r = agents.run_transitions(p)

        assert r["rendered_count"] == 1
        assert r["skipped_count"] == 1
        first_row, second_row = r["transitions"]
        assert first_row["mode"] == "local"
        assert "path" in first_row
        assert second_row["mode"] == "skipped"
        assert second_row.get("install_hint") == "install hint"
        assert "path" not in second_row


def test_run_transitions_no_op_with_fewer_than_two_motion_clips():
    """With only one motion clip there is no boundary to transition - the
    stage returns a note instead of raising."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp, transitions_backend="wan21_flf2v")
        motion_dir = os.path.join(p.artifacts_dir, "motion")
        os.makedirs(motion_dir, exist_ok=True)
        mp4 = os.path.join(motion_dir, "s01_sh1.mp4")
        _write_fake_clip(mp4)
        p.write_artifact("motion_shots", {"motion_shots": [{
            "scene_id": "s01", "shot_id": "sh1",
            "path": os.path.relpath(mp4, p.dir).replace(os.sep, "/"),
            "backend": "wan22", "rendered": True,
        }]})

        with patch.object(agents, "_extract_motion_endpoint",
                           side_effect=AssertionError("no extraction")), \
             patch.object(agents, "_render_transition_flf2v",
                           side_effect=AssertionError("no render")):
            r = agents.run_transitions(p)

        assert r["transitions"] == []
        assert "boundaries" in r["note"].lower()


# ---------------------------------------------------------------------------
# run_editor splice: transitions_art rows weave between motion clips
# ---------------------------------------------------------------------------

def _seed_editor_fixture(project: _FakeProject, *, transitions_mode: str) -> None:
    """Seed shots, motion_shots, and (optionally) transitions artifacts
    plus the backing files on disk for a 3-shot cut.

    ``transitions_mode``:
      - ``"absent"``  - no transitions artifact at all
      - ``"both"``    - 2 rendered rows (one per boundary)
      - ``"first_only"`` - boundary 1 rendered, boundary 2 skipped
    """
    project.write_artifact("storyboard", {
        "scenes": {"s01": [
            {"id": "sh1", "description": "shot one",
             "subject": "a robot", "action": "waves", "dialogue": ""},
            {"id": "sh2", "description": "shot two",
             "subject": "the robot", "action": "turns", "dialogue": ""},
            {"id": "sh3", "description": "shot three",
             "subject": "the robot", "action": "leaves", "dialogue": ""},
        ]},
    })
    project.write_artifact("cinematographer", {
        "scenes": {"s01": {
            "sh1": {"framing": "MS", "lens_mm": 35, "camera_move": "static",
                     "lighting": "soft", "palette": "warm", "duration_sec": 4},
            "sh2": {"framing": "CU", "lens_mm": 50, "camera_move": "pan",
                     "lighting": "hard", "palette": "cool", "duration_sec": 4},
            "sh3": {"framing": "WS", "lens_mm": 24, "camera_move": "static",
                     "lighting": "soft", "palette": "warm", "duration_sec": 4},
        }},
    })

    # Three shot keyframes on disk + the shots artifact.
    shots_scene_dir = os.path.join(project.shots_dir, "s01")
    os.makedirs(shots_scene_dir, exist_ok=True)
    shots: List[Dict[str, Any]] = []
    for shot_id in ("sh1", "sh2", "sh3"):
        png = os.path.join(shots_scene_dir, f"{shot_id}.png")
        with open(png, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"x" * 128)
        shots.append({
            "scene_id": "s01", "shot_id": shot_id,
            "path": os.path.relpath(png, project.dir).replace(os.sep, "/"),
            "duration_sec": 4, "rendered": True,
        })
    project.write_artifact("shots", {"shots": shots})

    # Three motion mp4s on disk + the motion_shots artifact (so run_editor
    # takes the motion branch and never calls Ken Burns).
    motion_dir = os.path.join(project.artifacts_dir, "motion")
    os.makedirs(motion_dir, exist_ok=True)
    motions: List[Dict[str, Any]] = []
    for shot_id in ("sh1", "sh2", "sh3"):
        mp4 = os.path.join(motion_dir, f"s01_{shot_id}.mp4")
        _write_fake_clip(mp4)
        motions.append({
            "scene_id": "s01", "shot_id": shot_id,
            "path": os.path.relpath(mp4, project.dir).replace(os.sep, "/"),
            "duration_sec": 4, "backend": "wan22", "rendered": True,
        })
    project.write_artifact("motion_shots", {"motion_shots": motions})

    if transitions_mode == "absent":
        return

    tr_dir = os.path.join(project.artifacts_dir, "transitions")
    os.makedirs(tr_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    pairs = [("sh1", "sh2"), ("sh2", "sh3")]
    for idx, (frm, to) in enumerate(pairs):
        rendered = (transitions_mode == "both"
                     or (transitions_mode == "first_only" and idx == 0))
        if rendered:
            mp4 = os.path.join(tr_dir, f"s01_{frm}_to_{to}.mp4")
            _write_fake_clip(mp4)
            rows.append({
                "from_key": f"s01::{frm}",
                "to_key": f"s01::{to}",
                "mode": "local",
                "path": os.path.relpath(mp4, project.dir).replace(os.sep, "/"),
            })
        else:
            rows.append({
                "from_key": f"s01::{frm}",
                "to_key": f"s01::{to}",
                "mode": "skipped",
                "error": "no FLF2V available",
                "install_hint": "install this",
            })
    project.write_artifact("transitions",
                            {"transitions": rows, "backend": "wan21_flf2v"})


def _editor_patches(agents_mod):
    """Patch the three ffmpeg-shelling helpers run_editor touches so no
    subprocess fires. _standardize_motion_clip is used for BOTH motion
    clips and transition clips; _render_kenburns_clip must never fire
    when a motion mp4 is present; the concat subprocess.run is stubbed
    to just create its output file. Returns a tuple of patchers to
    enter under one ``with`` statement."""
    def _fake_standardize(src, dst, duration_sec):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(b"\x00" * 2048)

    def _fake_subprocess_run(cmd, *args, **kwargs):
        # cmd[-1] is the concat output path; create it so run_editor's
        # relpath call succeeds.
        out = cmd[-1]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "wb") as f:
            f.write(b"\x00" * 2048)
        return SimpleNamespace(stderr="", stdout="")

    return (
        patch.object(agents_mod, "_standardize_motion_clip",
                      side_effect=_fake_standardize),
        patch.object(agents_mod, "_render_kenburns_clip",
                      side_effect=AssertionError(
                          "Ken Burns must not run when motion mp4s are present")),
        patch.object(agents_mod.subprocess, "run",
                      side_effect=_fake_subprocess_run),
    )


def _read_concat_lines(project: _FakeProject) -> List[str]:
    concat_txt = os.path.join(project.artifacts_dir, "editor.concat.txt")
    with open(concat_txt, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def test_run_editor_splices_transition_clips_between_motion_shots():
    """Three motion shots + two rendered boundaries weaves to
    motion1, trans1, motion2, trans2, motion3 in the concat list, and
    the stage result reports transition_clips_used == 2."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp)
        _seed_editor_fixture(p, transitions_mode="both")

        a, b, c = _editor_patches(agents)
        with a, b, c:
            r = agents.run_editor(p)

        lines = _read_concat_lines(p)
        assert len(lines) == 5, lines
        # concat lines look like: file 'C:/.../000_s01_sh1.mp4'
        assert lines[0].endswith("000_s01_sh1.mp4'")
        assert lines[1].endswith("000_s01_sh1_T.mp4'")
        assert lines[2].endswith("001_s01_sh2.mp4'")
        assert lines[3].endswith("001_s01_sh2_T.mp4'")
        assert lines[4].endswith("002_s01_sh3.mp4'")
        assert r["motion_clips_used"] == 3
        assert r["transition_clips_used"] == 2
        assert r["shot_count"] == 3


def test_run_editor_without_transitions_artifact_behaves_as_before():
    """No transitions artifact at all: three motion clips, no morph
    splices, and transition_clips_used == 0."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp)
        _seed_editor_fixture(p, transitions_mode="absent")

        a, b, c = _editor_patches(agents)
        with a, b, c:
            r = agents.run_editor(p)

        lines = _read_concat_lines(p)
        assert len(lines) == 3, lines
        assert lines[0].endswith("000_s01_sh1.mp4'")
        assert lines[1].endswith("001_s01_sh2.mp4'")
        assert lines[2].endswith("002_s01_sh3.mp4'")
        assert r["motion_clips_used"] == 3
        assert r["transition_clips_used"] == 0


def test_run_editor_skips_unrendered_transition_rows():
    """Boundary 1 rendered, boundary 2 skipped: concat is
    motion1, trans1, motion2, motion3 and transition_clips_used == 1."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp)
        _seed_editor_fixture(p, transitions_mode="first_only")

        a, b, c = _editor_patches(agents)
        with a, b, c:
            r = agents.run_editor(p)

        lines = _read_concat_lines(p)
        assert len(lines) == 4, lines
        assert lines[0].endswith("000_s01_sh1.mp4'")
        assert lines[1].endswith("000_s01_sh1_T.mp4'")
        assert lines[2].endswith("001_s01_sh2.mp4'")
        assert lines[3].endswith("002_s01_sh3.mp4'")
        assert r["motion_clips_used"] == 3
        assert r["transition_clips_used"] == 1
