"""Tests for talking_head.py (audio-driven talking-head wrappers)."""
from __future__ import annotations

import os
import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_png(tmp_path, name="portrait.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    return str(p)


def _mk_wav(tmp_path, name="voice.wav"):
    p = tmp_path / name
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 128)
    return str(p)


# ---------------------------------------------------------------------------
# argv builders (pure; no weights needed)
# ---------------------------------------------------------------------------


def test_sadtalker_argv_core_flags(tmp_path):
    from talking_head import build_sadtalker_argv
    img = _mk_png(tmp_path)
    aud = _mk_wav(tmp_path)
    out = str(tmp_path)
    argv = build_sadtalker_argv(img, aud, out)
    assert "--source_image" in argv and img in argv
    assert "--driven_audio" in argv and aud in argv
    assert "--result_dir" in argv and out in argv
    assert "--enhancer" in argv


def test_sadtalker_argv_exposes_expression_scale(tmp_path):
    from talking_head import build_sadtalker_argv
    argv = build_sadtalker_argv(
        _mk_png(tmp_path), _mk_wav(tmp_path), str(tmp_path),
        expression_scale=1.4,
    )
    i = argv.index("--expression_scale")
    assert argv[i + 1] == "1.4"


def test_sadtalker_argv_still_flag_toggle(tmp_path):
    from talking_head import build_sadtalker_argv
    with_still = build_sadtalker_argv(_mk_png(tmp_path), _mk_wav(tmp_path),
                                      str(tmp_path), still=True)
    no_still = build_sadtalker_argv(_mk_png(tmp_path), _mk_wav(tmp_path),
                                    str(tmp_path), still=False)
    assert "--still" in with_still and "--no-still" in no_still


def test_hallo3_argv_core_flags(tmp_path):
    from talking_head import build_hallo3_argv
    img = _mk_png(tmp_path)
    aud = _mk_wav(tmp_path)
    out = str(tmp_path / "out.mp4")
    argv = build_hallo3_argv(img, aud, out)
    assert "--source_image" in argv and img in argv
    assert "--driving_audio" in argv and aud in argv
    assert "--output" in argv and out in argv
    assert "--checkpoint" in argv


def test_hallo3_argv_inference_knobs(tmp_path):
    from talking_head import build_hallo3_argv
    argv = build_hallo3_argv(
        _mk_png(tmp_path), _mk_wav(tmp_path), str(tmp_path / "o.mp4"),
        inference_steps=40, guidance_scale=3.5,
    )
    i = argv.index("--inference_steps")
    assert argv[i + 1] == "40"
    j = argv.index("--cfg_scale")
    assert argv[j + 1] == "3.5"


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_validate_missing_image_raises(tmp_path):
    from talking_head import _validate_inputs
    with pytest.raises(FileNotFoundError):
        _validate_inputs(str(tmp_path / "nope.png"), _mk_wav(tmp_path))


def test_validate_missing_audio_raises(tmp_path):
    from talking_head import _validate_inputs
    with pytest.raises(FileNotFoundError):
        _validate_inputs(_mk_png(tmp_path), str(tmp_path / "nope.wav"))


def test_validate_rejects_webp(tmp_path):
    from talking_head import _validate_inputs
    webp = tmp_path / "shot.webp"
    webp.write_bytes(b"\x00" * 512)
    with pytest.raises(ValueError, match="png"):
        _validate_inputs(str(webp), _mk_wav(tmp_path))


# ---------------------------------------------------------------------------
# generate_talking_head orchestration (runner-injected; no actual inference)
# ---------------------------------------------------------------------------


def test_generate_without_any_backend_raises(tmp_path, monkeypatch):
    from talking_head import generate_talking_head
    import talking_head as th
    monkeypatch.setattr(th, "sadtalker_available", lambda: False)
    monkeypatch.setattr(th, "hallo3_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs ONE"):
        generate_talking_head(_mk_png(tmp_path), _mk_wav(tmp_path),
                              runner=lambda _a: _FakeProc(0))


def test_generate_auto_prefers_hallo3_when_available(tmp_path, monkeypatch):
    from talking_head import generate_talking_head
    import talking_head as th
    monkeypatch.setattr(th, "sadtalker_available", lambda: True)
    monkeypatch.setattr(th, "hallo3_available", lambda: True)
    picked = {}
    def _fake_runner(argv):
        # Inspect which inference script is being called
        picked["argv"] = argv
        # Pretend the output appeared
        out = argv[argv.index("--output") + 1]
        with open(out, "wb") as f:
            f.write(b"\x00" * 2048)
        return _FakeProc(0)
    out_path = str(tmp_path / "result.mp4")
    r = generate_talking_head(_mk_png(tmp_path), _mk_wav(tmp_path), out_path,
                              runner=_fake_runner)
    assert "hallo3" in picked["argv"][1] or "inference_long" in picked["argv"][1]
    assert r == out_path


def test_generate_falls_back_to_sadtalker(tmp_path, monkeypatch):
    from talking_head import generate_talking_head
    import talking_head as th
    monkeypatch.setattr(th, "sadtalker_available", lambda: True)
    monkeypatch.setattr(th, "hallo3_available", lambda: False)
    out_path = str(tmp_path / "result.mp4")

    def _fake_runner(argv):
        # SadTalker writes into result_dir
        out_dir = argv[argv.index("--result_dir") + 1]
        subdir = os.path.join(out_dir, "20260101_000000")
        os.makedirs(subdir, exist_ok=True)
        fake_out = os.path.join(subdir, "final.mp4")
        with open(fake_out, "wb") as f:
            f.write(b"\x00" * 4096)
        return _FakeProc(0)

    r = generate_talking_head(_mk_png(tmp_path), _mk_wav(tmp_path), out_path,
                              backend="auto", runner=_fake_runner)
    assert os.path.isfile(r)


def test_generate_explicit_backend_not_installed_raises(tmp_path, monkeypatch):
    from talking_head import generate_talking_head
    import talking_head as th
    monkeypatch.setattr(th, "sadtalker_available", lambda: True)
    monkeypatch.setattr(th, "hallo3_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        generate_talking_head(_mk_png(tmp_path), _mk_wav(tmp_path),
                              backend="hallo3",
                              runner=lambda _a: _FakeProc(0))


def test_generate_surfaces_subprocess_failure(tmp_path, monkeypatch):
    from talking_head import generate_talking_head
    import talking_head as th
    monkeypatch.setattr(th, "sadtalker_available", lambda: False)
    monkeypatch.setattr(th, "hallo3_available", lambda: True)
    with pytest.raises(RuntimeError, match="inference failed"):
        generate_talking_head(
            _mk_png(tmp_path), _mk_wav(tmp_path), str(tmp_path / "o.mp4"),
            runner=lambda _a: _FakeProc(1, "", "CUDA out of memory"),
        )


def test_available_backends_returns_list(monkeypatch):
    import talking_head as th
    monkeypatch.setattr(th, "sadtalker_available", lambda: True)
    monkeypatch.setattr(th, "hallo3_available", lambda: False)
    assert th.available_backends() == ["sadtalker"]
    monkeypatch.setattr(th, "sadtalker_available", lambda: False)
    monkeypatch.setattr(th, "hallo3_available", lambda: True)
    assert th.available_backends() == ["hallo3"]
    monkeypatch.setattr(th, "sadtalker_available", lambda: False)
    monkeypatch.setattr(th, "hallo3_available", lambda: False)
    assert th.available_backends() == []


def test_install_hint_mentions_both_backends():
    from talking_head import install_hint
    h = install_hint()
    assert "SadTalker" in h and "Hallo3" in h
