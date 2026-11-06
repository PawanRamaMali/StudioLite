"""Tests for ambient_audio.py (AudioLDM2 / Stable Audio Open wrapper)."""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_empty_prompt():
    from ambient_audio import AmbientConfig, _validate_config
    with pytest.raises(ValueError, match="prompt"):
        _validate_config(AmbientConfig(prompt=""))


def test_validate_whitespace_prompt():
    from ambient_audio import AmbientConfig, _validate_config
    with pytest.raises(ValueError, match="prompt"):
        _validate_config(AmbientConfig(prompt="   "))


def test_validate_duration_bounds():
    from ambient_audio import AmbientConfig, _validate_config
    with pytest.raises(ValueError, match="duration"):
        _validate_config(AmbientConfig(prompt="x", duration_s=0))
    with pytest.raises(ValueError, match="duration"):
        _validate_config(AmbientConfig(prompt="x", duration_s=61))


def test_validate_sample_rate_must_be_standard():
    from ambient_audio import AmbientConfig, _validate_config
    with pytest.raises(ValueError, match="sample_rate"):
        _validate_config(AmbientConfig(prompt="x", sample_rate=12345))


def test_validate_inference_steps_bounds():
    from ambient_audio import AmbientConfig, _validate_config
    with pytest.raises(ValueError, match="num_inference_steps"):
        _validate_config(AmbientConfig(prompt="x", num_inference_steps=2))
    with pytest.raises(ValueError, match="num_inference_steps"):
        _validate_config(AmbientConfig(prompt="x", num_inference_steps=500))


def test_validate_accepts_sane_config():
    from ambient_audio import AmbientConfig, _validate_config
    _validate_config(AmbientConfig(prompt="forest rain", duration_s=5.0))


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------


def test_resolve_output_path_uses_explicit(tmp_path):
    from ambient_audio import AmbientConfig, _resolve_output_path
    out = str(tmp_path / "foo.wav")
    assert _resolve_output_path(AmbientConfig(prompt="x", output_path=out)) == out


def test_resolve_output_path_default_lives_under_mp():
    from ambient_audio import AmbientConfig, _resolve_output_path
    p = _resolve_output_path(AmbientConfig(prompt="x"))
    assert p.endswith(".wav")
    assert ".mp" in p.replace("\\", "/")


def test_resolve_output_path_varies_with_prompt():
    from ambient_audio import AmbientConfig, _resolve_output_path
    a = _resolve_output_path(AmbientConfig(prompt="forest"))
    b = _resolve_output_path(AmbientConfig(prompt="desert"))
    assert a != b


# ---------------------------------------------------------------------------
# pick_backend
# ---------------------------------------------------------------------------


def test_pick_backend_auto_prefers_stable_audio(monkeypatch):
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: True)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: True)
    assert aa.pick_backend("auto") == "stable_audio"


def test_pick_backend_auto_falls_back_to_audioldm2(monkeypatch):
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: True)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    assert aa.pick_backend("auto") == "audioldm2"


def test_pick_backend_none_available_raises(monkeypatch):
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: False)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs ONE"):
        aa.pick_backend("auto")


def test_pick_backend_explicit_missing_raises(monkeypatch):
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: True)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        aa.pick_backend("stable_audio")


def test_available_backends_empty(monkeypatch):
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: False)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    assert aa.available_backends() == []


# ---------------------------------------------------------------------------
# write_silent_wav + generate_ambient dispatch
# ---------------------------------------------------------------------------


def test_write_silent_wav_round_trip(tmp_path):
    from ambient_audio import write_silent_wav
    import wave
    p = str(tmp_path / "s.wav")
    write_silent_wav(p, duration_s=0.5, sample_rate=22050)
    with wave.open(p, "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0


def test_generate_ambient_uses_runner(tmp_path, monkeypatch):
    from ambient_audio import AmbientConfig, generate_ambient, write_silent_wav
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: True)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    calls = {}

    def _fake_runner(cfg, out_path, backend, _cb):
        calls["backend"] = backend
        calls["out"] = out_path
        write_silent_wav(out_path, cfg.duration_s, cfg.sample_rate)

    out = str(tmp_path / "amb.wav")
    r = generate_ambient(
        AmbientConfig(prompt="distant traffic", duration_s=1.0, output_path=out),
        runner=_fake_runner,
    )
    assert r == out
    assert os.path.isfile(r)
    assert calls["backend"] == "audioldm2"


def test_generate_ambient_raises_if_no_output_written(tmp_path, monkeypatch):
    from ambient_audio import AmbientConfig, generate_ambient
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: True)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)

    def _noop(_cfg, _out, _backend, _cb):
        pass  # no file written

    with pytest.raises(RuntimeError, match="no usable output"):
        generate_ambient(
            AmbientConfig(prompt="x", duration_s=1.0,
                          output_path=str(tmp_path / "nothing.wav")),
            runner=_noop,
        )


def test_install_hint_mentions_backends(monkeypatch):
    # Force "nothing installed" so we exercise the hint text path.
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: False)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    monkeypatch.setattr(aa, "_has_diffusers", lambda: False)
    h = aa.install_hint()
    assert "audioldm2" in h.lower() or "stable audio" in h.lower()


def test_install_hint_reports_ready_when_installed(monkeypatch):
    import ambient_audio as aa
    monkeypatch.setattr(aa, "audioldm2_available", lambda: True)
    monkeypatch.setattr(aa, "stable_audio_available", lambda: False)
    assert "set up" in aa.install_hint()
