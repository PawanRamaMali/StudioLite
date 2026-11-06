"""Tests for audio_effects.py (denoise + equalizer + background music).

Covers:
- Filter-string construction for both denoise methods (afftdn clamps, anlmdn).
- Equalizer preset resolution, custom gains override, per-band clamp,
  zero-gain pruning, empty case falls back to ``anull``.
- Background music filter_complex graph shape for ducked and un-ducked
  mixes, music fade/trim when voice duration is known.
- Argv builders include correct codec flags for wav and mp3.
- Dispatcher failure modes (missing source, missing ffmpeg, non-zero exit,
  no output) and one success path per effect.
- probe_duration behaviour (missing ffprobe, non-zero exit, junk stdout).
"""
from __future__ import annotations

import os

import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_audio(tmp_path, name="in.wav"):
    p = tmp_path / name
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 128)
    return str(p)


def _touch(path, size=4096):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


# ---------------------------------------------------------------------------
# Denoise filter + argv
# ---------------------------------------------------------------------------


def test_denoise_filter_afftdn_shape():
    from audio_effects import DenoiseStyle, build_denoise_filter
    f = build_denoise_filter(DenoiseStyle(method="afftdn",
                                          noise_reduction_db=-25,
                                          noise_floor_db=-40))
    assert f.startswith("afftdn=nr=25.0:nf=-40.0")


def test_denoise_filter_afftdn_clamps_extremes():
    from audio_effects import DenoiseStyle, build_denoise_filter
    f = build_denoise_filter(DenoiseStyle(method="afftdn",
                                          noise_reduction_db=-500,  # -> clamp to 97
                                          noise_floor_db=-500))     # -> -80
    assert "nr=97.0" in f
    assert "nf=-80.0" in f


def test_denoise_filter_anlmdn_shape():
    from audio_effects import DenoiseStyle, build_denoise_filter
    f = build_denoise_filter(DenoiseStyle(method="anlmdn", smoothing=0.003))
    assert f.startswith("anlmdn=s=0.00300")


def test_denoise_filter_unknown_method_falls_back_to_afftdn():
    from audio_effects import DenoiseStyle, build_denoise_filter
    f = build_denoise_filter(DenoiseStyle(method="bogus"))
    assert f.startswith("afftdn=")


def test_build_denoise_cmd_wav():
    from audio_effects import DenoiseStyle, build_denoise_cmd
    cmd = build_denoise_cmd("in.wav", "out.wav", DenoiseStyle(output_format="wav"))
    assert cmd[0] == "ffmpeg"
    assert "-af" in cmd
    assert "pcm_s16le" in cmd
    assert cmd[-1] == "out.wav"


def test_build_denoise_cmd_mp3_switches_encoder():
    from audio_effects import DenoiseStyle, build_denoise_cmd
    cmd = build_denoise_cmd("in.wav", "out.mp3", DenoiseStyle(output_format="mp3"))
    assert "libmp3lame" in cmd
    assert "-b:a" in cmd
    assert "192k" in cmd


# ---------------------------------------------------------------------------
# Denoise dispatcher
# ---------------------------------------------------------------------------


def test_denoise_audio_missing_source(tmp_path):
    from audio_effects import denoise_audio
    r = denoise_audio(str(tmp_path / "ghost.wav"), str(tmp_path / "out.wav"))
    assert r["ok"] is False
    assert "not found" in r["error"].lower()


def test_denoise_audio_missing_ffmpeg(monkeypatch, tmp_path):
    from audio_effects import denoise_audio
    monkeypatch.setattr("audio_effects.shutil.which", lambda name: None)
    r = denoise_audio(_mk_audio(tmp_path), str(tmp_path / "out.wav"))
    assert r["ok"] is False
    assert r.get("install_hint")


def test_denoise_audio_non_zero_exit(monkeypatch, tmp_path):
    from audio_effects import denoise_audio
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffmpeg")

    def runner(cmd):
        return _FakeProc(returncode=1, stderr="afftdn: bad args")
    r = denoise_audio(_mk_audio(tmp_path), str(tmp_path / "out.wav"), runner=runner)
    assert r["ok"] is False
    assert "afftdn" in r["error"]


def test_denoise_audio_empty_output(monkeypatch, tmp_path):
    from audio_effects import denoise_audio
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffmpeg")
    out = str(tmp_path / "out.wav")

    def runner(cmd):
        return _FakeProc(returncode=0)   # does NOT produce the file
    r = denoise_audio(_mk_audio(tmp_path), out, runner=runner)
    assert r["ok"] is False
    assert "no output" in r["error"].lower()


def test_denoise_audio_success(monkeypatch, tmp_path):
    from audio_effects import denoise_audio
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffmpeg")
    out = str(tmp_path / "out.wav")

    def runner(cmd):
        _touch(out, 2048)
        return _FakeProc(returncode=0)

    r = denoise_audio(_mk_audio(tmp_path), out, runner=runner)
    assert r["ok"] is True
    assert r["backend"].startswith("ffmpeg:")
    assert r["output_path"] == out


# ---------------------------------------------------------------------------
# Equalizer
# ---------------------------------------------------------------------------


def test_equalizer_filter_flat_returns_anull():
    from audio_effects import EqStyle, build_equalizer_filter
    assert build_equalizer_filter(EqStyle(preset="flat")) == "anull"


def test_equalizer_filter_bass_boost_has_low_band_gain():
    from audio_effects import EqStyle, build_equalizer_filter
    f = build_equalizer_filter(EqStyle(preset="bass_boost"))
    assert "equalizer=f=31:t=q:w=1:g=8.00" in f
    # High bands at 0 dB are pruned
    assert "f=8000" not in f
    assert "f=16000" not in f


def test_equalizer_filter_custom_gains_override_preset():
    from audio_effects import EqStyle, build_equalizer_filter, EQ_BANDS_HZ
    gains = [0, 0, 0, 0, 6, 0, 0, 0, 0, 0]
    f = build_equalizer_filter(EqStyle(preset="rock", gains_db=gains))
    assert f == f"equalizer=f={EQ_BANDS_HZ[4]}:t=q:w=1:g=6.00"


def test_equalizer_filter_clamps_gain_to_18():
    from audio_effects import EqStyle, build_equalizer_filter
    # Request +30 dB on band 0; should clamp to +18
    f = build_equalizer_filter(EqStyle(gains_db=[30, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert "g=18.00" in f
    assert "g=30" not in f


def test_equalizer_unknown_preset_falls_back_to_flat():
    from audio_effects import EqStyle, build_equalizer_filter
    assert build_equalizer_filter(EqStyle(preset="imaginary")) == "anull"


def test_build_equalizer_cmd_contains_af_and_encoder():
    from audio_effects import EqStyle, build_equalizer_cmd
    cmd = build_equalizer_cmd("in.wav", "out.wav", EqStyle(preset="vocal_boost"))
    assert "-af" in cmd
    assert "pcm_s16le" in cmd


def test_equalize_audio_missing_source(tmp_path):
    from audio_effects import equalize_audio
    r = equalize_audio(str(tmp_path / "ghost.wav"), str(tmp_path / "out.wav"))
    assert r["ok"] is False


def test_equalize_audio_non_zero_exit(monkeypatch, tmp_path):
    from audio_effects import equalize_audio
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffmpeg")

    def runner(cmd):
        return _FakeProc(returncode=2, stderr="equalizer: parse")
    r = equalize_audio(_mk_audio(tmp_path), str(tmp_path / "out.wav"), runner=runner)
    assert r["ok"] is False
    assert "equalizer" in r["error"].lower()


def test_equalize_audio_success(monkeypatch, tmp_path):
    from audio_effects import equalize_audio, EqStyle
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffmpeg")
    out = str(tmp_path / "out.wav")

    def runner(cmd):
        _touch(out, 1024)
        return _FakeProc(returncode=0)

    r = equalize_audio(_mk_audio(tmp_path), out, style=EqStyle(preset="vocal_boost"), runner=runner)
    assert r["ok"] is True
    assert r["bands_hz"] and r["gains_db"]


# ---------------------------------------------------------------------------
# Background music mix
# ---------------------------------------------------------------------------


def test_bgmusic_filter_without_duck_has_single_amix():
    from audio_effects import BackgroundMusicStyle, build_background_music_filter_complex
    g = build_background_music_filter_complex(
        voice_duration=10.0,
        style=BackgroundMusicStyle(duck=False, fade_in=0.5, fade_out=0.5),
    )
    assert "[v0][m0]amix" in g
    assert "sidechaincompress" not in g
    # trim/fade applied to music bed
    assert "atrim=0:10.000" in g
    assert "afade=t=in:st=0:d=0.50" in g
    assert "afade=t=out:st=9.50:d=0.50" in g


def test_bgmusic_filter_with_duck_uses_sidechain():
    from audio_effects import BackgroundMusicStyle, build_background_music_filter_complex
    g = build_background_music_filter_complex(
        voice_duration=5.0,
        style=BackgroundMusicStyle(duck=True, duck_threshold=0.1, duck_ratio=6),
    )
    assert "sidechaincompress=threshold=0.100:ratio=6.0" in g
    assert "[va][md]amix" in g
    assert "asplit=2" in g


def test_bgmusic_filter_without_voice_duration_omits_trim():
    from audio_effects import BackgroundMusicStyle, build_background_music_filter_complex
    g = build_background_music_filter_complex(
        voice_duration=None, style=BackgroundMusicStyle(fade_in=0.2),
    )
    assert "atrim" not in g
    # fade_in still applied
    assert "afade=t=in" in g


def test_bgmusic_filter_clamps_volumes():
    from audio_effects import BackgroundMusicStyle, build_background_music_filter_complex
    g = build_background_music_filter_complex(
        voice_duration=1.0,
        style=BackgroundMusicStyle(music_volume=-5, voice_volume=99, duck=False),
    )
    # voice_volume clamped at 4.0, music_volume at 0.0
    assert "volume=4.000" in g
    assert "volume=0.000" in g


def test_build_bgmusic_cmd_loops_music_when_enabled():
    from audio_effects import BackgroundMusicStyle, build_background_music_cmd
    cmd = build_background_music_cmd(
        "voice.wav", "bg.mp3", "out.wav",
        voice_duration=12.5,
        style=BackgroundMusicStyle(loop_music=True),
    )
    assert "-stream_loop" in cmd
    assert "-1" in cmd
    assert cmd[-1] == "out.wav"
    assert "-filter_complex" in cmd
    assert "-map" in cmd
    i = cmd.index("-map")
    assert cmd[i + 1] == "[mix]"
    # When duration is known, -t is used rather than -shortest
    assert "-t" in cmd
    assert "-shortest" not in cmd


def test_build_bgmusic_cmd_no_loop_uses_shortest_when_no_duration():
    from audio_effects import BackgroundMusicStyle, build_background_music_cmd
    cmd = build_background_music_cmd(
        "voice.wav", "bg.mp3", "out.wav",
        voice_duration=None,
        style=BackgroundMusicStyle(loop_music=False),
    )
    assert "-shortest" in cmd
    assert "-stream_loop" not in cmd


def test_probe_duration_missing_ffprobe(monkeypatch):
    from audio_effects import probe_duration
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: None)
    assert probe_duration("x.wav") is None


def test_probe_duration_parses_value(monkeypatch):
    from audio_effects import probe_duration
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffprobe")

    def runner(cmd):
        return _FakeProc(returncode=0, stdout="12.345\n")
    assert probe_duration("x.wav", runner=runner) == pytest.approx(12.345)


def test_probe_duration_junk_stdout_returns_none(monkeypatch):
    from audio_effects import probe_duration
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffprobe")

    def runner(cmd):
        return _FakeProc(returncode=0, stdout="not-a-float")
    assert probe_duration("x.wav", runner=runner) is None


def test_probe_duration_non_zero_exit_returns_none(monkeypatch):
    from audio_effects import probe_duration
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/ffprobe")

    def runner(cmd):
        return _FakeProc(returncode=1, stdout="12.0", stderr="boom")
    assert probe_duration("x.wav", runner=runner) is None


def test_mix_background_music_missing_music(monkeypatch, tmp_path):
    from audio_effects import mix_background_music
    voice = _mk_audio(tmp_path, "voice.wav")
    r = mix_background_music(voice, str(tmp_path / "ghost.mp3"), str(tmp_path / "out.wav"))
    assert r["ok"] is False
    assert "music" in r["error"].lower()


def test_mix_background_music_missing_voice(tmp_path):
    from audio_effects import mix_background_music
    music = _mk_audio(tmp_path, "music.mp3")
    r = mix_background_music(str(tmp_path / "ghost.wav"), music, str(tmp_path / "out.wav"))
    assert r["ok"] is False
    assert "voice" in r["error"].lower()


def test_mix_background_music_success(monkeypatch, tmp_path):
    from audio_effects import mix_background_music, BackgroundMusicStyle
    voice = _mk_audio(tmp_path, "voice.wav")
    music = _mk_audio(tmp_path, "music.mp3")
    out = str(tmp_path / "out.wav")

    def which(name):
        return "/bin/" + name
    monkeypatch.setattr("audio_effects.shutil.which", which)

    def runner(cmd):
        if cmd[0].endswith("ffprobe") or "ffprobe" in cmd[0]:
            return _FakeProc(returncode=0, stdout="7.5")
        _touch(out, 2048)
        return _FakeProc(returncode=0)

    r = mix_background_music(voice, music, out,
                             style=BackgroundMusicStyle(duck=True), runner=runner)
    assert r["ok"] is True
    assert r["backend"].endswith("ducked")
    assert r["duration"] == pytest.approx(7.5)


def test_mix_background_music_non_zero_exit(monkeypatch, tmp_path):
    from audio_effects import mix_background_music
    voice = _mk_audio(tmp_path, "voice.wav")
    music = _mk_audio(tmp_path, "music.mp3")
    monkeypatch.setattr("audio_effects.shutil.which", lambda n: "/bin/" + n)
    calls = {"n": 0}

    def runner(cmd):
        calls["n"] += 1
        if calls["n"] == 1:  # ffprobe
            return _FakeProc(returncode=0, stdout="5.0")
        return _FakeProc(returncode=1, stderr="amix: fatal")

    r = mix_background_music(voice, music, str(tmp_path / "out.wav"), runner=runner)
    assert r["ok"] is False
    assert "amix" in r["error"].lower()


# ---------------------------------------------------------------------------
# Endpoint smoke tests
# ---------------------------------------------------------------------------


def _fresh_api(monkeypatch, tmp_path):
    import importlib
    import sys as _sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", "off")
    for name in list(_sys.modules):
        if name == "api_server" or name.startswith("api_server."):
            del _sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


class _Immediate:
    def __init__(self, *, target, args, daemon=False):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


_WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 128
_MP3_BYTES = b"ID3\x03\x00\x00\x00" + b"\x00" * 128


def test_denoise_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)

    seen = {}

    def fake_run(job_id, saved_path, style):
        seen["method"] = style.method
        seen["nr"] = style.noise_reduction_db
    monkeypatch.setattr(api, "_run_denoise", fake_run, raising=True)
    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/audio/denoise?method=anlmdn&noise_reduction_db=-15",
        files={"file": ("tiny.wav", _WAV_BYTES, "audio/wav")},
    )
    assert r.status_code == 200
    assert seen["method"] == "anlmdn"
    assert seen["nr"] == -15.0


def test_equalizer_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)

    seen = {}

    def fake_run(job_id, saved_path, style):
        seen["preset"] = style.preset
    monkeypatch.setattr(api, "_run_equalizer", fake_run, raising=True)
    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/audio/equalizer?preset=rock",
        files={"file": ("tiny.wav", _WAV_BYTES, "audio/wav")},
    )
    assert r.status_code == 200
    assert seen["preset"] == "rock"


def test_bgmusic_endpoint_requires_both_files(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/audio/background-music",
        files={"voice": ("v.wav", _WAV_BYTES, "audio/wav")},
    )
    assert r.status_code in (400, 422)


def test_bgmusic_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)

    seen = {}

    def fake_run(job_id, voice_path, music_path, style):
        seen["voice"] = voice_path
        seen["music"] = music_path
        seen["music_volume"] = style.music_volume
    monkeypatch.setattr(api, "_run_background_music", fake_run, raising=True)
    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/audio/background-music?music_volume=0.5&duck=false",
        files={
            "voice": ("v.wav", _WAV_BYTES, "audio/wav"),
            "music": ("m.mp3", _MP3_BYTES, "audio/mpeg"),
        },
    )
    assert r.status_code == 200
    assert seen["music_volume"] == 0.5
