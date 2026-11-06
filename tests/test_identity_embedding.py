"""Tests for identity_embedding.py (PuLID + IP-Adapter fallback)."""
from __future__ import annotations

import os
import pytest


def _mk_png(tmp_path, name="ref.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    return str(p)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_missing_reference_raises(tmp_path):
    from identity_embedding import IdentityConfig, _validate_config
    cfg = IdentityConfig(reference_image=str(tmp_path / "missing.png"),
                         prompt="x")
    with pytest.raises(FileNotFoundError):
        _validate_config(cfg)


def test_validate_empty_prompt_raises(tmp_path):
    from identity_embedding import IdentityConfig, _validate_config
    cfg = IdentityConfig(reference_image=_mk_png(tmp_path), prompt="")
    with pytest.raises(ValueError, match="prompt"):
        _validate_config(cfg)


def test_validate_dims_must_be_multiple_of_8(tmp_path):
    from identity_embedding import IdentityConfig, _validate_config
    cfg = IdentityConfig(reference_image=_mk_png(tmp_path), prompt="x",
                         width=777, height=1024)
    with pytest.raises(ValueError, match="multiples of 8"):
        _validate_config(cfg)


def test_validate_rejects_webp(tmp_path):
    from identity_embedding import IdentityConfig, _validate_config
    ref = tmp_path / "ref.webp"
    ref.write_bytes(b"\x00" * 256)
    cfg = IdentityConfig(reference_image=str(ref), prompt="x")
    with pytest.raises(ValueError, match="png"):
        _validate_config(cfg)


def test_validate_id_scale_bounds(tmp_path):
    from identity_embedding import IdentityConfig, _validate_config
    cfg = IdentityConfig(reference_image=_mk_png(tmp_path), prompt="x",
                         id_scale=5.0)
    with pytest.raises(ValueError, match="id_scale"):
        _validate_config(cfg)


def test_validate_passes_for_sane_config(tmp_path):
    from identity_embedding import IdentityConfig, _validate_config
    cfg = IdentityConfig(reference_image=_mk_png(tmp_path), prompt="x")
    _validate_config(cfg)  # no raise


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------


def test_resolve_output_path_explicit_wins(tmp_path):
    from identity_embedding import IdentityConfig, _resolve_output_path
    out = str(tmp_path / "forced.png")
    cfg = IdentityConfig(reference_image=_mk_png(tmp_path), prompt="x",
                         output_path=out)
    assert _resolve_output_path(cfg) == out


def test_resolve_output_path_default_sits_beside_ref(tmp_path):
    from identity_embedding import IdentityConfig, _resolve_output_path
    ref = _mk_png(tmp_path, "target.png")
    cfg = IdentityConfig(reference_image=ref, prompt="x")
    out = _resolve_output_path(cfg)
    assert os.path.dirname(out) == os.path.dirname(ref)
    assert out.endswith(".png")
    assert "target" in out


def test_resolve_output_path_varies_with_prompt(tmp_path):
    from identity_embedding import IdentityConfig, _resolve_output_path
    ref = _mk_png(tmp_path)
    a = _resolve_output_path(IdentityConfig(reference_image=ref, prompt="cat"))
    b = _resolve_output_path(IdentityConfig(reference_image=ref, prompt="dog"))
    assert a != b


# ---------------------------------------------------------------------------
# generate_with_identity dispatch (runner-injected)
# ---------------------------------------------------------------------------


def _writing_runner(expected_tag: str):
    """Return a runner callable that records how it was called and writes a stub file."""
    log: dict = {"calls": 0, "tag": None, "paths": []}

    def _run(cfg, out_path):
        log["calls"] += 1
        log["tag"] = expected_tag
        log["paths"].append(out_path)
        with open(out_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048)
        return out_path

    return _run, log


def test_dispatch_uses_pulid_when_available(tmp_path, monkeypatch):
    from identity_embedding import IdentityConfig, generate_with_identity
    import identity_embedding as ie
    monkeypatch.setattr(ie, "pulid_available", lambda: True)
    runner, log = _writing_runner("pulid")
    out = str(tmp_path / "forced.png")
    cfg = IdentityConfig(
        reference_image=_mk_png(tmp_path), prompt="portrait",
        output_path=out,
    )
    r = generate_with_identity(cfg, runner=runner)
    assert r == out
    assert log["calls"] == 1


def test_dispatch_falls_back_when_pulid_unavailable(tmp_path, monkeypatch):
    from identity_embedding import IdentityConfig, generate_with_identity
    import identity_embedding as ie
    monkeypatch.setattr(ie, "pulid_available", lambda: False)
    runner, _log = _writing_runner("fallback")
    out = str(tmp_path / "forced.png")
    cfg = IdentityConfig(
        reference_image=_mk_png(tmp_path), prompt="portrait",
        output_path=out,
    )
    r = generate_with_identity(cfg, runner=runner)
    assert os.path.isfile(r)


def test_dispatch_raises_when_output_missing(tmp_path, monkeypatch):
    from identity_embedding import IdentityConfig, generate_with_identity
    import identity_embedding as ie
    monkeypatch.setattr(ie, "pulid_available", lambda: True)
    def _noop_runner(_cfg, out_path):
        # Simulate a failed sampling run - no output written
        return out_path
    cfg = IdentityConfig(
        reference_image=_mk_png(tmp_path), prompt="x",
        output_path=str(tmp_path / "nothing.png"),
    )
    with pytest.raises(RuntimeError, match="no usable output"):
        generate_with_identity(cfg, runner=_noop_runner)


def test_install_hint_mentions_pulid_components():
    from identity_embedding import install_hint
    h = install_hint()
    assert "PuLID" in h
