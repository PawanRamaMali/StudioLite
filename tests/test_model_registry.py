"""Tests for the cross-backend model registry.

Covers:
  - every REGISTRY entry has the required fields populated,
  - probe_model on a non-existent path reports present=False,
  - probe_model on a real file reports present=True + its byte size,
  - the HTTP endpoints behind /api/v1/models/registry and
    /api/v1/models/{id}/open-folder serialize the probe cleanly.

Nothing in here hits the network. The registry probe is a pure
filesystem check, which is the invariant the UI relies on when it
calls it on tab-open.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

from models.registry import (
    REGISTRY,
    ModelSpec,
    expected_path_for,
    probe_model,
    probe_registry,
    registry_summary,
)


# ---------------------------------------------------------------------------
# Registry data invariants
# ---------------------------------------------------------------------------

REQUIRED_KINDS = {"text", "image", "video", "audio", "face", "other"}


class TestRegistryShape:
    def test_registry_is_non_empty(self):
        # The deliverable asks for a minimum of 10 entries - we ship 20+.
        assert len(REGISTRY) >= 10

    def test_registry_keys_match_spec_ids(self):
        for key, spec in REGISTRY.items():
            assert key == spec.id, (
                f"REGISTRY dict key {key!r} must match ModelSpec.id {spec.id!r}"
            )

    def test_every_entry_has_required_fields(self):
        for spec in REGISTRY.values():
            assert isinstance(spec, ModelSpec)
            assert spec.id and isinstance(spec.id, str)
            assert spec.name and isinstance(spec.name, str)
            assert spec.description and isinstance(spec.description, str)
            assert spec.kind in REQUIRED_KINDS, f"bad kind: {spec.kind} on {spec.id}"
            # expected_path is either a non-empty str or a callable.
            assert spec.expected_path and (
                isinstance(spec.expected_path, str) or callable(spec.expected_path)
            )
            # size floor + source URL both populated so the UI can render.
            assert isinstance(spec.min_size_bytes, int) and spec.min_size_bytes > 0
            assert spec.source_url.startswith("http"), spec.source_url
            assert spec.backend_tag and isinstance(spec.backend_tag, str)

    def test_expected_path_resolves_to_absolute_string(self):
        # Every entry - callable or not - must resolve to an abspath string.
        for spec in REGISTRY.values():
            p = expected_path_for(spec)
            assert isinstance(p, str) and p
            assert os.path.isabs(p), f"{spec.id}: expected absolute path, got {p}"

    def test_core_models_present(self):
        # The brief lists these as the minimum required coverage.
        for expected_id in [
            "clip_vit_b32",
            "whisper_tiny", "whisper_base", "whisper_small", "whisper_medium",
            "yunet", "sface",
            "sdxl_turbo",
            "wan22_ti2v_5b",
            "musicgen_small",
            "audioldm2",
            "realesrgan_x2", "realesrgan_x4",
            "piper_amy",
        ]:
            assert expected_id in REGISTRY, f"missing core model: {expected_id}"


# ---------------------------------------------------------------------------
# probe_model behavior
# ---------------------------------------------------------------------------

class TestProbe:
    def test_missing_path_reports_absent(self, tmp_path):
        bogus = tmp_path / "does_not_exist.bin"
        spec = ModelSpec(
            id="probe_missing",
            name="probe test (missing)",
            kind="other",
            description="fixture",
            expected_path=str(bogus),
            min_size_bytes=1,
            source_url="https://example.com",
            backend_tag="test",
        )
        r = probe_model(spec)
        assert r["present"] is False
        assert r["size_bytes"] == 0
        assert r["path"] == str(bogus)
        assert r["matched_path"] is None

    def test_real_file_reports_present_with_size(self, tmp_path):
        f = tmp_path / "weights.bin"
        payload = b"x" * 4096
        f.write_bytes(payload)
        spec = ModelSpec(
            id="probe_file",
            name="probe test (file)",
            kind="other",
            description="fixture",
            expected_path=str(f),
            # Set floor well below payload so present-check passes.
            min_size_bytes=100,
            source_url="https://example.com",
            backend_tag="test",
        )
        r = probe_model(spec)
        assert r["present"] is True
        assert r["size_bytes"] == len(payload)
        assert r["matched_path"] == str(f)

    def test_file_below_min_size_reports_absent(self, tmp_path):
        # A half-finished download should NOT count as present. The probe
        # uses min_size_bytes to flag that case.
        f = tmp_path / "partial.bin"
        f.write_bytes(b"x" * 10)
        spec = ModelSpec(
            id="probe_partial",
            name="probe test (partial)",
            kind="other",
            description="fixture",
            expected_path=str(f),
            min_size_bytes=1024 * 1024,  # 1 MB floor
            source_url="https://example.com",
            backend_tag="test",
        )
        r = probe_model(spec)
        assert r["present"] is False
        # Size still reported honestly - the UI uses it in a hover tooltip.
        assert r["size_bytes"] == 10

    def test_directory_sums_file_sizes(self, tmp_path):
        d = tmp_path / "modeldir"
        d.mkdir()
        (d / "a.bin").write_bytes(b"a" * 100)
        (d / "sub").mkdir()
        (d / "sub" / "b.bin").write_bytes(b"b" * 200)
        spec = ModelSpec(
            id="probe_dir",
            name="probe test (dir)",
            kind="other",
            description="fixture",
            expected_path=str(d),
            min_size_bytes=50,
            source_url="https://example.com",
            backend_tag="test",
        )
        r = probe_model(spec)
        assert r["present"] is True
        assert r["size_bytes"] == 300

    def test_callable_expected_path_is_evaluated(self, tmp_path):
        f = tmp_path / "lazy.bin"
        f.write_bytes(b"y" * 2048)
        spec = ModelSpec(
            id="probe_callable",
            name="probe test (callable)",
            kind="other",
            description="fixture",
            expected_path=lambda: str(f),
            min_size_bytes=100,
            source_url="https://example.com",
            backend_tag="test",
        )
        r = probe_model(spec)
        assert r["present"] is True
        assert r["size_bytes"] == 2048

    def test_alt_paths_matched_when_primary_missing(self, tmp_path):
        primary = tmp_path / "missing.bin"
        alt = tmp_path / "alt.bin"
        alt.write_bytes(b"z" * 500)
        spec = ModelSpec(
            id="probe_alt",
            name="probe test (alt)",
            kind="other",
            description="fixture",
            expected_path=str(primary),
            min_size_bytes=100,
            source_url="https://example.com",
            backend_tag="test",
            alt_paths=[lambda: str(alt)],
        )
        r = probe_model(spec)
        assert r["present"] is True
        assert r["matched_path"] == str(alt)
        # The primary path is still reported so the UI can show it as
        # the "canonical" expected location.
        assert r["path"] == str(primary)


# ---------------------------------------------------------------------------
# Summary + probe_registry
# ---------------------------------------------------------------------------

class TestProbeRegistry:
    def test_probe_registry_returns_one_row_per_spec(self):
        rows = probe_registry()
        assert len(rows) == len(REGISTRY)
        ids = {r["id"] for r in rows}
        assert ids == set(REGISTRY.keys())

    def test_registry_summary_fields(self):
        s = registry_summary()
        assert set(s.keys()) == {"total", "present", "missing", "bytes_on_disk"}
        assert s["total"] == len(REGISTRY)
        assert s["present"] + s["missing"] == s["total"]
        assert isinstance(s["bytes_on_disk"], int) and s["bytes_on_disk"] >= 0

    def test_probe_registry_with_custom_dict(self, tmp_path):
        f = tmp_path / "weights.bin"
        f.write_bytes(b"x" * 2048)
        spec = ModelSpec(
            id="probe_custom",
            name="probe test (custom)",
            kind="other",
            description="fixture",
            expected_path=str(f),
            min_size_bytes=100,
            source_url="https://example.com",
            backend_tag="test",
        )
        rows = probe_registry({"probe_custom": spec})
        assert len(rows) == 1
        assert rows[0]["id"] == "probe_custom"
        assert rows[0]["present"] is True
        assert rows[0]["size_bytes"] == 2048


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def _client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", "off")
    for name in list(sys.modules):
        if name == "api_server":
            del sys.modules[name]
    api = importlib.import_module("api_server")
    return api, TestClient(api.app)


class TestEndpoints:
    def test_registry_endpoint_returns_models_and_summary(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.get("/api/v1/models/registry")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "models" in body and "summary" in body
        assert len(body["models"]) == len(REGISTRY)
        for row in body["models"]:
            # Fields the UI relies on.
            for key in (
                "id", "name", "kind", "description", "expected_path",
                "present", "size_bytes", "min_size_bytes", "source_url",
                "backend_tag",
            ):
                assert key in row, f"missing {key!r} in {row['id']}"
        s = body["summary"]
        assert s["total"] == len(REGISTRY)
        assert s["present"] + s["missing"] == s["total"]

    def test_open_folder_returns_parent_dir(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        # Pick any model id - CLIP is guaranteed present per the spec test.
        r = client.post("/api/v1/models/clip_vit_b32/open-folder")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_id"] == "clip_vit_b32"
        # The expected path points at the HF-hub cache dir for this repo;
        # the folder we hand back is a real string path.
        assert body["folder"] and isinstance(body["folder"], str)
        assert "expected_path" in body
        assert isinstance(body["exists"], bool)

    def test_open_folder_unknown_id_404s(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.post("/api/v1/models/no_such_model_xyz/open-folder")
        assert r.status_code == 404
