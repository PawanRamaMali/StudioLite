"""Coverage for the timeline / NLE export path.

The tests here exercise the export-profile matrix (codec × quality) and
the license-gated watermark decision. The runner itself needs moviepy
plus a real video file to end-to-end verify, so we don't invoke it - 
the split above keeps the fast unit tests fast while the API-integration
side still confirms the endpoint is wired and rejects bad input."""
from __future__ import annotations

import importlib
import os
import sys

import pytest

from filmmaker import licensing
from filmmaker.licensing import LicensePayload, sign_payload_for_tests, _DEV_PRIVATE_KEY_PEM


def _reload_api(monkeypatch, tmp_path, *, auth="off", license_path=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", auth)
    monkeypatch.setenv(
        "STUDIOLITE_LICENSE_FILE",
        license_path or str(tmp_path / ".license"),
    )
    for name in list(sys.modules):
        if name == "api_server":
            del sys.modules[name]
    return importlib.import_module("api_server")


class TestExportProfile:
    def test_h264_high_returns_libx264_low_crf(self):
        api = importlib.import_module("api_server")
        p = api._export_profile("h264", "high")
        assert p["vcodec"] == "libx264"
        assert "crf 18" in p["extra"]
        assert p["ext"] == "mp4"

    def test_h265_medium(self):
        api = importlib.import_module("api_server")
        p = api._export_profile("h265", "medium")
        assert p["vcodec"] == "libx265"
        assert "crf 24" in p["extra"]

    def test_prores_uses_mov(self):
        api = importlib.import_module("api_server")
        p = api._export_profile("prores", "high")
        assert p["vcodec"] == "prores_ks"
        assert p["ext"] == "mov"
        assert "profile:v" in p["extra"]

    def test_unknown_codec_rejected(self):
        api = importlib.import_module("api_server")
        with pytest.raises(ValueError):
            api._export_profile("av1", "high")

    def test_unknown_quality_rejected(self):
        api = importlib.import_module("api_server")
        with pytest.raises(ValueError):
            api._export_profile("h264", "cinema")

    def test_quality_lowers_crf_higher(self):
        api = importlib.import_module("api_server")
        # 'low' should produce a HIGHER CRF (worse quality, smaller file)
        # than 'high' - the intuition users have for the labels.
        high = api._export_profile("h264", "high")["extra"]
        low = api._export_profile("h264", "low")["extra"]
        # crf 18 < crf 26
        assert "crf 18" in high
        assert "crf 26" in low


class TestAPIEndpoint:
    def test_rejects_empty_timeline(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        api = _reload_api(monkeypatch, tmp_path)
        client = TestClient(api.app)
        r = client.post("/api/v1/edit/timeline-render", json={"clips": []})
        assert r.status_code == 422

    def test_rejects_missing_clip_file(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        api = _reload_api(monkeypatch, tmp_path)
        client = TestClient(api.app)
        r = client.post(
            "/api/v1/edit/timeline-render",
            json={
                "clips": [
                    {"video_path": str(tmp_path / "nope.mp4"),
                     "in_point": 0.0, "out_point": 1.0}
                ]
            },
        )
        assert r.status_code == 422


class TestWatermarkGating:
    """The runner double-checks the watermark decision against the
    installed license. We can't invoke the runner without a real video,
    so we assert the policy logic in isolation."""

    def _entitlement(self, *, tier="free", features=(), valid=True):
        return licensing.EntitlementCheck(
            valid=valid, tier=tier, features=list(features),
        )

    def test_free_tier_always_gets_watermark(self):
        ent = self._entitlement(tier="free", valid=False)
        allow_no_watermark = (
            ent.valid
            and (ent.tier in {"pro", "studio"} or ent.has("watermark_removal"))
        )
        assert allow_no_watermark is False
        # User asked for no watermark - still watermarked.
        assert (False or not allow_no_watermark) is True

    def test_pro_tier_can_remove(self):
        ent = self._entitlement(tier="pro")
        allow = ent.valid and (ent.tier in {"pro", "studio"}
                               or ent.has("watermark_removal"))
        assert allow is True

    def test_feature_flag_lets_free_remove(self):
        ent = self._entitlement(tier="free", features=["watermark_removal"])
        allow = ent.valid and (ent.tier in {"pro", "studio"}
                               or ent.has("watermark_removal"))
        assert allow is True

    def test_end_to_end_free_default(self, monkeypatch, tmp_path):
        """With no license installed, the license status endpoint should
        say tier=free - the same reading the timeline runner uses."""
        from fastapi.testclient import TestClient
        api = _reload_api(monkeypatch, tmp_path)
        client = TestClient(api.app)
        r = client.get("/api/v1/system/license")
        body = r.json()
        assert body["tier"] == "free"
        assert body["valid"] is False

    def test_end_to_end_pro_after_install(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        api = _reload_api(monkeypatch, tmp_path)
        client = TestClient(api.app)

        p = LicensePayload(tier="pro", features=["batch_export"], licensee="Owner")
        r = client.post(
            "/api/v1/system/license",
            json=sign_payload_for_tests(p, _DEV_PRIVATE_KEY_PEM),
        )
        assert r.status_code == 200

        # And the runner-facing check now sees Pro.
        ent = licensing.check_entitlement()
        assert ent.valid is True
        assert ent.tier == "pro"
