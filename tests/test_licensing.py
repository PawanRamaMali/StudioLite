"""Coverage for the offline license verifier.

Signs a payload with the DEV private key baked in ``filmmaker.licensing``
so the round-trip is real end-to-end - same signature format the release
publisher would produce."""
from __future__ import annotations

import base64
import json
import os
import time

import pytest

from filmmaker import licensing
from filmmaker.licensing import (
    LicensePayload,
    check_entitlement,
    device_fingerprint,
    install_license,
    sign_payload_for_tests,
    verify_license,
    _DEV_PRIVATE_KEY_PEM,
)


def _sign(payload: LicensePayload) -> dict:
    return sign_payload_for_tests(payload, _DEV_PRIVATE_KEY_PEM)


class TestVerify:
    def test_valid_perpetual_license(self):
        p = LicensePayload(tier="pro", features=["upscale", "batch"],
                            issued_at=1.0, licensee="Test Studio")
        check = verify_license(_sign(p))
        assert check.valid is True
        assert check.tier == "pro"
        assert "upscale" in check.features
        assert check.has("upscale") is True
        assert check.has("missing_feature") is False

    def test_tampered_signature_rejected(self):
        p = LicensePayload(tier="pro", features=["upscale"])
        bundle = _sign(p)
        # Flip a bit in the signature
        raw = base64.b64decode(bundle["signature"])
        tampered = bytearray(raw)
        tampered[0] ^= 0xFF
        bundle["signature"] = base64.b64encode(bytes(tampered)).decode("ascii")
        check = verify_license(bundle)
        assert check.valid is False
        assert "signature" in check.reason.lower()

    def test_tampered_payload_rejected(self):
        p = LicensePayload(tier="free", features=[])
        bundle = _sign(p)
        # Rewrite tier after signing - should not verify.
        bundle["payload"]["tier"] = "studio"
        check = verify_license(bundle)
        assert check.valid is False

    def test_expired_beyond_grace_rejected(self):
        p = LicensePayload(tier="pro", features=["batch"],
                            issued_at=1000.0, expires_at=2000.0,
                            grace_days=1)
        # 2000 + 1 day grace = 86400 + 2000 = 88400; now well past.
        check = verify_license(_sign(p), now=1_000_000)
        assert check.valid is False
        assert "expire" in check.reason.lower()

    def test_expired_within_grace_still_valid_with_warning(self):
        p = LicensePayload(tier="pro", features=["batch"],
                            issued_at=1000.0, expires_at=2000.0,
                            grace_days=7)
        # Now is 4000 - past expiry (2000) but within grace (2000 + 7*86400).
        check = verify_license(_sign(p), now=4000.0)
        assert check.valid is True
        assert check.warning
        assert "grace" in check.warning.lower()

    def test_device_bound_license_wrong_fingerprint_rejected(self):
        p = LicensePayload(tier="studio", features=["batch"],
                            device_hash="a" * 64)
        check = verify_license(_sign(p), fingerprint="b" * 64)
        assert check.valid is False
        assert "device" in check.reason.lower()

    def test_device_bound_license_right_fingerprint_ok(self):
        fp = "c" * 64
        p = LicensePayload(tier="studio", features=["batch"],
                            device_hash=fp)
        check = verify_license(_sign(p), fingerprint=fp)
        assert check.valid is True


class TestFingerprint:
    def test_is_stable_across_calls(self):
        a = device_fingerprint()
        b = device_fingerprint()
        assert a == b
        assert len(a) == 64  # sha256 hex


class TestFileIO:
    def test_check_entitlement_defaults_to_free_without_file(self, tmp_path):
        path = str(tmp_path / ".license")
        check = check_entitlement(path=path)
        assert check.valid is False
        assert check.tier == "free"

    def test_install_persists_and_reads_back(self, tmp_path):
        path = str(tmp_path / ".license")
        p = LicensePayload(tier="pro", features=["upscale"],
                            licensee="Owner")
        installed = install_license(_sign(p), path=path)
        assert installed.valid is True
        assert os.path.exists(path)
        again = check_entitlement(path=path)
        assert again.valid is True
        assert again.tier == "pro"

    def test_install_of_invalid_bundle_writes_nothing(self, tmp_path):
        path = str(tmp_path / ".license")
        bogus = {"payload": {"tier": "studio"}, "signature": "nope"}
        check = install_license(bogus, path=path)
        assert check.valid is False
        assert not os.path.exists(path)


class TestAPIIntegration:
    """Round-trip through the FastAPI endpoints."""

    def _fresh_api(self, monkeypatch, tmp_path, *, auth="off"):
        import importlib
        import sys
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STUDIOLITE_AUTH", auth)
        monkeypatch.setenv("STUDIOLITE_LICENSE_FILE",
                           str(tmp_path / ".license"))
        for name in list(sys.modules):
            if name == "api_server":
                del sys.modules[name]
        api = importlib.import_module("api_server")
        from fastapi.testclient import TestClient
        return api, TestClient(api.app)

    def test_status_when_no_license(self, monkeypatch, tmp_path):
        _api, client = self._fresh_api(monkeypatch, tmp_path)
        r = client.get("/api/v1/system/license")
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is False
        assert body["tier"] == "free"

    def test_install_and_deactivate(self, monkeypatch, tmp_path):
        _api, client = self._fresh_api(monkeypatch, tmp_path)
        p = LicensePayload(tier="pro", features=["upscale"], licensee="X")
        bundle = _sign(p)

        r = client.post("/api/v1/system/license", json=bundle)
        assert r.status_code == 200, r.text
        assert r.json()["tier"] == "pro"

        r = client.get("/api/v1/system/license")
        assert r.json()["valid"] is True

        r = client.delete("/api/v1/system/license")
        assert r.status_code == 200
        assert r.json()["deactivated"] is True

        r = client.get("/api/v1/system/license")
        assert r.json()["valid"] is False

    def test_install_rejects_malformed(self, monkeypatch, tmp_path):
        _api, client = self._fresh_api(monkeypatch, tmp_path)
        r = client.post("/api/v1/system/license",
                        json={"payload": {"tier": "pro"}, "signature": "AAAA"})
        assert r.status_code == 400
