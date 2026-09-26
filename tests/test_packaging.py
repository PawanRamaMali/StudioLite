"""Coverage for delivery packaging + portable export/import.

We build real Projects on tmp_path (no mocks) so the tests exercise
the same JSON layout the orchestrator produces, then walk the zip
outputs to confirm each expected file is present and the manifest
carries what a downstream reader would need.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import sys
import zipfile

import pytest

from filmmaker import packaging as pkg
from filmmaker.project import Project, ProjectConfig


def _fresh_project(tmp_path, title="My Film", brief="A one-paragraph brief."):
    Project.set_store(None)          # force fresh sqlite in new root
    root = str(tmp_path / "films")
    os.makedirs(root, exist_ok=True)
    proj = Project.create(root, brief=brief, title=title,
                          config=ProjectConfig())
    return proj, root


def _write_fake_final(films_root, project_id, *, mixed=True):
    d = os.path.join(films_root, project_id)
    os.makedirs(d, exist_ok=True)
    name = "final_mixed.mp4" if mixed else "final.mp4"
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"\x00" * 128)      # a byte-shaped stand-in for a real render
    return p


class TestDeliveryBundle:
    def test_includes_final_video_and_credits(self, tmp_path):
        proj, root = _fresh_project(tmp_path)
        _write_fake_final(root, proj.project_id, mixed=True)
        out = str(tmp_path / "delivery.zip")
        result = pkg.build_delivery_bundle(
            proj, root, out, entitlement_tier="pro", watermarked=False,
        )
        assert result.warning == ""
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            assert "CREDITS.md" in names
            assert "manifest.json" in names
            assert any(n.endswith(".mp4") for n in names)
            manifest = json.loads(z.read("manifest.json"))
            assert manifest["kind"] == "delivery"
            assert manifest["watermarked"] is False
            assert manifest["entitlement_tier"] == "pro"
            credits = z.read("CREDITS.md").decode("utf-8")
            assert "# My Film" in credits
            assert "pro tier" in credits

    def test_missing_final_still_emits_bundle_with_warning(self, tmp_path):
        proj, root = _fresh_project(tmp_path)
        out = str(tmp_path / "delivery.zip")
        result = pkg.build_delivery_bundle(proj, root, out)
        assert "No final render" in result.warning
        with zipfile.ZipFile(out) as z:
            manifest = json.loads(z.read("manifest.json"))
            assert manifest["final_video"] is None
            assert manifest["warning"]

    def test_characters_land_in_credits(self, tmp_path):
        proj, root = _fresh_project(tmp_path)
        _write_fake_final(root, proj.project_id)
        out = str(tmp_path / "delivery.zip")
        pkg.build_delivery_bundle(
            proj, root, out,
            characters=[
                {"name": "Alice", "description": "A curious astronomer"},
                {"name": "Bob"},
            ],
        )
        with zipfile.ZipFile(out) as z:
            credits = z.read("CREDITS.md").decode("utf-8")
        assert "Alice" in credits
        assert "curious astronomer" in credits
        assert "Bob" in credits


class TestExportImport:
    def test_export_writes_zip_with_project_json(self, tmp_path):
        proj, root = _fresh_project(tmp_path)
        proj.write_artifact("outline",
                            {"beats": ["beat A", "beat B"]},
                            snapshot=False)
        out = str(tmp_path / "export.studioproj")
        result = pkg.build_project_export(proj, out)
        assert result.file_count > 1
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
        assert "_studioproj/manifest.json" in names
        assert "project/project.json" in names
        assert any(n.startswith("project/artifacts/") for n in names)

    def test_import_creates_new_id_and_preserves_artifacts(self, tmp_path):
        proj, root = _fresh_project(tmp_path)
        proj.write_artifact("outline", {"beats": ["beat A"]},
                            snapshot=False)
        out = str(tmp_path / "export.studioproj")
        pkg.build_project_export(proj, out)

        # Import into a fresh root dir to prove nothing was cached in-place.
        Project.set_store(None)
        dest_root = str(tmp_path / "films2")
        os.makedirs(dest_root, exist_ok=True)
        result = pkg.import_project(out, dest_root,
                                    title_override="Copied Film")
        assert result.project_id != proj.project_id
        assert result.title == "Copied Film"
        assert result.source_project_id == proj.project_id
        loaded = Project.load(dest_root, result.project_id)
        # Artifact traveled over intact.
        art = loaded.read_artifact("outline")
        assert art == {"beats": ["beat A"]}
        # Meta rewritten with new id.
        assert loaded.meta.id == result.project_id

    def test_import_refuses_non_studioproj(self, tmp_path):
        bogus = str(tmp_path / "bogus.zip")
        with zipfile.ZipFile(bogus, "w") as z:
            z.writestr("hello.txt", "not a project")
        with pytest.raises(ValueError):
            pkg.import_project(bogus, str(tmp_path / "target"))

    def test_import_rejects_zip_slip(self, tmp_path):
        """A zip with a member escaping the target must fail cleanly
        rather than write files outside the project root."""
        malicious = str(tmp_path / "evil.studioproj")
        header = {"format_version": pkg.EXPORT_FORMAT_VERSION,
                  "kind": "studioproj_export",
                  "source_project_id": "spoof",
                  "exported_at": 0}
        with zipfile.ZipFile(malicious, "w") as z:
            z.writestr("_studioproj/manifest.json", json.dumps(header))
            z.writestr("project/../../etc/passwd_stub", "you got popped")
        with pytest.raises(ValueError):
            pkg.import_project(malicious, str(tmp_path / "target"))


class TestPackagingEndpoints:
    def _client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STUDIOLITE_AUTH", "off")
        for name in list(sys.modules):
            if name == "api_server":
                del sys.modules[name]
        api = importlib.import_module("api_server")
        Project.set_store(None)
        return api, TestClient(api.app)

    def test_package_missing_project_404s(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        r = client.post("/api/v1/films/no-such-id/package", json={})
        assert r.status_code == 404

    def test_export_then_import_roundtrip(self, monkeypatch, tmp_path):
        api, client = self._client(monkeypatch, tmp_path)
        # Spawn a project via the real endpoint so the film dir/store are
        # bound the way the running server does it.
        r = client.post("/api/v1/films", json={
            "brief": "A quick end-to-end export test project.",
            "title": "Roundtrip",
        })
        assert r.status_code == 200, r.text
        pid = r.json()["project"]["id"]

        # Download the export.
        r = client.get(f"/api/v1/films/{pid}/export")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        blob = r.content
        assert len(blob) > 100

        # POST it back as an import - should get a new id back.
        files = {"file": ("roundtrip.studioproj", io.BytesIO(blob),
                          "application/zip")}
        r = client.post("/api/v1/films/import", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source_project_id"] == pid
        assert body["project"]["id"] != pid
        assert body["project"]["title"] == "Roundtrip (imported)"

    def test_import_rejects_garbage(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        files = {"file": ("junk.studioproj",
                           io.BytesIO(b"not a zip at all"),
                           "application/zip")}
        r = client.post("/api/v1/films/import", files=files)
        assert r.status_code == 400
