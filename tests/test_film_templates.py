"""Coverage for the Film Studio template catalog + spawn endpoint.

We assert the catalog's public shape (kebab-case ids, non-empty
sample_brief, template_config keys line up with ProjectConfig fields)
so a bad template entry can't slip into a release."""
from __future__ import annotations

import importlib
import re
import sys

import pytest

from filmmaker import film_templates
from filmmaker.project import ProjectConfig


class TestCatalog:
    def test_at_least_three_templates(self):
        tpls = film_templates.list_templates()
        assert len(tpls) >= 3

    def test_ids_are_unique_and_kebab_case(self):
        tpls = film_templates.list_templates()
        ids = [t.id for t in tpls]
        assert len(ids) == len(set(ids)), "Template ids must be unique."
        for tid in ids:
            assert re.fullmatch(r"[a-z][a-z0-9-]{2,}", tid), tid

    def test_categories_are_known(self):
        allowed = {"narrative", "explainer", "promo", "experimental"}
        for t in film_templates.list_templates():
            assert t.category in allowed, f"{t.id}: {t.category}"

    def test_sample_briefs_are_meaningful(self):
        for t in film_templates.list_templates():
            assert len(t.sample_brief) >= 8, t.id

    def test_target_minutes_within_sane_range(self):
        for t in film_templates.list_templates():
            assert 0.1 < t.target_minutes < 30, t.id


class TestLookup:
    def test_get_returns_the_named_entry(self):
        t = film_templates.get_template("short-story")
        assert t.name == "Short Story"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            film_templates.get_template("does-not-exist")

    def test_template_config_matches_project_config_fields(self):
        cfg = film_templates.template_config("short-story")
        # Every key we emit must be a real field on ProjectConfig — else
        # ProjectConfig(**cfg) will explode at project-create time.
        allowed = set(ProjectConfig.__dataclass_fields__.keys())
        for k in cfg:
            assert k in allowed, f"Template emits unknown config field: {k}"


class TestAPI:
    def _client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STUDIOLITE_AUTH", "off")
        for name in list(sys.modules):
            if name == "api_server":
                del sys.modules[name]
        api = importlib.import_module("api_server")
        return api, TestClient(api.app)

    def test_list_templates(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        r = client.get("/api/v1/films/templates")
        assert r.status_code == 200
        body = r.json()
        assert "templates" in body
        assert len(body["templates"]) >= 3
        for t in body["templates"]:
            assert set(["id", "name", "description", "category"]).issubset(t)

    def test_spawn_from_template_uses_sample_brief(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        r = client.post(
            "/api/v1/films/from-template",
            json={"template_id": "short-story"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["template"]["id"] == "short-story"
        assert body["project"]["title"] == "Short Story"
        # The stored config picked up the template's target_minutes.
        assert body["project"]["config"]["target_minutes"] == 1.0

    def test_spawn_with_brief_override(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        r = client.post(
            "/api/v1/films/from-template",
            json={
                "template_id": "explainer",
                "title": "Battery Explainer v2",
                "brief_override": "A tighter 90-second explainer that "
                                  "focuses on why lithium-ion is dying.",
                "config_override": {"target_minutes": 1.5},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["project"]["title"] == "Battery Explainer v2"
        assert body["project"]["brief"].startswith("A tighter 90-second")
        assert body["project"]["config"]["target_minutes"] == 1.5

    def test_unknown_template_returns_404(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        r = client.post(
            "/api/v1/films/from-template",
            json={"template_id": "not-a-real-template"},
        )
        assert r.status_code == 404

    def test_too_short_brief_rejected(self, monkeypatch, tmp_path):
        _api, client = self._client(monkeypatch, tmp_path)
        r = client.post(
            "/api/v1/films/from-template",
            json={"template_id": "short-story", "brief_override": "hi"},
        )
        assert r.status_code == 422
