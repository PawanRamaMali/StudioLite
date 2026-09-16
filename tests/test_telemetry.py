"""Unit tests for filmmaker/telemetry.py.

The telemetry module writes to a fixed path (`.telemetry/` in the repo).
We monkey-patch that path per-test so we never touch the real one and
tests can run in parallel."""
from __future__ import annotations

import io
import json
import os
import zipfile

import pytest

from filmmaker import telemetry


@pytest.fixture(autouse=True)
def _isolate_telemetry_dir(tmp_path, monkeypatch):
    """Point telemetry storage at a fresh tmp dir for every test.
    The module reads its state on every call, so patching the module-
    level constants is enough — no cache to invalidate."""
    d = tmp_path / ".telemetry"
    monkeypatch.setattr(telemetry, "TELEMETRY_DIR", str(d))
    monkeypatch.setattr(telemetry, "CONSENT_FILE", str(d / "consent.json"))
    monkeypatch.setattr(telemetry, "EVENTS_FILE", str(d / "events.jsonl"))


class TestConsent:
    def test_defaults_to_opt_out(self):
        state = telemetry.get_state()
        assert state["consent"] is False
        assert state["installation_id"] == ""

    def test_grant_mints_installation_id(self):
        telemetry.record_consent(True)
        state = telemetry.get_state()
        assert state["consent"] is True
        assert state["installation_id"], "should have a uuid"

    def test_revoke_keeps_installation_id(self):
        telemetry.record_consent(True)
        install_id = telemetry.get_state()["installation_id"]
        telemetry.record_consent(False)
        state = telemetry.get_state()
        assert state["consent"] is False
        # ID sticks around so the user can reactivate the same install.
        # Reset is a separate call.
        assert state["installation_id"] == install_id

    def test_reset_id_rotates_when_consent_on(self):
        telemetry.record_consent(True)
        first = telemetry.get_state()["installation_id"]
        second = telemetry.reset_installation_id()["installation_id"]
        assert first and second and first != second

    def test_reset_id_clears_when_consent_off(self):
        telemetry.record_consent(True)
        telemetry.record_consent(False)
        state = telemetry.reset_installation_id()
        assert state["installation_id"] == ""


class TestEventRecording:
    def test_records_dropped_when_opted_out(self):
        assert telemetry.record_event(event="stage_start", stage="shots") is False
        assert telemetry.recent_events() == []

    def test_records_written_when_opted_in(self):
        telemetry.record_consent(True)
        ok = telemetry.record_event(
            event="stage_success", stage="shots", duration_seconds=42.5,
        )
        assert ok is True
        events = telemetry.recent_events()
        # Consent grant itself logs an event, then our stage_success.
        assert any(e["event"] == "stage_success" for e in events)
        target = next(e for e in events if e["event"] == "stage_success")
        assert target["stage"] == "shots"
        assert target["duration_seconds"] == 42.5

    def test_disallowed_keys_are_dropped(self):
        """The whole point of the whitelist: a caller cannot smuggle a
        prompt or filename through record_event by accident."""
        telemetry.record_consent(True)
        telemetry.record_event(
            event="stage_start",
            prompt="THIS SHOULD NEVER BE LOGGED",
            filename="secret.mp4",
        )
        events = telemetry.recent_events()
        target = next(e for e in events if e["event"] == "stage_start")
        assert "prompt" not in target
        assert "filename" not in target

    def test_installation_id_not_leaked_after_opt_out(self):
        telemetry.record_consent(True)
        install_id = telemetry.get_state()["installation_id"]
        telemetry.record_event(event="stage_start", stage="shots")
        telemetry.record_consent(False)
        # After opt-out record_event should return False and never write
        assert (
            telemetry.record_event(event="stage_start", stage="editor") is False
        )
        # And the consent_revoked event that DOES land carries a blank
        # installation_id since consent is off at write time.
        events = telemetry.recent_events()
        revoked = next(e for e in events if e["event"] == "consent_revoked")
        assert revoked["installation_id"] == ""
        # But earlier grant-time events still carry their id — that's
        # the history the user asked to log.
        earlier = next(
            e for e in events
            if e["event"] == "stage_start" and e.get("stage") == "shots"
        )
        assert earlier["installation_id"] == install_id

    def test_string_values_are_capped(self):
        """A rogue caller passing a giant string should not blow up the
        events log; the module truncates defensively."""
        telemetry.record_consent(True)
        big = "x" * 5000
        telemetry.record_event(event="stage_start", stage=big)
        events = telemetry.recent_events()
        target = next(
            e for e in events
            if e["event"] == "stage_start" and isinstance(e.get("stage"), str)
        )
        assert len(target["stage"]) <= 200


class TestRedaction:
    def test_hf_token_redacted(self):
        line = "loaded model with token hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert "hf_[REDACTED]" in telemetry._redact_line(line)
        assert "hf_ABCDEFGH" not in telemetry._redact_line(line)

    def test_path_outside_root_redacted(self):
        line = "Opening C:\\Users\\alice\\Documents\\secret.txt"
        redacted = telemetry._redact_line(line)
        assert "<user-path>" in redacted
        assert "alice" not in redacted


class TestDiagnosticBundle:
    def test_bundle_includes_readme_and_consent(self):
        telemetry.record_consent(True)
        telemetry.record_event(event="stage_start", stage="shots")
        blob = telemetry.build_diagnostic_bundle(log_tail=5, event_tail=5)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = set(zf.namelist())
            assert "README.txt" in names
            assert "consent.json" in names
            assert "events.jsonl" in names
            consent = json.loads(zf.read("consent.json"))
            assert consent["consent"] is True
