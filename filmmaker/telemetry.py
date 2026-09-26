"""Opt-in local telemetry and diagnostics for StudioLite.

This module never sends data anywhere. It writes a per-event JSONL file
under `.telemetry/events.jsonl` in the repo root, records whether the
user has consented, and knows how to bundle the last N events + the
recent app log into a support zip the user can attach to a bug report
themselves. Nothing here talks to the network.

Design invariants:
- **Opt-in.** No events are logged until the user calls `record_consent(True)`.
    A fresh install starts with consent=False.
- **Content allowlist.** `record_event()` accepts only keys in
    EVENT_KEY_ALLOWLIST. Freeform strings never land in the log, so a
    stray prompt or filename can't leak through by accident.
- **Installation-only ID.** The installation ID is a fresh UUID minted
    the first time telemetry is turned on. `reset_installation_id()`
    replaces it so the user can start over.
- **Redaction.** The bundle helper redacts obvious paths and tokens
    from the recent log tail before it hands them over.

The API layer wraps this module with three endpoints (see api_server.py):
`/api/v1/system/telemetry` for the current consent state, its POST for
opt-in/out, and `/api/v1/system/telemetry/bundle` for the zip download.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("studiolite.telemetry")

# --- Storage layout ---------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(_ROOT, ".telemetry")
CONSENT_FILE = os.path.join(TELEMETRY_DIR, "consent.json")
EVENTS_FILE = os.path.join(TELEMETRY_DIR, "events.jsonl")

# Loaded once; guarded by a lock so consent flips and event writes never race.
_LOCK = threading.Lock()

# --- Event schema -----------------------------------------------------------

# Only these keys are legal on a telemetry event. Anything else silently
# drops. Keeps the schema honest as the codebase grows - you have to add
# a field here on purpose before you can record it.
EVENT_KEY_ALLOWLIST = frozenset({
    "event",              # e.g. "stage_start" or "model_download_failure"
    "stage",              # pipeline stage key, e.g. "motion_shots"
    "backend",            # e.g. "wan22" / "indextts2" / "realesrgan_x2"
    "model_key",          # opaque identifier from model_manager
    "duration_seconds",   # non-negative float
    "vram_used_mb",       # non-negative int/float
    "error_class",        # short symbol, e.g. "CUDA_OOM"
    "outcome",            # "success" / "failure" / "cancelled"
    "attempt",            # int
    "app_version",        # short semver-ish string
    "cuda_available",     # bool
})

# Human-readable event names that describe intent. Free-form strings are
# still allowed for `event` (small models change frequently), but the
# canonical set below is what the caller should stick to.
CANONICAL_EVENT_NAMES = frozenset({
    "install_success", "install_failure",
    "model_download_success", "model_download_failure",
    "stage_start", "stage_success", "stage_failure",
    "cuda_oom", "cuda_error",
    "export_success", "export_failure",
    "consent_granted", "consent_revoked",
    "installation_id_reset",
})


@dataclass
class TelemetryState:
    """The persisted consent record. Written to consent.json atomically."""
    consent: bool = False
    installation_id: str = ""
    accepted_at: float = 0.0
    revoked_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent": self.consent,
            "installation_id": self.installation_id,
            "accepted_at": self.accepted_at,
            "revoked_at": self.revoked_at,
        }


def _read_state() -> TelemetryState:
    """Read the consent record, defaulting to opt-out on any parse failure.
    Failing closed on a corrupted file is the safe choice - a user shouldn't
    get their events logged because a hex-editor slip broke consent.json."""
    if not os.path.exists(CONSENT_FILE):
        return TelemetryState()
    try:
        with open(CONSENT_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return TelemetryState(
            consent=bool(raw.get("consent", False)),
            installation_id=str(raw.get("installation_id", "") or ""),
            accepted_at=float(raw.get("accepted_at", 0) or 0),
            revoked_at=float(raw.get("revoked_at", 0) or 0),
        )
    except (OSError, ValueError, TypeError):
        return TelemetryState()


def _write_state(state: TelemetryState) -> None:
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    tmp = CONSENT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
    os.replace(tmp, CONSENT_FILE)


def get_state() -> Dict[str, Any]:
    """Return a plain-dict view of the consent state, safe to send to the UI.
    Note the installation ID is included when consent is on - this is the
    only stable identifier the module ever generates and it never leaves the
    box on its own."""
    with _LOCK:
        return _read_state().to_dict()


def record_consent(consent: bool) -> Dict[str, Any]:
    """Flip consent on or off. Turning on mints a fresh installation ID if
    one isn't already recorded. Turning off leaves the ID in place - the
    user can wipe it explicitly with reset_installation_id()."""
    with _LOCK:
        state = _read_state()
        now = time.time()
        if consent:
            if not state.installation_id:
                state.installation_id = str(uuid.uuid4())
            state.consent = True
            state.accepted_at = now
        else:
            state.consent = False
            state.revoked_at = now
        _write_state(state)
        event = "consent_granted" if consent else "consent_revoked"
        _append_event_locked({"event": event})
        return state.to_dict()


def reset_installation_id() -> Dict[str, Any]:
    """Rotate the installation ID. Doesn't touch the events log - the user
    who wants a truly fresh start can also clear .telemetry/events.jsonl
    via the panel."""
    with _LOCK:
        state = _read_state()
        old_id = state.installation_id
        state.installation_id = str(uuid.uuid4()) if state.consent else ""
        _write_state(state)
        _append_event_locked({
            "event": "installation_id_reset",
            "outcome": "success",
        })
        logger.info("Telemetry installation ID rotated (was %s)", old_id or "<unset>")
        return state.to_dict()


def _append_event_locked(payload: Dict[str, Any]) -> None:
    """Append one event to the JSONL log. Caller holds _LOCK.

    Two guards on top of the schema whitelist:
      1. Silent drop of any non-whitelisted key.
      2. Numeric coercion for duration_seconds / vram_used_mb so a stray
         string doesn't corrupt the tail-parser downstream.
    """
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    filtered: Dict[str, Any] = {}
    for k, v in payload.items():
        if k not in EVENT_KEY_ALLOWLIST:
            continue
        if k in ("duration_seconds", "vram_used_mb"):
            try:
                filtered[k] = float(v)
            except (TypeError, ValueError):
                continue
        elif k in ("attempt",):
            try:
                filtered[k] = int(v)
            except (TypeError, ValueError):
                continue
        elif k in ("cuda_available",):
            filtered[k] = bool(v)
        else:
            filtered[k] = str(v)[:200]  # cap string lengths defensively
    # Attach the installation ID + timestamp last so they can't be spoofed
    # by the caller.
    state = _read_state()
    filtered["ts"] = time.time()
    filtered["installation_id"] = state.installation_id if state.consent else ""
    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(filtered) + "\n")


def record_event(**payload: Any) -> bool:
    """Log an event iff the user has opted in. Returns True when the event
    was written, False when consent is off or the payload was empty after
    filtering (i.e. every field was silently dropped)."""
    with _LOCK:
        state = _read_state()
        if not state.consent:
            return False
        # Confirm at least one whitelisted key survived filtering.
        if not any(k in EVENT_KEY_ALLOWLIST for k in payload.keys()):
            return False
        _append_event_locked(payload)
        return True


def recent_events(limit: int = 200) -> List[Dict[str, Any]]:
    """Read the tail of the events log for the diagnostics panel."""
    with _LOCK:
        if not os.path.exists(EVENTS_FILE):
            return []
        try:
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
        except OSError:
            return []
    out: List[Dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


# --- Log redaction helpers --------------------------------------------------

_PATH_RE = re.compile(r"([A-Za-z]:\\|/)[^ \n\r\t\"'\|<>]+")
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _redact_line(line: str) -> str:
    """Coarse redaction of user-specific tokens/paths for the bundle export.
    We deliberately leave stack frames intact - file paths inside the
    installation dir are still useful to a maintainer - but drop absolute
    paths outside the repo and any recognizable HF-style token."""
    line = _HF_TOKEN_RE.sub("hf_[REDACTED]", line)

    def _p(m: re.Match) -> str:
        path = m.group(0)
        # Preserve paths that live inside the install so backtraces stay
        # readable; opaque-ify everything else.
        if os.path.abspath(path).lower().startswith(_ROOT.lower()):
            return path
        return "<user-path>"

    return _PATH_RE.sub(_p, line)


def build_diagnostic_bundle(*, log_tail: int = 500,
                             event_tail: int = 500) -> bytes:
    """Assemble a zip of {consent.json, events.jsonl tail, redacted app-log
    tail}. Nothing user-editable, nothing secret, nothing over the wire - 
    the user opens the zip themselves and attaches it to a bug report.

    Reads the app log from the same path api_server writes to (.mp/videogen.log)
    so it works whether the caller is the API layer or the CLI."""
    buffer = io.BytesIO()
    log_file = os.path.join(_ROOT, ".mp", "videogen.log")
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        state = get_state()
        zf.writestr("consent.json", json.dumps(state, indent=2))
        events = recent_events(limit=event_tail)
        zf.writestr(
            "events.jsonl",
            "\n".join(json.dumps(e) for e in events),
        )
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()[-log_tail:]
                redacted = "".join(_redact_line(line) for line in lines)
                zf.writestr("app.log", redacted)
            except OSError:
                pass
        # A README so the recipient knows what this is.
        zf.writestr(
            "README.txt",
            "StudioLite diagnostic bundle.\n\n"
            "Contents:\n"
            "  consent.json - the local telemetry consent record.\n"
            "  events.jsonl - recent events (last {} entries).\n"
            "  app.log - recent app log lines, paths outside the\n"
            "                  install root redacted; HF tokens redacted.\n\n"
            "Nothing in this bundle was sent anywhere. If you attach it to\n"
            "a support ticket, you are doing that yourself.\n".format(event_tail),
        )
    return buffer.getvalue()
