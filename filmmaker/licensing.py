"""Offline license verification for StudioLite Pro.

Design in one paragraph
-----------------------

A StudioLite Pro license is a small JSON payload — tier, features, expiry,
optional device fingerprint — plus an Ed25519 signature made by the
publisher's release key. The install ships with the corresponding public
key baked in (``LICENSE_PUBLIC_KEY_PEM`` below). At runtime we read the
license file, verify the signature, and expose an ``EntitlementCheck``
that tells the API layer whether a feature is unlocked. Nothing here
talks to the network — activation is intentionally offline so a StudioLite
box can run behind an air gap and still be legit.

For the community build we ship an obviously-fake dev keypair in
``dev_keys/`` (never used for real releases) so the whole verify path
runs green in tests. Real signing lives elsewhere.

Fields on the payload
---------------------

  tier            "free" | "pro" | "studio"
  features        list of enum-ish strings — anything the API layer wants
                  to gate (e.g. "batch_export", "cloud_render").
  device_hash     optional; when set, only a machine whose fingerprint
                  matches will pass verify. Unbound licenses (device_hash
                  empty) are legal too for site licenses.
  issued_at       unix seconds
  expires_at      unix seconds. Zero means perpetual.
  grace_days      how long after expires_at the license still works but
                  emits a warning in the check result.
  licensee        display-only string (name / org).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger("studiolite.licensing")


# --- Publisher key ---------------------------------------------------------

# Placeholder DEV public key — DO NOT ship a Pro-signed license against this
# key in production. Real releases replace this string at package time. The
# key here matches ``dev_keys/dev_ed25519_private.pem`` in the repo so tests
# and demos work end-to-end.
LICENSE_PUBLIC_KEY_PEM = os.environ.get(
    "STUDIOLITE_LICENSE_PUBKEY_PEM",
    (
        "-----BEGIN PUBLIC KEY-----\n"
        "MCowBQYDK2VwAyEAxMNpHYRAL5bGddJQutGkcOUJ/XCL1DwnC8pJK5M/BNo=\n"
        "-----END PUBLIC KEY-----\n"
    ),
)

# Matching DEV private key. Only present in the community repo so the
# test suite and the demo installer can sign a payload the module then
# verifies. A real release replaces LICENSE_PUBLIC_KEY_PEM (via env var
# or a source patch) and keeps its private key completely off the box.
_DEV_PRIVATE_KEY_PEM = (
    b"-----BEGIN PRIVATE KEY-----\n"
    b"MC4CAQAwBQYDK2VwBCIEIHV+Obuh/HemEwcEV418wx0rNXzx42zITQIycnRVP9Th\n"
    b"-----END PRIVATE KEY-----\n"
)


def _load_public_key() -> Ed25519PublicKey:
    """Parse LICENSE_PUBLIC_KEY_PEM once. Failing to parse is fatal at
    import time because a broken pubkey means the licensing module can't
    do its only job."""
    key = serialization.load_pem_public_key(LICENSE_PUBLIC_KEY_PEM.encode("utf-8"))
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Configured license pubkey is not Ed25519.")
    return key


_PUBKEY: Optional[Ed25519PublicKey] = None


def get_public_key() -> Ed25519PublicKey:
    global _PUBKEY
    if _PUBKEY is None:
        _PUBKEY = _load_public_key()
    return _PUBKEY


# --- Fingerprint -----------------------------------------------------------

def device_fingerprint() -> str:
    """Coarse, non-PII, stable machine fingerprint. We hash the MAC of the
    primary NIC together with the hostname. Not tamper-proof against a
    determined attacker with root — a real anti-piracy story would need
    a TPM binding or a HSM — but it's enough to make casually-shared
    license files fail verification."""
    try:
        node = uuid.getnode()  # MAC as an int (falls back to a random 48-bit)
    except Exception:  # noqa: BLE001
        node = 0
    material = f"{node:012x}|{platform.node()}|{platform.system()}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


# --- License data model ----------------------------------------------------

@dataclass
class LicensePayload:
    tier: str = "free"
    features: List[str] = field(default_factory=list)
    device_hash: str = ""
    issued_at: float = 0.0
    expires_at: float = 0.0     # 0 = perpetual
    grace_days: int = 7
    licensee: str = ""

    def to_json_bytes(self) -> bytes:
        """Canonical, sorted-key JSON so signatures verify byte-for-byte."""
        return json.dumps(self.__dict__, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")


@dataclass
class EntitlementCheck:
    """What the caller actually looks at."""
    valid: bool
    tier: str = "free"
    features: List[str] = field(default_factory=list)
    reason: str = ""
    warning: str = ""
    expires_at: float = 0.0
    licensee: str = ""

    def has(self, feature: str) -> bool:
        return self.valid and feature in self.features


# --- File I/O and verification --------------------------------------------

LICENSE_FILE_ENV = "STUDIOLITE_LICENSE_FILE"


def default_license_path() -> str:
    """The install root's ``.license`` file. Overridable per-user via env."""
    from_env = os.environ.get(LICENSE_FILE_ENV)
    if from_env:
        return from_env
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".license",
    )


def _read_license_file(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def verify_license(license_bundle: Dict[str, Any],
                   *, now: Optional[float] = None,
                   fingerprint: Optional[str] = None) -> EntitlementCheck:
    """Verify a parsed license bundle (payload + signature).

    ``license_bundle`` shape:
        {
          "payload": { … LicensePayload fields … },
          "signature": "<base64 Ed25519 signature over the canonical payload json>"
        }

    ``now`` and ``fingerprint`` exist so tests can pin them without
    touching the wall clock or the real machine ID."""
    now = time.time() if now is None else now
    fingerprint = device_fingerprint() if fingerprint is None else fingerprint

    payload_raw = license_bundle.get("payload")
    sig_b64 = license_bundle.get("signature", "")
    if not isinstance(payload_raw, dict) or not isinstance(sig_b64, str):
        return EntitlementCheck(valid=False, reason="License bundle malformed.")

    # Rehydrate through the dataclass so an unknown key doesn't slip past.
    try:
        payload = LicensePayload(
            tier=str(payload_raw.get("tier", "free")),
            features=list(payload_raw.get("features") or []),
            device_hash=str(payload_raw.get("device_hash", "") or ""),
            issued_at=float(payload_raw.get("issued_at", 0) or 0),
            expires_at=float(payload_raw.get("expires_at", 0) or 0),
            grace_days=int(payload_raw.get("grace_days", 7) or 7),
            licensee=str(payload_raw.get("licensee", "")),
        )
    except (TypeError, ValueError):
        return EntitlementCheck(valid=False, reason="License payload types are wrong.")

    try:
        sig = base64.b64decode(sig_b64.encode("ascii"), validate=True)
    except (ValueError, TypeError):
        return EntitlementCheck(valid=False, reason="License signature not base64.")

    try:
        get_public_key().verify(sig, payload.to_json_bytes())
    except InvalidSignature:
        return EntitlementCheck(
            valid=False, reason="License signature failed verification.",
        )
    except Exception as e:  # noqa: BLE001
        return EntitlementCheck(valid=False, reason=f"License verify error: {e}")

    # Signature valid — apply policy checks.

    if payload.device_hash and payload.device_hash != fingerprint:
        return EntitlementCheck(
            valid=False, tier=payload.tier, licensee=payload.licensee,
            reason="License is bound to a different device.",
        )

    warning = ""
    if payload.expires_at:
        grace = payload.grace_days * 86400
        if now > payload.expires_at + grace:
            return EntitlementCheck(
                valid=False, tier=payload.tier, licensee=payload.licensee,
                reason="License expired past its grace window.",
                expires_at=payload.expires_at,
            )
        if now > payload.expires_at:
            warning = "License expired; running on grace period."

    return EntitlementCheck(
        valid=True,
        tier=payload.tier,
        features=list(payload.features),
        expires_at=payload.expires_at,
        licensee=payload.licensee,
        warning=warning,
    )


def check_entitlement(*, path: Optional[str] = None) -> EntitlementCheck:
    """Read the on-disk license (if any) and report the current
    entitlement. Absence of a license is not an error — it just means
    ``valid=False, tier="free"``, and the API layer can gate paid
    features accordingly."""
    if path is None:
        path = default_license_path()
    raw = _read_license_file(path)
    if raw is None:
        return EntitlementCheck(valid=False, tier="free",
                                reason="No license installed.")
    return verify_license(raw)


def install_license(bundle: Dict[str, Any], *,
                    path: Optional[str] = None) -> EntitlementCheck:
    """Verify a caller-supplied license bundle and, if valid, persist it.
    On invalid input we return the failure reason without writing
    anything — no partial state on disk."""
    check = verify_license(bundle)
    if not check.valid:
        return check
    if path is None:
        path = default_license_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    os.replace(tmp, path)
    logger.info("License installed for %s tier=%s", check.licensee, check.tier)
    return check


def deactivate_license(*, path: Optional[str] = None) -> None:
    """Remove the on-disk license. Idempotent — a missing file is fine."""
    if path is None:
        path = default_license_path()
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# --- Dev-mode signer -------------------------------------------------------
# Used only by tests / the demo installer script. Real releases sign
# offline with a private key that never lives in this repo.

def sign_payload_for_tests(payload: LicensePayload,
                           private_key_pem: bytes) -> Dict[str, Any]:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("sign_payload_for_tests needs an Ed25519 private key.")
    sig = key.sign(payload.to_json_bytes())
    return {
        "payload": payload.__dict__,
        "signature": base64.b64encode(sig).decode("ascii"),
    }
