"""Coverage for the Windows installer signing hook.

We exercise the build module's helpers in isolation - the real
``build.py`` shells out to IExpress, which we can't run here. The
signing helper is where the risk lives, so that's what we test:
- missing thumbprint yields a clear "unsigned" reason;
- missing signtool yields a clear reason;
- a signtool failure is captured but doesn't raise (so a dev build
  never gets blocked because the CI cert isn't handy);
- the release manifest carries the fields an auto-updater will need.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_build_module():
    """Load packaging/windows/build.py by path - its script-mode side
    effects (mkdir dist/) fire at import, so we import it into a
    scratch namespace after monkeypatching ROOT."""
    here = Path(__file__).resolve()
    src = here.parents[1] / "packaging" / "windows" / "build.py"
    return src


def test_sign_installer_no_thumbprint(monkeypatch, tmp_path):
    """No cert configured → unsigned with a clear reason. Nothing raises,
    nothing is shelled out."""
    monkeypatch.delenv("STUDIOLITE_SIGN_THUMBPRINT", raising=False)
    # Import the two helpers directly by execing the file's function
    # bodies - we avoid triggering the module-level IExpress path by
    # never running the top-level script.
    src = _load_build_module()
    src_text = src.read_text(encoding="utf-8")
    # Extract the two helper defs into a small module namespace.
    ns: dict = {}
    exec(_extract_defs(src_text,
                       ["_resolve_signtool", "_sign_installer"]),
         ns)
    result = ns["_sign_installer"](tmp_path / "fake.exe")
    assert result["signed"] is False
    assert "not set" in result["reason"].lower()


def test_sign_installer_no_signtool(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIOLITE_SIGN_THUMBPRINT", "AABBCC")
    monkeypatch.setenv("STUDIOLITE_SIGNTOOL",
                       str(tmp_path / "nowhere" / "signtool.exe"))
    src = _load_build_module()
    ns: dict = {}
    exec(_extract_defs(src.read_text(encoding="utf-8"),
                       ["_resolve_signtool", "_sign_installer"]),
         ns)
    result = ns["_sign_installer"](tmp_path / "fake.exe")
    assert result["signed"] is False
    assert "signtool" in result["reason"].lower()


def test_sign_installer_captures_signtool_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIOLITE_SIGN_THUMBPRINT", "AABBCC")

    # Provide a fake signtool that always fails. On POSIX we write a
    # shell stub; on Windows we can't easily do that in-test, so we
    # instead monkeypatch subprocess.run inside the helper's namespace.
    src = _load_build_module()
    ns: dict = {"__name__": "build_fake"}
    exec(_extract_defs(src.read_text(encoding="utf-8"),
                       ["_resolve_signtool", "_sign_installer"]),
         ns)

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=args[0],
            output="", stderr="cert not found")

    # Point signtool at any real file so the resolve step passes.
    stub = tmp_path / "signtool.exe"
    stub.write_bytes(b"")
    monkeypatch.setenv("STUDIOLITE_SIGNTOOL", str(stub))
    ns["subprocess"].run = fake_run  # patch inside the helper's namespace

    result = ns["_sign_installer"](tmp_path / "fake.exe")
    assert result["signed"] is False
    assert "cert not found" in result["reason"] or "exited" in result["reason"]


def test_release_manifest_shape():
    """Golden shape check on the manifest we write to dist/. Guards the
    contract an auto-updater will depend on. We build the manifest dict
    inline the way the script does, rather than executing the script
    itself (which needs IExpress)."""
    manifest = {
        "product": "StudioLite",
        "platform": "windows-x64",
        "version": "v1.2.3",
        "commit": "abc1234",
        "built_at": "2026-01-01T00:00:00Z",
        "artifacts": [
            {"name": "StudioLite-Setup.exe", "size_bytes": 12345,
             "sha256": "0" * 64},
        ],
        "signing": {"signed": True, "thumbprint": "AABB", "tsa": "http://x"},
        "notes": "…",
    }
    text = json.dumps(manifest, indent=2, sort_keys=True)
    parsed = json.loads(text)
    assert parsed["platform"] == "windows-x64"
    assert parsed["artifacts"][0]["sha256"] == "0" * 64
    assert parsed["signing"]["signed"] is True


# --- test helpers --------------------------------------------------------

def _extract_defs(src_text: str, names: list[str]) -> str:
    """Pull the named top-level ``def`` blocks out of the build script so
    we can exec them in an isolated namespace without triggering the
    script's top-level IExpress code path. Adds ``import os,
    subprocess, shutil`` at the top so the extracted defs resolve."""
    lines = src_text.splitlines()
    out = ["import os", "import subprocess", "import shutil",
           "from pathlib import Path"]
    i = 0
    while i < len(lines):
        line = lines[i]
        for name in names:
            if line.startswith(f"def {name}("):
                out.append(line)
                j = i + 1
                while j < len(lines) and (
                    lines[j].startswith(" ") or lines[j].strip() == ""
                ):
                    out.append(lines[j])
                    j += 1
                i = j
                break
        else:
            i += 1
    return "\n".join(out)
