# Windows setup

Build on Windows with Python, Git and the Windows IExpress utility:

```powershell
python packaging/windows/build.py
```

Output: `dist/StudioLite-Setup.exe`, a ZIP alternative and SHA-256 checksums.
The payload uses an allowlist of tracked application sources plus the API
requirements file. Local credentials, configuration, models, caches, generated
media, virtual environments and node_modules are excluded.

On the target computer install 64-bit Python 3.11, Node.js 22 or newer, and
FFmpeg with ffprobe on PATH. Run the EXE, or extract the ZIP and run `setup.cmd`.
Setup downloads CPU dependencies, creates an isolated Python environment,
runs `npm ci` and a production build, and adds a desktop shortcut. Internet
access and several GB of free disk space are required. GPU acceleration needs
a separate compatible PyTorch installation; this package defaults to CPU.
Model weights are downloaded separately through the app.

Default destination: `%LOCALAPPDATA%\StudioLite`. For a custom destination,
use the extracted ZIP:

```powershell
.\setup.cmd -InstallDir "D:\Apps\StudioLite"
```

Prerequisite-only check (no installation):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -CheckOnly
```

Failures are recorded in `setup.log` in the installation directory. Correct
the issue and rerun setup to resume an incomplete installation. Completed
installations are never overwritten; choose a separate directory for updates.
Close the launcher with Ctrl+C to stop the servers. To uninstall, close the app,
remove its desktop shortcut, and remove its installation folder after backing
up projects and models. This installer does not bundle runtimes or provide
an offline install or an automatic uninstaller.

## Signed release builds

`build.py` signs the installer with Authenticode when a code-signing
certificate is available. Absent one, it emits an unsigned build and
records the reason in `dist/release-manifest.json` - the CI pipeline
can decide whether to promote or block it.

Environment variables the build reads:

- `STUDIOLITE_SIGN_THUMBPRINT` - SHA-1 thumbprint of the code-signing cert
  in the current user's certificate store. Required to trigger signing.
- `STUDIOLITE_SIGN_TSA` - RFC 3161 timestamp authority URL. Defaults to
  `http://timestamp.digicert.com`.
- `STUDIOLITE_SIGNTOOL` - full path to `signtool.exe`. Optional; when
  unset, `signtool` is looked up on PATH (Windows SDK installs it).

## Release manifest

Every successful build writes `dist/release-manifest.json` alongside the
installer. It carries the product name, `git describe` version string,
commit sha, build timestamp, per-artifact size + SHA-256, and a signing
summary. This is the file a future auto-updater checks to decide whether
a running install needs to pull a new release.
