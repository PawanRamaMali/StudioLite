"""Build a source-based Windows setup; never package local data or credentials.

Optional Authenticode signing kicks in when ``STUDIOLITE_SIGNTOOL`` is set
in the environment. Absent the variable this build stays unsigned - which
is fine for dogfooding but means end users get a SmartScreen warning.
See ``packaging/windows/README.md`` for the release setup.
"""
from pathlib import Path
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dist'
STAGE = OUT / 'windows-setup'
STAGE.mkdir(parents=True, exist_ok=True)
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
directories = {'api', 'filmmaker', 'fonts', 'library', 'mpv2', 'web', '.streamlit'}
root_files = {'LICENSE', 'THIRD_PARTY_NOTICES.md', 'config.example.json', 'README.md',
              'launch.bat', 'launch.ps1', 'install_shortcut.ps1'}
files = set()
for name in filter(None, tracked):
    p = Path(name)
    if (len(p.parts) == 1 and (p.suffix == '.py' or name.startswith('requirements') or name in root_files)) or (
        p.parts[0] in directories and p.suffix.lower() in {'.py', '.ts', '.tsx', '.js', '.mjs', '.json', '.css', '.ico', '.svg', '.png', '.ttf', '.txt', '.toml'}
    ):
        files.add(name)
files.add('packaging/windows/requirements-api.txt')
with zipfile.ZipFile(STAGE / 'payload.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(files):
        archive.write(ROOT / name, name)
for name in ('setup.cmd', 'install.ps1'):
    (STAGE / name).write_bytes((ROOT / 'packaging/windows' / name).read_bytes())
with zipfile.ZipFile(OUT / 'StudioLite-Setup.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('setup.cmd', 'install.ps1', 'payload.zip'):
        archive.write(STAGE / name, name)
exe = OUT / 'StudioLite-Setup.exe'
build_exe = OUT / f'StudioLite-Setup-{uuid.uuid4().hex}.exe'
sed = f'''[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=0
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=Install StudioLite? Internet, Python 3.11, Node.js 22+ and FFmpeg are required. CPU dependencies will be installed.
DisplayLicense=
FinishMessage=
TargetName={build_exe}
FriendlyName=StudioLite Setup
AppLaunched=cmd.exe /c setup.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles
[SourceFiles]
SourceFiles0={STAGE}\\
[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
[Strings]
FILE0="setup.cmd"
FILE1="install.ps1"
FILE2="payload.zip"
'''
sed_path = STAGE / 'setup.sed'
sed_path.write_text(sed, encoding='ascii')
subprocess.run(['iexpress.exe', '/N', '/Q', str(sed_path)], check=True)
if not build_exe.exists() or build_exe.stat().st_size == 0:
    raise RuntimeError('IExpress did not produce the installer')
try:
    build_exe.replace(exe)
except PermissionError:
    # Windows prevents replacing an installer while it is open or locked.
    print('Existing installer is locked; keeping the new build under a unique name.')
    exe = build_exe
def _resolve_signtool() -> Path | None:
    """Find signtool.exe. Honors STUDIOLITE_SIGNTOOL first (an explicit
    path lets a release pipeline pin the Windows SDK build it wants),
    then falls back to the PATH."""
    override = os.environ.get('STUDIOLITE_SIGNTOOL')
    if override:
        p = Path(override)
        return p if p.is_file() else None
    found = shutil.which('signtool')
    return Path(found) if found else None


def _sign_installer(installer: Path) -> dict:
    """Sign the installer if a cert is configured. Returns a summary
    dict for the release manifest - always non-None so downstream code
    can just consult ``signed``."""
    thumb = os.environ.get('STUDIOLITE_SIGN_THUMBPRINT', '').strip()
    if not thumb:
        return {
            'signed': False,
            'reason': 'STUDIOLITE_SIGN_THUMBPRINT not set; unsigned build.',
        }
    signtool = _resolve_signtool()
    if signtool is None:
        return {
            'signed': False,
            'reason': 'signtool.exe not found (install Windows SDK, or set '
                      'STUDIOLITE_SIGNTOOL to its full path).',
        }
    tsa = os.environ.get('STUDIOLITE_SIGN_TSA',
                          'http://timestamp.digicert.com')
    cmd = [
        str(signtool), 'sign',
        '/sha1', thumb,
        '/tr', tsa, '/td', 'sha256',
        '/fd', 'sha256',
        '/d', 'StudioLite Setup',
        str(installer),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Don't fail the whole build on a signing miss - emit the reason
        # so CI can decide what to do. Keeps local dev builds unbroken.
        return {'signed': False,
                'reason': f'signtool exited {e.returncode}: '
                          f'{(e.stderr or e.stdout).strip()[:400]}'}
    return {'signed': True, 'thumbprint': thumb, 'tsa': tsa}


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return ''


def _git_describe() -> str:
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--always'],
            cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return ''


artifacts: list[dict] = []
for path in (exe, OUT / 'StudioLite-Setup.zip'):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + '.sha256').write_text(
        f'{digest}  {path.name}\n')
    artifacts.append({
        'name': path.name,
        'size_bytes': path.stat().st_size,
        'sha256': digest,
    })
    print(f'{path} ({path.stat().st_size:,} bytes)')

sign_result = _sign_installer(exe)
# Re-hash after signing - the exe bytes changed if signtool ran.
if sign_result.get('signed'):
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    exe.with_suffix(exe.suffix + '.sha256').write_text(
        f'{digest}  {exe.name}\n')
    for a in artifacts:
        if a['name'] == exe.name:
            a['sha256'] = digest
            a['size_bytes'] = exe.stat().st_size

# Release manifest - the file an auto-updater will fetch to compare
# the local install against what's on the release channel. Sits next to
# the installer so a downstream pipeline can publish the whole dist/
# directory as one unit.
manifest = {
    'product': 'StudioLite',
    'platform': 'windows-x64',
    'version': _git_describe() or '0.0.0-dev',
    'commit': _git_rev(),
    'built_at': _dt.datetime.now(_dt.timezone.utc)
                  .strftime('%Y-%m-%dT%H:%M:%SZ'),
    'artifacts': artifacts,
    'signing': sign_result,
    'notes': 'Requires internet, Python 3.11, Node.js 22+, and FFmpeg on '
             'first launch to fetch CPU dependencies.',
}
manifest_path = OUT / 'release-manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8')
print(f'{manifest_path} (signing: {sign_result})')
