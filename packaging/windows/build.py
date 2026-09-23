"""Build a source-based Windows setup; never package local data or credentials."""
from pathlib import Path
import hashlib
import subprocess
import zipfile

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
TargetName={exe}
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
if not exe.exists() or exe.stat().st_size == 0:
    raise RuntimeError('IExpress did not produce the installer')
for path in (exe, OUT / 'StudioLite-Setup.zip'):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + '.sha256').write_text(f'{digest}  {path.name}\n')
    print(f'{path} ({path.stat().st_size:,} bytes)')
