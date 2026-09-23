[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'StudioLite'),
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Setup can be rerun after correcting the error." }
}
try {
    Write-Host 'StudioLite Setup - checking prerequisites'
    foreach ($command in 'python.exe', 'node.exe', 'npm.cmd', 'ffmpeg.exe', 'ffprobe.exe') {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Missing $command. Install Python 3.11, Node.js 22 LTS and FFmpeg (including ffprobe), add them to PATH, then rerun setup."
        }
    }
    Invoke-Checked 'python.exe' @('-c', 'import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32, "64-bit Python 3.11 is required for this setup"')
    Invoke-Checked 'node.exe' @('-e', 'if(parseInt(process.versions.node)<22) process.exit(1)')
    Invoke-Checked 'ffmpeg.exe' @('-version')
    Invoke-Checked 'ffprobe.exe' @('-version')
    if ($CheckOnly) { Write-Host 'Prerequisites passed.'; exit 0 }
    $InstallDir = [IO.Path]::GetFullPath($InstallDir)
    if ((Test-Path (Join-Path $InstallDir 'api_server.py')) -and -not (Test-Path (Join-Path $InstallDir '.setup-incomplete'))) {
        throw "An installation already exists at $InstallDir. Choose a new -InstallDir to preserve the existing application and data."
    }
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Set-Content -LiteralPath (Join-Path $InstallDir '.setup-incomplete') -Value '1'
    $log = Join-Path $InstallDir 'setup.log'
    Start-Transcript -Path $log -Append | Out-Null
    try {
        Expand-Archive -LiteralPath (Join-Path $PSScriptRoot 'payload.zip') -DestinationPath $InstallDir -Force
        Push-Location $InstallDir
        try {
            Invoke-Checked 'python.exe' @('-m', 'venv', 'venv')
            $python = Join-Path $InstallDir 'venv\Scripts\python.exe'
            Invoke-Checked $python @('-m', 'pip', 'install', '--upgrade', 'pip')
            Invoke-Checked $python @('-m', 'pip', 'install', '-r', 'requirements-cpu.txt')
            Invoke-Checked $python @('-m', 'pip', 'install', '-r', 'requirements.txt')
            Invoke-Checked $python @('-m', 'pip', 'install', '-r', 'packaging/windows/requirements-api.txt')
            Push-Location 'web'
            try {
                Invoke-Checked 'npm.cmd' @('ci')
                Invoke-Checked 'npm.cmd' @('run', 'build')
            } finally { Pop-Location }
            Set-Content -LiteralPath '.production' -Value '1' -Encoding ascii
            & (Join-Path $InstallDir 'install_shortcut.ps1')
            Remove-Item -LiteralPath (Join-Path $InstallDir '.setup-incomplete')
            Write-Host "Installed to $InstallDir. Models download when requested."
        } finally { Pop-Location }
    } finally { Stop-Transcript | Out-Null }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
