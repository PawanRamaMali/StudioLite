# Creates a Desktop shortcut "StudioLite.lnk" that runs launch.bat.
# Re-running this overwrites the existing shortcut.

[CmdletBinding()]
param(
    [string]$ShortcutName = 'StudioLite'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $root 'launch.bat'
$icon = Join-Path $root 'web\app\favicon.ico'

if (-not (Test-Path $target)) {
    Write-Host "launch.bat not found at $target" -ForegroundColor Red
    exit 1
}

$desktop = [Environment]::GetFolderPath('Desktop')
$linkPath = Join-Path $desktop "$ShortcutName.lnk"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($linkPath)
$lnk.TargetPath = $target
$lnk.WorkingDirectory = $root
$lnk.Description = 'Launch StudioLite (FastAPI + Next.js) and open the web UI'
$lnk.WindowStyle = 1  # Normal window
if (Test-Path $icon) { $lnk.IconLocation = "$icon,0" }
$lnk.Save()

# Mark the shortcut "Run as Administrator" off; just to be safe set the byte
# flag clear (some setups inherit elevated flag from a parent context).
try {
    $bytes = [System.IO.File]::ReadAllBytes($linkPath)
    if ($bytes.Length -ge 22) {
        # Byte 21 bit 0x20 = "Run as administrator". Clear it.
        $bytes[21] = $bytes[21] -band (-bnot 0x20)
        [System.IO.File]::WriteAllBytes($linkPath, $bytes)
    }
} catch {}

Write-Host ''
Write-Host "Desktop shortcut created:" -ForegroundColor Green
Write-Host "  $linkPath"
Write-Host ''
Write-Host 'To pin to taskbar:' -ForegroundColor Cyan
Write-Host '  1. Right-click the desktop shortcut'
Write-Host '  2. Choose "Show more options" (Windows 11) -> "Pin to taskbar"'
Write-Host '     (or just drag the shortcut onto the taskbar)'
Write-Host ''
