# StudioLite one-click launcher.
# Starts FastAPI (uvicorn) on :8000 and Next.js dev on :3000, waits for both
# to come up, opens the browser, then tails interleaved logs in this window.
# Ctrl+C cleanly stops both servers.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$logDir = Join-Path $root '.mp\launcher'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$apiOut = Join-Path $logDir 'api.out.log'
$apiErr = Join-Path $logDir 'api.err.log'
$webOut = Join-Path $logDir 'web.out.log'
$webErr = Join-Path $logDir 'web.err.log'
foreach ($f in $apiOut, $apiErr, $webOut, $webErr) {
    '' | Set-Content -Path $f -Encoding ascii
}

$venvPython = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "venv missing at $venvPython" -ForegroundColor Red
    Write-Host "Run: python -m venv venv ; .\venv\Scripts\Activate.ps1 ; pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

$webDir = Join-Path $root 'web'
if (-not (Test-Path (Join-Path $webDir 'node_modules'))) {
    Write-Host "web/node_modules missing - running 'npm install' first..." -ForegroundColor Yellow
    Push-Location $webDir
    try { & cmd.exe /c "npm install" } finally { Pop-Location }
}

Write-Host ''
Write-Host '  StudioLite launcher' -ForegroundColor Cyan
Write-Host '  ===================' -ForegroundColor Cyan
Write-Host "  Root : $root"
Write-Host "  Logs : $logDir"
Write-Host ''

function Test-LocalPort([int]$port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $c.BeginConnect('127.0.0.1', $port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(400, $false)
        return ($ok -and $c.Connected)
    } catch { return $false } finally { $c.Close() }
}

# Refuse to start if ports are already taken (likely a previous launcher).
foreach ($p in 8000, 3000) {
    if (Test-LocalPort $p) {
        Write-Host "Port $p is already in use. Close the existing server (or kill old uvicorn/node) and retry." -ForegroundColor Red
        Read-Host 'Press Enter to close'
        exit 1
    }
}

Write-Host 'Starting API server  (uvicorn) on http://localhost:8000 ...'
$apiProc = Start-Process -FilePath $venvPython `
    -ArgumentList @('-u', '-m', 'uvicorn', 'api_server:app', '--host', '127.0.0.1', '--port', '8000') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $apiOut `
    -RedirectStandardError $apiErr `
    -WindowStyle Hidden `
    -PassThru

Write-Host 'Starting Next.js dev server  on http://localhost:3000 ...'
$webProc = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/c', 'npm', 'run', 'dev') `
    -WorkingDirectory $webDir `
    -RedirectStandardOutput $webOut `
    -RedirectStandardError $webErr `
    -WindowStyle Hidden `
    -PassThru

function Stop-Tree([System.Diagnostics.Process]$p) {
    if (-not $p) { return }
    try {
        $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($p.Id)" -ErrorAction SilentlyContinue
        foreach ($k in $kids) {
            try {
                $grand = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($k.ProcessId)" -ErrorAction SilentlyContinue
                foreach ($g in $grand) { Stop-Process -Id $g.ProcessId -Force -ErrorAction SilentlyContinue }
                Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue
            } catch {}
        }
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    } catch {}
}

try {
    $deadline = (Get-Date).AddMinutes(3)
    $apiUp = $false; $webUp = $false
    while ((Get-Date) -lt $deadline -and -not ($apiUp -and $webUp)) {
        if (-not $apiUp) { $apiUp = Test-LocalPort 8000 }
        if (-not $webUp) { $webUp = Test-LocalPort 3000 }
        $apiTag = if ($apiUp) { '[OK]' } else { '[..]' }
        $webTag = if ($webUp) { '[OK]' } else { '[..]' }
        Write-Host ("`r  api {0}   web {1}   " -f $apiTag, $webTag) -NoNewline
        Start-Sleep -Milliseconds 600
        if ($apiProc.HasExited) {
            Write-Host ''
            Write-Host 'API process exited unexpectedly.' -ForegroundColor Red
            foreach ($f in $apiErr, $apiOut) {
                if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
                    Write-Host ("--- {0} (last 30 lines) ---" -f (Split-Path $f -Leaf)) -ForegroundColor DarkGray
                    Get-Content -Path $f -Tail 30 | ForEach-Object { Write-Host "  $_" }
                }
            }
            throw 'api start failed'
        }
        if ($webProc.HasExited) {
            Write-Host ''
            Write-Host 'Web process exited unexpectedly.' -ForegroundColor Red
            foreach ($f in $webErr, $webOut) {
                if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
                    Write-Host ("--- {0} (last 30 lines) ---" -f (Split-Path $f -Leaf)) -ForegroundColor DarkGray
                    Get-Content -Path $f -Tail 30 | ForEach-Object { Write-Host "  $_" }
                }
            }
            throw 'web start failed'
        }
    }
    Write-Host ''

    if (-not $apiUp) { Write-Host "API did not respond on :8000 within 3 min. See $logDir" -ForegroundColor Red; throw 'api timeout' }
    if (-not $webUp) { Write-Host "Next.js did not respond on :3000 within 3 min. See $logDir" -ForegroundColor Red; throw 'web timeout' }

    Write-Host ''
    Write-Host '  Both servers are up.' -ForegroundColor Green
    Write-Host '  Web : http://localhost:3000'
    Write-Host '  API : http://localhost:8000/docs'
    Write-Host ''
    Start-Process 'http://localhost:3000' | Out-Null

    Write-Host 'Press Ctrl+C to stop both servers.' -ForegroundColor Cyan
    Write-Host '=================================================' -ForegroundColor DarkGray

    $streams = @(
        @{ Tag = 'api'; Path = $apiOut; Color = 'DarkCyan' },
        @{ Tag = 'api'; Path = $apiErr; Color = 'DarkCyan' },
        @{ Tag = 'web'; Path = $webOut; Color = 'DarkMagenta' },
        @{ Tag = 'web'; Path = $webErr; Color = 'DarkMagenta' }
    )
    foreach ($s in $streams) { $s.Pos = (Get-Item $s.Path).Length }

    while ($true) {
        Start-Sleep -Milliseconds 600
        foreach ($s in $streams) {
            $size = (Get-Item $s.Path).Length
            if ($size -gt $s.Pos) {
                $fs = [System.IO.File]::Open($s.Path, 'Open', 'Read', 'ReadWrite')
                try {
                    $fs.Seek($s.Pos, 'Begin') | Out-Null
                    $sr = New-Object System.IO.StreamReader($fs)
                    $chunk = $sr.ReadToEnd()
                } finally { $fs.Close() }
                $s.Pos = $size
                foreach ($line in ($chunk -split "`r?`n")) {
                    if ($line.Length -gt 0) {
                        Write-Host ("[{0}] " -f $s.Tag) -NoNewline -ForegroundColor $s.Color
                        Write-Host $line
                    }
                }
            }
        }
        if ($apiProc.HasExited -or $webProc.HasExited) {
            Write-Host ''
            Write-Host 'A server exited. Stopping the other.' -ForegroundColor Yellow
            break
        }
    }
}
finally {
    Write-Host ''
    Write-Host 'Stopping services...' -ForegroundColor Yellow
    Stop-Tree $apiProc
    Stop-Tree $webProc
    Write-Host 'Done.' -ForegroundColor Yellow
}
