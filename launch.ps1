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
    $hasGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasGpu) {
        Write-Host "NVIDIA GPU detected. Run:" -ForegroundColor Yellow
        Write-Host "  python -m venv venv ; .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
        Write-Host "  pip install -r requirements-cuda.txt ; pip install -r requirements.txt" -ForegroundColor Yellow
    } else {
        Write-Host "No NVIDIA GPU detected - installing CPU-only torch wheels:" -ForegroundColor Yellow
        Write-Host "  python -m venv venv ; .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
        Write-Host "  pip install -r requirements-cpu.txt ; pip install -r requirements.txt" -ForegroundColor Yellow
    }
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

function Get-PortHolder([int]$port) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -First 1
        if (-not $conn) { return $null }
        $pid_ = $conn.OwningProcess
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $pid_" -ErrorAction SilentlyContinue
        $name = if ($cim -and $cim.Name) { $cim.Name } else { '?' }
        $cmd  = if ($cim -and $cim.CommandLine) { $cim.CommandLine } else { '' }
        return [pscustomobject]@{ Pid = $pid_; Name = $name; CommandLine = $cmd }
    } catch { return $null }
}

# A port holder is "ours" (a stale StudioLite) when the command line clearly
# names our server entrypoints. We only auto-reclaim those; a stranger (an
# unrelated dev server from another project) is left alone and we move to
# the next free port instead.
function Test-IsStudioLiteHolder($holder, [string]$kind) {
    if (-not $holder) { return $false }
    $cmd = $holder.CommandLine
    if (-not $cmd) { return $false }
    if ($kind -eq 'api') {
        return ($cmd -match 'uvicorn\s+api_server:app' -or $cmd -match 'api_server\.py')
    }
    if ($kind -eq 'web') {
        $webDir = (Join-Path $root 'web').Replace('\', '\\')
        return ($cmd -match "next.*(dev|start)" -and $cmd -match $webDir) `
            -or ($cmd -match 'next-server')
    }
    return $false
}

function Find-FreePort([int]$start, [int]$maxTries, [string]$kind) {
    for ($i = 0; $i -lt $maxTries; $i++) {
        $p = $start + $i
        if (-not (Test-LocalPort $p)) { return $p }
        if ($i -eq 0) {
            $holder = Get-PortHolder $p
            if (Test-IsStudioLiteHolder $holder $kind) {
                Write-Host ("  Port {0} held by a stale StudioLite (pid {1}); reclaiming." -f $p, $holder.Pid) -ForegroundColor Yellow
                try { Stop-Process -Id $holder.Pid -Force -ErrorAction Stop } catch {}
                Start-Sleep -Milliseconds 400
                if (-not (Test-LocalPort $p)) { return $p }
            } elseif ($holder) {
                Write-Host ("  Port {0} held by {1} (pid {2}); using an alternate port." -f $p, $holder.Name, $holder.Pid) -ForegroundColor Yellow
            } else {
                Write-Host ("  Port {0} appears taken; using an alternate port." -f $p) -ForegroundColor Yellow
            }
        }
    }
    return 0
}

$apiPort = Find-FreePort 8000 20 'api'
$webPort = Find-FreePort 3000 20 'web'
if ($apiPort -eq 0) {
    Write-Host 'No free API port found in 8000-8019.' -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
if ($webPort -eq 0) {
    Write-Host 'No free web port found in 3000-3019.' -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}

# Mint the API token up front (api_server reuses .auth) so the web server
# can hand the same token to the UI; see web/app/layout.tsx.
$authFile = Join-Path $root '.auth'
if (-not "$(if (Test-Path $authFile) { Get-Content -Raw $authFile })".Trim()) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    [System.IO.File]::WriteAllText($authFile, $token)
}
$env:STUDIOLITE_API_TOKEN = "$(Get-Content -Raw $authFile)".Trim()

$apiUrl = "http://localhost:$apiPort"
$webUrl = "http://localhost:$webPort"

# API's CORS allowlist defaults to :3000; when the launcher moves the web
# port, teach the API about the new origin before it boots or every browser
# fetch gets rejected at the CORS preflight.
$env:STUDIOLITE_ALLOWED_ORIGINS = "http://localhost:$webPort,http://127.0.0.1:$webPort"

Write-Host "Starting API server  (uvicorn) on $apiUrl ..."
$apiProc = Start-Process -FilePath $venvPython `
    -ArgumentList @('-u', '-m', 'uvicorn', 'api_server:app', '--host', '127.0.0.1', '--port', "$apiPort", '--ws-ping-interval', '30', '--ws-ping-timeout', '90') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $apiOut `
    -RedirectStandardError $apiErr `
    -WindowStyle Hidden `
    -PassThru

# Children of this launcher inherit our environment, so we point the web
# server at the chosen API port via two env vars: STUDIOLITE_API_URL feeds
# next.config.ts's rewrite target, NEXT_PUBLIC_API_URL is the direct
# browser-side base read by web/lib/api.ts.
$env:STUDIOLITE_API_URL = $apiUrl
$env:NEXT_PUBLIC_API_URL = $apiUrl

$webMode = if (Test-Path (Join-Path $root '.production')) { 'start' } else { 'dev' }
Write-Host "Starting Next.js ($webMode) on $webUrl ..."
$webProc = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/c', 'npm', 'run', $webMode, '--', '--hostname', '127.0.0.1', '--port', "$webPort") `
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
        if (-not $apiUp) { $apiUp = Test-LocalPort $apiPort }
        if (-not $webUp) { $webUp = Test-LocalPort $webPort }
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

    if (-not $apiUp) { Write-Host "API did not respond on :$apiPort within 3 min. See $logDir" -ForegroundColor Red; throw 'api timeout' }
    if (-not $webUp) { Write-Host "Next.js did not respond on :$webPort within 3 min. See $logDir" -ForegroundColor Red; throw 'web timeout' }

    Write-Host ''
    Write-Host '  Both servers are up.' -ForegroundColor Green
    Write-Host "  Web : $webUrl"
    Write-Host "  API : $apiUrl/docs"
    Write-Host ''
    Start-Process $webUrl | Out-Null

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
