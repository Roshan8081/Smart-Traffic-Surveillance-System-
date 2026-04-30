$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$root = $PSScriptRoot
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$mongoDataDir = Join-Path $root ".mongodb-data"
$logDir = Join-Path $root ".logs"
$mongoExe = "C:\Program Files\MongoDB\Server\8.2\bin\mongod.exe"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
if (-not (Test-Path $mongoDataDir)) {
    New-Item -ItemType Directory -Path $mongoDataDir | Out-Null
}

# Start MongoDB if not already running.
$mongoRunning = Get-NetTCPConnection -LocalPort 27017 -State Listen -ErrorAction SilentlyContinue
if (-not $mongoRunning) {
    if (-not (Test-Path $mongoExe)) {
        throw "MongoDB executable not found at: $mongoExe"
    }
    Start-Process -FilePath $mongoExe `
        -ArgumentList "--dbpath `"$mongoDataDir`" --bind_ip 127.0.0.1 --port 27017" `
        -RedirectStandardOutput (Join-Path $logDir "mongo.out.log") `
        -RedirectStandardError (Join-Path $logDir "mongo.err.log") `
        -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
    Write-Host "[OK] MongoDB started."
}
else {
    Write-Host "[OK] MongoDB already running."
}

# Install backend deps if needed and start backend in background.
if (-not (Test-Path (Join-Path $backendDir "node_modules"))) {
    Write-Host "[INFO] Installing backend dependencies..."
    & npm install --prefix $backendDir
}
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run dev --prefix `"$backendDir`"" `
    -RedirectStandardOutput (Join-Path $logDir "backend.out.log") `
    -RedirectStandardError (Join-Path $logDir "backend.err.log") `
    -WindowStyle Hidden | Out-Null
Write-Host "[OK] Backend starting on http://localhost:5000"

# Install frontend deps if needed and start frontend in background.
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "[INFO] Installing frontend dependencies..."
    & npm install --prefix $frontendDir
}
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run dev --prefix `"$frontendDir`"" `
    -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $logDir "frontend.err.log") `
    -WindowStyle Hidden | Out-Null
Write-Host "[OK] Frontend starting on http://localhost:5173"

# Ensure detection API target points to backend.
$env:BACKEND_API_BASE_URL = "http://localhost:5000/api"
Write-Host "[OK] BACKEND_API_BASE_URL=$env:BACKEND_API_BASE_URL"

Write-Host ""
Write-Host "Dashboard: http://localhost:5173"
Write-Host "API health: http://localhost:5000/api/health"
Write-Host ""
Write-Host "[INFO] Launching detection pipeline (press 'q' in video window to exit)..."

# Run detection in foreground so you can see video output.
& python (Join-Path $root "detection\main.py")
