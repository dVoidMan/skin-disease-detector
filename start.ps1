# PowerShell Start Script for Skin Disease Detection App
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting Skin Disease Detection App" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Test if Docker daemon is running
$dockerAvailable = $false
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerAvailable = $true
    }
} catch {
    $dockerAvailable = $false
}

if ($dockerAvailable) {
    Write-Host "Starting container with Docker Compose..." -ForegroundColor Green
    docker compose up -d
    Write-Host "`nApplication running at: http://localhost:5000" -ForegroundColor Green
} else {
    Write-Host "Docker is not running. Starting directly with Python..." -ForegroundColor Yellow
    if (Test-Path ".venv\Scripts\python.exe") {
        Write-Host "Using virtual environment (.venv)..." -ForegroundColor Green
        & .venv\Scripts\python.exe app.py
    } else {
        Write-Host "Using system Python..." -ForegroundColor Green
        python app.py
    }
}
