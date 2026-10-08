@echo off
echo ========================================================
echo Starting Skin Disease Detection App
echo ========================================================

:: Check if Docker is available and running
docker info >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Starting container with Docker Compose...
    docker compose up -d
    echo.
    echo Application running at: http://localhost:5000
    goto :end
)

echo Docker is not running. Starting directly with Python...
if exist .venv\Scripts\python.exe (
    echo Using virtual environment (.venv)...
    .venv\Scripts\python.exe app.py
) else (
    echo Using system Python...
    python app.py
)

:end
