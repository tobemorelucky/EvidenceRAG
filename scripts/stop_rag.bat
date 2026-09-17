@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
if not exist "%PROJECT_ROOT%\docker-compose.yml" (
    echo [ERROR] Project root is invalid: "%PROJECT_ROOT%"
    echo         docker-compose.yml was not found next to the scripts directory.
    exit /b 1
)

pushd "%PROJECT_ROOT%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot enter project root: "%PROJECT_ROOT%"
    exit /b 1
)

set "RUN_DIR=%TEMP%\EvidenceRAG"
set "PID_FILE=%RUN_DIR%\backend.pid"
set "BACKEND_STATUS=not running"
set "DOCKER_STATUS=not running"
set "BACKEND_PID="

rem Always identify the live backend from its health endpoint and listening socket.
rem A PID file alone can be stale after Windows recycles a process identifier.
call :backend_pid

if defined BACKEND_PID (
    echo [INFO] Stopping EvidenceRAG backend PID !BACKEND_PID!...
    taskkill /PID !BACKEND_PID! /T /F >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Backend PID !BACKEND_PID! could not be stopped or had already exited.
        set "BACKEND_STATUS=not running"
    ) else (
        set "BACKEND_STATUS=stopped"
    )
) else (
    echo [INFO] EvidenceRAG backend is not running; skipping.
)

if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1

where docker >nul 2>&1
if errorlevel 1 (
    echo [WARN] Docker CLI was not found; Docker Compose stop was skipped.
    set "DOCKER_STATUS=skipped ^(Docker CLI unavailable^)"
    goto :summary
)
docker info >nul 2>&1
if errorlevel 1 (
    echo [INFO] Docker daemon is not running; Docker Compose stop was skipped.
    set "DOCKER_STATUS=not running ^(data preserved^)"
    goto :summary
)

echo [INFO] Stopping Docker Compose services without removing containers or volumes...
docker compose stop
if errorlevel 1 (
    echo [WARN] One or more Docker Compose services could not be stopped.
    set "DOCKER_STATUS=stop reported an error ^(data preserved^)"
) else (
    set "DOCKER_STATUS=stopped ^(data preserved^)"
)

:summary
echo.
echo ========================================================
echo EvidenceRAG stopped
echo.
echo Backend:
echo   !BACKEND_STATUS!
echo.
echo Docker:
echo   !DOCKER_STATUS!
echo ========================================================
popd >nul 2>&1
exit /b 0

:backend_pid
set "BACKEND_PID="
curl.exe --silent --fail --max-time 2 http://127.0.0.1:8000/config/frontend >nul 2>&1
if errorlevel 1 exit /b 0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do if not defined BACKEND_PID set "BACKEND_PID=%%P"
exit /b 0
