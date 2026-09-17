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
set "BACKEND_LOG=%RUN_DIR%\backend.log"
set "BACKEND_ERROR_LOG=%RUN_DIR%\backend-error.log"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%" >nul 2>&1

where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker CLI was not found. Install Docker Desktop and add docker.exe to PATH.
    goto :fail
)

call :docker_ok
if errorlevel 1 (
    echo [INFO] Docker daemon is not ready. Attempting to start Docker Desktop...
    call :docker_start
    if errorlevel 1 goto :fail
    call :wait_docker 120
    if errorlevel 1 (
        echo [ERROR] Docker daemon did not become ready within 120 seconds.
        echo         Start Docker Desktop manually, wait until it is ready, then rerun this script.
        goto :fail
    )
)
set "DOCKER_STATUS=OK"

docker compose version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose v2 is unavailable. Verify that "docker compose" works.
    goto :fail
)

set /a EXPECTED_SERVICES=0
set /a RUNNING_SERVICES=0
for /f "usebackq delims=" %%S in (`docker compose config --services 2^>nul`) do set /a EXPECTED_SERVICES+=1
if !EXPECTED_SERVICES! LEQ 0 (
    echo [ERROR] No services were found in docker-compose.yml.
    goto :fail
)
for /f "usebackq delims=" %%S in (`docker compose ps --status running --services 2^>nul`) do set /a RUNNING_SERVICES+=1

if !RUNNING_SERVICES! LSS !EXPECTED_SERVICES! (
    echo [INFO] Starting Docker Compose services ^(!RUNNING_SERVICES!/!EXPECTED_SERVICES! already running^)...
    docker compose up -d
    if errorlevel 1 (
        echo [ERROR] Failed to start Docker Compose services.
        goto :fail
    )
) else (
    echo [INFO] Docker Compose services are already running.
)

call :backend_pid
if defined BACKEND_PID (
    >"%PID_FILE%" echo !BACKEND_PID!
    set "BACKEND_STATUS=OK ^(already running, PID !BACKEND_PID!^)"
    goto :success
)

call :port_used
if not errorlevel 1 (
    echo [ERROR] Port 8000 is occupied by a process that is not the EvidenceRAG backend.
    echo         Stop that process or change its port before starting EvidenceRAG.
    goto :fail
)

where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda was not found in PATH. Open an Anaconda Prompt or initialize Conda first.
    goto :fail
)

echo [INFO] Starting EvidenceRAG backend in Conda environment "rag"...
if exist "%BACKEND_LOG%" del /q "%BACKEND_LOG%" >nul 2>&1
if exist "%BACKEND_ERROR_LOG%" del /q "%BACKEND_ERROR_LOG%" >nul 2>&1

rem Resolve the environment interpreter once, then start it directly. Capturing the
rem output of Start-Process through FOR /F can keep the pipe open for the lifetime of
rem uvicorn, which makes this batch file appear to hang even though port 8000 is ready.
set "RAG_PYTHON="
for /f "usebackq delims=" %%P in (`conda run -n rag python -c "import sys; sys.stdout.write(sys.executable)" 2^>nul`) do for %%Q in ("%%P") do set "RAG_PYTHON=%%~fQ"
if not defined RAG_PYTHON (
    echo [ERROR] Conda environment "rag" is unavailable or its Python could not be resolved.
    goto :fail
)
if not exist "!RAG_PYTHON!" (
    echo [ERROR] Python executable does not exist: "!RAG_PYTHON!"
    goto :fail
)

set "PYTHONPATH=%PROJECT_ROOT%\backend;%PROJECT_ROOT%;%PYTHONPATH%"
set "EVIDENCERAG_PYTHON=!RAG_PYTHON!"
set "EVIDENCERAG_PID_FILE=%PID_FILE%"
set "EVIDENCERAG_BACKEND_LOG=%BACKEND_LOG%"
set "EVIDENCERAG_BACKEND_ERROR_LOG=%BACKEND_ERROR_LOG%"
powershell -NoProfile -Command "$process = Start-Process -FilePath $env:EVIDENCERAG_PYTHON -ArgumentList @('-m','uvicorn','backend.app:app','--host','127.0.0.1','--port','8000') -WorkingDirectory '%PROJECT_ROOT%' -RedirectStandardOutput $env:EVIDENCERAG_BACKEND_LOG -RedirectStandardError $env:EVIDENCERAG_BACKEND_ERROR_LOG -WindowStyle Hidden -PassThru; Set-Content -LiteralPath $env:EVIDENCERAG_PID_FILE -Value $process.Id -Encoding Ascii" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to create the backend process.
    goto :fail
)
set /p BACKEND_PID=<"%PID_FILE%"
if not defined BACKEND_PID (
    echo [ERROR] Backend process started without returning a PID.
    goto :fail
)

set /a BACKEND_WAIT=180
:backend_loop
curl.exe --silent --fail --max-time 2 http://127.0.0.1:8000/config/frontend >nul 2>&1
if not errorlevel 1 goto :backend_ready
tasklist /FI "PID eq !BACKEND_PID!" /NH 2>nul | findstr /C:"!BACKEND_PID!" >nul
if errorlevel 1 (
    echo [ERROR] Backend process exited before it became ready.
    echo         Standard output: "%BACKEND_LOG%"
    echo         Error output:    "%BACKEND_ERROR_LOG%"
    goto :fail
)
if !BACKEND_WAIT! LEQ 0 (
    echo [ERROR] Backend did not listen on http://127.0.0.1:8000 within 180 seconds.
    echo         Standard output: "%BACKEND_LOG%"
    echo         Error output:    "%BACKEND_ERROR_LOG%"
    goto :fail
)
set /a BACKEND_ELAPSED=180-BACKEND_WAIT
set /a BACKEND_PROGRESS=BACKEND_ELAPSED%%10
if !BACKEND_PROGRESS! EQU 0 echo [INFO] Waiting for backend readiness... !BACKEND_ELAPSED!/180 seconds
ping 127.0.0.1 -n 3 >nul
set /a BACKEND_WAIT-=2
goto :backend_loop

:backend_ready
set "BACKEND_STATUS=OK ^(PID !BACKEND_PID!^)"
goto :success

:success
echo.
echo ========================================================
echo EvidenceRAG Started
echo.
echo Docker:
echo   !DOCKER_STATUS!
echo.
echo Backend:
echo   !BACKEND_STATUS!
echo.
echo Frontend:
echo   http://127.0.0.1:8000
echo.
echo API Docs:
echo   http://127.0.0.1:8000/docs
echo ========================================================
popd >nul 2>&1
exit /b 0

:docker_ok
docker info >nul 2>&1
exit /b %errorlevel%

:docker_start
set "DOCKER_DESKTOP_EXE="
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not defined DOCKER_DESKTOP_EXE if exist "%LOCALAPPDATA%\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP_EXE=%LOCALAPPDATA%\Docker\Docker Desktop.exe"
if not defined DOCKER_DESKTOP_EXE (
    echo [ERROR] Docker Desktop executable was not found in its standard locations.
    echo         Start Docker Desktop manually and rerun this script.
    exit /b 1
)
powershell -NoProfile -Command "Start-Process -FilePath '%DOCKER_DESKTOP_EXE%' -WindowStyle Hidden" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop could not be started: "%DOCKER_DESKTOP_EXE%"
    exit /b 1
)
exit /b 0

:wait_docker
set /a DOCKER_WAIT=%~1
:docker_loop
call :docker_ok
if not errorlevel 1 exit /b 0
if !DOCKER_WAIT! LEQ 0 exit /b 1
ping 127.0.0.1 -n 3 >nul
set /a DOCKER_WAIT-=2
goto :docker_loop

:backend_pid
set "BACKEND_PID="
curl.exe --silent --fail --max-time 2 http://127.0.0.1:8000/config/frontend >nul 2>&1
if errorlevel 1 exit /b 0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do if not defined BACKEND_PID set "BACKEND_PID=%%P"
exit /b 0

:port_used
set "PORT_8000_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do if not defined PORT_8000_PID set "PORT_8000_PID=%%P"
if defined PORT_8000_PID exit /b 0
exit /b 1

:fail
if defined BACKEND_PID call :kill_backend !BACKEND_PID!
popd >nul 2>&1
exit /b 1

:kill_backend
set "FAILED_BACKEND_PID=%~1"
taskkill /PID !FAILED_BACKEND_PID! /T /F >nul 2>&1
if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
exit /b 0
