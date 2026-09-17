from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts" / "start_rag.bat"
STOP = ROOT / "scripts" / "stop_rag.bat"


def _normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower().replace("^", "")


def test_start_script_has_docker_backend_and_duplicate_guards():
    script = _normalized(START)

    assert "docker info" in script
    assert "docker compose ps" in script
    assert "docker compose up -d" in script
    assert "'uvicorn','backend.app:app'" in script
    assert "'--host','127.0.0.1','--port','8000'" in script
    assert 'set "pythonpath=' in script
    assert "backend_pid" in script
    assert "curl.exe --silent --fail" in script
    assert "netstat -ano" in script
    assert "already running" in script
    assert "backend.pid" in script
    assert "sys.executable" in script
    assert "start-process -filepath $env:evidencerag_python" in script
    assert "waiting for backend readiness" in script
    assert "backend process exited before it became ready" in script
    assert "backend-launcher.pid" not in script
    assert "%~dp0.." in script
    assert "timeout /t" not in script


def test_stop_script_preserves_docker_data():
    script = _normalized(STOP)

    assert "docker compose stop" in script
    assert "taskkill /pid" in script
    assert "backend_pid" in script
    assert "curl.exe --silent --fail" in script
    assert "netstat -ano" in script
    assert "data preserved" in script
    assert "docker compose down" not in script
    assert "docker volume" not in script
    assert "docker image" not in script
    assert "docker system prune" not in script


def test_pid_file_is_outside_repository_and_shared_by_both_scripts():
    start = _normalized(START)
    stop = _normalized(STOP)

    marker = "%temp%\\evidencerag"
    assert marker in start
    assert marker in stop
    assert "backend.pid" in start and "backend.pid" in stop


def test_readme_documents_start_stop_and_logs():
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert "scripts\\start_rag.bat" in readme
    assert "scripts\\stop_rag.bat" in readme
    assert "docker compose stop" in readme
    assert "%TEMP%\\EvidenceRAG" in readme
