from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

from runtime_profile import apply_runtime_profile, print_feature_summary

apply_runtime_profile()

import api as api_module
from database import init_db
from conversation_memory_v5 import conversation_memory_v5_enabled
from memory.persistent_memory_store import init_memory_db

FRONTEND_DIR = BASE_DIR / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(
        title="EvidenceRAG API",
        description="可检索、可引用、可验证、可审计的专业 RAG 服务。",
        version="0.2.0",
    )

    @app.on_event("startup")
    async def _startup_init_db():
        print_feature_summary()
        init_db()
        if conversation_memory_v5_enabled():
            init_memory_db()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/config/frontend")
    async def _frontend_config():
        conversation_memory_enabled = conversation_memory_v5_enabled()
        return {
            "use_conversation_api": os.getenv("USE_CONVERSATION_API", "false").strip().lower()
            in {"1", "true", "yes", "on"} and conversation_memory_enabled,
            "conversation_memory_enabled": conversation_memory_enabled,
        }

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)

    # serve frontend static files at root
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
