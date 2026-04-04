"""
法律智能助手 Chat UI —— 基于 FastAPI 的前端服务。

启动方式:
    python -m UI.app
    # 或
    uvicorn UI.app:app --host 0.0.0.0 --port 8009
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Legal Agent Chat UI")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/config")
async def config():
    """向前端暴露 agent API 地址，避免硬编码。"""
    return {
        "agent_base_url": os.getenv("agent_base_url", "http://localhost:8008/v1"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "UI.app:app",
        host="0.0.0.0",
        port=8009,
        log_level="info",
    )
