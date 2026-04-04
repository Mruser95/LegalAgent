"""
FastAPI 服务 —— 以 OpenAI 兼容的 /v1/chat/completions 接口暴露 LegalAgent。

启动方式:
    python -m Agent.api
    # 或
    uvicorn Agent.api:app --host 0.0.0.0 --port 8008
"""

from __future__ import annotations

import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from Agent.build_agent import LegalAgent

logger = logging.getLogger(__name__)

# ══════════════════════════════ Schemas ═══════════════════════════════

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "legal-agent"
    messages: list[Message]
    stream: bool = False
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = "default"
    temperature: float | None = None

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Usage()

# ══════════════════════════════ Agent Singleton ═══════════════════════

agent: LegalAgent | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = LegalAgent()
    await agent.setup()
    logger.info("LegalAgent initialized")
    yield
    await agent.close()
    logger.info("LegalAgent closed")

# ══════════════════════════════ App ═══════════════════════════════════

app = FastAPI(title="Legal Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════ Helpers ═══════════════════════════════

def _extract_user_input(messages: list[Message]) -> str:
    """取最后一条 user 消息作为输入。"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


async def _stream_sse(
    request: ChatCompletionRequest, user_input: str
) -> AsyncGenerator[str, None]:
    """生成 OpenAI 兼容的 SSE 流。"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = request.model

    yield (
        f"data: {{"
        f'"id":"{completion_id}",'
        f'"object":"chat.completion.chunk",'
        f'"created":{created},'
        f'"model":"{model}",'
        f'"choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]'
        f"}}\n\n"
    )

    try:
        async for token in agent.astream(
            user_input,
            thread_id=request.thread_id,
            user_id=request.user_id,
        ):
            escaped = token.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            yield (
                f"data: {{"
                f'"id":"{completion_id}",'
                f'"object":"chat.completion.chunk",'
                f'"created":{created},'
                f'"model":"{model}",'
                f'"choices":[{{"index":0,"delta":{{"content":"{escaped}"}},"finish_reason":null}}]'
                f"}}\n\n"
            )
    except Exception as exc:
        logger.error("Streaming error: %s", exc, exc_info=True)
        err_msg = f"[服务端错误] {type(exc).__name__}: {exc}"
        escaped = err_msg.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        yield (
            f"data: {{"
            f'"id":"{completion_id}",'
            f'"object":"chat.completion.chunk",'
            f'"created":{created},'
            f'"model":"{model}",'
            f'"choices":[{{"index":0,"delta":{{"content":"{escaped}"}},"finish_reason":null}}]'
            f"}}\n\n"
        )

    yield (
        f"data: {{"
        f'"id":"{completion_id}",'
        f'"object":"chat.completion.chunk",'
        f'"created":{created},'
        f'"model":"{model}",'
        f'"choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]'
        f"}}\n\n"
    )
    yield "data: [DONE]\n\n"


# ══════════════════════════════ Routes ════════════════════════════════

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_input = _extract_user_input(request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_sse(request, user_input),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        full_reply = await agent.ainvoke(
            user_input,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )
    except Exception as exc:
        logger.error("Invoke error: %s", exc, exc_info=True)
        full_reply = f"[服务端错误] {type(exc).__name__}: {exc}"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=Message(role="assistant", content=full_reply),
            )
        ],
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "legal-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "legal-agent",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": agent is not None and agent._graph is not None}


# ══════════════════════════════ CLI Entry ═════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Agent.api:app",
        host="0.0.0.0",
        port=8008,
        log_level="info",
    )
