"""
FastMCP 工具服务 —— 法律Agent的统一工具注册中心。

包含三类工具：
  1. search_legal_docs  — 本地法律文档检索（实际执行由 retriever_node 处理）
  2. extract_memory     — 长期记忆提取（实际执行由 memory_node 处理）
  3. tavily_search      — 互联网搜索（通过 Tavily REST API 实际执行）

独立运行 MCP 服务:
    python Agent/mcp.py               # stdio 传输
    python Agent/mcp.py --transport sse --port 8080  # SSE 传输

在 LangGraph 中使用:
    from Agent.mcp import TOOL_SCHEMAS, run_tavily_search
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "legal-agent-tools",
    instructions="法律智能助手的工具服务，提供法律文档检索、记忆提取和互联网搜索能力。",
)

# ═══════════════════════════ Tool: search_legal_docs ══════════════════


@mcp.tool()
async def search_legal_docs(
    query: Annotated[str, Field(description="详细的法律检索查询，应包含相关法律名称、条款号、关键法律术语")],
) -> str:
    """搜索法律文档数据库，获取相关法律条文、司法解释和案例。当用户询问具体法律条文、需要准确引用法律规定、或涉及案例分析时使用。"""
    return f"[由 retriever_node 处理] query={query}"


# ═══════════════════════════ Tool: extract_memory ═════════════════════


@mcp.tool()
async def extract_memory(
    content: Annotated[str, Field(description="需要从中提取记忆的用户对话内容（原文）")],
) -> str:
    """从用户对话中提取关键事实和偏好并保存为长期记忆。当用户陈述了案件事实、个人情况、或明确偏好时使用。不要对用户的提问使用此工具。"""
    return f"[由 memory_node 处理] content={content}"


# ═══════════════════════════ Tool: tavily_search ══════════════════════


async def run_tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> str:
    """调用 Tavily REST API 执行互联网搜索，返回格式化结果。"""
    import httpx

    api_key = os.getenv("tavily_api_key")
    if not api_key:
        return "[错误] 未配置 tavily_api_key 环境变量"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"[Tavily API 错误] HTTP {exc.response.status_code}"
    except Exception as exc:
        return f"[Tavily 请求失败] {exc}"

    parts: list[str] = []
    if data.get("answer"):
        parts.append(f"**摘要**: {data['answer']}")
    for i, r in enumerate(data.get("results", []), 1):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        parts.append(f"[{i}] {title}\n{content}\n来源: {url}")

    return "\n\n".join(parts) or "未找到相关结果"


@mcp.tool()
async def tavily_search(
    query: Annotated[str, Field(description="互联网搜索查询关键词")],
    max_results: Annotated[int, Field(description="返回的最大结果数量", default=5)] = 5,
    search_depth: Annotated[
        str, Field(description="搜索深度: basic 或 advanced", default="advanced")
    ] = "advanced",
) -> str:
    """搜索互联网获取最新的法律新闻、判例、政策变化和法律解读。当本地法律数据库信息不足，或需要了解最新动态时使用。"""
    return await run_tavily_search(query, max_results, search_depth)


# ═══════════════════════════ TOOL_SCHEMAS (OpenAI format) ═════════════
# 供 nodes.py 导入后 bind 到 ChatModel

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_legal_docs",
            "description": (
                "搜索法律文档数据库，获取相关法律条文、司法解释和案例。"
                "当用户询问具体法律条文、需要准确引用法律规定、或涉及案例分析时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "详细的法律检索查询，应包含相关法律名称、条款号、关键法律术语",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_memory",
            "description": (
                "从用户对话中提取关键事实和偏好并保存为长期记忆。"
                "当用户陈述了案件事实、个人情况、或明确偏好时使用。"
                "不要对用户的提问使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "需要从中提取记忆的用户对话内容（原文）",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "搜索互联网获取最新的法律新闻、判例、政策变化和法律解读。"
                "当本地法律数据库信息不足，或需要了解最新动态时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "互联网搜索查询关键词",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回的最大结果数量",
                        "default": 5,
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "搜索深度: basic 或 advanced",
                        "default": "advanced",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ═══════════════════════════ CLI ══════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
