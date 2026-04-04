"""
法律Agent的状态定义、节点实现和路由函数。

图结构:
    START → manager → route:
        → retriever → manager        (检索循环)
        → memory   → manager        (记忆循环)
        → tavily   → manager        (网络搜索循环)
        → short_mem → user_input    (压缩后等待输入)
        → user_input                (直接等待输入)
    user_input → manager            (新一轮对话)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages
from langgraph.types import interrupt

from Agent.mcp import TOOL_SCHEMAS, run_tavily_search
from Agent.prompt import MANAGER_SYSTEM_PROMPT, MEMORY_EXTRACT_PROMPT, SHORT_MEM_COMPRESS_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ========================================== State =============================================


class AgentState(TypedDict):
    """法律Agent全局状态"""

    messages: Annotated[list[AnyMessage], add_messages]
    tasks: list[str]                 # 执行轨迹
    memories: list[dict[str, Any]]   # 长期记忆 [{type, content, priority, timestamp}, ...]
    compressed_history: list[dict]   # 压缩后的早期对话摘要
    retrieval_context: str           # 最近一次检索结果
    turn_count: int                  # 用户对话轮次


# ========================================== Helpers =============================================


def _get_llm(config: RunnableConfig | None = None) -> ChatOpenAI:
    cfg = (config or {}).get("configurable", {})
    return ChatOpenAI(
        model=cfg.get("model", os.getenv("GEN_MODEL") or os.getenv("VLLM_LORA_NAME", "legal-lora")),
        base_url=cfg.get("base_url", os.getenv("base_url")),
        api_key=cfg.get("api_key", os.getenv("api_key")),
        temperature=0.7,
        streaming=True,
        timeout=120,
        max_retries=1,
    )


def _load_skills(skill_names: list[str]) -> str:
    """从 Skills/ 目录加载指定的 skill 文件，拼接为一段文本。"""
    skills_dir = PROJECT_ROOT / "Skills"
    parts: list[str] = []
    for name in skill_names:
        path = skills_dir / f"{name}.md"
        if path.exists():
            parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def _build_manager_messages(
    state: AgentState, config: RunnableConfig | None = None
) -> list[AnyMessage]:
    """将压缩历史、记忆、检索上下文、Skills 注入 system prompt，拼接会话消息。"""
    parts: list[str] = [MANAGER_SYSTEM_PROMPT]

    # 注入 Skills（工具使用指南）
    skills = (config or {}).get("configurable", {}).get("skills", [])
    if skills:
        skill_text = _load_skills(skills)
        if skill_text:
            parts.append(f"\n## 工具使用指南\n{skill_text}")

    if state.get("compressed_history"):
        parts.append("\n## 历史对话摘要（已压缩的早期对话）")
        for idx, ch in enumerate(state["compressed_history"], 1):
            parts.append(f"### 摘要 {idx}")
            parts.append(json.dumps(ch, ensure_ascii=False, indent=2))

    memories = state.get("memories", [])
    if memories:
        parts.append("\n## 已记录的用户信息")
        facts = sorted(
            [m for m in memories if m.get("type") == "fact"],
            key=lambda m: m.get("priority", 1),
            reverse=True,
        )
        prefs = [m for m in memories if m.get("type") == "preference"]
        if facts:
            parts.append("### 事实记忆（按优先级排序）")
            parts.extend(f"- {f['content']}" for f in facts)
        if prefs:
            parts.append("### 用户偏好")
            parts.extend(f"- {p['content']}" for p in prefs)

    if state.get("retrieval_context"):
        parts.append(f"\n## 检索到的法律文档\n{state['retrieval_context']}")

    system_msg = SystemMessage(content="\n".join(parts))
    conv = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    return [system_msg] + conv


def _count_user_turns(messages: list[AnyMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _find_compress_boundary(messages: list[AnyMessage], keep_turns: int = 4) -> int:
    """返回压缩边界索引：该索引之前的消息将被压缩，之后的保留。"""
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= keep_turns:
        return 0
    return human_indices[-keep_turns]


def _strip_code_fence(text: str) -> str:
    """移除 LLM 输出中可能包裹的 markdown 代码块标记。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ========================================== Nodes =============================================


async def manager_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    主Agent节点 —— 分析用户输入，决定直接回答还是调用子Agent。
    通过 tool calling 与子Agent交互，每次最多调用一个工具。
    """
    logger.info("manager_node: START (turn=%s, msg_count=%s)",
                state.get("turn_count"), len(state.get("messages", [])))
    llm = _get_llm(config)

    store = config.get("configurable", {}).get("store")
    memories = state.get("memories", [])
    if not memories and store is not None:
        user_id = config.get("configurable", {}).get("user_id", "default")
        try:
            items = await store.asearch(("memories", user_id), limit=50)
            memories = [item.value for item in items]
        except Exception:
            logger.debug("Failed to load memories from store", exc_info=True)

    effective_state = {**state, "memories": memories} if memories else state
    messages = _build_manager_messages(effective_state, config)

    logger.info("manager_node: calling LLM (%s messages)...", len(messages))
    response = await llm.bind_tools(
        TOOL_SCHEMAS, parallel_tool_calls=False
    ).ainvoke(messages, config=config)

    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    logger.info("manager_node: LLM returned (content_len=%s, tool_calls=%s)",
                len(response.content) if response.content else 0,
                [tc["name"] for tc in tool_calls] if tool_calls else "none")

    result: dict[str, Any] = {
        "messages": [response],
        "tasks": state.get("tasks", []) + [f"manager:{datetime.now().isoformat()}"],
    }
    if memories and not state.get("memories"):
        result["memories"] = memories
    return result


async def memory_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    记忆子Agent —— 提取"事实"与"用户偏好"两类关键记忆。
    不记录模型回答，不记录用户疑问，越近的记忆优先级越高。
    使用 langmem 模式存储到 PostgreSQL。
    """
    last_ai = state["messages"][-1]
    if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
        return {}

    tc = next((t for t in last_ai.tool_calls if t["name"] == "extract_memory"), None)
    if not tc:
        return {}

    content = tc["args"]["content"]
    user_id = config.get("configurable", {}).get("user_id", "default")
    store = config.get("configurable", {}).get("store")

    llm = _get_llm(config)
    extract_resp = await llm.ainvoke([
        SystemMessage(content=MEMORY_EXTRACT_PROMPT),
        HumanMessage(content=content),
    ])

    new_memories = list(state.get("memories", []))
    result_text = "未提取到新记忆"

    try:
        raw = _strip_code_fence(extract_resp.content)
        extracted: list[dict] = json.loads(raw)
        if isinstance(extracted, list) and extracted:
            for mem in extracted:
                mem["timestamp"] = datetime.now().isoformat()
                new_memories.append(mem)
                # 持久化到 PostgreSQL store（langmem 兼容）
                if store is not None:
                    ns = ("memories", user_id, mem.get("type", "fact"))
                    key = f"{mem['type']}_{int(datetime.now().timestamp() * 1000)}"
                    await store.aput(namespace=ns, key=key, value=mem)
            result_text = f"已提取并保存 {len(extracted)} 条记忆"
            logger.info("Memory: %d items extracted for user=%s", len(extracted), user_id)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Memory extraction parse error: %s", exc)
        result_text = f"记忆提取完成: {extract_resp.content[:200]}"

    return {
        "messages": [ToolMessage(content=result_text, tool_call_id=tc["id"])],
        "memories": new_memories,
        "tasks": state.get("tasks", []) + [f"memory:{datetime.now().isoformat()}"],
    }


async def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    检索子Agent —— 接收主Agent发出的详细检索查询，
    调用 LegalRetriever 多路检索并将结果返回给主Agent。
    """
    last_ai = state["messages"][-1]
    if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
        return {}

    tc = next((t for t in last_ai.tool_calls if t["name"] == "search_legal_docs"), None)
    if not tc:
        return {}

    query = tc["args"]["query"]
    retriever = config.get("configurable", {}).get("retriever")

    if retriever is not None:
        try:
            results = retriever.retrieve(query)
            parts = []
            for i, r in enumerate(results, 1):
                title = r.metadata.get("title", "未知法律")
                article = r.metadata.get("article_label", "")
                score_str = f"(相关度: {r.score:.3f})" if hasattr(r, "score") else ""
                parts.append(f"【{i}】{title} {article} {score_str}\n{r.text}")
            context = "\n\n".join(parts) if parts else "未找到相关法律文档"
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            context = f"检索出错: {exc}"
    else:
        context = f"[检索系统未连接] 查询内容: {query}"
        logger.warning("retriever_node: no retriever in config")

    return {
        "messages": [ToolMessage(content=context, tool_call_id=tc["id"])],
        "retrieval_context": context,
        "tasks": state.get("tasks", []) + [f"retriever:{datetime.now().isoformat()}"],
    }


async def tavily_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Tavily 搜索子Agent —— 搜索互联网获取最新法律新闻、判例和政策变化。
    当本地法律数据库信息不足时由主Agent调用。
    """
    last_ai = state["messages"][-1]
    if not isinstance(last_ai, AIMessage) or not last_ai.tool_calls:
        return {}

    tc = next((t for t in last_ai.tool_calls if t["name"] == "tavily_search"), None)
    if not tc:
        return {}

    query = tc["args"]["query"]
    max_results = tc["args"].get("max_results", 5)
    search_depth = tc["args"].get("search_depth", "advanced")

    try:
        result = await run_tavily_search(query, max_results, search_depth)
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc, exc_info=True)
        result = f"互联网搜索出错: {exc}"

    return {
        "messages": [ToolMessage(content=result, tool_call_id=tc["id"])],
        "tasks": state.get("tasks", []) + [f"tavily:{datetime.now().isoformat()}"],
    }


async def short_mem_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    短期记忆子Agent —— 当对话超过 8 轮时触发，
    将最早且未压缩的对话压缩成 JSON 结构记忆，只保留最近 4 轮不压缩。
    """
    messages = state["messages"]
    boundary = _find_compress_boundary(messages, keep_turns=4)
    if boundary == 0:
        return {}

    to_compress = messages[:boundary]

    conv_lines: list[str] = []
    for m in to_compress:
        if isinstance(m, HumanMessage):
            conv_lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            conv_lines.append(f"助手: {m.content}")
        elif isinstance(m, ToolMessage):
            preview = m.content[:100] + "..." if len(m.content) > 100 else m.content
            conv_lines.append(f"[工具结果] {preview}")

    if not conv_lines:
        return {}

    llm = _get_llm(config)
    summary_resp = await llm.ainvoke([
        SystemMessage(content=SHORT_MEM_COMPRESS_PROMPT),
        HumanMessage(content="\n".join(conv_lines)),
    ])

    try:
        summary = json.loads(_strip_code_fence(summary_resp.content))
    except json.JSONDecodeError:
        summary = {
            "summary": summary_resp.content,
            "turn_count": _count_user_turns(to_compress),
        }
    summary["compressed_at"] = datetime.now().isoformat()

    removals = [RemoveMessage(id=m.id) for m in to_compress if hasattr(m, "id") and m.id]
    compressed = list(state.get("compressed_history", []))
    compressed.append(summary)

    logger.info(
        "ShortMem: compressed %d messages (%d turns), keeping last 4 turns",
        len(to_compress),
        _count_user_turns(to_compress),
    )
    return {
        "messages": removals,
        "compressed_history": compressed,
        "tasks": state.get("tasks", []) + [f"short_mem:{datetime.now().isoformat()}"],
    }


def user_input_node(state: AgentState) -> dict:
    """
    用户输入节点 —— 使用 interrupt 暂停图执行等待用户输入。
    恢复时将新消息注入 messages 并递增 turn_count。
    """
    user_msg: str = interrupt(value="awaiting_user_input")
    return {
        "messages": [HumanMessage(content=user_msg)],
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ========================================== Routing =============================================


def route_after_manager(
    state: AgentState,
) -> Literal["memory", "retriever", "tavily", "short_mem", "user_input"]:
    """
    Manager 响应后路由：
      有 tool call → 对应子Agent
      无 tool call → 检查是否需要压缩 → short_mem 或 user_input
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        name = last.tool_calls[0]["name"]
        if name == "search_legal_docs":
            return "retriever"
        if name == "extract_memory":
            return "memory"
        if name == "tavily_search":
            return "tavily"

    # 无工具调用：检查是否需要压缩对话历史
    if _count_user_turns(state["messages"]) > 8:
        if _find_compress_boundary(state["messages"], keep_turns=4) > 0:
            return "short_mem"

    return "user_input"
