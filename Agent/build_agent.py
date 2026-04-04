"""
法律Agent —— 基于 LangGraph 的多子Agent架构。

图结构:
    START ──► manager ──► route_after_manager
                ▲            │
                │            ├─► retriever ──► manager  (检索循环)
                │            ├─► memory    ──► manager  (记忆循环)
                │            ├─► short_mem ──► user_input (压缩)
                │            └─► user_input             (直接)
                │                    │
                └────────────────────┘

用法:
    from Agent.build_agent import LegalAgent

    agent = LegalAgent(pg_url="postgresql://...", retriever=my_retriever)
    await agent.setup()

    async for token in agent.astream("请问盗窃罪的构成要件是什么？", thread_id="t1"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from Agent.nodes import (
    AgentState,
    manager_node,
    memory_node,
    retriever_node,
    route_after_manager,
    short_mem_node,
    tavily_node,
    user_input_node,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ========================================== LegalAgent =============================================


class LegalAgent:
    """法律智能Agent —— 封装 LangGraph 图的编译、运行和流式输出。"""

    DEFAULT_SKILLS = ["search_legal_docs", "extract_memory", "tavily_search"]

    def __init__(
        self,
        pg_url: str | None = None,
        retriever: Any = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        skills: list[str] | None = None,
    ):
        self._pg_url = os.getenv("PG_URL") if pg_url is None else pg_url
        self._retriever = retriever
        self._model = model or os.getenv("GEN_MODEL") or os.getenv("VLLM_LORA_NAME", "legal-lora")
        self._base_url = (
            os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")
            if base_url is None
            else base_url
        )
        self._api_key = (
            os.getenv("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key is None
            else api_key
        )
        self._skills = skills if skills is not None else self.DEFAULT_SKILLS

        self._store = None
        self._checkpointer = None
        self._graph = None
        self._resource_stack: AsyncExitStack | None = None

    # ────────────────────── lifecycle ──────────────────────

    async def setup(self) -> None:
        """初始化 PostgreSQL store / checkpointer 并编译图。"""
        await self.close()

        if self._pg_url:
            stack = AsyncExitStack()
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from langgraph.store.postgres import AsyncPostgresStore

                self._store = await stack.enter_async_context(
                    AsyncPostgresStore.from_conn_string(self._pg_url)
                )
                await self._store.setup()

                self._checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(self._pg_url)
                )
                await self._checkpointer.setup()
                self._resource_stack = stack
                logger.info("PostgreSQL store & checkpointer ready")
            except Exception as exc:
                await stack.aclose()
                logger.warning(
                    "PostgreSQL init failed, falling back to in-memory checkpointing: %s",
                    exc,
                )
                self._store = None
                self._checkpointer = None

        if self._checkpointer is None:
            self._checkpointer = InMemorySaver()

        self._graph = self._build_graph()

    async def close(self) -> None:
        """释放连接资源。"""
        if self._resource_stack is not None:
            try:
                await self._resource_stack.aclose()
            except Exception:
                pass
            self._resource_stack = None

        self._store = None
        self._checkpointer = None
        self._graph = None

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # ────────────────────── graph construction ──────────────────────

    def _build_graph(self) -> Any:
        """构建并编译 StateGraph。"""
        builder = StateGraph(AgentState)

        # ── 添加节点 ──
        builder.add_node("manager", manager_node)
        builder.add_node("memory", memory_node)
        builder.add_node("retriever", retriever_node)
        builder.add_node("tavily", tavily_node)
        builder.add_node("short_mem", short_mem_node)
        builder.add_node("user_input", user_input_node)

        # ── 连接边 ──
        builder.add_edge(START, "manager")

        builder.add_conditional_edges(
            "manager",
            route_after_manager,
            {
                "memory": "memory",
                "retriever": "retriever",
                "tavily": "tavily",
                "short_mem": "short_mem",
                "user_input": "user_input",
            },
        )

        # 子Agent完成后回到 manager
        builder.add_edge("memory", "manager")
        builder.add_edge("retriever", "manager")
        builder.add_edge("tavily", "manager")

        # 压缩完成后进入用户输入等待
        builder.add_edge("short_mem", "user_input")

        # 用户输入后回到 manager 开启新一轮
        builder.add_edge("user_input", "manager")

        compile_kwargs: dict[str, Any] = {}
        if self._checkpointer is not None:
            compile_kwargs["checkpointer"] = self._checkpointer
        if self._store is not None:
            compile_kwargs["store"] = self._store

        return builder.compile(**compile_kwargs)

    # ============================ config builder ===============================

    def _make_config(self, thread_id: str, user_id: str = "default") -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "retriever": self._retriever,
                "store": self._store,
                "model": self._model,
                "base_url": self._base_url,
                "api_key": self._api_key,
                "skills": self._skills,
            }
        }

    # ============================ streaming ==================================

    STREAM_TIMEOUT = 300  # 整个流式会话最长 5 分钟

    async def astream(
        self,
        user_input: str,
        thread_id: str = "default",
        user_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        增量流式输出 —— 只返回 manager 节点生成的回复文本 token。

        首次调用通过 initial state 注入用户消息；
        后续调用通过 Command(resume=...) 恢复 interrupt 的 user_input 节点。
        """
        if self._graph is None:
            raise RuntimeError("请先调用 await agent.setup()")

        config = self._make_config(thread_id, user_id)

        graph_state = None
        if self._checkpointer is not None:
            graph_state = await self._graph.aget_state(config)

        if graph_state and graph_state.values and graph_state.next:
            input_data: Any = Command(resume=user_input)
        else:
            input_data = {
                "messages": [HumanMessage(content=user_input)],
                "turn_count": 1,
                "memories": [],
                "compressed_history": [],
                "retrieval_context": "",
                "tasks": [],
            }

        logger.info("astream: starting graph execution (thread=%s)", thread_id)
        token_count = 0
        thinking_notified = False

        try:
            async with asyncio.timeout(self.STREAM_TIMEOUT):
                async for event in self._graph.astream_events(
                    input_data, config=config, version="v2"
                ):
                    if event["event"] != "on_chat_model_stream":
                        continue
                    node = event.get("metadata", {}).get("langgraph_node")
                    if node != "manager":
                        continue
                    chunk = event["data"].get("chunk")
                    if not isinstance(chunk, AIMessageChunk):
                        continue

                    if chunk.content:
                        token_count += 1
                        yield chunk.content
                    elif not thinking_notified and token_count == 0:
                        reasoning = (
                            chunk.additional_kwargs.get("reasoning_content")
                            or getattr(chunk, "reasoning_content", None)
                        )
                        if reasoning:
                            thinking_notified = True
                            logger.info("astream: model is in thinking/reasoning phase")
                            yield "💭 *正在深度思考中…*\n\n"
        except TimeoutError:
            logger.error("astream: timed out after %ss (yielded %d tokens)",
                         self.STREAM_TIMEOUT, token_count)
            yield f"\n\n[错误] 处理超时（{self.STREAM_TIMEOUT}秒），请重试。"

        logger.info("astream: finished (yielded %d tokens)", token_count)

    # ============================== single invoke =====================================

    async def ainvoke(
        self,
        user_input: str,
        thread_id: str = "default",
        user_id: str = "default",
    ) -> str:
        """非流式调用，返回完整回复文本。"""
        tokens: list[str] = []
        async for t in self.astream(user_input, thread_id, user_id):
            tokens.append(t)
        return "".join(tokens)

    # ============================= convenience REPL ===============================

    async def chat(self, thread_id: str = "default", user_id: str = "default") -> None:
        """交互式命令行对话（便于调试）。"""
        print("=" * 60)
        print("法律智能助手 —— 输入问题，输入 q 退出")
        print("=" * 60)

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n用户> ")
            except (EOFError, KeyboardInterrupt):
                break
            user_input = user_input.strip()
            if not user_input or user_input.lower() == "q":
                break

            print("\n助手> ", end="", flush=True)
            async for token in self.astream(user_input, thread_id, user_id):
                print(token, end="", flush=True)
            print()

    # ========================== graph visualization ================================

    def get_graph_image(self) -> bytes | None:
        """返回图结构的 PNG 图片字节（需要 pygraphviz）。"""
        if self._graph is None:
            return None
        try:
            return self._graph.get_graph().draw_mermaid_png()
        except Exception:
            return None


# ========================================== CLI entry point =============================================


async def _main() -> None:
    async with LegalAgent() as agent:
        await agent.chat()


if __name__ == "__main__":
    asyncio.run(_main())
