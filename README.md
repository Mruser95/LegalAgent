# LegalAgent — 法律智能助手

基于 LangGraph 多子 Agent 架构的法律咨询系统，集成 RAG 检索、长期记忆、互联网搜索和 LoRA 微调推理能力。

## 系统架构

```
                         ┌─────────────────┐
                         │   Chat UI :8009  │
                         └────────┬────────┘
                                  │ SSE 流式
                         ┌────────▼────────┐
                         │ Agent API :8008  │
                         │   (LangGraph)    │
                         └──┬─────┬─────┬──┘
                            │     │     │
               ┌────────────┘     │     └────────────┐
               ▼                  ▼                   ▼
        ┌─────────────┐  ┌──────────────┐   ┌──────────────┐
        │  PostgreSQL  │  │  vLLM :8888  │   │  Tavily API  │
        │  + pgvector  │  │ Qwen3.5-9B   │   │  (互联网搜索) │
        │  :5432       │  │ + LoRA       │   └──────────────┘
        └─────────────┘  └──────────────┘
```

### Agent 图结构

```
START ──► manager ──► route
             ▲          ├─► retriever ──► manager    (法律文档检索)
             │          ├─► memory    ──► manager    (记忆提取)
             │          ├─► tavily    ──► manager    (互联网搜索)
             │          ├─► short_mem ──► user_input (历史压缩)
             │          └─► user_input              (等待输入)
             │                 │
             └─────────────────┘
```

- **manager** — 主 Agent，分析用户意图，决定直接回答或调用工具
- **retriever** — 法律文档多路检索（BM25 + 向量 + RRF 融合 + Rerank）
- **memory** — 从对话中提取事实/偏好，持久化到 PostgreSQL
- **tavily** — 互联网搜索，获取最新法律动态
- **short_mem** — 超过 8 轮自动压缩早期对话
- **user_input** — interrupt 机制暂停图执行等待用户输入

## 目录结构

```
LegalAgent/
├── Agent/                  # LangGraph Agent 核心
│   ├── api.py              # FastAPI 服务 (OpenAI 兼容接口)
│   ├── build_agent.py      # Agent 构建与生命周期管理
│   ├── nodes.py            # 节点实现与路由逻辑
│   ├── mcp.py              # MCP 工具定义 & Tavily 搜索
│   └── prompt.py           # 系统提示词
├── UI/                     # Chat 前端
│   ├── app.py              # FastAPI 静态文件服务
│   └── static/index.html   # 聊天界面 (SPA)
├── RAG/                    # 检索增强生成
│   ├── build_chunks.py     # 法律文档分块
│   ├── load_store.py       # 向量数据库加载
│   ├── retriver.py         # 多路检索器
│   ├── chunks_json/        # 分块后的法律文档 JSON
│   └── origin_doc/         # 原始法律文档 (.docx)
├── LLM/                    # 模型与训练
│   ├── deploy.py           # vLLM 部署脚本
│   ├── train.py            # LoRA 微调训练
│   ├── qwen3.5-9b/         # 基座模型权重
│   └── qwen3.5-9b-lora/    # LoRA 适配器
├── Skills/                 # Agent 工具使用指南 (.md)
├── Docker/                 # 容器化部署
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── init-postgres.sql
│   ├── .env.docker         # Docker 环境变量（实际使用）
│   └── .env.docker.example # Docker 环境变量模板
├── requirements.txt
└── .env.example
```

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL 16 + pgvector 扩展
- NVIDIA GPU（本地推理时需要）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少配置以下项：
#   PG_URL          — PostgreSQL 连接地址
#   base_url        — LLM API 地址（本地 vLLM 或 OpenAI）
#   api_key         — LLM API Key
#   GEN_MODEL       — 模型名称（本地 LoRA 名或 gpt-4o 等）
#   tavily_api_key  — Tavily 搜索 Key（可选）
```

### 3. 初始化数据库

```bash
# 启动 PostgreSQL 并创建 pgvector 扩展
psql -U postgres -d legal_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 加载法律文档到向量数据库
python RAG/load_store.py
```

### 4. 启动 LLM 推理服务

```bash
# 本地 vLLM 部署（需要 GPU）
python LLM/deploy.py

# 或使用 OpenAI API，在 .env 中配置：
#   OPENAI_API_KEY=your_openai_api_key_here
#   OPENAI_BASE_URL=https://api.openai.com/v1
#   GEN_MODEL=gpt-4o
```

### 5. 启动服务

```bash
# 启动 Agent API (端口 8008)
python -m Agent.api

# 启动 Chat UI (端口 8009)
python -m UI.app
```

浏览器访问 http://localhost:8009 开始对话。

## Docker 部署（推荐）

Docker 方式可在一台**全新机器**上一键部署所有服务（PostgreSQL、vLLM、Agent API、Chat UI），无需预装任何运行环境，只需项目代码、模型文件和 Docker。

### 前置条件

- Docker Engine 20.10+ 和 Docker Compose V2
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)（使用本地 vLLM 时需要）
- 项目目录下已有模型文件 `LLM/qwen3.5-9b/` 和 `LLM/qwen3.5-9b-lora/`
- 首次构建镜像和初始化 RAG 数据需要**网络连接**（pip 安装依赖、下载 embedding 模型）

### 一键启动全部服务

```bash
cd Docker

# 1. 配置环境变量
cp .env.docker.example .env.docker
# 编辑 .env.docker 中的 tavily_api_key 等

# 2. 启动全部服务（需 NVIDIA GPU）
docker compose up -d

# 3. 查看启动状态
docker compose ps

# 4. 首次部署：加载法律文档到向量数据库
#    （需要网络连接，首次会下载 ~1.2GB 的 embedding 模型）
docker compose run --rm agent-api python RAG/load_store.py
```

启动顺序：`postgres` (就绪) → `vllm` (模型加载) → `agent-api` (健康检查通过) → `ui`

### 不使用本地 GPU（外部 API）

如果没有 GPU 或想使用 OpenAI 等外部 API：

```bash
# 跳过 vLLM 容器
docker compose up -d --scale vllm=0
```

同时修改 `.env.docker`：

```env
# 注释掉本地 vLLM 相关配置，设置外部 API
base_url=https://api.openai.com/v1
api_key=sk-your-key-here
GEN_MODEL=gpt-4o
```

### 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| Chat UI | 8009 | 浏览器访问 http://localhost:8009 |
| Agent API | 8008 | OpenAI 兼容接口 http://localhost:8008/v1 |
| vLLM | 8888 | LLM 推理服务 |
| PostgreSQL | 5432 | 数据库（pgvector + checkpoint） |

### 远程访问

默认配置仅支持本机浏览器访问。如需从其他机器访问，修改 `.env.docker`：

```env
# 将 localhost 替换为服务器实际 IP
agent_base_url=http://192.168.1.100:8008/v1
```

### 常用运维命令

```bash
# 查看日志
docker compose logs -f agent-api
docker compose logs -f vllm

# 重启单个服务
docker compose restart agent-api

# 停止全部服务（保留数据）
docker compose down

# 停止并清除全部数据（包括数据库）
docker compose down -v

# 重新构建镜像（代码更新后）
docker compose build --no-cache
docker compose up -d
```

### 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GEN_MODEL` | LLM 模型名称 | `legal-lora` |
| `base_url` | LLM API 地址 | `http://vllm:8888/v1` |
| `api_key` | LLM API Key | `EMPTY` |
| `PG_URL` | PostgreSQL 连接地址 | `postgresql://postgres:123456@postgres:5432/legal_db` |
| `tavily_api_key` | Tavily 搜索 API Key | 空（可选） |
| `agent_base_url` | 浏览器访问 Agent API 的地址 | `http://localhost:8008/v1` |
| `VLLM_TP_SIZE` | vLLM tensor parallel 数量 | `1` |
| `VLLM_MAX_MODEL_LEN` | vLLM 最大序列长度 | `4096` |
| `VLLM_GPU_MEMORY_UTILIZATION` | vLLM GPU 显存利用率 | `0.90` |
| `VLLM_QUANTIZATION` | vLLM 量化方式 | `bitsandbytes` |

## API 使用

Agent API 兼容 OpenAI Chat Completions 协议，可直接对接 OpenAI SDK：

### 流式调用

```bash
curl -N http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "legal-agent",
    "messages": [{"role": "user", "content": "盗窃罪的构成要件是什么？"}],
    "stream": true,
    "thread_id": "session-1"
  }'
```

### Python SDK

```python
from Agent.build_agent import LegalAgent

async with LegalAgent() as agent:
    async for token in agent.astream("劳动合同解除的条件？", thread_id="t1"):
        print(token, end="", flush=True)
```

## RAG 检索流程

```
Query → 实体提取（法律名/条款号） → 元信息过滤
      → BM25 + Vector 双路召回
      → RRF 融合排序
      → Cross-Encoder Rerank
      → Top-K 结果返回
```

- **BM25** — jieba 分词 + BM25Okapi，支持元信息过滤
- **向量检索** — BGE-large-zh-v1.5 嵌入 + pgvector HNSW 索引
- **融合** — Reciprocal Rank Fusion (k=60)
- **精排** — BGE-reranker-v2-m3 交叉编码重排序

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangGraph + LangChain |
| LLM | Qwen3.5-9B + LoRA (vLLM) / OpenAI API |
| 向量数据库 | PostgreSQL + pgvector |
| Embedding | BGE-large-zh-v1.5 |
| Reranker | BGE-reranker-v2-m3 |
| API | FastAPI (OpenAI 兼容) |
| 前端 | 原生 HTML/JS + marked.js |
| 工具 | FastMCP + Tavily Search |
| 容器化 | Docker Compose |
