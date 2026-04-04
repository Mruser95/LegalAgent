# ── LegalAgent 应用镜像 ──
# 用于 agent-api (端口 8008) 和 ui (端口 8009) 两个服务
# vLLM 推理服务使用独立的官方镜像，不包含在此镜像中
# 环境变量通过 docker-compose env_file 注入，不烘焙进镜像

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 安装依赖（排除 vllm / bitsandbytes，它们由独立 GPU 容器提供）──
COPY requirements.txt .
RUN grep -vE "^(vllm|bitsandbytes)" requirements.txt > /tmp/requirements-app.txt \
    && pip install --no-cache-dir -r /tmp/requirements-app.txt \
    && rm /tmp/requirements-app.txt

# ── 复制应用代码 ──
COPY Agent/ Agent/
COPY UI/ UI/
COPY Skills/ Skills/
COPY RAG/*.py RAG/

# 确保 Python 包目录可被正确导入
RUN touch Agent/__init__.py UI/__init__.py RAG/__init__.py

# Skills 和 RAG/chunks_json 通过 volume 挂载可覆盖

EXPOSE 8008 8009

# 默认启动 agent-api，可在 docker-compose 中通过 command 覆盖
CMD ["uvicorn", "Agent.api:app", "--host", "0.0.0.0", "--port", "8008"]
