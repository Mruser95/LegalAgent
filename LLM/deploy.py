"""
使用 vLLM 部署 Qwen3.5-9B（可选挂载 LoRA 适配器），提供 OpenAI 兼容的 Chat Completions API。

启动:
  python LLM/deploy.py

服务默认监听 http://0.0.0.0:8000/v1
可通过 .env 中的 DEPLOY_VLLM_* 变量自定义配置。

LoRA 适配器:
  在 .env 中设置 DEPLOY_VLLM_ENABLE_LORA=true 即可启用。
  请求时将 model 设为 DEPLOY_VLLM_LORA_NAME（默认 legal-lora）即可使用适配器推理。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

_DEFAULT_MODEL = str(LLM_DIR / "qwen3.5-9b")

MODEL_PATH = os.getenv("VLLM_MODEL_PATH", _DEFAULT_MODEL)
HOST = os.getenv("VLLM_HOST", "0.0.0.0")
PORT = os.getenv("VLLM_PORT", "8000")
TP_SIZE = os.getenv("VLLM_TP_SIZE", "1")
MAX_MODEL_LEN = os.getenv("VLLM_MAX_MODEL_LEN", "32768")
GPU_MEM_UTIL = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.80")
DTYPE = os.getenv("VLLM_DTYPE", "bfloat16")
QUANTIZATION = os.getenv("VLLM_QUANTIZATION", "")  # awq / gptq / bitsandbytes / ""
TOOL_CALL = os.getenv("VLLM_ENABLE_TOOL_CALL", "true").lower() == "true"
LANGUAGE_ONLY = os.getenv("VLLM_LANGUAGE_MODEL_ONLY", "true").lower() == "true"
ENFORCE_EAGER = os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true"
PREFIX_CACHING = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower() == "true"
CHUNKED_PREFILL = os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "true").lower() == "true"

ENABLE_LORA = os.getenv("VLLM_ENABLE_LORA", "false").lower() == "true"
LORA_PATH = os.getenv("VLLM_LORA_PATH", str(LLM_DIR / "qwen3.5-9b-lora"))
LORA_NAME = os.getenv("VLLM_LORA_NAME", "legal-lora")
MAX_LORA_RANK = os.getenv("VLLM_MAX_LORA_RANK", "64")


def build_cmd() -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", "qwen3.5-9b",
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", TP_SIZE,
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--dtype", DTYPE,
        "--trust-remote-code",
        "--reasoning-parser", "qwen3",
    ]

    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")

    if PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")

    if CHUNKED_PREFILL:
        cmd.append("--enable-chunked-prefill")

    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]

    if TOOL_CALL:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]

    if LANGUAGE_ONLY:
        cmd.append("--language-model-only")

    if ENABLE_LORA:
        lora_spec = json.dumps({
            "name": LORA_NAME,
            "path": LORA_PATH,
            "base_model_name": MODEL_PATH,
        })
        cmd += [
            "--enable-lora",
            "--max-lora-rank", MAX_LORA_RANK,
            "--lora-modules", lora_spec,
        ]

    return cmd


def main() -> None:
    cmd = build_cmd()

    print("=" * 60)
    print("vLLM OpenAI-Compatible Server")
    print("=" * 60)
    print(f"  Model        : {MODEL_PATH}")
    print(f"  Served Name  : qwen3.5-9b")
    print(f"  Endpoint     : http://{HOST}:{PORT}/v1")
    print(f"  TP Size      : {TP_SIZE}")
    print(f"  Max Model Len: {MAX_MODEL_LEN}")
    print(f"  GPU Mem Util : {GPU_MEM_UTIL}")
    print(f"  Dtype        : {DTYPE}")
    print(f"  Quantization : {QUANTIZATION or 'none'}")
    print(f"  Enforce Eager: {ENFORCE_EAGER}")
    print(f"  Prefix Cache : {PREFIX_CACHING}")
    print(f"  Chunked Pre. : {CHUNKED_PREFILL}")
    print(f"  Tool Call    : {TOOL_CALL}")
    print(f"  Language Only: {LANGUAGE_ONLY}")
    if ENABLE_LORA:
        print(f"  LoRA Adapter : {LORA_NAME} -> {LORA_PATH}")
        print(f"  Max LoRA Rank: {MAX_LORA_RANK}")
    else:
        print(f"  LoRA         : disabled")
    print("=" * 60)
    print()

    try:
        process = subprocess.run(cmd)
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print("Error: vllm 未安装")
        sys.exit(1)


if __name__ == "__main__":
    main()
