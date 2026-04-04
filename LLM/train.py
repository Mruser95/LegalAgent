"""
Qwen3.5-9B LoRA 微调脚本 —— 中国法律助手 SFT

双卡并行启动：
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 LLM/train.py \
      --output_dir LLM/qwen3.5-9b-lora \
      --num_train_epochs 3 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --learning_rate 2e-4 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type cosine \
      --weight_decay 0.01 \
      --bf16 true \
      --gradient_checkpointing true \
      --logging_steps 10 \
      --save_strategy epoch \
      --save_total_limit 3 \
      --report_to tensorboard \
      --ddp_find_unused_parameters false

可选 DeepSpeed ZeRO-2（显存更紧张时）：
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 LLM/train.py \
      --deepspeed LLM/ds_config_zero2.json \
      --output_dir LLM/qwen3.5-9b-lora \
      ... (其余参数同上)
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

IGNORE_INDEX = -100
PROJECT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    model_path: str = field(
        default=str(PROJECT_DIR / "qwen3.5-9b"),
        metadata={"help": "基座模型路径"},
    )
    data_path: str = field(
        default=str(PROJECT_DIR / "train_data.json"),
        metadata={"help": "训练数据 JSON 路径"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"},
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "逗号分隔的 LoRA 目标模块名"},
    )


# ---------------------------------------------------------------------------
# Data utils
# ---------------------------------------------------------------------------
def _parse_json_like(value):
    """尽量将 JSON/近似 JSON 的字符串解析为 Python 对象。"""
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    stripped = value.strip()
    if stripped and not stripped.startswith(("{", "[")) and ":" in stripped:
        try:
            return json.loads("{" + stripped + "}")
        except json.JSONDecodeError:
            pass

    return value


def _normalize_tool_arguments(tool_name: str, arguments):
    arguments = _parse_json_like(arguments)
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}

    key = "query" if "search" in tool_name.lower() else "input"
    return {key: arguments}


def _normalize_tool_call(tc: dict) -> dict:
    tc = copy.deepcopy(tc)

    function = tc.get("function")
    if isinstance(function, str):
        function = {"name": function}
    elif isinstance(function, dict):
        function = dict(function)
    else:
        function = {}

    if "arguments" in tc and "arguments" not in function:
        function["arguments"] = tc.pop("arguments")

    name = function.get("name")
    parsed_name = _parse_json_like(name)
    if isinstance(parsed_name, dict):
        function["arguments"] = function.get("arguments", parsed_name)
        function["name"] = "tavily_search" if "query" in parsed_name else "unknown_tool"
    elif isinstance(name, str) and name:
        function["name"] = name
    else:
        function["name"] = "unknown_tool"

    function["arguments"] = _normalize_tool_arguments(
        function["name"], function.get("arguments")
    )
    tc["function"] = function
    return tc


def fix_tool_calls(messages: list[dict]) -> list[dict]:
    """将不同形态的消息/tool_calls 统一为 chat_template 可消费的结构。"""
    result = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("content") is None:
            msg["content"] = ""

        if msg.get("role") == "tool":
            # tool 消息只需要 content/tool_call_id，额外 tool_calls 可能是脏数据。
            msg.pop("tool_calls", None)
            result.append(msg)
            continue

        if msg.get("tool_calls"):
            fixed = []
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fixed.append(_normalize_tool_call(tc))
            msg["tool_calls"] = fixed
        result.append(msg)
    return result


class LegalSFTDataset(Dataset):
    """多轮对话 + 工具调用 SFT 数据集，仅对 assistant 回复计算 loss。"""

    def __init__(self, data_path: str, tokenizer, max_seq_length: int):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_seq_length

        self.im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.role_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
        self.data = []
        skipped: list[tuple[int, str]] = []

        # 训练前先过滤掉会让 chat_template/tokenizer 失败的脏样本。
        for idx, sample in enumerate(raw_data):
            try:
                messages = fix_tool_calls(sample["messages"])
                self._encode_messages(messages)
                self.data.append({"messages": messages})
            except Exception as exc:
                skipped.append((idx, f"{type(exc).__name__}: {exc}"))

        if not self.data:
            raise ValueError("没有可用训练样本，所有数据都在预处理阶段被跳过。")

        if skipped:
            logger.warning("检测到 %d 条脏数据，训练时将直接跳过。", len(skipped))
            for idx, reason in skipped[:10]:
                logger.warning("跳过样本 idx=%d，原因：%s", idx, reason)
            if len(skipped) > 10:
                logger.warning("其余 %d 条脏数据已省略日志。", len(skipped) - 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 双保险：若仍遇到漏网脏样本，就顺延取下一个，避免整轮训练中断。
        for offset in range(len(self.data)):
            real_idx = (idx + offset) % len(self.data)
            try:
                messages = self.data[real_idx]["messages"]
                enc = self._encode_messages(messages)
                break
            except Exception as exc:
                logger.warning(
                    "样本 idx=%d 运行时预处理失败，已跳过：%s: %s",
                    real_idx,
                    type(exc).__name__,
                    exc,
                )
        else:
            raise RuntimeError("没有可用样本可供当前 batch 使用。")

        input_ids = enc["input_ids"]
        labels = self._mask_non_assistant(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    def _encode_messages(self, messages: list[dict]) -> dict:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )

    def _mask_non_assistant(self, ids: list[int]) -> list[int]:
        """只保留 <|im_start|>assistant\\n ... <|im_end|> 之间的 token 作为 label，
        其余位置设为 IGNORE_INDEX(-100)。"""
        labels = [IGNORE_INDEX] * len(ids)
        n = len(ids)
        rlen = len(self.role_ids)

        i = 0
        while i < n:
            if ids[i] == self.im_start:
                ok = all(
                    i + 1 + j < n and ids[i + 1 + j] == self.role_ids[j]
                    for j in range(rlen)
                )
                if ok:
                    content_start = i + 1 + rlen
                    k = content_start
                    while k < n and ids[k] != self.im_end:
                        k += 1
                    for p in range(content_start, min(k + 1, n)):
                        labels[p] = ids[p]
                    i = k + 1
                    continue
            i += 1

        return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    arch_name = config.architectures[0]
    model_cls = getattr(transformers, arch_name, None)
    if model_cls is None:
        raise ImportError(
            f"找不到模型类 {arch_name}，请确认 transformers 版本 >= 4.57: "
            "pip install transformers>=4.57"
        )

    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()

    # ── LoRA ──
    targets = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──
    dataset = LegalSFTDataset(args.data_path, tokenizer, args.max_seq_length)
    logger.info("训练样本数: %d", len(dataset))

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
        ),
    )

    trainer.train()

    # ── 保存 LoRA adapter ──
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("LoRA adapter 已保存到 %s", training_args.output_dir)


if __name__ == "__main__":
    main()
