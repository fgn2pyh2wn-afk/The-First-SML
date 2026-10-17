#!/usr/bin/env python3
"""Qwen3-4B-Instruct-2507 人格特征提取 · QLoRA SFT（RTX 5090 32GB 配置）
在 AutoDL 服务器上运行：python train_sft.py"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType

try:
    from peft import prepare_model_for_kbit_training
except ImportError:  # 新版本移除后手动处理
    def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        return model

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ID = "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507"
OUT = "/root/autodl-tmp/output/sft-v1"

# ---------- 数据 ----------
ds = load_dataset("json", data_files={
    "train": "/root/autodl-tmp/data/sft_train.jsonl",
    "validation": "/root/autodl-tmp/data/sft_val.jsonl",
})

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.model_max_length = 2048  # 控制序列长度，适配 32GB 显存


def fmt(ex):
    msgs = [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["query"]},
        {"role": "assistant", "content": ex["response"]},
    ]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )


# ---------- QLoRA ----------
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quant,
    device_map="auto", dtype="auto",
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    run_name="qwen3-4b-persona-sft-v1",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=args,
    peft_config=peft_config,
    formatting_func=fmt,
)

trainer.train()
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("SFT done ->", OUT)
