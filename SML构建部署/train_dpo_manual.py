#!/usr/bin/env python3
"""手写 DPO 训练：绕过 trl fused 兼容问题，公式：
L = -log σ( β[(logp_w - logp_ref_w) - (logp_l - logp_ref_l)] )"""
import json, math, os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507"
SFT_ADAPTER = "/root/autodl-tmp/output/sft-v1"
OUT = "/root/autodl-tmp/output/dpo-v1"
PAIRS = "/root/autodl-tmp/data/dpo_pairs.jsonl"
BETA = 0.5
SFT_WEIGHT = 0.3          # 混入 SFT 正则，防止格式崩溃
LR = 5e-6
EPOCHS = 1
BATCH = 1

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True,
                           modules_to_not_convert=["lm_head"])
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

base_p = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=quant,
                                              device_map="auto", dtype="auto")
policy = PeftModel.from_pretrained(base_p, SFT_ADAPTER, is_trainable=True)

base_r = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=quant,
                                              device_map="auto", dtype="auto")
ref = PeftModel.from_pretrained(base_r, SFT_ADAPTER)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

pairs = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
pairs = pairs[:400]
print("偏好对:", len(pairs), flush=True)

opt = torch.optim.AdamW(policy.parameters(), lr=LR)
steps = math.ceil(len(pairs) / BATCH)


def seq_logp(model, prompt, resp):
    """返回模型对 resp 部分 token 的平均 log 概率（数值稳定版）"""
    enc = tok([prompt], return_tensors="pt").to("cuda")
    all_ids = tok([prompt + resp], return_tensors="pt").input_ids.to("cuda")
    all_ids = all_ids[:, :1536]
    resp_len = all_ids.shape[1] - enc.input_ids.shape[1]
    if resp_len <= 0:
        return torch.tensor(0.0, device="cuda")
    logits = model(input_ids=all_ids).logits[:, -resp_len-1:-1, :]  # 预测 resp 各 token
    logp = torch.gather(F.log_softmax(logits.float(), dim=-1), -1,
                        all_ids[:, -resp_len:].unsqueeze(-1)).squeeze(-1)
    return logp.sum() / resp_len


policy.train()
ref_logps = []
with torch.no_grad():
    for p in pairs:
        rw = seq_logp(ref, p["prompt"], p["chosen"]).item()
        rl = seq_logp(ref, p["prompt"], p["rejected"]).item()
        ref_logps.append((rw, rl))
del ref
torch.cuda.empty_cache()
print("参考模型 logps 预算完成，ref 已卸载", flush=True)

for ep in range(EPOCHS):
    running = 0.0
    for i, p in enumerate(pairs):
        lw = seq_logp(policy, p["prompt"], p["chosen"])
        ll = seq_logp(policy, p["prompt"], p["rejected"])
        rw, rl = ref_logps[i]
        rw = torch.tensor(rw, device="cuda")
        rl = torch.tensor(rl, device="cuda")
        dpo = -F.logsigmoid(BETA * ((lw - rw) - (ll - rl))).mean()
        sft = -lw.mean()
        loss = dpo + SFT_WEIGHT * sft
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        running += loss.item()
        if (i + 1) % 10 == 0:
            print(f"step {i+1}/{len(pairs)} loss={loss.item():.4f} "
                  f"logp_w={lw.mean().item():.2f} logp_l={ll.mean().item():.2f}", flush=True)
    print(f"epoch {ep+1} avg_loss={running/max(1,steps):.4f}", flush=True)

policy.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("DPO done ->", OUT, flush=True)
