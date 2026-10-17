# Qwen 基座模型下载方案（本地优先版）

> 配套：《Qwen人格特征提取小模型构建部署方案.md》《Qwen人格提取_方案修订_RTX_PRO_6000.md》。
> 本次修订：**基座相关的代码与权重全部先在本地 Mac 下载、校验，再上传云服务器**；所有代码一律以 **Jupyter Notebook（.ipynb）** 保存与执行。

---

## 0. 一页总览

| 优先级 | 模型 | 用途 | 来源 | 大小 | 本地目录 |
|---|---|---|---|---|---|
| P0 | Qwen3-4B-Instruct-2507 (bf16) | SFT 全参微调基座 | ModelScope | ~8.5 GB | `~/models/parallax/Qwen3-4B-Instruct-2507/` |
| P0 | Qwen3-32B-AWQ (4bit) | 自托管蒸馏打标 / 兜底 | ModelScope | ~19 GB | `~/models/parallax/Qwen3-32B-AWQ/` |
| P1 | Qwen3-8B (base, bf16) | 4B 不达标时 LoRA 备选（ModelScope 无 8B Instruct，用 base 或走 hf-mirror） | ModelScope | ~16 GB | `~/models/parallax/Qwen3-8B/` |
| P2 | Qwen3-1.7B-GGUF (Q4_K_M) | 端侧 / CPU 降级 | hf-mirror | ~1.2 GB | `~/models/parallax/` |
| P2 | Qwen3-0.6B-GGUF (Q4_K_M) | iPhone/轻端侧 | hf-mirror | ~0.5 GB | `~/models/parallax/` |

**本地磁盘要求**：模型合计 ~44GB，`~/models/parallax` 分区可用空间 ≥ 60GB（本机当前可用约 797GB，满足）。

---

## 1. 工作流（本地 → 校验 → 上传 → 服务器冒烟）

```text
[本地 Mac]
  00_环境准备.ipynb      安装工具 + 镜像配置 + 磁盘检查 + 建目录
  01_下载Qwen权重.ipynb   ModelScope / hf-mirror 断点续传下载全部权重
  02_本地校验.ipynb       safetensors 逐分片解析 + 体积核对 + 可选 MLX 冒烟
        │  rsync 断点续传上传（03 notebook 内）
        ▼
[云服务器 RTX PRO 6000]
  /data/models/           03_上传云服务器.ipynb 的 ssh 步骤
  GPU 冒烟                vLLM 起 4B 提取服务 + 32B 打标服务
```

**代码规范：所有脚本一律 .ipynb 保存**，命名 `00_xxx.ipynb` 递增，代码与说明同在一个 notebook 内，便于复现与交接。

---

## 2. 下载源选择

| 源 | 速度 | 适用 |
|---|---|---|
| **ModelScope（首选）** | 国内 CDN，常跑满带宽 | Qwen 官方全系权重 |
| **hf-mirror.com** | 国内镜像 | GGUF 社区量化等 ModelScope 没有的模型 |
| HuggingFace 直连 | 差 | 不推荐 |

规则：官方 Qwen 权重走 ModelScope；GGUF 走 hf-mirror。

---

## 3. 执行步骤（对应 notebook）

### 3.1 `00_环境准备.ipynb`
- 安装 `modelscope` / `huggingface_hub[cli]` / `hf_transfer`
- 把 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_ENABLE_HF_TRANSFER=1` 写入 `~/.zshrc`
- 磁盘检查（<60GB 报警）并创建 `~/models/parallax`

### 3.2 `01_快速下载_多线程并行.ipynb`（主用，推荐）

- **实测结论（2026-10-16 本机）**：modelscope SDK 单连接只有 0.8–2.2 KB/s（表现为"完全不动"）；改用 **8 线程 + HTTP Range 分段 + .part 断点续传**后达到 **28–32 MB/s**。
- 纯 Python 实现，零依赖；重跑 cell 自动续传；末尾附 aria2 16 连接极速模式（可选，`brew install aria2`）。
- 旧 `01_下载Qwen权重.ipynb` 保留作 ModelScope SDK 备用。

### 3.2b `01_下载Qwen权重.ipynb`（备用）
- 用 `modelscope.hub.snapshot_download` 下载 4B-Instruct-2507 / 32B-AWQ / 8B（自带断点续传，中断重跑同一 cell 即可）
- 用 `hf_hub_download` 拉两个 GGUF（Q4_K_M）
- 末尾 cell 打印每个目录的实际体积，快速核对

### 3.3 `02_本地校验.ipynb`
- `safe_open` 逐分片解析表头：结构损坏会立刻抛错
- 体积阈值检查（4B ≥8GB、32B ≥18GB、8B ≥15GB）+ `config.json` 存在性
- 可选：Apple Silicon 上用 MLX 加载 4B 做冒烟（内存 <16GB 跳过）

### 3.4 `03_上传云服务器.ipynb`
- `rsync -avP --append-verify` 断点续传上传到服务器 `/data/models/`
- ssh 到服务器核对磁盘与文件
- 服务器端起 vLLM：4B 提取服务 `:8001`、32B 打标服务 `:8002`，curl 冒烟

---

## 4. 常见坑

0. **仓库名 404**：ModelScope 上不存在 `Qwen/Qwen3-4B-Instruct`（无日期后缀），官方命名是 `Qwen/Qwen3-4B-Instruct-2507`；8B 也没有 Instruct 版仓库。下载前先按 §3.2 的表核对仓库名。

1. **磁盘**：Mac 与服务器都要先 `df -h`；别把 44GB 装进小分区。
2. **半截文件**：`*.incomplete` 或体积不符 → 重跑下载 cell，续传自动接上。
3. **GGUF 文件名**：仓库更新后先看文件列表再改 cell 里的文件名。
4. **代理干扰**：ModelScope 走代理反而慢，notebook 内临时 `unset http_proxy https_proxy`。
5. **版本锁定**：下载后记录 commit/hash 到 `02_本地校验.ipynb` 的备注，复现实验用。
6. **rsync 中断**：`--append-verify` 保证重跑不重传；大文件（32B）建议 tmux 内执行。

---

## 5. 时间表（按 30MB/s 计）

| 模型 | 预计耗时 |
|---|---|
| 4B-Instruct-2507（8.5GB） | ~5 分钟 |
| 32B-AWQ（19GB） | ~11 分钟 |
| 8B（16GB） | ~9 分钟 |
| 合计（全部模型） | 25 分钟内 |



| 时段 | 动作 | 预期 |
|---|---|---|
| 白天 | 00 + 01：下载 4B（P0） | 千兆带宽 10–30 分钟 |
| 下午 | 02 校验 + MLX 冒烟 | 通过后即可准备上传 |
| 夜间 | 01：续跑 32B-AWQ + 8B | 2–4 小时 |
| 次日 | 03：rsync 上传 + 服务器 vLLM 冒烟 | 通过后开 SFT |
