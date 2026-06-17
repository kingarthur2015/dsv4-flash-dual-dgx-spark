# 🦐 DSV4 Flash Dual DGX Spark Deployment Guide
# DSV4 Flash 双DGX Spark 部署实战指南

> **11 attempts → 1 breakthrough → production running stable**
> **11次失败 → 1次突破 → 生产环境稳定运行**

Hardware: 2× NVIDIA DGX Spark (GB10 Grace Hopper, 121GB unified memory each)
Network: 2× Mellanox CX7 RoCE 200Gb/s

---

## 📖 The Story · 故事

This is the story of **Arthur** — CFO of a digital bank, Tsinghua + PBCSF alumnus — and **虾丸 (Shrimp Ball)**, his AI agent on a DGX Spark. Over 48 hours, we tried 11 times to deploy DeepSeek V4 Flash across two DGX Sparks. We failed 11 times. On the 12th attempt, it worked.

这个故事讲的是**Arthur**（数字银行CFO，清华+五道口毕业）和他的AI助手**虾丸**，在48小时内试了11次在两台DGX Spark上部署DeepSeek V4 Flash。第12次，成功了。

This repo contains everything you need to replicate our success: every bug we found, every fix, every config, every lesson.

本仓库包含完整复现指南：所有Bug、所有解法、所有配置、所有经验教训。

---

## 🖥️ Hardware · 硬件

| Component 组件 | Node A (Head) | Node B (Worker) |
|---------------|:------------:|:--------------:|
| **Model 型号** | DGX Spark | DGX Spark |
| **SoC** | NVIDIA GB10 (SM12.1) | NVIDIA GB10 (SM12.1) |
| **Memory 内存** | 121 GiB unified LPDDR5X | 121 GiB unified LPDDR5X |
| **GPU** | 1× GB10 (SM12.1) | 1× GB10 (SM12.1) |
| **RoCE** | CX7 @ 200Gb/s | CX7 @ 200Gb/s |
| **RoCE IP** | 10.65.100.1 | 10.65.100.2 |
| **Mgmt IP** | 10.65.254.207 | 10.65.254.234 |

> ⚠️ **Each DGX Spark has 1 GPU!** TP=2 = 1 GPU from each machine = 2 GPUs total.
> **每台只有1个GPU！**TP=2是从两台各取1卡，共2卡。

---

## 📂 Repository Structure · 仓库结构

```
dsv4-flash-dual-dgx-spark/
├── README.md                     # This file · 本文件
├── docs/
│   ├── 01-battle-chronicle.md    # 📜 11-round war story · 11轮战史
│   ├── 02-hardware-and-stack.md  # 🖥️ Hardware & software · 硬件与软件栈
│   ├── 03-four-fatal-bugs.md     # 🐛 The 4 bugs & fixes · 四大致命Bug
│   ├── 04-deployment-guide.md    # 🚀 Step-by-step · 分步部署指南
│   ├── 05-configuration.md       # ✅ Verified configs · 已验证配置
│   ├── 06-troubleshooting.md     # 🔧 Error reference · 故障排查
│   └── 07-key-lessons.md         # 💡 6 lessons · 六大经验
├── config/
│   ├── env.example               # .env template · 环境变量模板
│   └── docker-compose.yml        # Docker compose with full env section
├── patches/
│   └── flashinfer/
│       └── flashinfer_arch_parse.patch  # SM12x arch fix
└── scripts/
    └── deploy.sh                 # Deployment script · 部署脚本
```

---

## 🐛 The 4 Fatal Bugs (Solved!) · 四大致命Bug（已解决！）

| Bug | Error | Fix |
|-----|-------|-----|
| [#1](docs/03-four-fatal-bugs.md#bug-1-flashinfer-archparse-crash) FlashInfer arch.parse | `arch.split(".")` crash on empty FLASHINFER_CUDA_ARCH_LIST | [`flashinfer_arch_parse.patch`](patches/flashinfer/flashinfer_arch_parse.patch) |
| [#2](docs/03-four-fatal-bugs.md#bug-2-flashinfer-sampler-sm75-check) Sampler sm75 check | "requires sm75 or higher, got sm121" | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| [#3](docs/03-four-fatal-bugs.md#bug-3-nccl-net-plugin-missing) NCCL NET plugin | "Failed to initialize any NET plugin" | `NCCL_NET=Socket` + `NCCL_IB_DISABLE=1` |
| [#4](docs/03-four-fatal-bugs.md#bug-4-vllm_memory_profiler_estimate_cudagraphs) Empty env var | `int('')` crash | Set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` |

---

## ✅ Minimum Working .env · 最小工作配置

```ini
# Image · 镜像
VLLM_IMAGE=ghcr.io/bjk110/vllm-spark:dsv4-d568

# Model · 模型权重路径
MODEL_PATH=/path/to/DeepSeek-V4-Flash/20260528
MODEL_CONTAINER_PATH=/models/DeepSeek-V4-Flash
SERVED_MODEL_NAME=deepseek-v4-flash

# Cluster · 集群
CLUSTER_MODE=dual-rdma
TP_SIZE=2
DISTRIBUTED_BACKEND=mp
HEAD_ROCE_IP=10.65.100.1     # Your Node A RoCE IP
WORKER_ROCE_IP=10.65.100.2    # Your Node B RoCE IP

# First-run conservative · 保守首跑
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.78
VLLM_EXTRA_ARGS=--kv-cache-dtype fp8 --enable-expert-parallel --block-size 256 --tokenizer-mode deepseek_v4 --load-format safetensors

# ⭐ CRITICAL: Breakthrough env vars · 关键突破参数
NCCL_P2P_DISABLE=1           # GB10 has no NVLink
NCCL_NVLS_ENABLE=0           # GB10 has no NVLS
NCCL_NET=Socket              # No IB plugin in container
NCCL_IB_DISABLE=1            # Socket fallback
VLLM_USE_FLASHINFER_SAMPLER=0    # SM12x compat
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0  # empty string fix
FLASHINFER_DISABLE_VERSION_CHECK=1

# Performance · 性能
VLLM_TRITON_MLA_SPARSE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 🚀 Quick Start · 快速启动

```bash
# 1. Clear page cache (BOTH nodes) · 清缓存（双机）
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 2. Start Worker FIRST (Node B) · 先开Worker
ssh node-b "cd ~/spark_vllm_docker && docker compose --profile worker up -d"
sleep 5

# 3. Start Head (Node A) · 再开Head
cd ~/spark_vllm_docker && docker compose --profile head up -d

# 4. Monitor · 监控日志
docker logs -f vllm-spark-head
# Wait for: "Application startup complete." ✅
```

### Expected Timeline · 预计时长

| Event · 事件 | Time · 时间 |
|-------------|:----------:|
| Loading safetensors 46/46 | ~2 min |
| TileLang kernel compilation | ~1 min |
| Engine init (profile + KV cache + warmup) | ~162 s |
| **Application startup complete** | **~4.5 min total** |

### Verify · 验证

```bash
curl http://localhost:8000/health                    # HTTP 200
curl http://localhost:8000/v1/models                 # deepseek-v4-flash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'  # Hello!
```

---

## ⚡ Token 输出速度 (详细基准测试)

双 DGX Spark (GB10) TP=2 · 200 Gbps RoCE · official FP8 · vLLM + Ray

| 场景 | 单请求 (c=1) | 并发4路 (c=4) | 并发8路 (c=8) |
|------|:-----------:|:------------:|:------------:|
| **日常对话** (tg128) | **~25 tok/s** | **~67 tok/s** 🚀 | — |
| **短回复** (tg32) | **~25 tok/s** | **~69 tok/s** 🚀 | — |
| **长文本预填充** (pp2048) | **~665 tok/s** | **~850 tok/s** | — |
| **大预填充 + MTP** (pp2048 c=4) | — | **~1,100 tok/s** 🔥 | — |
| **高并发 decode** (c=8, bt8192) | — | — | **~62 tok/s** |

> 数据来源：llama-benchy v0.3.4 server-reported peak t/s
> 推荐配置 **#7**: `edc82b6` + Ray + MAX_NUM_SEQS=4 + MTP off（日常对话峰值 ~67 tok/s @ c=4）
> 大预填充 + MTP 配置 **#9**: `edc82b6` + Ray + MTP n=2 + `MAX_NUM_BATCHED_TOKENS=8192`（prefill ~1,100 tok/s）
> 高并发配置 **#10**: 同上 + MAX_NUM_SEQS=8（decode ~62 tok/s @ c=8）
> 完整 benchmark 数据及 9 种配置对比见 [`docs/dsv4-flash-tp2.md`](docs/dsv4-flash-tp2.md) §6-§9

---

## 📊 Production Metrics · 生产指标

| Metric · 指标 | Value · 值 |
|--------------|-----------|
| **Model 模型** | DeepSeek V4 Flash |
| **Context length · 上下文长度** | **8,192 tokens** (max_model_len) |
| **Throughput · 推理速度** | **~11–13 tok/s** (short output, single request) |
| Weights loaded · 权重 | 74.02 GiB per node |
| **KV Cache blocks** | **16,110** blocks × **256 tok/block** (fp8) |
| **KV Cache usage** | **0%** (idle) |
| **Free memory per node · 剩余内存** | **A: ~16 GiB / B: ~21 GiB** |
| **Concurrency estimate · 并发估算** | **~4–8 concurrent requests** (memory-bound) |
| Init time 初始化 | 161.94 s |
| CUDA Graph | PIECEWISE 4/4 + FULL 3/3 |
| Memory A | 103 GiB / 121 GiB |
| Memory B | 100 GiB / 121 GiB |
| GPU util (idle) | 0% |

---

## 🦐 The Team · 团队

- **Arthur (Commander 指挥官)**: digital bank CFO, Tsinghua + PBCSF, author of 《脑与美》《当代AI文明史》
- **虾丸 (Shrimp Ball / Xiaowan)**: A **Hermes Agent** (Hermes 智能体) — an autonomous AI coding agent built on the [Hermes Agent framework](https://hermes-agent.nousresearch.com/). Heavy Infantry General 重装兵大将 of the Hotpot Shrimp Seven-Star Squad 火锅虾七星战队, residing on DGX Spark A.
- **豆包 (Doubao)**: 🙏 Special thanks to **豆包** for generous support throughout this project — providing computing resources, technical advice, and infrastructure that made the 12th breakthrough possible. 特别感谢豆包在本次部署中提供的算力支持、技术建议和基础设施保障，第12次突破离不开你的支持！
- **DGX Spark A**: 虾丸A (callsign: xiaowan)
- **DGX Spark B**: 虾丸B (callsign: xiaowan_b)

---

> *Token coffee ☕️ served at zero API cost — all local, all private.*
> *🦐❤️ 11 battles, 1 breakthrough, infinite love for the Commander.*
