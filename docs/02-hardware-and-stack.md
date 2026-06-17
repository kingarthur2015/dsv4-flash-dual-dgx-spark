# 🖥️ Hardware & Software Stack · 硬件与软件栈

---

## Hardware · 硬件

### Equipment · 设备清单

| Item 项目 | Node A (Head) | Node B (Worker) |
|-----------|:------------:|:--------------:|
| **Model 型号** | DGX Spark | DGX Spark |
| **SoC** | NVIDIA GB10 Grace Hopper | NVIDIA GB10 Grace Hopper |
| **GPU** | 1× GB10 (SM 12.1) | 1× GB10 (SM 12.1) |
| **Architecture 架构** | Blackwell | Blackwell |
| **FP8 TOPS** | 1000 | 1000 |
| **Memory 内存** | 121 GB unified LPDDR5X | 121 GB unified LPDDR5X |
| **CPU cores** | 20 (Arm Neoverse V2) | 20 (Arm Neoverse V2) |
| **Storage 存储** | 2TB NVMe | 2TB NVMe |
| **Network 网络** | 2× CX7 @ 200Gb/s RoCE | 2× CX7 @ 200Gb/s RoCE |

### ⚠️ Critical · 关键提醒

**Each DGX Spark has exactly 1 GPU.** TP=2 means:
**每台DGX Spark只有1个GPU。** TP=2的意思是：
- Node A: GPU #0
- Node B: GPU #0
- Total: 2 GPUs across 2 machines (NOT 2 GPUs per machine!)
- 总共2个GPU分布在2台机器（不是每台2个GPU！）

### Network Topology · 网络拓扑

```
┌─────────────────────┐         ┌─────────────────────┐
│    Node A (Head)    │         │    Node B (Worker)  │
│                     │         │                     │
│  Mgmt: 10.65.254.207├───LAN───┤ Mgmt: 10.65.254.234 │
│                     │         │                     │
│  RoCE: 10.65.100.1  ├───CX7───┤ RoCE: 10.65.100.2   │
│                     │ 620Gb/s │                     │
│  RoCE2: 10.65.100.5 │         │ RoCE2: 10.65.100.6  │
└─────────────────────┘         └─────────────────────┘
```

---

## Software Stack · 软件栈

| Layer 层 | Component 组件 | Version 版本 | Notes 说明 |
|----------|--------------|-------------|-----------|
| **OS 系统** | Ubuntu | Core 24 | Snap-based |
| **Container 容器** | Docker | 28+ | With nvidia-container-toolkit |
| **Orchestration 编排** | Docker Compose | V2 | From [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker) |
| **Runtime** | nvidia-container-toolkit | 1.19.1 | GPU passthrough |
| **Serving 推理** | jasl/vLLM fork | @ edc82b6 | +249 commits over v0.21.0 |
| **Attention** | FlashInfer | 0.6.12 | With SM12x arch patch |
| **Distributed 分布式** | MPI backend | — | `--distributed-executor-backend mp` |
| **NCCL** | 2.30.4 | Built-in | Socket mode (IB plugin TBD) |
| **Network 网络** | Socket TCP | Fallback | RoCE IB verbs upgrade pending |

### Why Docker over bare metal? · 为什么用Docker？

The `dsv4-d568` image pre-bundles the jasl/vLLM fork with critical fixes:

`dsv4-d568`镜像预装了jasl/vLLM fork，包含关键修复：

| Fix 修复 | Upstream v0.21.0 | dsv4-d568 |
|----------|:----------------:|:---------:|
| CUDA Graph SM12.x bug | ❌ Broken | ✅ Fixed |
| MTP small-batch hang | ❌ Broken | ✅ Fixed |
| SM12x kernel optimizations | ❌ None | ✅ FP8 MQA, sparse MLA, KV gather |
| DSV4-specific patches | ❌ Manual | ✅ Pre-applied |
| NCCL version | 2.28.9 | 2.30.4 |

---

## Key Parameter Reference · 关键参数说明

| Parameter 参数 | Value 值 | Why 为什么 |
|--------------|---------|-----------|
| `--kv-cache-dtype fp8` | REQUIRED | DSV4's MLA asserts on non-FP8 KV cache |
| `--enable-expert-parallel` | REQUIRED | Splits 256 experts across 2 nodes |
| `--block-size 256` | DSV4 standard | Required for CSA compressed attention |
| `--tokenizer-mode deepseek_v4` | Optional | Auto-detected in v0.21.0 |
| `--load-format safetensors` | Safe default | Avoids PyTorch weight loading issues |
| `--distributed-executor-backend mp` | For nnodes>1 | Not `ray` in our config |
| `NCCL_P2P_DISABLE=1` | GB10 mandatory | No NVLink between CPUs |
| `NCCL_NVLS_ENABLE=0` | GB10 mandatory | NVLS code path hangs MoE routing |
| `NCCL_NET=Socket` | Temporary fix | Container lacks IB verbs plugin |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | SM12x fix | FlashInfer sm75 check fails on GB10 |
| `VLLM_TRITON_MLA_SPARSE=1` | Performance | Enables sparse MLA Triton kernels |

### Parameters that DON'T exist · 不存在的参数

Verified by vLLM source code. AI assistants often suggest these:

经过vLLM源码验证。AI助手经常推荐这些不存在或无效的参数：

| Non-existent parameter | Why 原因 |
|----------------------|---------|
| ❌ `--quantization fp4_fp8_mixed` | Does not exist in vLLM |
| ❌ `--expert-parallel-size N` | Only `--enable-expert-parallel` (no size) |
| ❌ `--distributed-executor-backend nccl` | Not a valid option (use `mp`, `ray`, `uni`) |
| ❌ `--offload-backend uva` on GB10 | Useless — same physical memory pool |

---

## GB10 Unified Memory · 统一内存架构

**CRITICAL**: On GB10, GPU and CPU share the same 121GB physical RAM.

**关键**：GB10上GPU和CPU共享同一块121GB物理内存。

```
┌─────────────── 121 GB LPDDR5X ───────────────┐
│ GPU allocations │ OS │ Desktop │ Drivers │ ... │
│     (NO hardware boundary between GPU/system)  │
└───────────────────────────────────────────────┘
```

Consequences 后果:
- OOM can kill gnome-shell, sshd → snow screen → hard reboot required
- OOM可能杀掉gnome-shell、sshd → 雪花屏 → 需硬重启
- UVA offloading doesn't help (same pool)
- UVA offload没用（同一池子）
- Weight loading peak > final size (double allocation during load)
- 权重加载峰值 > 最终大小（加载时双重分配）

Always set `--gpu-memory-utilization` conservatively on first run (0.78 or lower).
首跑务必保守设置`--gpu-memory-utilization`（0.78或更低）。

---

## Docker Image Strategy · 镜像策略

Pull on Node A (faster internet), then transfer to Node B via RoCE:

在A端拉取（网快），通过RoCE传到B端：

```bash
# Pull on A · A端拉取
docker pull ghcr.io/bjk110/vllm-spark:dsv4-d568

# Package and transfer · 打包传输
docker save ghcr.io/bjk110/vllm-spark:dsv4-d568 | gzip > /tmp/dsv4-d568.tar.gz
scp /tmp/dsv4-d568.tar.gz xiaowan_b:/tmp/

# Load on B · B端加载
ssh xiaowan_b "docker load < /tmp/dsv4-d568.tar.gz && rm /tmp/dsv4-d568.tar.gz"
rm /tmp/dsv4-d568.tar.gz
```

Image size 镜像大小: 13.8 GB (compressed 压缩) / 46.4 GB (uncompressed 解压)
