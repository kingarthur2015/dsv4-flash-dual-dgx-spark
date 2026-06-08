# ✅ Verified Configuration · 已验证配置

## Complete `.env` · 完整环境变量文件

```ini
# ===== Docker Image 镜像 =====
VLLM_IMAGE=ghcr.io/bjk110/vllm-spark:dsv4-d568

# ===== Model Paths 模型路径 =====
MODEL_PATH=/home/arthur/models/deepseek/deepseek-v4-flash/20260528
MODEL_CONTAINER_PATH=/models/DeepSeek-V4-Flash
SERVED_MODEL_NAME=deepseek-v4-flash

# ===== Cluster 集群 =====
CLUSTER_MODE=dual-rdma
TP_SIZE=2
DISTRIBUTED_BACKEND=mp
HEAD_ROCE_IP=10.65.100.1
WORKER_ROCE_IP=10.65.100.2
ROCE_IF_NAME=enp1s0f0np0
IB_HCA_NAME=rocep1s0f0,roceP2p1s0f0

# ===== Serving Parameters 服务参数 =====
HOST_PORT=8000
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.78
VLLM_EXTRA_ARGS=--kv-cache-dtype fp8 --enable-expert-parallel --block-size 256 --tokenizer-mode deepseek_v4 --load-format safetensors

# ===== ⭐ Breakthrough GB10 Params 关键突破参数 =====
NCCL_P2P_DISABLE=1
NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1
NCCL_NET=Socket
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
FLASHINFER_DISABLE_VERSION_CHECK=1

# ===== Performance 性能 =====
VLLM_TRITON_MLA_SPARSE=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## docker-compose.yml — Environment Section

⚠️ **Every env var in .env must ALSO be declared in docker-compose.yml**

**.env的每个环境变量必须在docker-compose.yml中也声明一次！**

```yaml
services:
  head:
    image: ${VLLM_IMAGE}
    container_name: vllm-spark-head
    profiles: ["head"]
    network_mode: host
    ipc: host
    environment:
      - ROLE=head
      - CLUSTER_MODE=${CLUSTER_MODE:-dual-rdma}
      - TP_SIZE=${TP_SIZE:-2}
      - DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND:-mp}
      - MODEL_CONTAINER_PATH=${MODEL_CONTAINER_PATH}
      - SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
      - MAX_MODEL_LEN=${MAX_MODEL_LEN}
      - MAX_NUM_SEQS=${MAX_NUM_SEQS}
      - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
      - VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
      
      # ⭐ GB10 Critical 关键参数
      - NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
      - NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
      - NCCL_NET=${NCCL_NET:-Socket}
      - NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
      - VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}
      - FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
```

## Env Var Verification · 环境变量验证

```bash
# Check what actually reached the container
# 检查容器实际收到的环境变量
docker exec vllm-spark-head env | grep -E 'NCCL|VLLM|FLASHINEF'
# Expected output 预期输出:
# NCCL_P2P_DISABLE=1
# NCCL_NVLS_ENABLE=0
# NCCL_NET=Socket
# VLLM_USE_FLASHINFER_SAMPLER=0
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# FLASHINFER_DISABLE_VERSION_CHECK=1
```

## Memory Budget · 内存预算

```
Physical memory 物理内存:          121 GB (unified 统一内存)
System reserved 系统预留:           ~14 GB
Available for vLLM vLLM可用:        ~107 GB

DSV4 Flash TP=2 (per node 每节点):
  Expert weights 专家权重:          70.7 GB  (128/256 experts)
  Attention weights 注意力权重:      2.6 GB
  Embed + LMHead:                   2.2 GB
  PyTorch/CUDA overhead:           ~8 GB
  KV Cache (8K, FP8):               1.3 GB
  ────────────────────────────────
  Subtotal 小计:                   ~84.8 GB
  Free remaining 剩余:              ~22 GB
```

## Parameter Graduation Path · 参数升级路线

| Stage 阶段 | Context 上下文 | GPU Util | MTP | CUDA Graph | Status 状态 |
|:---------:|:-------------:|:--------:|:---:|:----------:|:----------:|
| 🟢 First run | 8K | 0.78 | Off | Off | ✅ Verified |
| 🟡 Mid-term | 32K-128K | 0.85 | Off | ON | ⬜ Planned |
| 🔵 Production | 128K-524K | 0.90 | ON | ON | ⬜ Future |
