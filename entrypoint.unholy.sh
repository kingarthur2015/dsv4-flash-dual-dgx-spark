#!/usr/bin/env bash
# Entrypoint for aidendle94/sparkrun-vllm-ds4-gb10 (unholy-fusion) image.
# Uses --distributed-executor-backend mp (SPMD) instead of Ray.
# Conda env at /opt/env; dsv4-vllm-entrypoint wrapper replicated inline.
set -euo pipefail

# ── conda env setup (mirrors dsv4-vllm-entrypoint) ──────────────────────────
export PATH="/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/opt/env/targets/sbsa-linux}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="/opt/env/lib:/opt/env/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/opt/env/bin/aarch64-conda-linux-gnu-g++}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin ${CUDAHOSTCXX} -I${CUDA_HOME}/include/cccl -I${CUDA_HOME}/include}"
export DG_JIT_NVCC_COMPILER="${DG_JIT_NVCC_COMPILER:-/opt/env/bin/nvcc}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-/opt/env/bin/nvcc}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ── node rank from ROLE if not explicit ─────────────────────────────────────
if [ -z "${NODE_RANK:-}" ]; then
  case "${ROLE:-head}" in
    head)   NODE_RANK=0 ;;
    worker) NODE_RANK=1 ;;
    *)      NODE_RANK=0 ;;
  esac
fi
export NODE_RANK

# ── per-rank cache dirs (prevent rank-0/1 collision) ────────────────────────
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-rank${NODE_RANK}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-rank${NODE_RANK}}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ── NCCL GID index: auto-detect RoCE v2 IPv4-mapped entry ───────────────────
for HCA in rocep1s0f0 roceP2p1s0f0; do
  for i in $(seq 0 7); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null) || continue
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null) || continue
    case "$t" in *"RoCE v2"*)
      case "$g" in *0000:0000:0000:0000:0000:ffff:*)
        export NCCL_IB_GID_INDEX=$i
        echo "[unholy] NCCL_IB_GID_INDEX=${i} (${HCA})"
        break 2
      ;; esac
    ;; esac
  done
done

# ── headless flag for worker node ───────────────────────────────────────────
if [ "${NODE_RANK}" = "0" ]; then
  echo "[unholy] ROLE=head NODE_RANK=0 MASTER=${HEAD_ROCE_IP:-?}:${MASTER_PORT:-25000}"
  HEADLESS_FLAG=""
else
  echo "[unholy] ROLE=worker NODE_RANK=${NODE_RANK} MASTER=${HEAD_ROCE_IP:-?}:${MASTER_PORT:-25000}"
  HEADLESS_FLAG="--headless"
fi

# ── cutlass DSL install if missing (aidendle94 pattern) ─────────────────────
PY=/opt/env/bin/python
if ! $PY -c "import cutlass; assert cutlass.__version__ == '4.5.1'" 2>/dev/null; then
  echo "[unholy] Installing nvidia-cutlass-dsl 4.5.1 ..."
  $PY -m pip install --no-cache-dir \
    "nvidia-cutlass-dsl==4.5.1" \
    "nvidia-cutlass-dsl-libs-base==4.5.1" \
    "nvidia-cutlass-dsl-libs-cu13==4.5.1" > /tmp/cutlass-install.log 2>&1 || true
fi

echo "[unholy] Starting vllm serve (mp backend, NODE_RANK=${NODE_RANK})"
exec vllm serve "${MODEL_CONTAINER_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 --port "${HOST_PORT:-8000}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len "${MAX_MODEL_LEN:-200000}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":true}' \
  --enable-flashinfer-autotune \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${HEAD_ROCE_IP}" \
  --master-port "${MASTER_PORT:-25000}" \
  ${HEADLESS_FLAG}
