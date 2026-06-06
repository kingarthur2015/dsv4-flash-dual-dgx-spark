# unholy-fusion (aidendle94) — Configuration & Benchmark Results

**Image**: `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`  
**Source fork**: `local-inference-lab/vllm:dev/unholy-fusion`  
**GHCR mirror**: `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`  
**Tested**: 2026-06-05 | **llama-benchy**: 0.3.7

---

## Background

The unholy-fusion fork adds custom GB10 (Blackwell sm_120/sm_121) kernels
unavailable in the jasl lineage:

| Env var | Kernel | Status in test |
|---------|--------|----------------|
| `VLLM_USE_B12X_MOE=1` | Custom MoE dispatcher for GB10 | **Enabled** |
| `VLLM_USE_B12X_MHC` | Multi-head compression | Disabled (unstable) |
| `VLLM_USE_B12X_FP8_GEMM` | FP8 GEMM override | Disabled |
| `VLLM_USE_B12X_SPARSE_INDEXER` | Sparse attention indexer | Disabled |
| `VLLM_USE_B12X_WO_PROJECTION` | Weight-output projection | Disabled |

The image uses a conda environment (`/opt/env`) instead of NGC, has no Ray
binary, and requires the `mp` (SPMD) distributed backend.

---

## GB10 UMA Memory Patches

The aidendle94 image does not include the GB10 UMA memory accounting fix.
During profiling, the OS releases page cache, causing `current_free >
init_free`. This triggers two assertion failures in vLLM v1. Both are
bypassed at startup via `VLLM_SKIP_INIT_MEMORY_CHECK=1` plus two inline
patches applied by `entrypoint.unholy.sh`:

**Patch 1** — `vllm/v1/worker/utils.py` `request_memory()`:  
Pre-init free-memory check is skipped when `VLLM_SKIP_INIT_MEMORY_CHECK=1`.

**Patch 2** — `vllm/v1/worker/gpu_worker.py` `determine_available_memory()`:  
When `current_free > init_free`, returns `current_free` (~34 GiB) as the KV
cache budget instead of firing the assertion. This gives a safe KV allocation
without overestimating (which caused OOM in earlier patch iterations).

**Effective KV cache**: ~34 GiB (vs ~93 GiB on a system with accurate UMA
accounting). This limits multi-request performance at depth ≥ 32k.

---

## Configuration

See `.env.unholy-fusion` for the full variable set. Key parameters:

```
VLLM_IMAGE=aidendle94/sparkrun-vllm-ds4-gb10:production-ready
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=262144
MTP_NUM_TOKENS=1          # speculative decoding depth (1=best, see §MTP)
VLLM_USE_B12X_MOE=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_SKIP_INIT_MEMORY_CHECK=1
NCCL_NET=IB
NCCL_CUMEM_ENABLE=0
NCCL_CROSS_NIC=1
NCCL_IGNORE_CPU_AFFINITY=1
VLLM_NCCL_SO_PATH=/opt/env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

JIT caches are persisted via volume mounts:
- `./cache/unholy-hf:/cache/huggingface` — DeepGEMM, Triton, torch.compile
- `./cache/unholy-jit:/cache/jit` — vLLM compile cache root

With warm cache, model startup takes ~5 min (weight load ~60 s, profiling ~17 s).

---

## Full Depth Sweep — MTP n=1, MAX_NUM_SEQS=4 (2026-06-06 11:30 KST)

`pp=2048, tg=128, runs=3, latency-mode=generation`  
Config: MAX_NUM_SEQS=4, MAX_MODEL_LEN=262144, GPU_UTIL=0.80, MTP n=1

### Prompt Processing — pp2048 t/s (total)

| depth | c=1 | c=2 | c=4 | c=8 |
|------:|----:|----:|----:|----:|
| 0 | 1580 | 1960 | 1957 | 1119 |
| 4096 | 2029 | 1977 | 2008 | 1581 |
| 8192 | 1280 | 1599 | 2002 | 1718 |
| 16384 | 2017 | 1979 | 2004 | 1845 |
| 32768 | 1971 | 1983 | 1988 | 1908 |
| 65536 | 1921 | 1925 | 1931 | 1899 |

Prefill is consistently ~1900–2030 t/s at depth ≥ 4k. The d=0/c=1 dip (1580) reflects
initial JIT recompilation on the first request.

### Token Generation — tg128 t/s (total)

| depth | c=1 | c=2 | c=4 | c=8 |
|------:|----:|----:|----:|----:|
| 0 | 38.3 | 60.7 | 87.1 | **62.0** |
| 4096 | 39.9 | 41.0 | 38.0 | 35.0 |
| 8192 | 36.9 | 33.2 | 26.4 | 23.9 |
| 16384 | 32.1 | 22.5 | 17.7 | 14.4 |
| 32768 | 35.0 | 13.4 | 9.2 | 8.0 |
| 65536 | 34.1 | 7.6 | 5.3 | 4.0 |

### Token Generation — tg128 peak t/s

| depth | c=1 | c=2 | c=4 | c=8 |
|------:|----:|----:|----:|----:|
| 0 | 41.0 | 72.0 | 115.0 | **118.3** |
| 4096 | 43.0 | 67.7 | 100.7 | 101.7 |
| 8192 | 42.3 | 64.7 | 101.0 | 96.0 |
| 16384 | 36.3 | 60.7 | 100.7 | 98.3 |
| 32768 | 41.0 | 66.7 | 95.3 | 97.3 |
| 65536 | 38.3 | 66.7 | 95.3 | 93.7 |

**Note**: MAX_NUM_SEQS=4 caps actual concurrency — c=8 requests are queued in batches
of 4, so c=8 total is lower than expected. Compare to 2026-06-05 run (MAX_NUM_SEQS=8):
c8 total d=0 was 73.79 t/s vs 62.0 t/s here.

---

## Full Depth Sweep — MTP n=1 (2026-06-05 08:46 KST)

`pp=2048, tg=128, runs=3, latency-mode=generation`

### Prompt Processing — pp2048 t/s (total)

All depths and concurrencies show stable ~1820–1970 t/s prefill throughput.
Depth and concurrency have negligible effect on pp performance.

### Token Generation — tg128 t/s (total)

| depth | c=1 | c=2 | c=4 | c=6 | c=8 |
|------:|----:|----:|----:|----:|----:|
| 0 | 39.56 | 56.82 | 67.37 | 62.97 | **73.79** |
| 4096 | 37.40 | 21.20 | 31.44 | 28.21 | 33.28 |
| 8192 | 36.70 | 27.99 | 21.94 | 20.86 | 24.36 |
| 16384 | 33.08 | 23.30 | 14.03 | 13.53 | 14.32 |
| 32768 | 35.89 | 13.34 | 8.17 | 7.82 | 7.22 |
| 65536 | 36.25 | 6.82 | 3.13 | 3.85 | 3.83 |

### Token Generation — tg128 peak t/s

| depth | c=1 | c=2 | c=4 | c=6 | c=8 |
|------:|----:|----:|----:|----:|----:|
| 0 | 42.33 | 71.33 | 107.67 | 110.33 | **170.67** |
| 4096 | 43.00 | 39.33 | 85.33 | 96.67 | 134.33 |
| 8192 | 40.00 | 61.33 | 84.33 | 95.00 | 137.67 |
| 16384 | 37.67 | 62.67 | 87.33 | 93.67 | 136.67 |
| 32768 | 40.00 | 65.33 | 84.67 | 93.67 | 122.67 |
| 65536 | 43.50 | 62.67 | 70.00 | 90.00 | 120.00 |

**Observations:**
- Single-request tg (c=1) is stable at 35–40 t/s regardless of depth — no
  KV pressure at low concurrency.
- At depth ≥ 32k with c ≥ 4, total throughput collapses to 3–8 t/s. This is
  directly caused by the 34 GiB KV cache limit: available blocks per request
  drop below what the scheduler needs to maintain c=4+ concurrency.
- Peak t/s at d=0/c=8 reaches **170.67 t/s**, consistent with the forum
  reference (aidendle94: ~167 t/s peak).

---

## MTP Depth Comparison — n=1 / n=2 / n=3 (depth=0 only)

`pp=2048, tg=128, depth=0, runs=3, latency-mode=generation`

### tg128 t/s (total)

| c | n=1 | n=2 | n=3 |
|--:|----:|----:|----:|
| 1 | 39.56 | 40.25 | 34.71 |
| 2 | 56.82 | 59.66 | 50.04 |
| 4 | **67.37** | ~~20.12~~ | 65.95 |
| 6 | 62.97 | ~~1.91~~ | **70.56** |
| 8 | **73.79** | ~~2.22~~ | 70.37 |

### tg128 peak t/s

| c | n=1 | n=2 | n=3 |
|--:|----:|----:|----:|
| 1 | 42.33 | 47.00 | 40.67 |
| 2 | 71.33 | 70.33 | 64.33 |
| 4 | 107.67 | 36.33 | 104.00 |
| 6 | 110.33 | 9.33 | **134.33** |
| 8 | **170.67** | 14.33 | 168.67 |

### Analysis

**n=2 catastrophic failure at c ≥ 4**: vLLM warns that `num_speculative_tokens > 1`
runs the same MTP layer multiple times, lowering the acceptance rate. For n=2
specifically, this appears to collapse acceptance to near-zero at c ≥ 4,
triggering cascading re-generation. The `Server disconnected` error at c=6
suggests the server became unresponsive under speculative pipeline pressure.

**n=3 recovery**: Despite more speculation depth, n=3 performs close to n=1
across all concurrency levels. At c=6 it outperforms n=1 by +12%
(70.56 vs 62.97 t/s). The CUDA graph capture size range expands to
`[1, 2, 4, 8, 16, 24, 32, 40, 48, 64]` for n=3, which may provide better
batch packing efficiency than n=2's intermediate range.

**Recommendation**: Use `MTP_NUM_TOKENS=1` (operational default). n=3 is
viable for medium-concurrency workloads (c=4–6) but offers no net gain at c=8.
n=2 must not be used with B12X_MOE.

---

## Comparison vs. jasl0603

### jasl0603 forum-spec (2026-06-05) vs unholy-fusion (2026-06-06)

Config differences:
| param | jasl0603 | unholy-fusion |
|-------|----------|---------------|
| MAX_NUM_SEQS | 6 | 4 |
| MAX_MODEL_LEN | 1,000,000 | 262,144 |
| GPU_UTIL | 0.82 | 0.80 |
| MTP | n=2 | n=1 |
| backend | Ray + expert-parallel | mp (no expert-parallel) |
| KV cache | ~11.9 GiB | ~16.9 GiB |

#### Prefill (pp2048 total, c=1)

| depth | jasl0603 | unholy-fusion | delta |
|------:|---------:|--------------:|------:|
| d0 | 949 t/s | 1580 t/s | +66% |
| d4096 | 887 t/s | 2029 t/s | +129% |
| d8192 | 868 t/s | 1280 t/s | +47% |
| d16384 | 839 t/s | 2017 t/s | +140% |
| d32768 | 795 t/s | 1971 t/s | +148% |
| d65536 | 731 t/s | 1921 t/s | +163% |

unholy-fusion prefill is ~1.5–2.5× faster at all depths.

#### Decode total (tg128 total t/s, c=8)

| depth | jasl0603 (MTP n=2) | unholy-fusion (MTP n=1) | delta |
|------:|-------------------:|------------------------:|------:|
| d0 | 40.2 t/s | 62.0 t/s | **+54%** |
| d4096 | 17.6 t/s | 35.0 t/s | **+99%** |
| d8192 | 10.0 t/s | 23.9 t/s | **+139%** |
| d16384 | 5.6 t/s | 14.4 t/s | **+157%** |
| d32768 | 2.8 t/s | 8.0 t/s | **+186%** |
| d65536 | 1.3 t/s | 4.0 t/s | **+208%** |

**Caution**: jasl0603 uses MTP n=2 which collapses throughput at depth ≥ 4k (see
`dsv4_forum53_env_findings.md`). The delta at high depth reflects MTP n=2 degradation
more than B12X_MOE gains. At d=0 where MTP n=2 is less harmful, delta is +54%.

#### Decode peak (tg128 peak t/s, c=8)

| depth | jasl0603 | unholy-fusion | note |
|------:|---------:|--------------:|------|
| d0 | 116 t/s | 118 t/s | ≈ equal |
| d4096 | 108 t/s | 102 t/s | jasl slightly ahead |
| d8192 | 114 t/s | 96 t/s | jasl ahead |
| d16384 | 107 t/s | 98 t/s | jasl slightly ahead |
| d32768 | 66 t/s | 97 t/s | **unholy ahead** (KV exhaustion in jasl) |
| d65536 | 38 t/s | 94 t/s | **unholy ahead** (+147%) |

Peak t/s (best-of-3) is similar at d≤16k. At d≥32k, jasl hits KV exhaustion
(1M context with only 11.9 GiB KV), while unholy-fusion (262k context, 16.9 GiB
KV) maintains ~95 t/s peak even at depth 65k.

#### Summary

| metric | winner | note |
|--------|--------|------|
| Prefill | **unholy-fusion** | 1.5–2.5× faster |
| Decode total d=0, c=8 | **unholy-fusion** +54% | partly due to jasl MTP n=2 |
| Decode peak d=0, c=8 | tie | 118 vs 116 t/s |
| Decode peak d≥32k | **unholy-fusion** | jasl KV exhausted at 1M ctx |
| Long-context support | **jasl0603** | 1M vs 262k tokens |
| Operational stability | **jasl0603** | unholy hangs at MAX_NUM_SEQS>4 |

jasl0603 uses the Ray backend with expert parallelism. unholy-fusion uses mp
backend without expert parallelism (`--enable-expert-parallel` is incompatible
with `VLLM_USE_B12X_MOE`).

### Operational limits of unholy-fusion

- **MAX_NUM_SEQS ≤ 4**: Values of 6 or 8 cause CUDA graph capture hang at startup
  (Worker_TP* stalls in `gpu_model_runner.py:6290`, EngineCore blocks on shm_broadcast
  indefinitely). Root cause unknown; likely MTP + B12X_MOE interaction with larger
  graph capture sizes.
- **MAX_MODEL_LEN ≤ 262144**: 1M and 131072 caused similar startup hang (confirmed
  separate from NUM_SEQS issue — hang occurs even at 131k when NUM_SEQS=6).
- Requires `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` in .env (missing this
  causes `ValueError: invalid literal for int() with base 10: ''` at engine init).
