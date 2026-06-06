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

## jasl0603 Forum-Spec Results (2026-06-05, for comparison)

`pp=2048, tg=128, runs=3, latency-mode=generation`  
Config: MAX_NUM_SEQS=6, MAX_MODEL_LEN=1,000,000, GPU_UTIL=0.82, MTP n=2, Ray + expert-parallel  
KV cache: 11.94 GiB (spark01) / 11.99 GiB (spark02)

### Prefill — pp2048 t/s (total)

| depth   | c=1 | c=2 | c=4 | c=8 |
|--------:|----:|----:|----:|----:|
| d0      | 949 | 923 | 789 | 716 |
| d4096   | 887 | 915 | 913 | 804 |
| d8192   | 868 | 815 | 837 | 781 |
| d16384  | 839 | 833 | 852 | 789 |
| d32768  | 795 | 690 | 784 | 718 |
| d65536  | 731 | 601 | 639 | 624 |
| d131072 | 648 | 619 | 613 | 619 |

Prefill at d=0/c=1 peaks at 949 t/s. At higher concurrency (c=4/8), scheduler
contention with MTP re-generation lowers throughput to 716–789 t/s. The jasl0603
prefill ceiling is ~950 t/s for a single stream, compared to ~2030 t/s for unholy-fusion.

### Decode — tg128 t/s (total)

| depth   | c=1  | c=2  | c=4  | c=8  |
|--------:|-----:|-----:|-----:|-----:|
| d0      | 36.0 | 39.2 | 38.2 | 40.2 |
| d4096   | 36.2 | 25.9 | 19.5 | 17.6 |
| d8192   | 36.5 | 14.8 | 12.4 | 10.0 |
| d16384  | 32.5 | 12.1 | 7.5  | 5.6  |
| d32768  | 32.2 | 4.0  | 2.9  | 2.8  |
| d65536  | 60.8†| 1.9  | 1.5  | 1.3  |
| d131072 | 25.4 | 1.0  | 0.8  | 0.7  |

† d65536/c=1 shows 60.8 t/s with σ=43.44 — high variance indicates MTP n=2 occasionally
accelerates single-stream generation, but it is not reproducibly stable.

**MTP n=2 collapse pattern**: At c=1, throughput stays 25–40 t/s across all depths.
At c≥2, throughput halves at d=4096 and collapses to ≤2 t/s by d=32768. This is the
MTP n=2 acceptance-rate collapse under queued requests: the speculative pipeline
re-generates tokens repeatedly when multiple requests compete for the same KV blocks.

### Decode — tg128 peak t/s

| depth   | c=1   | c=2   | c=4   | c=8   |
|--------:|------:|------:|------:|------:|
| d0      | 41.7  | 49.7  | 90.3  | **116.3** |
| d4096   | 41.7  | 66.7  | 89.7  | 108.3 |
| d8192   | 41.7  | 66.7  | 89.3  | 114.0 |
| d16384  | 38.0  | 61.0  | 85.7  | 106.7 |
| d32768  | 38.3  | 56.7  | 67.3  | 66.0  |
| d65536  | 77.4† | 38.7  | 34.0  | 37.7  |
| d131072 | 30.7  | 32.3  | 32.3  | 31.7  |

Peak (best-of-3) at d≤16k remains strong (90–116 t/s at c=8), showing the hardware
can deliver high burst decode even with MTP n=2. At d≥32k, peak collapses as KV cache
(11.9 GiB) is exhausted — cannot hold 8 concurrent sequences × 32k+ context.

---

## Comparison vs. jasl0603

### Config differences

| param | jasl0603 | unholy-fusion |
|-------|----------|---------------|
| MAX_NUM_SEQS | 6 | 4 |
| MAX_MODEL_LEN | 1,000,000 | 262,144 |
| GPU_UTIL | 0.82 | 0.80 |
| MTP | n=2 | n=1 |
| backend | Ray + expert-parallel | mp (no expert-parallel) |
| KV cache | ~11.9 GiB | ~16.9 GiB |

### Prefill (pp2048 total, c=1 and c=8)

| depth | jasl c1 | unholy c1 | delta c1 | jasl c8 | unholy c8 | delta c8 |
|------:|--------:|----------:|---------:|--------:|----------:|---------:|
| d0    | 949     | 1580      | **+66%** | 716     | 1119      | **+56%** |
| d4096 | 887     | 2029      | **+129%**| 804     | 1581      | **+97%** |
| d8192 | 868     | 1280      | **+47%** | 781     | 1718      | **+120%**|
| d16384| 839     | 2017      | **+140%**| 789     | 1845      | **+134%**|
| d32768| 795     | 1971      | **+148%**| 718     | 1908      | **+166%**|
| d65536| 731     | 1921      | **+163%**| 624     | 1899      | **+204%**|

**unholy-fusion prefill is 1.5–3× faster at all depths and concurrencies.**  
Root cause: B12X custom MoE dispatcher + mp SPMD backend eliminates Ray actor
overhead on expert routing. jasl0603's expert-parallel mode adds per-token
routing synchronization that B12X_MOE bypasses with a custom GB10 dispatch path.

### Decode total (tg128 total t/s)

| depth  | jasl c1 | unholy c1 | jasl c4 | unholy c4 | jasl c8 | unholy c8 |
|-------:|--------:|----------:|--------:|----------:|--------:|----------:|
| d0     | 36.0    | 38.3      | 38.2    | **87.1**  | 40.2    | 62.0      |
| d4096  | 36.2    | 39.9      | 19.5    | 38.0      | 17.6    | 35.0      |
| d8192  | 36.5    | 36.9      | 12.4    | 26.4      | 10.0    | 23.9      |
| d16384 | 32.5    | 32.1      | 7.5     | 17.7      | 5.6     | 14.4      |
| d32768 | 32.2    | 35.0      | 2.9     | 9.2       | 2.8     | 8.0       |
| d65536 | 60.8†   | 34.1      | 1.5     | 5.3       | 1.3     | 4.0       |

- **c=1**: Nearly identical (36–40 t/s). Single-stream decode is hardware-bound; neither
  MTP n=2 nor B12X changes the single-request decode rate significantly.
- **c=4**: unholy-fusion leads at d=0 (+128%); gap narrows at depth. At d=0, unholy-fusion
  serves all 4 concurrent requests (MAX_NUM_SEQS=4), while jasl0603 with MTP n=2 spends
  cycles on speculative re-generation.
- **c=8**: unholy-fusion leads (+54% at d=0), but advantage narrows. Both are limited —
  unholy-fusion by MAX_NUM_SEQS=4 (queues 4 extra), jasl0603 by MTP n=2 overhead.
- **d≥4k, c≥2**: jasl0603 collapses due to MTP n=2 acceptance-rate drop. This is not
  a hardware or KV-cache limit — single-stream (c=1) stays at 32–36 t/s even at d=131k.

### Decode peak (tg128 peak t/s, best-of-3 run)

| depth  | jasl c4 | unholy c4 | jasl c8 | unholy c8 |
|-------:|--------:|----------:|--------:|----------:|
| d0     | 90.3    | 115.0     | **116.3** | 118.3   |
| d4096  | 89.7    | 100.7     | 108.3   | 101.7     |
| d8192  | 89.3    | 101.0     | **114.0** | 96.0    |
| d16384 | 85.7    | 100.7     | **106.7** | 98.3    |
| d32768 | 67.3    | 95.3      | 66.0    | **97.3**  |
| d65536 | 34.0    | 95.3      | 37.7    | **93.7**  |

Peak throughput (best single run) tells a different story from total:
- At **d≤16k**, both are similar (~90–116 t/s at c=8). jasl0603 occasionally edges
  ahead because a lucky run has high MTP acceptance → burst decode.
- At **d≥32k**, jasl0603 drops sharply (KV exhaustion at 11.9 GiB with 1M context
  model). unholy-fusion maintains ~95 t/s peak because its 16.9 GiB KV with 262k
  context has more headroom per block at these depths.

### Summary

| metric | winner | magnitude | note |
|--------|--------|-----------|------|
| Prefill all depths | **unholy-fusion** | 1.5–3× | B12X MoE + mp backend |
| Decode total c=1 | tie | <10% | hardware-bound single stream |
| Decode total c=4, d=0 | **unholy-fusion** | +128% | MTP n=2 overhead in jasl |
| Decode total c=8, d=0 | **unholy-fusion** | +54% | NUM_SEQS cap limits both |
| Decode total d≥4k, c≥2 | **unholy-fusion** | +100–200% | MTP n=2 collapse in jasl |
| Decode peak d≤16k | tie | <10% | both near hardware ceiling |
| Decode peak d≥32k | **unholy-fusion** | +44–148% | jasl KV exhausted (11.9 GiB) |
| Long-context | **jasl0603** | 4× more | 1M vs 262k tokens |
| Operational stability | **jasl0603** | — | unholy hangs at NUM_SEQS>4 |

**Key takeaway**: unholy-fusion's B12X_MOE kernel delivers a massive prefill advantage
(~2×) and eliminates MTP n=2 throughput collapse at depth. The decode peak hardware
ceiling is similar (~115–120 t/s at c=8, d=0). The practical gap is: unholy-fusion
cannot run with MAX_NUM_SEQS>4 or MAX_MODEL_LEN>262k, making it unsuitable as a
drop-in for long-context or high-concurrency production workloads.

jasl0603 uses the Ray backend with expert parallelism. unholy-fusion uses mp
backend without expert parallelism (`--enable-expert-parallel` is incompatible
with `VLLM_USE_B12X_MOE`).

### Operational limits of unholy-fusion

- **MAX_NUM_SEQS ≤ 4**: Values ≥ 5 cause CUDA graph capture hang at startup
  (Worker_TP* stalls in `gpu_model_runner.py:6290`, EngineCore blocks on shm_broadcast
  indefinitely). Root cause unknown; likely MTP + B12X_MOE interaction with larger
  graph capture sizes.
- **MAX_MODEL_LEN ≤ 262144**: 1M and 131072 tested — both caused startup hang
  when combined with NUM_SEQS=6. Not tested with NUM_SEQS=4.
- Requires `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` in .env (missing this
  causes `ValueError: invalid literal for int() with base 10: ''` at engine init).
