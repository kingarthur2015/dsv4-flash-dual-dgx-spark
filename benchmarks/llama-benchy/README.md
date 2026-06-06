# llama-benchy Raw Results

This directory contains raw `llama-benchy` output files (Markdown tables) from
DGX Spark / GB10 experiments on DSV4-Flash, unholy-fusion, and Qwen3.5-397B-INT4-TQ.

Files are retained for traceability. The presence of a file does **not** imply
the test conditions it encodes are currently recommended.

For the interpreted DSV4 / unholy-fusion comparison, read
[`docs/unholy-fusion-benchmark.md`](../../docs/unholy-fusion-benchmark.md).

---

## Filename legend

Filenames encode test conditions using dash-separated tokens.

| Token | Meaning |
|---|---|
| `ray` | Ray distributed backend experiment |
| `mp` | Multiprocessing (SPMD) backend experiment |
| `maxseq4` | `MAX_NUM_SEQS=4` — the current documented safe limit for unholy-fusion |
| `maxseq8` | `MAX_NUM_SEQS=8` — **experimental / not the safe default**; causes CUDA graph capture hang on unholy-fusion |
| `mtp1` | `MTP_NUM_TOKENS=1` — recommended |
| `mtp2` | `MTP_NUM_TOKENS=2` — **known to cause throughput collapse at `c≥4`** in some configs |
| `nomtp` | Speculative decoding disabled |
| `bt8192` | `MAX_NUM_BATCHED_TOKENS=8192` — current safe default |
| `bt12288` | `MAX_NUM_BATCHED_TOKENS=12288` — A/B experiment |
| `bt16384` | `MAX_NUM_BATCHED_TOKENS=16384` — A/B experiment |
| `c1to4` | Concurrency sweep: c=1, 2, 4 |
| `c1to8` | Concurrency sweep: c=1, 2, 4, 8 |
| `DEPTH` | Depth sweep (prompt length varies; not concurrency sweep) |
| `edc82b6` | Specific vLLM/jasl commit hash (dsv4-d568 cherry-sched variant tested at that SHA) |
| `cherry-sched` | `dsv4-d568-cherry-sched` image variant |
| `OMP8` | `OMP_NUM_THREADS=8` tuning experiment |
| `flashsampler` | Flash sampler A/B experiment |
| `no-custom-allreduce` | NCCL custom all-reduce disabled (A/B) |
| `k3v4` / `k8v4` | TurboQuant KV quantization config (3-bit K / 4-bit K + 4-bit V) |
| `3bit` / `4bit` | TurboQuant weight bitwidth variant |

Tokens not listed above should be read from the file header comment.

---

## File inventory (21 files)

### DSV4-Flash FP8 TP=2

| File | Image / backend | MAX_NUM_SEQS | MTP | Notes |
|---|---|---|---|---|
| `results_dsv4-flash-fp8-tp2.md` | early dsv4-d568 | — | — | earliest DSV4 baseline |
| `results_dsv4-flash-fp8-tp2-c1to4.md` | dsv4-d568 | — | — | baseline concurrency sweep |
| `results_dsv4-flash-fp8-tp2-maxseq4-c1to4.md` | dsv4-d568 | 4 | — | maxseq=4 sweep |
| `results_dsv4-flash-fp8-tp2-maxseq4-mtp1-c1to4.md` | dsv4-d568 | 4 | 1 | mtp=1 test |
| `results_dsv4-flash-fp8-tp2-maxseq4-mtp2-c1to4.md` | dsv4-d568 | 4 | 2 | mtp=2 test |
| `results_dsv4-flash-fp8-tp2-mp-maxseq4-mtp2-c1to4.md` | dsv4-d568, mp | 4 | 2 | mp backend |
| `results_dsv4-flash-fp8-tp2-edc82b6-mp-maxseq4-mtp2-c1to4.md` | edc82b6, mp | 4 | 2 | commit-pinned mp |
| `results_dsv4-flash-fp8-tp2-edc82b6-mp-maxseq4-nomtp-c1to4.md` | edc82b6, mp | 4 | off | no-MTP baseline |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-c1to4.md` | edc82b6, ray | 4 | 2 | ray bt=8192 |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt12288-c1to4.md` | edc82b6, ray | 4 | 2 | bt=12288 A/B |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-no-custom-allreduce-c1to4.md` | edc82b6, ray | 4 | 2 | no custom allreduce |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-OMP8-only-c1to4.md` | edc82b6, ray | 4 | 2 | OMP=8 only |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-mtp2-bt8192-OMP8-flashsampler-c1to4.md` | edc82b6, ray | 4 | 2 | OMP=8 + flashsampler |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq4-nomtp-c1to4.md` | edc82b6, ray | 4 | off | no-MTP baseline |
| `results_dsv4-flash-fp8-tp2-edc82b6-ray-maxseq8-mtp2-bt8192-c1to8.md` | edc82b6, ray | **8** | 2 | **experimental — maxseq8 not safe default** |
| `results_dsv4-flash-fp8-tp2-cherry-sched-ray-maxseq8-mtp2-bt8192-c1to8.md` | cherry-sched, ray | **8** | 2 | **experimental — maxseq8 not safe default** |
| `results_dsv4-flash-fp8-tp2-cherry-sched-ray-maxseq8-mtp2-bt8192-DEPTH.md` | cherry-sched, ray | **8** | 2 | depth sweep, experimental |

### Qwen3.5-397B INT4 TurboQuant

| File | TQ config | Notes |
|---|---|---|
| `results_397b-int4-tq-3bit-c1to4.md` | 3-bit weight | TQ weight bitwidth A/B |
| `results_397b-int4-tq-4bit-c1to4.md` | 4-bit weight | TQ weight bitwidth A/B |
| `results_397b-int4-tq-k3v4-c1to4.md` | KV 3K/4V quant | TQ KV config A/B |
| `results_397b-int4-tq-k8v4-c1to4.md` | KV 8K/4V quant | TQ KV config A/B |

---

## Notes

- Files with `jasl` in the `edc82b6` identifier refer to a specific dsv4-d568 image
  SHA used during testing. `edc82b6` is a git commit hash, not a currently deployed tag.
- Files with `maxseq8` reflect experiments run before `MAX_NUM_SEQS=4` was confirmed
  as the safe limit for unholy-fusion. Do not use `maxseq8` with the unholy-fusion image.
- Files with `mtp2` reflect experiments before `MTP_NUM_TOKENS=1` was confirmed as
  the safe default. `MTP_NUM_TOKENS=2` causes throughput collapse at `c≥4`.
