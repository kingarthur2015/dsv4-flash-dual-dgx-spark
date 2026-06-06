# Benchmark Artifacts

This directory stores raw benchmark artifacts and experiment outputs accumulated
during DGX Spark / GB10 vLLM stack development and tuning.

Raw files are preserved for traceability. They are **not necessarily recommended
runtime settings** — many files represent A/B experiments, failed attempts, or
superseded configurations.

## Where to read interpreted results

For the current interpreted summary, read:

- [`README.md`](../README.md) — top-level image roles, quick-start, and tuning notes
- [`docs/unholy-fusion-benchmark.md`](../docs/unholy-fusion-benchmark.md) — full
  DSV4 vs unholy-fusion depth × concurrency comparison and analysis

## Current unholy-fusion safe defaults

The following defaults remain documented as safe unless a later document explicitly
supersedes them:

| Variable | Safe default |
|---|---|
| `MAX_MODEL_LEN` | `262144` |
| `MAX_NUM_SEQS` | `4` |
| `MAX_NUM_BATCHED_TOKENS` | `8192` |
| `GPU_MEMORY_UTILIZATION` | `0.80` |
| `MTP_NUM_TOKENS` | `1` |

Do not infer production readiness from a raw benchmark filename alone. A file
with `maxseq8` or `mtp2` in its name reflects an experiment, not a safe default.

## Contents

```
benchmarks/
└── llama-benchy/        Raw llama-benchy outputs; see llama-benchy/README.md
```

To list all files:

```bash
find benchmarks -maxdepth 2 -type f | sort
```
