# Model Environment Presets

This directory contains Docker Compose environment preset files for model-serving
configurations.

It does **not** contain actual Hugging Face model weights.

## What these files are

Each `.env` file in this directory defines model-specific runtime settings passed to
`docker compose --env-file models/<preset>.env`. Typical settings include:

- `MODEL_PATH` — host path to the model weight directory
- `MODEL_CONTAINER_PATH` — container-internal mount point
- `SERVED_MODEL_NAME` — name served on the OpenAI-compatible API
- `TP_SIZE` — tensor-parallel degree
- `CLUSTER_MODE` — `single` (one DGX Spark) or `dual-rdma` (two nodes over RoCE)
- `VLLM_IMAGE` — which container image to use
- Quantization options, MTP settings, and other vLLM flags

## Where to store actual model weights

Keep model weights outside this repository. Example locations:

```text
/mnt/data/llm-models/deepseek-ai/DeepSeek-V4-Flash
/mnt/data/llm-models/Qwen/<model-name>
/home/<user>/Documents/Models/<model-name>
```

Point the preset to that location by editing `MODEL_PATH` in the chosen `.env` file:

```bash
# Edit directly:
sed -i 's|/path/to/model|/mnt/data/llm-models/deepseek-ai/DeepSeek-V4-Flash|' \
  models/dsv4-flash-fp8-tp2.env

# Or copy to .env and edit there:
cp models/dsv4-flash-fp8-tp2.env .env
# then edit MODEL_PATH in .env
```

## Usage

```bash
# Launch with a preset directly (no copy needed):
docker compose --env-file models/dsv4-flash-fp8-tp2.env --profile head up -d

# Or copy to .env and use the default:
cp models/redhatai-122b-nvfp4.env .env
docker compose --profile head up -d
```

## Directory name

The directory is named `models/` for backward compatibility with existing docs and
scripts. The name may change to `presets/` or `model-presets/` in a future cleanup,
but no rename is happening in the current release.
