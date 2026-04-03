#!/usr/bin/env python3
"""
AutoRound INT4 quantization for wangzhang/Qwen3.5-122B-A10B-abliterated.
Matches Intel/Qwen3.5-122B-A10B-int4-AutoRound config:
  bits=4, group_size=128, sym=True, seqlen=512, batch_size=1,
  gradient_accumulate_steps=8, shared_expert layers kept at FP16.

Usage:
  python3 quantize_autoround_int4.py \
    --model /path/to/bf16/model \
    --output /path/to/output \
    [--device cuda:0] [--seqlen 512] [--nsamples 128]
"""

import argparse
import torch
from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to BF16 source model")
    parser.add_argument("--output", required=True, help="Output directory for INT4 model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model to CPU from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Keep shared_expert layers at FP16 (matching Intel config)
    layer_config = {}
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        prefix = f"model.language_model.layers.{i}.mlp"
        for suffix in ["shared_expert.down_proj", "shared_expert.gate_proj", "shared_expert.up_proj"]:
            layer_config[f"{prefix}.{suffix}"] = {"bits": 16, "data_type": "float"}
        layer_config[f"{prefix}.shared_expert_gate"] = {"bits": 16, "data_type": "fp"}

    print(f"Configuring AutoRound: bits=4, group_size=128, sym=True, seqlen={args.seqlen}")
    print(f"  nsamples={args.nsamples}, batch_size={args.batch_size}, iters={args.iters}")
    print(f"  {len(layer_config)} layers kept at FP16 (shared_expert)")

    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.grad_accum,
        iters=args.iters,
        layer_config=layer_config,
    )

    print("Starting quantization (this may take several hours)...")
    autoround.quantize()

    print(f"Saving quantized model to {args.output}...")
    autoround.save_quantized(args.output, format="auto_round")

    print("Done!")


if __name__ == "__main__":
    main()
