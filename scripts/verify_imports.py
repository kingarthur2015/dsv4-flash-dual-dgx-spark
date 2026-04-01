#!/usr/bin/env python3
"""
vLLM build verification — import, _C extensions, op registration.
Exit code 0 = all checks passed, non-zero = failure.

Usage:
    python3 verify_imports.py          # basic import checks
    python3 verify_imports.py --gpu    # include GPU/op registration checks
"""
import sys
import argparse

REQUIRED_OPS = [
    "_C.scaled_fp4_quant",
    "_C.cutlass_scaled_fp4_mm",
    "_C.per_token_group_fp8_quant",
    "_C.cutlass_scaled_mm",
    "_C.permute_cols",
    "_C.cutlass_scaled_mm_supports_fp4",
    "_C.scaled_fp4_experts_quant",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Run GPU-dependent checks (op registration)")
    args = parser.parse_args()

    failures = []

    # 1. Core imports
    print("=== Import Checks ===")
    for mod in ["vllm", "vllm._C", "vllm._C_stable_libtorch"]:
        try:
            __import__(mod)
            print(f"  {mod}: OK")
        except ImportError as e:
            err_str = str(e)
            # libcuda.so not available in build-time (no GPU) — expected, not a failure
            if "libcuda" in err_str or "CUDA" in err_str:
                print(f"  {mod}: SKIP (no GPU — {err_str})")
            else:
                print(f"  {mod}: FAIL — {e}")
                failures.append(mod)
        except Exception as e:
            print(f"  {mod}: FAIL — {e}")
            failures.append(mod)

    # 2. Version info
    print("\n=== Versions ===")
    try:
        import torch, vllm
        print(f"  vLLM:    {vllm.__version__}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA:    {torch.version.cuda}")
    except Exception as e:
        print(f"  Version check failed: {e}")
        failures.append("version_check")

    # 3. _C extension file exists
    print("\n=== Extension Files ===")
    from pathlib import Path
    import vllm
    vllm_dir = Path(vllm.__file__).parent
    for so_name in ["_C.abi3.so", "_C_stable_libtorch.abi3.so"]:
        p = vllm_dir / so_name
        if p.exists():
            print(f"  {so_name}: EXISTS ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  {so_name}: MISSING")
            failures.append(so_name)

    # 4. hoist patch verification
    print("\n=== Hoist Patch ===")
    torch_utils = vllm_dir / "utils" / "torch_utils.py"
    if torch_utils.exists():
        content = torch_utils.read_text()
        if "hoist=True" in content:
            print("  FAIL — hoist=True still present")
            failures.append("hoist_patch")
        else:
            print("  OK — hoist=True not found")
    else:
        print("  SKIP — torch_utils.py not found")

    # 5. Op registration (requires GPU)
    if args.gpu:
        print("\n=== Op Registration (GPU) ===")
        try:
            import torch
            torch.cuda.init()
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

            # Force extension loading
            import vllm._C_stable_libtorch  # noqa

            missing = []
            for name in REQUIRED_OPS:
                ns, op = name.split(".", 1)
                base = getattr(torch.ops, ns, None)
                ok = base is not None and hasattr(base, op)
                status = "OK" if ok else "MISSING"
                print(f"  {name}: {status}")
                if not ok:
                    missing.append(name)

            if missing:
                failures.extend(missing)
        except Exception as e:
            print(f"  GPU op check failed: {e}")
            failures.append("gpu_ops")
    else:
        print("\n=== Op Registration ===")
        print("  SKIP — run with --gpu to check (requires CUDA device)")

    # 6. Summary
    print(f"\n{'=' * 50}")
    if failures:
        print(f"FAIL — {len(failures)} issue(s): {failures}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
