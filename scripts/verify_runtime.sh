#!/bin/bash
# Runtime verification for vLLM DGX Spark image.
# Runs import checks + GPU op registration checks.
# Exit 0 = ready for staging, non-zero = issues found.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_PY="${SCRIPT_DIR}/verify_imports.py"

echo "=== vLLM Runtime Verification ==="
echo "Date: $(date -Iseconds)"
echo ""

# 1. Python + GPU basic
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute: SM{cap[0]}{cap[1]}')
else:
    print('WARNING: No CUDA device')
"

echo ""

# 2. Run verify_imports.py with --gpu
if [ -f "$VERIFY_PY" ]; then
    python3 "$VERIFY_PY" --gpu
else
    echo "ERROR: verify_imports.py not found at $VERIFY_PY"
    exit 1
fi

echo ""

# 3. FlashInfer check
python3 -c "
try:
    import flashinfer
    print(f'FlashInfer: {flashinfer.__version__}')
except ImportError:
    print('FlashInfer: NOT INSTALLED')
except Exception as e:
    print(f'FlashInfer: ERROR — {e}')
"

echo ""
echo "=== Verification Complete ==="
