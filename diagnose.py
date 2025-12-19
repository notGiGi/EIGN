#!/usr/bin/env python
"""Diagnose PyTorch and CUDA installation"""
import sys
from pathlib import Path

print("=== System Diagnostics ===\n")

# 1. Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}\n")

# 2. PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch location: {torch.__file__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Has torch.amp: {hasattr(torch, 'amp')}")
    if hasattr(torch, 'amp'):
        print(f"Has torch.amp.GradScaler: {hasattr(torch.amp, 'GradScaler')}")
    print()
except ImportError as e:
    print(f"ERROR: Cannot import torch: {e}\n")
    sys.exit(1)

# 3. Test basic operations
print("=== Testing Basic Operations ===\n")
try:
    x = torch.randn(10, 10)
    print(f"CPU tensor creation: OK")
    y = x @ x.T
    print(f"CPU matmul: OK")
except Exception as e:
    print(f"CPU operations FAILED: {e}")

if torch.cuda.is_available():
    try:
        x = torch.randn(10, 10, device='cuda')
        print(f"CUDA tensor creation: OK")
        y = x @ x.T
        print(f"CUDA matmul: OK")
        torch.cuda.synchronize()
        print(f"CUDA synchronize: OK")
    except Exception as e:
        print(f"CUDA operations FAILED: {e}")
        print("This is likely the cause of silent exits!")

print("\n=== Module Import Check ===\n")
# 4. Check if our modules are importing correctly
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from eign.training.loop import _create_grad_scaler, _get_summary_writer
    print("eign.training.loop: OK")

    # Test GradScaler
    scaler = _create_grad_scaler(enabled=False, device_type="cpu")
    print(f"GradScaler creation: OK (type: {type(scaler).__module__}.{type(scaler).__name__})")

    # Test SummaryWriter
    from pathlib import Path
    test_log_dir = Path("runs/diagnostic_test")
    writer, warning = _get_summary_writer(test_log_dir)
    if warning:
        print(f"SummaryWriter warning: {warning}")
    else:
        print(f"SummaryWriter: OK")
    writer.close()

except Exception as e:
    print(f"Module import FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Checking for __pycache__ ===\n")
import os
root = Path(__file__).parent
pycache_dirs = list(root.rglob("__pycache__"))
print(f"Found {len(pycache_dirs)} __pycache__ directories")
if pycache_dirs:
    print("Run: py clear_cache.py")
