#!/usr/bin/env python
"""Test TensorBoard imports"""

print("Testing TensorBoard imports...\n")

# Test 1: Direct import
try:
    from torch.utils.tensorboard import SummaryWriter
    print("✓ torch.utils.tensorboard.SummaryWriter imported successfully")
except ImportError as e:
    print(f"✗ ImportError: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check what's installed
import sys
print("\nChecking installed packages:")
try:
    import tensorboard
    print(f"✓ tensorboard {tensorboard.__version__} is installed")
except ImportError:
    print("✗ tensorboard is NOT installed")
    print("  Install with: pip install tensorboard")

try:
    import tensorflow
    print(f"✓ tensorflow {tensorflow.__version__} is installed")
except ImportError:
    print("✗ tensorflow is NOT installed (not required)")

# Test 3: Try creating a writer
try:
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter
    test_dir = Path("runs/test_tb")
    writer = SummaryWriter(log_dir=str(test_dir))
    writer.add_scalar("test", 1.0, 0)
    writer.close()
    print("\n✓ SummaryWriter works correctly")
except Exception as e:
    print(f"\n✗ SummaryWriter failed: {e}")
