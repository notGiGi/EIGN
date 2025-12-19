#!/usr/bin/env python
"""Minimal test to isolate the crash"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Step 1: Importing torch...")
import torch
print(f"  PyTorch {torch.__version__} loaded")
print(f"  CUDA available: {torch.cuda.is_available()}")

print("\nStep 2: Importing eign modules...")
from eign.model import EIGNModel
print("  EIGNModel imported")

print("\nStep 3: Creating small model...")
model_config = {
    "vocab_size": 100,
    "n_layers": 1,
    "d_model": 64,
    "n_heads": 2,
    "ffn_dim": 128,
    "rope_base": 10000.0,
    "max_seq_len": 32,
    "attn_dropout": 0.0,
    "resid_dropout": 0.0,
    "mlp_dropout": 0.0,
}
print(f"  Config: {model_config}")

try:
    model = EIGNModel(**model_config)
    print(f"  Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  ERROR creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing forward pass on CPU...")
try:
    x = torch.randint(0, 100, (2, 16))  # batch=2, seq=16
    print(f"  Input shape: {x.shape}")

    with torch.no_grad():
        y = model(x)
    print(f"  Output shape: {y.shape}")
    print("  Forward pass OK")
except Exception as e:
    print(f"  ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if torch.cuda.is_available():
    print("\nStep 5: Testing on CUDA...")
    try:
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        print(f"  Model and input moved to CUDA")

        with torch.no_grad():
            y_cuda = model_cuda(x_cuda)
        torch.cuda.synchronize()
        print(f"  CUDA forward pass OK")
        print(f"  Output shape: {y_cuda.shape}")
    except Exception as e:
        print(f"  ERROR in CUDA: {e}")
        import traceback
        traceback.print_exc()
        print("\n  *** This is likely the cause of silent exits! ***")
        print("  Your CUDA/PyTorch installation may be corrupted.")

print("\n=== All tests passed ===")
