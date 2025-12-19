#!/usr/bin/env python
"""Check GPU information"""
import subprocess
import sys

print("=== Checking GPU Information ===\n")

# Method 1: Try nvidia-smi
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print("NVIDIA GPU detected:")
        print(result.stdout)

        # Get CUDA version from driver
        result2 = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "CUDA Version:" in result2.stdout:
            for line in result2.stdout.split('\n'):
                if "CUDA Version:" in line:
                    print(line.strip())
    else:
        print("nvidia-smi command failed")
except FileNotFoundError:
    print("nvidia-smi not found - NVIDIA drivers may not be installed")
except Exception as e:
    print(f"Error running nvidia-smi: {e}")

# Method 2: Try wmic (Windows)
print("\n=== Checking via Windows Management ===")
try:
    result = subprocess.run(
        ["wmic", "path", "win32_VideoController", "get", "name"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print("Graphics cards:")
        print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

print("\n=== Recommended PyTorch Installation ===")
print("\nFor CUDA 11.8:")
print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("\nFor CUDA 12.1:")
print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("\nFor CUDA 12.4:")
print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
print("\nVisit https://pytorch.org/get-started/locally/ for more options")
