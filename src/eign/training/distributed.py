"""
Optional Distributed Data Parallel (DDP) support for multi-GPU training.

This module provides safe DDP setup that:
- Only activates when torch.cuda.device_count() > 1
- Falls back gracefully to single-GPU if setup fails
- Maintains exact same training semantics as single-GPU
- Uses environment variables for process coordination

Usage:
    from eign.training.distributed import setup_distributed, cleanup_distributed, is_distributed

    # At training start
    rank, world_size, device = setup_distributed()

    # Wrap model (only if distributed)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # At training end
    cleanup_distributed()
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[int, int, torch.device]:
    """
    Setup distributed training if multiple GPUs available.

    Returns:
        (rank, world_size, device): Process rank, total processes, and device to use

    Falls back to single-GPU if:
    - Only 1 GPU available
    - CUDA not available
    - Environment variables not set (not launched with torchrun)
    - DDP initialization fails
    """
    # Check if CUDA available
    if not torch.cuda.is_available():
        print("[DISTRIBUTED] CUDA not available, using CPU")
        return 0, 1, torch.device("cpu")

    # Check number of GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        print(f"[DISTRIBUTED] Only {n_gpus} GPU(s) available, using single-GPU mode")
        return 0, 1, torch.device("cuda:0")

    # Check if launched with torchrun/torch.distributed.launch
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(f"[DISTRIBUTED] {n_gpus} GPUs available but not launched with torchrun")
        print("[DISTRIBUTED] To use multi-GPU, launch with:")
        print(f"[DISTRIBUTED]   torchrun --nproc_per_node={n_gpus} scripts/train.py")
        print("[DISTRIBUTED] Falling back to single-GPU mode")
        return 0, 1, torch.device("cuda:0")

    # Get rank and world_size from environment
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    except (KeyError, ValueError) as e:
        print(f"[DISTRIBUTED] Failed to read environment variables: {e}")
        print("[DISTRIBUTED] Falling back to single-GPU mode")
        return 0, 1, torch.device("cuda:0")

    # Initialize process group
    try:
        # Use NCCL backend for GPU training
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"[DISTRIBUTED] Successfully initialized DDP with {world_size} GPUs")
            print(f"[DISTRIBUTED] Using NCCL backend")

        return rank, world_size, device

    except Exception as e:
        print(f"[DISTRIBUTED] Failed to initialize DDP: {e}")
        print("[DISTRIBUTED] Falling back to single-GPU mode")
        # Clean up any partial initialization
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0, 1, torch.device("cuda:0")


def cleanup_distributed() -> None:
    """Cleanup distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0
