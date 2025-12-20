#!/usr/bin/env python
"""EIGN Multi-GPU Training Script with DDP

This script demonstrates how to use Distributed Data Parallel (DDP) for multi-GPU training.

Usage:
    # Single GPU (automatic fallback)
    python scripts/train_distributed.py

    # Multi-GPU with torchrun (recommended for Kaggle 2x T4)
    torchrun --nproc_per_node=2 scripts/train_distributed.py

    # With smoke test
    torchrun --nproc_per_node=2 scripts/train_distributed.py --smoke-test

How it works:
    - Detects number of GPUs automatically
    - Falls back to single-GPU if not launched with torchrun
    - Each GPU process trains on different data batches
    - Gradients are synchronized across GPUs automatically
    - Only rank 0 saves checkpoints and logs to avoid conflicts
    - Training semantics remain identical to single-GPU

Requirements:
    - Launch with torchrun for multi-GPU
    - Each process gets its own GPU automatically
    - Same dataset must be accessible from all processes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

# Setup path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eign.data import DocumentDataset, SentencePieceTokenizer, train_tokenizer
from eign.env import get_artifacts_dir, get_cache_dir, get_data_dir, get_runs_dir
from eign.model import EIGNModel
from eign.training import (
    cleanup_distributed,
    is_distributed,
    is_main_process,
    setup_distributed,
    train,
)


def load_yaml_strict(path: Path) -> dict:
    """Load YAML configuration with strict validation (fail-fast, no silent failures).

    This function is used for critical training configs (model, data, train).
    It NEVER returns {} and always fails loudly if something is wrong.

    Args:
        path: Path to YAML file

    Returns:
        Loaded configuration dict

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file is empty or invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise RuntimeError(f"Configuration file is empty: {path}")

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML dict but got {type(data).__name__}: {path}")

    return data


def _hash_config(config: dict) -> str:
    """Generate deterministic hash of configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _list_txt_files(root: Path) -> list[Path]:
    """List all .txt files in directory recursively."""
    if not root.exists():
        raise FileNotFoundError(
            f"\nTraining data directory not found: {root}\n\n"
            f"Please create this directory and add your training data (.txt files).\n"
        )

    files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    if not files:
        raise ValueError(
            f"\nNo .txt files found in: {root}\n\n"
            f"Please add training data (.txt files) to this directory.\n"
        )
    return files


def _ensure_tokenizer(
    tokenizer_path: Path,
    training_data_dir: Path,
    vocab_size: int = 32000,
) -> Path:
    """Ensure tokenizer exists, training it if necessary (only on rank 0)."""
    # Only rank 0 should train the tokenizer
    if tokenizer_path.exists():
        return tokenizer_path

    if not is_main_process():
        # Wait for rank 0 to train tokenizer
        if is_distributed():
            import torch.distributed as dist

            dist.barrier()  # Wait for rank 0
        return tokenizer_path

    # Rank 0: Train tokenizer
    print("\n" + "=" * 70)
    print("TOKENIZER NOT FOUND - TRAINING NEW TOKENIZER (RANK 0)")
    print("=" * 70)
    print(f"Expected path: {tokenizer_path}")
    print(f"Training from: {training_data_dir}")
    print(f"Vocabulary size: {vocab_size:,}")
    print()

    training_files = _list_txt_files(training_data_dir)
    print(f"Found {len(training_files)} training files")

    model_prefix = str(tokenizer_path.with_suffix(""))
    print(f"Training tokenizer... (this may take a few minutes)")

    trained_model = train_tokenizer(
        input_files=training_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=3,
    )

    print(f"[OK] Tokenizer trained successfully: {trained_model}")
    print("=" * 70)
    print()

    if is_distributed():
        import torch.distributed as dist

        dist.barrier()  # Signal other ranks

    return trained_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EIGN language model with optional multi-GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config YAML files",
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=None,
        help="Training data directory containing .txt files (overrides config)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test (mini training with 5 steps)",
    )
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed()

    try:
        if is_main_process():
            print(f"Running on device: {device}")
            if is_distributed():
                print(f"Distributed training: {world_size} GPUs")
            else:
                print("Single-GPU mode")

        # Load configurations - STRICT loading (fail-fast, no silent failures)
        config_dir = REPO_ROOT / args.config_dir

        # Load configs ONCE using strict loader (never returns {})
        raw_model_cfg = load_yaml_strict(config_dir / "model.yaml")
        raw_data_cfg = load_yaml_strict(config_dir / "data.yaml")
        raw_train_cfg = load_yaml_strict(config_dir / "train.yaml")

        # Normalize: handle both {model: {...}} and {...} formats
        model_cfg = raw_model_cfg.get("model", raw_model_cfg)
        data_cfg = raw_data_cfg.get("data", raw_data_cfg)
        train_cfg = raw_train_cfg.get("train", raw_train_cfg)

        # Startup diagnostic logs (only on rank 0)
        if rank == 0:
            print(f"[CONFIG] model.yaml keys: {list(model_cfg.keys())}")
            print(f"[CONFIG] data.yaml keys: {list(data_cfg.keys())}")
            print(f"[CONFIG] train.yaml keys: {list(train_cfg.keys())}")

        # Validate seq_len consistency
        seq_len = model_cfg.get("max_seq_len", 512)
        if seq_len != data_cfg.get("seq_len", 512):
            raise ValueError(
                f"Sequence length mismatch: model.max_seq_len={seq_len} "
                f"!= data.seq_len={data_cfg.get('seq_len')}"
            )

        # Get training data directory
        if args.train_data_dir:
            train_dir = Path(args.train_data_dir).resolve()
        else:
            train_dir = get_data_dir() / data_cfg.get("train_dir", "train")

        # Get environment-aware directories
        artifacts_dir = get_artifacts_dir()
        cache_dir = get_cache_dir()
        runs_dir = get_runs_dir()

        # Tokenizer path
        tokenizer_path = (
            artifacts_dir / "tokenizer" / "v0001" / "eign_spm_unigram_32k.model"
        )

        # Ensure tokenizer exists
        vocab_size = 32000 if not args.smoke_test else 150
        tokenizer_path = _ensure_tokenizer(
            tokenizer_path=tokenizer_path,
            training_data_dir=train_dir,
            vocab_size=vocab_size,
        )

        # Load tokenizer
        tokenizer = SentencePieceTokenizer(tokenizer_path)
        model_cfg["vocab_size"] = tokenizer.vocab_size

        # Get training files
        file_paths = _list_txt_files(train_dir)

        # Create dataset
        dataset_cache_dir = cache_dir / ("smoke_test" if args.smoke_test else "train")
        shuffle = bool(data_cfg.get("shuffle", True))
        data_seed = int(data_cfg.get("seed", train_cfg["seed"]))

        dataset = DocumentDataset(
            file_paths,
            tokenizer,
            seq_len=seq_len,
            cache_dir=dataset_cache_dir,
            seed=data_seed,
            shuffle=shuffle,
        )

        # Create model
        if args.smoke_test:
            # Smaller model for smoke test
            model_cfg["n_layers"] = 2
            model_cfg["d_model"] = 256
            model_cfg["n_heads"] = 4
            model_cfg["ffn_dim"] = 512
            train_cfg["max_steps"] = 5

        model = EIGNModel(**model_cfg)

        # Wrap model with DDP if distributed
        if is_distributed():
            model = model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )
            if is_main_process():
                print(f"[DISTRIBUTED] Model wrapped with DDP")
        else:
            model = model.to(device)

        param_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Training banner (only on main process)
        if is_main_process():
            print("=" * 70)
            print("EIGN TRAINING STARTED" + (" (DISTRIBUTED)" if is_distributed() else ""))
            print(f"Model params: {param_count:,}")
            print(f"Sequence length: {seq_len}")
            print(f"Device: {device}")
            if is_distributed():
                print(f"World size: {world_size}")
                print(f"Rank: {rank}")
            print(f"Max steps: {train_cfg['max_steps']}")
            print("=" * 70)

        # Output directory
        output_dir = train_cfg.get("output_dir", str(runs_dir / "train"))

        # Config hashes
        train_cfg = dict(train_cfg)
        train_cfg["output_dir"] = output_dir
        train_cfg["config_hashes"] = {
            "model": _hash_config(model_cfg),
            "train": _hash_config(
                {k: v for k, v in train_cfg.items() if k != "config_hashes"}
            ),
            "data": _hash_config(data_cfg),
        }

        # Train
        train(
            model,
            dataset,
            train_cfg,
            output_dir=output_dir,
            device=device,
            pad_id=tokenizer.pad_id,
        )

        if is_main_process():
            print("\n" + "=" * 70)
            print("[OK] TRAINING COMPLETED")
            print("=" * 70)

    finally:
        # Cleanup distributed
        cleanup_distributed()


if __name__ == "__main__":
    main()
