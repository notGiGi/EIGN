#!/usr/bin/env python
"""EIGN Training Script - Production Ready

This script trains the EIGN language model with automatic tokenizer bootstrap.

Usage:
    # Smoke test (mini training, 5-10 steps)
    PYTHONPATH=src python scripts/train.py --smoke-test

    # Smoke test with custom data directory
    PYTHONPATH=src python scripts/train.py --smoke-test --train-data-dir /path/to/data

    # Full training
    PYTHONPATH=src python scripts/train.py

    # Full training with custom data directory
    PYTHONPATH=src python scripts/train.py --train-data-dir /path/to/data

Requirements:
    - Training data must exist as .txt files in specified directory
    - Tokenizer will be auto-trained on first run if missing
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
from eign.training.loop import train


def _load_yaml(path: Path) -> dict:
    """Load YAML configuration file."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _hash_config(config: dict) -> str:
    """Generate deterministic hash of configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_device(requested: str | None) -> torch.device:
    """Resolve compute device (CUDA/CPU)."""
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, using CPU")
            return torch.device("cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_txt_files(root: Path) -> list[Path]:
    """List all .txt files in directory recursively."""
    if not root.exists():
        raise FileNotFoundError(
            f"\nTraining data directory not found: {root}\n\n"
            f"Please create this directory and add your training data (.txt files).\n"
            f"Example structure:\n"
            f"  {root}/\n"
            f"    document1.txt\n"
            f"    document2.txt\n"
            f"    ...\n"
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
    """Ensure tokenizer exists, training it if necessary.

    Args:
        tokenizer_path: Expected path to tokenizer .model file
        training_data_dir: Directory containing training .txt files
        vocab_size: Tokenizer vocabulary size

    Returns:
        Path to tokenizer .model file
    """
    if tokenizer_path.exists():
        return tokenizer_path

    print("\n" + "=" * 70)
    print("TOKENIZER NOT FOUND - TRAINING NEW TOKENIZER")
    print("=" * 70)
    print(f"Expected path: {tokenizer_path}")
    print(f"Training from: {training_data_dir}")
    print(f"Vocabulary size: {vocab_size:,}")
    print()

    # Get training files
    training_files = _list_txt_files(training_data_dir)
    print(f"Found {len(training_files)} training files")

    # Train tokenizer
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

    print(f"✓ Tokenizer trained successfully: {trained_model}")
    print("=" * 70)
    print()

    return trained_model


def _resolve_train_data_dir(
    cli_arg: str | None,
    config_value: str | None,
) -> Path:
    """Resolve training data directory from CLI, config, or default.

    Priority:
        1. CLI argument (--train-data-dir)
        2. Config value (data.train_dir)
        3. Default (data/train)

    Args:
        cli_arg: CLI --train-data-dir argument value
        config_value: Config data.train_dir value

    Returns:
        Resolved Path to training data directory
    """
    if cli_arg:
        # CLI argument has highest priority
        return Path(cli_arg).resolve()

    if config_value:
        # Config has second priority
        # Handle both absolute and relative paths
        config_path = Path(config_value)
        if config_path.is_absolute():
            return config_path
        # Relative to project root or data dir
        if (get_data_dir() / config_value).exists():
            return get_data_dir() / config_value
        return REPO_ROOT / config_value

    # Default fallback
    return get_data_dir() / "train"


def _validate_seq_len_consistency(model_cfg: dict, data_cfg: dict) -> int:
    """Validate and return consistent seq_len from model and data configs.

    Args:
        model_cfg: Model configuration dict
        data_cfg: Data configuration dict

    Returns:
        The validated sequence length

    Raises:
        ValueError: If seq_len values don't match between configs
    """
    model_seq_len = model_cfg.get("max_seq_len")
    data_seq_len = data_cfg.get("seq_len")

    if model_seq_len is None:
        raise ValueError(
            "model.max_seq_len not found in configs/model.yaml\n"
            "Please add 'max_seq_len' under the 'model' section."
        )

    if data_seq_len is None:
        raise ValueError(
            "data.seq_len not found in configs/data.yaml\n"
            "Please add 'seq_len' under the 'data' section."
        )

    if model_seq_len != data_seq_len:
        raise ValueError(
            f"\nSequence length mismatch detected!\n"
            f"  model.max_seq_len (configs/model.yaml): {model_seq_len}\n"
            f"  data.seq_len (configs/data.yaml): {data_seq_len}\n\n"
            f"These values must match. Please update one of the configs so both are equal.\n"
        )

    return int(model_seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EIGN language model",
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
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
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

    # Load configurations
    config_dir = REPO_ROOT / args.config_dir
    model_cfg = _load_yaml(config_dir / "model.yaml").get("model", {})
    train_cfg = _load_yaml(config_dir / "train.yaml").get("train", {})
    data_cfg = _load_yaml(config_dir / "data.yaml").get("data", {})

    # Validate seq_len consistency BEFORE any processing
    seq_len = _validate_seq_len_consistency(model_cfg, data_cfg)

    # Resolve training data directory (CLI > config > default)
    train_dir = _resolve_train_data_dir(
        cli_arg=args.train_data_dir,
        config_value=data_cfg.get("train_dir"),
    )

    # Resolve device
    device = _resolve_device(args.device or train_cfg.get("device"))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"Training data: {train_dir}")
    print()

    # Get environment-aware directories
    artifacts_dir = get_artifacts_dir()
    cache_dir = get_cache_dir()
    runs_dir = get_runs_dir()

    # ========================================================================
    # SMOKE TEST MODE
    # ========================================================================
    if args.smoke_test:
        print("=" * 70)
        print("SMOKE TEST MODE")
        print("=" * 70)
        print("Running mini training (5 steps) to validate pipeline")
        print()

        # Override configs for smoke test
        train_cfg = dict(train_cfg)
        data_cfg = dict(data_cfg)
        model_cfg = dict(model_cfg)

        # Smoke test parameters (small but real)
        train_cfg["output_dir"] = str(runs_dir / "smoke_test")
        train_cfg["max_steps"] = 5
        train_cfg["batch_size"] = 2
        train_cfg["grad_accum_steps"] = 1
        train_cfg["log_every_steps"] = 1
        train_cfg["checkpoint_every_steps"] = 100  # No checkpoints in smoke test

        # Use validated seq_len for both model and data
        model_cfg["max_seq_len"] = seq_len
        data_cfg["seq_len"] = seq_len

        # Reduce model size for smoke test
        model_cfg["n_layers"] = 2
        model_cfg["d_model"] = 256
        model_cfg["n_heads"] = 4
        model_cfg["ffn_dim"] = 512

        # Tokenizer path
        tokenizer_path = artifacts_dir / "tokenizer" / "v0001" / "eign_spm_unigram_32k.model"

        # Ensure tokenizer exists (using resolved train_dir)
        tokenizer_path = _ensure_tokenizer(
            tokenizer_path=tokenizer_path,
            training_data_dir=train_dir,
            vocab_size=32000,
        )

        # Load tokenizer
        tokenizer = SentencePieceTokenizer(tokenizer_path)
        model_cfg["vocab_size"] = tokenizer.vocab_size

        # Get training files (using resolved train_dir)
        file_paths = _list_txt_files(train_dir)
        print(f"Training on {len(file_paths)} files from: {train_dir}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
        print(f"Sequence length: {seq_len} (model.max_seq_len = data.seq_len)")
        print()

        # Create dataset with validated seq_len
        dataset_cache_dir = cache_dir / "smoke_test"
        print("[DEBUG] Creating DocumentDataset (smoke test)...")
        dataset = DocumentDataset(
            file_paths,
            tokenizer,
            seq_len=seq_len,
            cache_dir=dataset_cache_dir,
            seed=int(train_cfg["seed"]),
            shuffle=True,
        )
        print(f"[DEBUG] DocumentDataset created, length={len(dataset)}")

        # Create model
        model = EIGNModel(**model_cfg)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")

        # Verify seq_len consistency
        print(f"✓ Model max_seq_len: {model.max_seq_len}")
        print(f"✓ Dataset seq_len: {dataset.seq_len}")
        assert model.max_seq_len == dataset.seq_len, (
            f"CRITICAL: model.max_seq_len ({model.max_seq_len}) != "
            f"dataset.seq_len ({dataset.seq_len})"
        )
        print()

        # Config hashes
        train_cfg["config_hashes"] = {
            "model": _hash_config(model_cfg),
            "train": _hash_config({k: v for k, v in train_cfg.items() if k != "config_hashes"}),
            "data": _hash_config(data_cfg),
        }

        # Train
        print("Starting smoke test training...")
        print("=" * 70)
        try:
            train(
                model,
                dataset,
                train_cfg,
                output_dir=train_cfg["output_dir"],
                device=device,
                pad_id=tokenizer.pad_id,
            )
            print("\n" + "=" * 70)
            print("✓ SMOKE TEST PASSED")
            print("=" * 70)
        finally:
            dataset.close()

        return

    # ========================================================================
    # FULL TRAINING MODE
    # ========================================================================
    print("=" * 70)
    print("FULL TRAINING MODE")
    print("=" * 70)
    print()

    # Tokenizer and cache paths
    tokenizer_path = artifacts_dir / "tokenizer" / "v0001" / "eign_spm_unigram_32k.model"
    dataset_cache_dir = cache_dir / data_cfg.get("cache_dir", "train")

    # Ensure tokenizer exists (using resolved train_dir)
    tokenizer_path = _ensure_tokenizer(
        tokenizer_path=tokenizer_path,
        training_data_dir=train_dir,
        vocab_size=32000,
    )

    # Load tokenizer
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if "vocab_size" in model_cfg and model_cfg["vocab_size"] != tokenizer.vocab_size:
        raise ValueError(
            f"Config vocab_size ({model_cfg['vocab_size']}) does not match "
            f"tokenizer vocab_size ({tokenizer.vocab_size})"
        )
    model_cfg["vocab_size"] = tokenizer.vocab_size

    # Get training files (using resolved train_dir)
    file_paths = _list_txt_files(train_dir)
    print(f"Training files: {len(file_paths)} from: {train_dir}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
    print(f"Sequence length: {seq_len} (model.max_seq_len = data.seq_len)")
    print()

    # Create dataset with validated seq_len
    shuffle = bool(data_cfg.get("shuffle", True))
    data_seed = int(data_cfg.get("seed", train_cfg["seed"]))

    print("[DEBUG] Creating DocumentDataset (full training)...")
    dataset = DocumentDataset(
        file_paths,
        tokenizer,
        seq_len=seq_len,
        cache_dir=dataset_cache_dir,
        seed=data_seed,
        shuffle=shuffle,
    )
    print(f"[DEBUG] DocumentDataset created, length={len(dataset)}")

    # Create model
    model = EIGNModel(**model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Verify seq_len consistency
    print(f"✓ Model max_seq_len: {model.max_seq_len}")
    print(f"✓ Dataset seq_len: {dataset.seq_len}")
    assert model.max_seq_len == dataset.seq_len, (
        f"CRITICAL: model.max_seq_len ({model.max_seq_len}) != "
        f"dataset.seq_len ({dataset.seq_len})"
    )
    print()

    # Output directory
    output_dir = train_cfg.get("output_dir", str(runs_dir / "train"))

    # Config hashes
    train_cfg = dict(train_cfg)
    train_cfg["output_dir"] = output_dir
    train_cfg["config_hashes"] = {
        "model": _hash_config(model_cfg),
        "train": _hash_config({k: v for k, v in train_cfg.items() if k != "config_hashes"}),
        "data": _hash_config(data_cfg),
    }

    # Train
    print("Starting full training...")
    print("=" * 70)
    train(
        model,
        dataset,
        train_cfg,
        output_dir=output_dir,
        device=device,
        pad_id=tokenizer.pad_id,
    )


if __name__ == "__main__":
    main()
