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
from eign.env import detect_environment, get_artifacts_dir, get_cache_dir, get_data_dir, get_runs_dir
from eign.model import EIGNModel
from eign.training.loop import train


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


def _check_tokenizer_exists(tokenizer_path: Path) -> None:
    """Verify tokenizer artifacts exist (strict check, no auto-training).

    Tokenizer training is a preprocessing step, not part of training runtime.

    Args:
        tokenizer_path: Expected path to tokenizer .model file

    Raises:
        FileNotFoundError: If tokenizer artifacts are missing
    """
    model_file = tokenizer_path
    vocab_file = tokenizer_path.with_suffix(".vocab")

    if not model_file.exists() or not vocab_file.exists():
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"TOKENIZER MISSING - CANNOT PROCEED\n"
            f"{'='*70}\n"
            f"Expected tokenizer artifacts:\n"
            f"  - {model_file} {'[FOUND]' if model_file.exists() else '[MISSING]'}\n"
            f"  - {vocab_file} {'[FOUND]' if vocab_file.exists() else '[MISSING]'}\n\n"
            f"Tokenizer must be trained as a preprocessing step before training.\n"
            f"Please train the tokenizer first using the tokenizer training script.\n"
            f"{'='*70}\n"
        )


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

    # Load configurations - STRICT loading (fail-fast, no silent failures)
    # Handle both directory path and file path (if file passed, use parent)
    config_path = REPO_ROOT / args.config_dir
    config_dir = config_path.parent if config_path.is_file() else config_path

    # Load configs ONCE using strict loader (never returns {})
    raw_model_cfg = load_yaml_strict(config_dir / "model.yaml")
    raw_data_cfg = load_yaml_strict(config_dir / "data.yaml")
    raw_train_cfg = load_yaml_strict(config_dir / "train.yaml")

    # Normalize: handle both {model: {...}} and {...} formats
    model_cfg = raw_model_cfg.get("model", raw_model_cfg)
    data_cfg = raw_data_cfg.get("data", raw_data_cfg)
    train_cfg = raw_train_cfg.get("train", raw_train_cfg)

    # Startup diagnostic logs
    print(f"[CONFIG] model.yaml keys: {list(model_cfg.keys())}")
    print(f"[CONFIG] data.yaml keys: {list(data_cfg.keys())}")
    print(f"[CONFIG] train.yaml keys: {list(train_cfg.keys())}")

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
        smoke_output_dir = str(runs_dir / "smoke_test")
        train_cfg["base_output_dir"] = smoke_output_dir
        train_cfg["max_steps"] = 5
        train_cfg["batch_size"] = 2
        train_cfg["grad_accum_steps"] = 1
        train_cfg["log_every_steps"] = 1
        train_cfg["checkpoint_every_steps"] = 100  # No checkpoints in smoke test
        train_cfg["auto_resume"] = False  # Disable auto-resume for smoke test

        # Use validated seq_len for both model and data
        model_cfg["max_seq_len"] = seq_len
        data_cfg["seq_len"] = seq_len

        # Reduce model size for smoke test
        model_cfg["n_layers"] = 2
        model_cfg["d_model"] = 256
        model_cfg["n_heads"] = 4
        model_cfg["ffn_dim"] = 512

        # Tokenizer path (environment-aware)
        env = detect_environment()
        if env == "kaggle":
            tokenizer_path = Path("/kaggle/input/eign-tokenizer/eign_spm_unigram_32k.model")
        else:
            tokenizer_path = artifacts_dir / "tokenizer" / "v0001" / "eign_spm_unigram_32k.model"

        # Verify tokenizer exists (strict check - no auto-training)
        _check_tokenizer_exists(tokenizer_path)

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
        dataset = DocumentDataset(
            file_paths,
            tokenizer,
            seq_len=seq_len,
            cache_dir=dataset_cache_dir,
            seed=int(train_cfg["seed"]),
            shuffle=True,
        )

        # Create model
        model = EIGNModel(**model_cfg)
        param_count = sum(p.numel() for p in model.parameters())

        # Verify seq_len consistency
        assert model.max_seq_len == dataset.seq_len, (
            f"CRITICAL: model.max_seq_len ({model.max_seq_len}) != "
            f"dataset.seq_len ({dataset.seq_len})"
        )

        # Config hashes
        train_cfg["config_hashes"] = {
            "model": _hash_config(model_cfg),
            "train": _hash_config({k: v for k, v in train_cfg.items() if k != "config_hashes"}),
            "data": _hash_config(data_cfg),
        }

        # Training banner
        print("=" * 70)
        print("EIGN SMOKE TEST TRAINING")
        print(f"Model params: {param_count:,}")
        print(f"Sequence length: {seq_len}")
        print(f"Device: {device}")
        print(f"Max steps: {train_cfg['max_steps']}")
        print("=" * 70)
        try:
            train(
                model,
                dataset,
                train_cfg,
                output_dir=smoke_output_dir,
                device=device,
                pad_id=tokenizer.pad_id,
            )
            print("\n" + "=" * 70)
            print("[OK] SMOKE TEST PASSED")
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

    # Tokenizer path (environment-aware)
    env = detect_environment()
    if env == "kaggle":
        tokenizer_path = Path("/kaggle/input/eign-tokenizer/eign_spm_unigram_32k.model")
    else:
        tokenizer_path = artifacts_dir / "tokenizer" / "v0001" / "eign_spm_unigram_32k.model"

    dataset_cache_dir = cache_dir / data_cfg.get("cache_dir", "train")

    # Verify tokenizer exists (strict check - no auto-training)
    print("=" * 70)
    print("TOKENIZER STATUS")
    _check_tokenizer_exists(tokenizer_path)
    print(f"[FOUND] {tokenizer_path}")
    print(f"[FOUND] {tokenizer_path.with_suffix('.vocab')}")
    print("=" * 70)
    print()

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

    print("=" * 70)
    print("DATASET & MODEL CONFIGURATION")
    print(f"Training files: {len(file_paths)}")
    print(f"Training data: {train_dir}")
    print(f"Tokenizer vocab: {tokenizer.vocab_size:,}")
    print(f"Sequence length: {seq_len}")
    print("=" * 70)
    print()

    # Create dataset with validated seq_len
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
    model = EIGNModel(**model_cfg)
    param_count = sum(p.numel() for p in model.parameters())

    # Verify seq_len consistency
    assert model.max_seq_len == dataset.seq_len, (
        f"CRITICAL: model.max_seq_len ({model.max_seq_len}) != "
        f"dataset.seq_len ({dataset.seq_len})"
    )

    # Output directory resolution (Kaggle-first design)
    # CRITICAL: On Kaggle, ONLY checkpoints go to /kaggle/working/
    # Everything else (code, data, cache, logs) stays outside to avoid bloating "Save Version"
    base_output_dir = train_cfg.get("base_output_dir")

    if base_output_dir:
        # Explicit override from config
        output_dir = str(Path(base_output_dir).resolve())
    elif Path("/kaggle/working").exists():
        # Kaggle environment: checkpoints ONLY in /kaggle/working/
        # (this directory gets saved by Kaggle's "Save Version")
        output_dir = "/kaggle/working"
    else:
        # Local development
        output_dir = str((runs_dir / "train").resolve())

    # Config hashes
    train_cfg = dict(train_cfg)
    train_cfg["output_dir"] = output_dir
    train_cfg["config_hashes"] = {
        "model": _hash_config(model_cfg),
        "train": _hash_config({k: v for k, v in train_cfg.items() if k != "config_hashes"}),
        "data": _hash_config(data_cfg),
    }

    # Training banner
    print("=" * 70)
    print("EIGN TRAINING STARTED")
    print(f"Model params: {param_count:,}")
    print(f"Sequence length: {seq_len}")
    print(f"Device: {device}")
    print(f"Max steps: {train_cfg['max_steps']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Grad accum: {train_cfg['grad_accum_steps']}")
    print(f"Output dir: {output_dir}")
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
