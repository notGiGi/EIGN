#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eign.data import DocumentDataset, SentencePieceTokenizer, generate_synthetic_corpus
from eign.env import get_cache_dir, get_data_dir, get_runs_dir
from eign.model import EIGNModel
from eign.training.loop import train


class SyntheticTokenizer:
    def __init__(self, vocab: list[str], pad_id: int = 3) -> None:
        self.pad_id = pad_id
        self._vocab = {token: i + pad_id + 1 for i, token in enumerate(vocab)}
        self._vocab_size = max(self._vocab.values()) + 1

    def encode(self, text: str) -> list[int]:
        return [self._vocab[tok] for tok in text.strip().split() if tok]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _hash_config(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_txt_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    if not files:
        raise ValueError(f"No .txt files found under {root}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    model_cfg = _load_yaml(config_dir / "model.yaml").get("model", {})
    train_cfg = _load_yaml(config_dir / "train.yaml").get("train", {})
    data_cfg = _load_yaml(config_dir / "data.yaml").get("data", {})

    device = _resolve_device(args.device or train_cfg.get("device"))

    if args.smoke_test:
        # Smoke test: use environment-aware paths
        train_cfg = dict(train_cfg)
        data_cfg = dict(data_cfg)
        model_cfg = dict(model_cfg)

        # Environment-aware output directory
        runs_dir = get_runs_dir()
        train_cfg["output_dir"] = str(runs_dir / "smoke_test")

        # Reduced config for smoke test (4GB VRAM compatibility)
        train_cfg["max_steps"] = 5
        train_cfg["num_epochs"] = 1
        train_cfg["batch_size"] = 2
        train_cfg["grad_accum_steps"] = 1
        train_cfg["log_every_steps"] = 1
        train_cfg["checkpoint_every_steps"] = 100  # No checkpoints during smoke test
        data_cfg["seq_len"] = 16
        model_cfg["max_seq_len"] = data_cfg["seq_len"]
        model_cfg["n_layers"] = 1
        model_cfg["d_model"] = 64
        model_cfg["n_heads"] = 2
        model_cfg["ffn_dim"] = 128

        # Environment-aware cache and data directories
        cache_dir = get_cache_dir() / "smoke_test_cache"
        doc_dir = runs_dir / "smoke_test_data" / "docs"

        seq_len = int(data_cfg["seq_len"])
        tokens_per_doc = (seq_len + 1) * 64
        doc_paths = generate_synthetic_corpus(
            doc_dir, num_docs=4, tokens_per_doc=tokens_per_doc
        )
        vocab = [f"DOC{i}" for i in range(4)]
        tokenizer = SyntheticTokenizer(vocab=vocab)
        model_cfg["vocab_size"] = tokenizer.vocab_size
        dataset = DocumentDataset(
            doc_paths,
            tokenizer,
            seq_len=seq_len,
            cache_dir=cache_dir,
            seed=int(train_cfg["seed"]),
            shuffle=True,
        )
        # Include config hashes in checkpoint metadata
        train_cfg["config_hashes"] = {
            "model": _hash_config(model_cfg),
            "train": _hash_config(
                {k: v for k, v in train_cfg.items() if k != "config_hashes"}
            ),
            "data": _hash_config(data_cfg),
        }
        model = EIGNModel(**model_cfg)
        try:
            train(
                model,
                dataset,
                train_cfg,
                output_dir=train_cfg["output_dir"],
                device=device,
                pad_id=tokenizer.pad_id,
            )
        finally:
            dataset.close()
        return

    # Full training: use environment-aware paths
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()

    # Resolve paths from config (fall back to environment defaults)
    train_dir = Path(data_cfg.get("train_dir", data_dir / "train"))
    tokenizer_model = Path(data_cfg.get("tokenizer_model", data_dir / "tokenizer.model"))
    dataset_cache_dir = Path(data_cfg.get("cache_dir", cache_dir / "train"))

    # Validate paths exist
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {train_dir}\n"
            f"Please place training .txt files in this directory."
        )
    if not tokenizer_model.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {tokenizer_model}\n"
            f"Please ensure the tokenizer model is available."
        )

    seq_len = int(data_cfg["seq_len"])
    shuffle = bool(data_cfg.get("shuffle", True))
    data_seed = int(data_cfg.get("seed", train_cfg["seed"]))

    tokenizer = SentencePieceTokenizer(tokenizer_model)
    if "vocab_size" in model_cfg and model_cfg["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("model.vocab_size must match tokenizer vocabulary size.")
    model_cfg["vocab_size"] = tokenizer.vocab_size

    file_paths = _list_txt_files(train_dir)
    dataset = DocumentDataset(
        file_paths,
        tokenizer,
        seq_len=seq_len,
        cache_dir=dataset_cache_dir,
        seed=data_seed,
        shuffle=shuffle,
    )

    # Environment-aware output directory
    runs_dir = get_runs_dir()
    output_dir = train_cfg.get("output_dir", str(runs_dir / "train"))

    # Include config hashes in checkpoint metadata
    train_cfg = dict(train_cfg)
    train_cfg["output_dir"] = output_dir
    train_cfg["config_hashes"] = {
        "model": _hash_config(model_cfg),
        "train": _hash_config(
            {k: v for k, v in train_cfg.items() if k != "config_hashes"}
        ),
        "data": _hash_config(data_cfg),
    }
    model = EIGNModel(**model_cfg)
    train(
        model,
        dataset,
        train_cfg,
        output_dir=train_cfg["output_dir"],
        device=device,
        pad_id=tokenizer.pad_id,
    )


if __name__ == "__main__":
    main()
