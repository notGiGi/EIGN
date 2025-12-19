#!/usr/bin/env python
"""Debug version of train.py with comprehensive error handling"""
import sys
import traceback
import faulthandler
from pathlib import Path

# Enable fault handler to catch segfaults
faulthandler.enable()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("DEBUGGING SMOKE TEST")
print("=" * 60)

try:
    print("\n[1/10] Importing modules...")
    import torch
    import yaml
    from eign.data import DocumentDataset, generate_synthetic_corpus
    from eign.model import EIGNModel
    from eign.training.loop import train
    print(f"✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    print("\n[2/10] Loading configs...")
    def _load_yaml(path: Path) -> dict:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    config_dir = Path("configs")
    model_cfg = _load_yaml(config_dir / "model.yaml").get("model", {})
    train_cfg = _load_yaml(config_dir / "train.yaml").get("train", {})
    data_cfg = _load_yaml(config_dir / "data.yaml").get("data", {})
    print("✓ Configs loaded")

    print("\n[3/10] Setting up smoke test config...")
    train_cfg = dict(train_cfg)
    data_cfg = dict(data_cfg)
    model_cfg = dict(model_cfg)
    train_cfg["output_dir"] = "runs/smoke_test"
    train_cfg["max_steps"] = 10  # Reduce to 10 steps for debugging
    train_cfg["batch_size"] = 2  # Reduce batch size
    train_cfg["grad_accum_steps"] = 1
    train_cfg["log_every_steps"] = 5
    train_cfg["checkpoint_every_steps"] = 10
    data_cfg["seq_len"] = 16  # Reduce sequence length
    model_cfg["max_seq_len"] = data_cfg["seq_len"]
    model_cfg["n_layers"] = 1  # Single layer for testing
    model_cfg["d_model"] = 64
    model_cfg["n_heads"] = 2
    model_cfg["ffn_dim"] = 128
    print(f"✓ Config: {train_cfg['max_steps']} steps, batch={train_cfg['batch_size']}, seq_len={data_cfg['seq_len']}")

    print("\n[4/10] Generating synthetic corpus...")
    smoke_test_root = Path("runs/smoke_test_data")
    doc_dir = smoke_test_root / "docs"
    cache_dir = smoke_test_root / "cache"
    seq_len = int(data_cfg["seq_len"])
    tokens_per_doc = (seq_len + 1) * 64
    doc_paths = generate_synthetic_corpus(doc_dir, num_docs=4, tokens_per_doc=tokens_per_doc)
    print(f"✓ Generated {len(doc_paths)} documents")

    print("\n[5/10] Creating tokenizer...")
    class SyntheticTokenizer:
        def __init__(self, vocab: list[str], pad_id: int = 3):
            self.pad_id = pad_id
            self._vocab = {token: i + pad_id + 1 for i, token in enumerate(vocab)}
            self._vocab_size = max(self._vocab.values()) + 1

        def encode(self, text: str) -> list[int]:
            return [self._vocab[tok] for tok in text.strip().split() if tok]

        @property
        def vocab_size(self) -> int:
            return self._vocab_size

    vocab = [f"DOC{i}" for i in range(4)]
    tokenizer = SyntheticTokenizer(vocab=vocab)
    model_cfg["vocab_size"] = tokenizer.vocab_size
    print(f"✓ Vocab size: {tokenizer.vocab_size}")

    print("\n[6/10] Creating dataset...")
    dataset = DocumentDataset(
        doc_paths,
        tokenizer,
        seq_len=seq_len,
        cache_dir=cache_dir,
        seed=int(train_cfg["seed"]),
        shuffle=True,
    )
    print(f"✓ Dataset created with {len(dataset)} samples")

    print("\n[7/10] Creating model...")
    import hashlib
    import json
    def _hash_config(config: dict) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    train_cfg["config_hashes"] = {
        "model": _hash_config(model_cfg),
        "train": _hash_config({k: v for k, v in train_cfg.items() if k != "config_hashes"}),
        "data": _hash_config(data_cfg),
    }

    model = EIGNModel(**model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count:,} parameters")

    print("\n[8/10] Setting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("\n[9/10] Starting training...")
    print("=" * 60)

    train(
        model,
        dataset,
        train_cfg,
        output_dir=train_cfg["output_dir"],
        device=device,
        pad_id=tokenizer.pad_id,
    )

    print("\n[10/10] Training completed successfully!")
    print("=" * 60)
    dataset.close()

except KeyboardInterrupt:
    print("\n\n!!! Training interrupted by user (Ctrl+C) !!!")
    sys.exit(1)

except Exception as e:
    print("\n" + "=" * 60)
    print("!!! FATAL ERROR !!!")
    print("=" * 60)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {e}\n")
    print("Full traceback:")
    print("-" * 60)
    traceback.print_exc()
    print("-" * 60)

    # Additional CUDA debugging
    if torch.cuda.is_available():
        print("\nCUDA Status:")
        try:
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        except:
            print("  (Unable to query CUDA memory)")

    sys.exit(1)
