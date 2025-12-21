from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset


def set_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def _get_amp_settings(device: torch.device) -> tuple[bool, torch.dtype, bool]:
    if device.type != "cuda":
        return False, torch.float32, False
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _create_grad_scaler(enabled: bool, device_type: str) -> torch.cuda.amp.GradScaler:
    # FIX E: Use torch.amp.GradScaler for PyTorch 2.0+
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        # PyTorch 2.0+ with unified GradScaler API
        return torch.amp.GradScaler(device_type, enabled=enabled)
    # Fallback for older PyTorch versions
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if max_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if max_steps <= warmup_steps:
            return 1.0
        progress = float(step + 1 - warmup_steps) / float(max_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _hash_config(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cpu_ram_gb() -> tuple[float, float] | None:
    try:
        import psutil
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    used_gb = vm.used / (1024**3)
    total_gb = vm.total / (1024**3)
    return used_gb, total_gb


class _GpuMonitor:
    def __init__(self) -> None:
        self.available = False
        self._pynvml = None
        self._handle = None
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return
        self.available = True
        self._pynvml = pynvml
        self._handle = handle

    def memory_gb(self) -> tuple[float, float] | None:
        if not self.available:
            return None
        mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        return used_gb, total_gb

    def shutdown(self) -> None:
        if self.available and self._pynvml is not None:
            self._pynvml.nvmlShutdown()


def _get_tensorboard_writer(log_dir: Path) -> object | None:
    """Attempt to create TensorBoard SummaryWriter. Returns None on failure."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as e:
        print(f"TensorBoard disabled: {e}")
        return None


def _safe_log_scalar(writer: object | None, tag: str, value: float, step: int) -> None:
    """Log scalar to TensorBoard if writer is available."""
    if writer is not None:
        try:
            writer.add_scalar(tag, value, step)
        except Exception:
            pass


def _safe_flush_writer(writer: object | None) -> None:
    """Flush TensorBoard writer if available."""
    if writer is not None:
        try:
            writer.flush()
        except Exception:
            pass


def _safe_close_writer(writer: object | None) -> None:
    """Close TensorBoard writer if available."""
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


def _save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    tokens_seen: int,
    config_hashes: dict[str, str],
) -> None:
    """Save checkpoint as single .pt file directly to output directory.

    Checkpoint files are saved directly to output_dir (typically /kaggle/working/)
    with no subdirectories, no ZIPs, no intermediate steps.

    Args:
        output_dir: Directory to save checkpoint (e.g., /kaggle/working/)
        model: Model to checkpoint
        optimizer: Optimizer to checkpoint
        scheduler: LR scheduler to checkpoint
        scaler: Gradient scaler to checkpoint (optional)
        step: Current training step
        tokens_seen: Total tokens processed
        config_hashes: Configuration hashes for validation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single .pt file per checkpoint (no subdirectories)
    checkpoint_filename = f"eign_step_{step:08d}.pt"
    checkpoint_path = output_dir / checkpoint_filename
    temp_checkpoint_path = output_dir / f".tmp_{checkpoint_filename}"

    # Prepare checkpoint payload
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "config_hashes": config_hashes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Atomic save: write to temp file, then rename (prevents corruption)
    torch.save(checkpoint, temp_checkpoint_path)
    temp_checkpoint_path.replace(checkpoint_path)

    # Verify checkpoint file exists and has reasonable size
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint save failed: {checkpoint_path} not found")

    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    # Sanity check: checkpoint should be at least 10MB
    if file_size_mb < 10:
        raise RuntimeError(
            f"Checkpoint suspiciously small: {checkpoint_path} "
            f"({file_size_mb:.2f} MB)"
        )

    print(f"[CHECKPOINT] Saved: {checkpoint_filename} ({file_size_mb:.1f} MB)")


def _load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, int]:
    """Load checkpoint from unified .pt file."""
    # Find latest checkpoint by lexicographic order
    checkpoint_files = sorted(checkpoint_dir.glob("eign_step_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    checkpoint_path = checkpoint_files[-1]  # Latest checkpoint
    print(f"[CHECKPOINT] Loading from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state
    if checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore AMP scaler state
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Move optimizer states to device
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    step = int(checkpoint.get("step", 0))
    tokens_seen = int(checkpoint.get("tokens_seen", 0))

    print(f"[CHECKPOINT] Resuming from step={step} tokens_seen={tokens_seen}")

    return {
        "step": step,
        "tokens_seen": tokens_seen,
    }


def _log_training_metrics(
    dataset: IterableDataset,
    batch_size: int,
    grad_accum_steps: int,
    max_steps: int,
    seq_len: int,
) -> None:
    """Log training token metrics for transparency."""
    tokens_per_step = batch_size * grad_accum_steps * seq_len

    # Try to get dataset length for epoch calculations
    try:
        dataset_len = len(dataset)
        samples_per_epoch = dataset_len
        tokens_per_epoch = samples_per_epoch * seq_len
        steps_per_epoch = samples_per_epoch // (batch_size * grad_accum_steps)
    except (TypeError, AttributeError):
        # IterableDataset may not have __len__
        samples_per_epoch = None
        tokens_per_epoch = None
        steps_per_epoch = None

    total_tokens = tokens_per_step * max_steps

    print("=" * 70)
    print("TRAINING METRICS")
    print(f"[INFO] Tokens/step: {tokens_per_step:,}")
    if tokens_per_epoch is not None:
        print(f"[INFO] Tokens/epoch: ~{tokens_per_epoch / 1e6:.1f}M")
        print(f"[INFO] Steps/epoch: ~{steps_per_epoch:,}")
    print(f"[INFO] Tokens (planned run): ~{total_tokens / 1e6:.1f}M")
    print(f"[INFO] Max steps: {max_steps:,}")
    print("=" * 70)


def train(
    model: nn.Module,
    dataset: IterableDataset,
    train_cfg: dict,
    output_dir: str | Path,
    device: torch.device,
    pad_id: int = 3,
) -> None:
    required_keys = [
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "lr",
        "betas",
        "weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "log_every_steps",
        "checkpoint_every_steps",
        "seed",
        "deterministic",
    ]
    for key in required_keys:
        if key not in train_cfg:
            raise KeyError(f"Missing train config key: {key}")

    batch_size = int(train_cfg["batch_size"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])
    max_steps = int(train_cfg["max_steps"])
    lr = float(train_cfg["lr"])
    betas = tuple(train_cfg["betas"])
    weight_decay = float(train_cfg["weight_decay"])
    warmup_steps = int(train_cfg["warmup_steps"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    log_every_steps = int(train_cfg["log_every_steps"])
    checkpoint_every_steps = int(train_cfg["checkpoint_every_steps"])
    seed = int(train_cfg["seed"])
    deterministic = bool(train_cfg["deterministic"])
    auto_resume = train_cfg.get("auto_resume", False)
    config_hashes = train_cfg.get("config_hashes")
    if not isinstance(config_hashes, dict) or not config_hashes:
        # Ensure checkpoints always include a config hash for reproducibility.
        train_cfg_hash = {
            k: v for k, v in train_cfg.items() if k != "config_hashes"
        }
        config_hashes = {"train": _hash_config(train_cfg_hash)}

    if batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch_size and grad_accum_steps must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if log_every_steps <= 0 or checkpoint_every_steps <= 0:
        raise ValueError("log_every_steps and checkpoint_every_steps must be positive.")

    set_determinism(seed, deterministic)

    # CRITICAL: Use absolute paths for Kaggle persistence
    output_dir = Path(output_dir).resolve()
    log_dir = output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING ENVIRONMENT")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Checkpoint location: {output_dir}/eign_step_*.pt")
    print(f"[INFO] TensorBoard logs: {log_dir}")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU: {props.name}")
        print(f"[INFO] GPU Memory: {props.total_memory / (1024**3):.1f} GB")
    print("=" * 70)

    # Safe optimization: Enable TF32 on Ampere GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Get seq_len from model for metrics calculation
    seq_len = getattr(model, "max_seq_len", 512)

    # Log training metrics
    _log_training_metrics(dataset, batch_size, grad_accum_steps, max_steps, seq_len)

    model.to(device)
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    if hasattr(dataset, "__len__"):
        try:
            if len(dataset) == 0:
                # Avoid infinite looping on empty IterableDataset.
                raise ValueError("Dataset is empty.")
        except TypeError:
            pass

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    param_groups = _build_param_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    scheduler = _build_scheduler(optimizer, warmup_steps, max_steps)

    use_amp, amp_dtype, use_scaler = _get_amp_settings(device)
    scaler = _create_grad_scaler(enabled=use_scaler, device_type=device.type)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
        if use_amp
        else nullcontext()
    )

    # TensorBoard writer is optional and may be None
    writer = _get_tensorboard_writer(log_dir)
    gpu_monitor = _GpuMonitor()

    global_step = 0
    tokens_seen = 0

    try:
        # Auto-resume: check if checkpoints exist and load latest
        if auto_resume and output_dir.exists():
            checkpoint_files = sorted(output_dir.glob("eign_step_*.pt"))
            if checkpoint_files:
                print("=" * 70)
                print("[AUTO-RESUME] Found existing checkpoints, resuming training")
                print("=" * 70)
                state = _load_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler if use_scaler else None,
                    device,
                )
                global_step = state["step"]
                tokens_seen = state["tokens_seen"]
            else:
                print("=" * 70)
                print("[TRAINING] No checkpoints found, starting from scratch")
                print("=" * 70)
        else:
            if not auto_resume:
                print("=" * 70)
                print("[TRAINING] Auto-resume disabled, starting from scratch")
                print("=" * 70)
            else:
                print("=" * 70)
                print("[TRAINING] Output directory not found, starting from scratch")
                print("=" * 70)

        optimizer.zero_grad(set_to_none=True)
        accum_steps = 0
        accum_loss = 0.0
        accum_tokens = 0
        window_start = None
        dataloader_iter = iter(dataloader)
        while global_step < max_steps:  # IterableDataset: terminate strictly by max_steps.
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)  # IterableDataset: restart iterator.
                continue

            if accum_steps == 0:
                window_start = time.monotonic()  # Measure tokens/sec per accumulation.
                accum_loss = 0.0
                accum_tokens = 0

            input_ids, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            tokens_seen += int(input_ids.numel())
            accum_tokens += int(input_ids.numel())

            with autocast_ctx:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss_value = float(loss.detach().cpu())

            accum_loss += loss_value
            loss = loss / grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_steps += 1
            if accum_steps < grad_accum_steps:
                continue

            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            if window_start is None:
                window_start = time.monotonic()
            window_time = time.monotonic() - window_start
            tokens_per_sec = accum_tokens / max(window_time, 1e-8)
            lr_value = scheduler.get_last_lr()[0]
            avg_loss = accum_loss / float(grad_accum_steps)

            # Clean production logging
            if global_step % log_every_steps == 0:
                print(
                    f"[TRAIN] step={global_step}/{max_steps} "
                    f"loss={avg_loss:.4f} "
                    f"tokens/s={tokens_per_sec:.1f} "
                    f"lr={lr_value:.6e}"
                )

            if global_step % log_every_steps == 0:
                _safe_log_scalar(writer, "train/loss", avg_loss, global_step)
                _safe_log_scalar(writer, "train/lr", lr_value, global_step)
                _safe_log_scalar(writer, "train/grad_norm", float(grad_norm), global_step)
                _safe_log_scalar(writer, "train/tokens_per_sec", tokens_per_sec, global_step)
                cpu_mem = _cpu_ram_gb()
                if cpu_mem is not None:
                    _safe_log_scalar(writer, "system/cpu_ram_used_gb", cpu_mem[0], global_step)
                    _safe_log_scalar(writer, "system/cpu_ram_total_gb", cpu_mem[1], global_step)
                gpu_mem = gpu_monitor.memory_gb()
                if gpu_mem is not None:
                    _safe_log_scalar(writer, "system/gpu_mem_used_gb", gpu_mem[0], global_step)
                    _safe_log_scalar(writer, "system/gpu_mem_total_gb", gpu_mem[1], global_step)

            # Early test checkpoint to validate infrastructure
            if global_step == 50:
                print("\n" + "=" * 70)
                print("[EARLY TEST CHECKPOINT] Saving at step 50 to validate disk write")
                print("=" * 70)
                _save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler if use_scaler else None,
                    global_step,
                    tokens_seen,
                    config_hashes,
                )
                print("[EARLY TEST CHECKPOINT] Validation complete. Training continues.")
                print("=" * 70 + "\n")

            if global_step % checkpoint_every_steps == 0:
                _save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler if use_scaler else None,
                    global_step,
                    tokens_seen,
                    config_hashes,
                )

            accum_steps = 0
            accum_loss = 0.0
            accum_tokens = 0
            window_start = None

        # Final checkpoint
        _save_checkpoint(
            output_dir,
            model,
            optimizer,
            scheduler,
            scaler if use_scaler else None,
            global_step,
            tokens_seen,
            config_hashes,
        )
    finally:
        # FIX B: Ensure dataset cleanup (closes memmaps on Windows)
        if hasattr(dataset, "close"):
            dataset.close()
        gpu_monitor.shutdown()
        _safe_flush_writer(writer)
        _safe_close_writer(writer)
