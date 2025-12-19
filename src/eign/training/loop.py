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
from safetensors.torch import load_file, save_file
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
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    tokens_seen: int,
    config_hashes: dict[str, str],
) -> None:
    # FIX A: Handle tied weights (tok_embeddings.weight and lm_head.weight share memory)
    step_dir = checkpoint_dir / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    state = {}
    seen_storage: dict[int, str] = {}

    for name, tensor in state_dict.items():
        tensor_cpu = tensor.detach().cpu()
        if not isinstance(tensor_cpu, torch.Tensor) or tensor_cpu.numel() == 0:
            state[name] = tensor_cpu
            continue

        # Detect shared storage by checking data pointer
        try:
            storage_ptr = tensor_cpu.untyped_storage().data_ptr()
        except AttributeError:
            storage_ptr = tensor_cpu.storage().data_ptr()

        if storage_ptr in seen_storage:
            # Clone tensors that share storage to avoid safetensors error
            state[name] = tensor_cpu.clone()
        else:
            seen_storage[storage_ptr] = name
            state[name] = tensor_cpu

    save_file(state, str(step_dir / "model.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        step_dir / "optimizer.pt",
    )
    meta = {
        "global_step": step,
        "tokens_seen": tokens_seen,
        "config_hashes": config_hashes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (step_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, int]:
    model_path = checkpoint_dir / "model.safetensors"
    optim_path = checkpoint_dir / "optimizer.pt"
    meta_path = checkpoint_dir / "metadata.json"

    model_state = load_file(str(model_path))
    # Load state dict - model will handle weight tying in its __init__
    model.load_state_dict(model_state, strict=False)
    optim_state = torch.load(optim_path, map_location="cpu")
    optimizer.load_state_dict(optim_state["optimizer"])
    scheduler.load_state_dict(optim_state["scheduler"])
    if scaler is not None and optim_state.get("scaler") is not None:
        scaler.load_state_dict(optim_state["scaler"])

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    step = int(meta.get("global_step", meta.get("step", 0)))
    tokens_seen = int(meta.get("tokens_seen", 0))
    return {
        "step": step,
        "tokens_seen": tokens_seen,
    }


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
        "resume_path",
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
    resume_path = train_cfg["resume_path"]
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

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    print("[DEBUG] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print("[DEBUG] DataLoader created successfully")
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
        if resume_path:
            resume_dir = Path(resume_path)
            state = _load_checkpoint(
                resume_dir,
                model,
                optimizer,
                scheduler,
                scaler if use_scaler else None,
                device,
            )
            global_step = state["step"]
            tokens_seen = state["tokens_seen"]

        optimizer.zero_grad(set_to_none=True)
        accum_steps = 0
        accum_loss = 0.0
        accum_tokens = 0
        window_start = None
        print("[DEBUG] Entering training loop...")
        print("[DEBUG] Creating dataloader iterator...")
        dataloader_iter = iter(dataloader)
        print("[DEBUG] Dataloader iterator created")
        while global_step < max_steps:  # IterableDataset: terminate strictly by max_steps.
            try:
                print(f"[DEBUG] Fetching next batch (step={global_step})...")
                batch = next(dataloader_iter)
                print(f"[DEBUG] Batch fetched successfully (step={global_step})")
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

            if global_step == 0 and accum_steps == 0:
                print(f"[DEBUG] Step 0: Running first forward pass...")
                print(f"[DEBUG] Step 0: input_ids shape={input_ids.shape}, device={input_ids.device}")

            with autocast_ctx:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss_value = float(loss.detach().cpu())

            if global_step == 0 and accum_steps == 0:
                print(f"[DEBUG] Step 0: Forward pass completed, loss={loss_value:.4f}")
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

            if global_step % checkpoint_every_steps == 0:
                _save_checkpoint(
                    checkpoint_dir,
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

        # Drop any partial accumulation to keep optimizer/scheduler synchronized.
        _save_checkpoint(
            checkpoint_dir,
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
