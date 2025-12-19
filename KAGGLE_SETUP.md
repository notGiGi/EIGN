# EIGN Training on Kaggle - Quick Setup Guide

## Environment Setup

### Kaggle Notebook Settings
1. **GPU:** 2x Tesla T4 (recommended) or 1x T4
2. **Accelerator:** GPU
3. **Internet:** On (for dependencies)

---

## Single GPU Training (Easiest)

### Step 1: Upload Code and Data
```python
# Upload EIGN repository to Kaggle dataset or use git
!git clone https://github.com/YOUR_USERNAME/EIGN.git
%cd EIGN
```

### Step 2: Install Dependencies
```python
!pip install -q sentencepiece safetensors tensorboard pyyaml psutil pynvml
```

### Step 3: Prepare Training Data
```python
# Option A: Use Kaggle dataset (recommended)
!ln -s /kaggle/input/your-text-data data/train

# Option B: Upload .txt files to data/train/
# (Drag and drop in Kaggle interface)
```

### Step 4: Run Training
```python
!PYTHONPATH=src python scripts/train.py
```

**Expected output:**
```
======================================================================
EIGN TRAINING STARTED
Model params: 138,145,792
Sequence length: 512
Device: cuda:0
Max steps: 1,000
Batch size: 8
Grad accum: 1
Output dir: runs/eign_v0
======================================================================
======================================================================
TRAINING METRICS
[INFO] Tokens/step: 4,096
[INFO] Tokens/epoch: ~23.6M
[INFO] Steps/epoch: ~5,770
[INFO] Tokens (planned run): ~4.1M
[INFO] Max steps: 1,000
======================================================================
[TRAIN] step=10/1000 loss=10.1234 tokens/s=2145.3 lr=3.000000e-05
[TRAIN] step=20/1000 loss=9.8765 tokens/s=2187.8 lr=6.000000e-05
...
[CHECKPOINT] Saved and zipped at step=200
```

### Step 5: Download Checkpoints
```python
# After training, download the ZIP
from IPython.display import FileLink
FileLink('/kaggle/working/runs/eign_v0/eign_checkpoints.zip')
```

---

## Multi-GPU Training (2x T4 - Faster)

### Step 1-3: Same as Single GPU

### Step 4: Run with torchrun
```python
!PYTHONPATH=src torchrun --nproc_per_node=2 scripts/train_distributed.py
```

**Expected output:**
```
[DISTRIBUTED] Successfully initialized DDP with 2 GPUs
[DISTRIBUTED] Using NCCL backend
======================================================================
EIGN TRAINING STARTED (DISTRIBUTED)
Model params: 138,145,792
Sequence length: 512
Device: cuda:0
World size: 2
Rank: 0
Max steps: 1,000
Batch size: 8
Grad accum: 1
Output dir: runs/eign_v0
======================================================================
======================================================================
TRAINING METRICS
[INFO] Tokens/step: 8,192  # ← 2x because 2 GPUs
[INFO] Tokens/epoch: ~47.2M
[INFO] Steps/epoch: ~5,770
[INFO] Tokens (planned run): ~8.2M  # ← 2x more data
[INFO] Max steps: 1,000
======================================================================
[TRAIN] step=10/1000 loss=10.1234 tokens/s=4290.6 lr=3.000000e-05  # ← 2x faster
```

**Benefits:**
- 2x throughput (tokens/sec)
- 2x more data processed per step
- ~50% faster wall-clock time

---

## Configuration for Kaggle

### Recommended: 10K steps (~163M tokens)
```python
# Edit configs/train.yaml BEFORE training
!cat > configs/train.yaml << 'EOF'
train:
  batch_size: 16           # Increase if fits in memory
  grad_accum_steps: 2      # Effective batch = 32 (or 64 with 2 GPUs)
  max_steps: 10000         # 163M tokens
  num_epochs: 1
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps: 200
  max_grad_norm: 1.0
  log_every_steps: 50      # Log less frequently
  checkpoint_every_steps: 1000  # Save every 1K steps
  output_dir: runs/eign_v0
  seed: 123
  deterministic: true
  resume_path: null
EOF
```

**Expected time on 2x T4:**
- ~1.5 hours
- ~10 checkpoints
- Final ZIP: ~2-3 GB

---

## Handling Kaggle Timeouts

### Save Progress Automatically
The checkpoint ZIP is updated after every checkpoint, so even if Kaggle disconnects:
1. Download `/kaggle/working/runs/eign_v0/eign_checkpoints.zip`
2. Extract latest checkpoint
3. Resume training with `resume_path`

### Resume Training
```yaml
# configs/train.yaml
train:
  resume_path: /kaggle/input/your-checkpoint/step_00005000
```

---

## Example Complete Notebook

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/EIGN.git
%cd EIGN
!pip install -q sentencepiece safetensors tensorboard pyyaml psutil pynvml

# Cell 2: Link data
!ln -s /kaggle/input/your-dataset data/train
!ls data/train/*.txt | wc -l  # Verify files

# Cell 3: Check GPU
!nvidia-smi

# Cell 4: Configure training
!cat > configs/train.yaml << 'EOF'
train:
  batch_size: 16
  grad_accum_steps: 2
  max_steps: 10000
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps: 200
  max_grad_norm: 1.0
  log_every_steps: 50
  checkpoint_every_steps: 1000
  output_dir: /kaggle/working/runs/eign_v0
  seed: 123
  deterministic: true
  resume_path: null
EOF

# Cell 5: Train (2x T4)
!PYTHONPATH=src torchrun --nproc_per_node=2 scripts/train_distributed.py

# Cell 6: Download checkpoints
from IPython.display import FileLink
FileLink('/kaggle/working/runs/eign_v0/eign_checkpoints.zip')
```

---

## Performance Expectations

| Setup | Tokens/s | Steps/hr | 10K steps time |
|-------|----------|----------|----------------|
| 1x T4 | ~2,000 | ~1,800 | ~5.5 hours |
| 2x T4 | ~4,000 | ~3,600 | ~2.8 hours |

---

## Questions?

See full documentation:
- `TRAINING_OPTIMIZATIONS.md` - Feature details
- `OPTIMIZATION_SUMMARY.md` - Implementation summary
- `PRODUCTION_LOGGING.md` - Logging details
