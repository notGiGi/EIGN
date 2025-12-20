# EIGN Training on Google Colab

This guide shows how to train EIGN on Google Colab with Google Drive for checkpoint persistence.

## Quick Start

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone Repository

```python
!git clone https://github.com/notGiGi/EIGN.git
%cd EIGN
```

### 3. Install Dependencies

```python
!pip install -q torch datasets pyyaml sentencepiece tqdm
```

### 4. Configure Training for Google Drive

Edit `configs/train.yaml`:

```yaml
train:
  # ... other settings ...

  # Set this to your Google Drive path for checkpoint persistence
  base_output_dir: /content/drive/MyDrive/eign

  # Enable auto-resume (will automatically continue from last checkpoint)
  auto_resume: true
```

**OR** set it programmatically:

```python
import yaml
from pathlib import Path

config_path = Path("configs/train.yaml")
config = yaml.safe_load(config_path.read_text())

# Use Google Drive for checkpoints
config["train"]["base_output_dir"] = "/content/drive/MyDrive/eign"
config["train"]["auto_resume"] = True

config_path.write_text(yaml.dump(config))
```

### 5. Run Training

```python
!python scripts/train.py
```

## How It Works

### Automatic Environment Detection

The training script automatically detects where it's running:

- **Google Colab** (with Drive mounted): Uses `/content/drive/MyDrive/eign`
- **Kaggle**: Uses `/kaggle/working`
- **Local**: Uses `runs/train`

You can override by setting `base_output_dir` in the config.

### Auto-Resume

When `auto_resume: true`:

1. Training starts
2. Checks `{base_output_dir}/checkpoints/` for existing checkpoints
3. If found: loads latest checkpoint and continues
4. If not found: starts from scratch

**Logs you'll see:**

```
[COLAB DETECTED] Using Google Drive: /content/drive/MyDrive/eign
======================================================================
CHECKPOINT DIRECTORIES (ABSOLUTE PATHS)
[INFO] Output dir: /content/drive/MyDrive/eign
[INFO] Checkpoint dir: /content/drive/MyDrive/eign/checkpoints
[INFO] Log dir: /content/drive/MyDrive/eign/tensorboard
======================================================================
```

If resuming:
```
======================================================================
[AUTO-RESUME] Found existing checkpoints, resuming training
======================================================================
[CHECKPOINT] Loading from: /content/drive/MyDrive/eign/checkpoints/eign_step_00000200.pt
[CHECKPOINT] Resuming from step=200 tokens_seen=819200
```

If starting fresh:
```
======================================================================
[TRAINING] No checkpoints found, starting from scratch
======================================================================
```

### Checkpoint Format

Checkpoints are saved as single `.pt` files:

```
/content/drive/MyDrive/eign/
├── checkpoints/
│   ├── eign_step_00000200.pt  (~558 MB each for 138M model)
│   ├── eign_step_00000400.pt
│   ├── eign_step_00000600.pt
│   └── ...
├── tensorboard/
│   └── events.out.tfevents...
└── eign_checkpoints.zip  (contains all .pt files)
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- AMP scaler state (if used)
- Training step
- Tokens seen
- Config hashes
- Timestamp

### Checkpoint Safety

- **Atomic saves**: Writes to temp file, then renames (prevents corruption)
- **Size verification**: Ensures checkpoint is at least 10MB
- **Never auto-deletes**: Old checkpoints are kept (you can manually delete if needed)

## Troubleshooting

### "No checkpoints found" but I have checkpoints

Make sure `base_output_dir` points to the correct location where your checkpoints are stored.

### Drive quota exceeded

Checkpoints are ~558 MB each for the 138M parameter model. You can manually delete old checkpoints:

```python
!rm /content/drive/MyDrive/eign/checkpoints/eign_step_00000200.pt
```

### Session disconnected, losing progress

All checkpoints are in Google Drive (`base_output_dir`), so you won't lose progress. Just restart and training will auto-resume.

## Training Monitoring

View TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/eign/tensorboard
```

## After Training

Download checkpoint ZIP for offline use:

```python
from google.colab import files
files.download('/content/drive/MyDrive/eign/eign_checkpoints.zip')
```

Extract locally:

```bash
unzip eign_checkpoints.zip -d ./checkpoints/
```
