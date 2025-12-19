# Kaggle Setup Instructions

## Quick Start

### 1. Clone the repository in Kaggle notebook

```python
!git clone https://github.com/YOUR_USERNAME/EIGN.git
%cd EIGN
```

### 2. Install dependencies

```python
!pip install -q -r requirements.txt
```

### 3. Prepare your data

Place your training data and tokenizer in Kaggle datasets:

- Training data: `.txt` files → `/kaggle/input/your-dataset/train/`
- Tokenizer model: `tokenizer.model` → `/kaggle/input/your-dataset/tokenizer.model`

Or use Kaggle dataset mounting.

### 4. Update configs (optional)

Edit `configs/data.yaml` to point to your Kaggle dataset paths:

```yaml
data:
  train_dir: "/kaggle/input/your-dataset/train"
  tokenizer_model: "/kaggle/input/your-dataset/tokenizer.model"
  cache_dir: "/kaggle/working/cache"
  seq_len: 2048
  shuffle: true
  seed: 42
```

### 5. Run smoke test

```python
!python scripts/train.py --smoke-test
```

### 6. Run full training

```python
!python scripts/train.py
```

## Environment Detection

The codebase automatically detects Kaggle environment via `KAGGLE_KERNEL_RUN_TYPE` and adjusts paths:

- **Local**: `./runs`, `./cache`, `./data`
- **Kaggle**: `/kaggle/working/runs`, `/kaggle/working/cache`, `/kaggle/input`

## Notes

- Checkpoints are saved to `/kaggle/working/runs/train/checkpoints`
- TensorBoard logs to `/kaggle/working/runs/train/tensorboard`
- GPU is auto-detected; use `--device cpu` to force CPU
