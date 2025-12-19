# EIGN - Efficient Implementation of Generative Networks

Production-ready implementation of a from-scratch language model training pipeline.

## Features

- **Automatic tokenizer bootstrap**: Trains SentencePiece tokenizer on first run
- **Environment-aware paths**: Runs identically on local machines and Kaggle
- **Smoke test mode**: Validates pipeline with mini training run
- **Production-ready error handling**: Clear, actionable error messages
- **Zero manual setup**: Just add data and run

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/EIGN.git
cd EIGN

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Training Data

Create `data/train/` directory and add your `.txt` files:

```bash
mkdir -p data/train
# Add your training data:
# data/train/document1.txt
# data/train/document2.txt
# ...
```

### 3. Run Smoke Test

Validates the entire pipeline with a 5-step mini training:

```bash
PYTHONPATH=src python scripts/train.py --smoke-test
```

This will:
- Auto-train tokenizer if missing (first run only)
- Build a small model (2 layers, 256 dim)
- Run 5 training steps
- Validate forward + backward passes

### 4. Run Full Training

```bash
PYTHONPATH=src python scripts/train.py
```

## Configuration

Edit YAML files in `configs/`:

- `configs/model.yaml` - Model architecture
- `configs/train.yaml` - Training hyperparameters
- `configs/data.yaml` - Data paths and preprocessing

## Project Structure

```
EIGN/
├── configs/          # YAML configuration files
├── scripts/          # Training scripts
│   └── train.py     # Main training script
├── src/eign/        # Source code
│   ├── model/       # Model architecture
│   ├── data/        # Dataset and tokenizer
│   ├── training/    # Training loop
│   └── env.py       # Environment detection
├── data/            # Training data (gitignored)
│   └── train/       # .txt files go here
├── artifacts/       # Tokenizer models (auto-generated)
├── cache/           # Tokenized data cache (auto-generated)
└── runs/            # Training outputs (auto-generated)
```

## Kaggle Usage

The codebase automatically detects Kaggle and adjusts paths:

```python
# In Kaggle notebook
!git clone https://github.com/YOUR_USERNAME/EIGN.git
%cd EIGN
!pip install -q -r requirements.txt

# Add your data to /kaggle/input or data/train/
# Run smoke test
!PYTHONPATH=src python scripts/train.py --smoke-test

# Run full training
!PYTHONPATH=src python scripts/train.py
```

## Environment Detection

Paths are automatically resolved based on environment:

| Resource | Local | Kaggle |
|----------|-------|--------|
| Data | `./data` | `/kaggle/input` |
| Artifacts | `./artifacts` | `/kaggle/working/artifacts` |
| Cache | `./cache` | `/kaggle/working/cache` |
| Runs | `./runs` | `/kaggle/working/runs` |

## Tokenizer Bootstrap

On first run, if tokenizer doesn't exist:

1. Scans `data/train/*.txt`
2. Trains SentencePiece tokenizer (unigram, 32k vocab)
3. Saves to `artifacts/tokenizer/v0001/eign_spm_unigram_32k.model`
4. Proceeds with training

Subsequent runs use the cached tokenizer.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- SentencePiece
- See `requirements.txt` for full list

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@software{eign2025,
  title={EIGN: Efficient Implementation of Generative Networks},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/EIGN}
}
```
