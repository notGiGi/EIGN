# Debug Instrumentation - Training Blockage Diagnosis

## Overview
Added `[DEBUG]` prints to diagnose where training blocks without changing functional behavior.

## Instrumentation Points

### 1. scripts/train.py

**Dataset creation (smoke test):**
- Line 340: `[DEBUG] Creating DocumentDataset (smoke test)...`
- Line 349: `[DEBUG] DocumentDataset created, length={len}`

**Dataset creation (full training):**
- Line 431: `[DEBUG] Creating DocumentDataset (full training)...`
- Line 440: `[DEBUG] DocumentDataset created, length={len}`

### 2. src/eign/training/loop.py

**DataLoader setup:**
- Line 320: `[DEBUG] Creating DataLoader...`
- Line 329: `[DEBUG] DataLoader created successfully`

**Training loop entry:**
- Line 377: `[DEBUG] Entering training loop...`
- Line 378: `[DEBUG] Creating dataloader iterator...`
- Line 380: `[DEBUG] Dataloader iterator created`

**Batch fetching (every step):**
- Line 383: `[DEBUG] Fetching next batch (step={global_step})...`
- Line 385: `[DEBUG] Batch fetched successfully (step={global_step})`

**First forward pass (step 0 only):**
- Line 402: `[DEBUG] Step 0: Running first forward pass...`
- Line 403: `[DEBUG] Step 0: input_ids shape={shape}, device={device}`
- Line 411: `[DEBUG] Step 0: Forward pass completed, loss={loss}`

**Loss monitoring (every 10 steps):**
- Line 415-416: `[DEBUG] step={step} loss={loss_value}`

**Optimizer step confirmation (every 50 steps):**
- Line 440-441: `[DEBUG] optimizer step completed at step={step}`

**Timing metrics (every 10 steps):**
- Line 455-458: `[DEBUG] step={step} avg_time_per_step={time}s tokens/s={rate}`

### 3. src/eign/data/datasets.py

**Iterator (every sample):**
- Line 116: `[DEBUG] __iter__ index={idx}, file={filename}, offset={offset}`

## Expected Output Sequence

```
[DEBUG] Creating DocumentDataset (...)
[DEBUG] DocumentDataset created, length=XXX
✓ Model max_seq_len: 512
✓ Dataset seq_len: 512
[DEBUG] Creating DataLoader...
[DEBUG] DataLoader created successfully
Starting ... training...
[DEBUG] Entering training loop...
[DEBUG] Creating dataloader iterator...
[DEBUG] Dataloader iterator created
[DEBUG] Fetching next batch (step=0)...
[DEBUG] __iter__ index=0, file=XXX.txt, offset=XXX
[DEBUG] __iter__ index=1, file=XXX.txt, offset=XXX
...
[DEBUG] Batch fetched successfully (step=0)
[DEBUG] Step 0: Running first forward pass...
[DEBUG] Step 0: input_ids shape=torch.Size([...]), device=cuda:0
[DEBUG] Step 0: Forward pass completed, loss=X.XXXX
[DEBUG] step=0 loss=X.XXXX
[DEBUG] optimizer step completed at step=0
[DEBUG] step=1 avg_time_per_step=X.XXXs tokens/s=XXXX.X
[DEBUG] Fetching next batch (step=1)...
...
[DEBUG] step=10 loss=X.XXXX
[DEBUG] step=10 avg_time_per_step=X.XXXs tokens/s=XXXX.X
[DEBUG] Fetching next batch (step=11)...
...
[DEBUG] optimizer step completed at step=50
[DEBUG] step=50 avg_time_per_step=X.XXXs tokens/s=XXXX.X
...
```

## Diagnostic Decision Tree

**If training blocks BEFORE:**
- `[DEBUG] Creating DocumentDataset` → Issue in previous code (model/tokenizer)
- `[DEBUG] DocumentDataset created` → Issue in DocumentDataset.__init__ (indexing/tokenization)
- `[DEBUG] Creating DataLoader` → Issue between dataset and dataloader
- `[DEBUG] DataLoader created` → Issue in DataLoader.__init__
- `[DEBUG] Entering training loop` → Issue in training loop setup
- `[DEBUG] Creating dataloader iterator` → Issue before iterator creation
- `[DEBUG] Dataloader iterator created` → Issue before first batch fetch
- `[DEBUG] Fetching next batch (step=0)` → Issue in next(dataloader)
- `[DEBUG] __iter__ index=X` → Issue in DocumentDataset.__iter__
- `[DEBUG] Batch fetched successfully` → Issue between batch fetch and forward pass
- `[DEBUG] Step 0: Running first forward pass` → Issue in forward pass
- `[DEBUG] Step 0: Forward pass completed` → First step succeeded!

## Removal Instructions

When diagnosis is complete, remove all lines containing `[DEBUG]`:

```bash
# Remove from datasets.py
git diff src/eign/data/datasets.py | grep "^\+" | grep DEBUG

# Remove from loop.py
git diff src/eign/training/loop.py | grep "^\+" | grep DEBUG

# Remove from train.py
git diff scripts/train.py | grep "^\+" | grep DEBUG
```

Or use git to revert specific lines after identifying the issue.

## Notes

- All prints are prefixed with `[DEBUG]` for easy identification
- No functional behavior changed (only observability)
- Safe to run in production (just noisy)
- Can be easily removed with search/replace or git revert
