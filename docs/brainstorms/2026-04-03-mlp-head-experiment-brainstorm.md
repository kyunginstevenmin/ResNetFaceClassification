---
date: 2026-04-03
topic: mlp-head-experiment
---

# MLP Head Architecture Experiment

## What We're Building

Replacing the single linear classifier in the ResNet head with deeper MLP variants to explore whether additional capacity improves classification performance. The backbone and global average pooling layer are kept frozen for this experiment to isolate the effect of the head architecture.

**Dataset:** 140,000 images, 7,000 classes (~20 images/class)  
**Backbone output:** 512-dim feature vector (after AdaptiveAvgPool2d + Flatten)  
**Current head:** `512 → 7000` (single LazyLinear)

## Why This Approach

The backbone already compresses spatial features to a 512-dim vector via global average pooling — replacing or removing that layer would explode parameter count (~175M). Alternatives like GeM pooling are worth exploring later as a follow-up, but keeping AdaptiveAvgPool2d fixed ensures this experiment cleanly isolates MLP head depth/width as the variable.

## MLP Head Candidates

Each hidden layer uses: `Linear → BatchNorm1d → ReLU → Dropout(0.3-0.5)`

| Config | Architecture | Added Params | Priority |
|--------|-------------|--------------|----------|
| **A (baseline)** | `512 → 7000` | 0 | Reference |
| **B** | `512 → 1024 → 7000` | ~530k | High |
| **C** | `512 → 512 → 7000` | ~270k | High |
| **D** | `512 → 1024 → 512 → 7000` | ~1.05M | Lower (needs more data) |

Start with A, B, C. Add D if B/C show clear improvement over baseline without overfitting.

## Key Decisions

- **Keep AdaptiveAvgPool2d:** Removing it without pooling yields 25,088 input features — intractable at 20 imgs/class. GeM pooling is a clean follow-up experiment after the best head is found.
- **Dropout 0.3-0.5:** Primary regularization given limited data per class.
- **BatchNorm1d on hidden layers:** Stabilizes training of new layers; do not apply to the final linear.
- **Stratified subsampling for Stage 1:** Ensures all 7,000 classes appear in the 20% subset.

## Data Staging Plan

**Stage 1 — Eliminate (cheap signal)**
- Data: 20% stratified sample (~28k images, ~4 imgs/class)
- Duration: 5 epochs, all configs in parallel
- Metric: val top-5 accuracy (top-1 is noisy at this scale with 4 imgs/class)
- Action: Eliminate bottom 1-2 configs

**Stage 2 — Select (full signal)**
- Data: 100% (140k images)
- Duration: 20-30 epochs, surviving configs
- Metrics: val top-1 accuracy, val top-5 accuracy, val loss curve shape
- Watch for: faster overfitting vs. baseline as a signal that head is too large

## Experiment Tracking

**Tool:** WandB (keep existing setup)

**Metrics to log per run:**
- `val/loss`
- `val/top1_acc`
- `val/top5_acc`
- `train/loss`
- Learning rate schedule

**Per-run config to tag:**
- Head architecture (A/B/C/D)
- Data stage (stage1 / stage2)
- Dropout value
- Hidden dim(s)

**Comparison basis:** Val top-5 accuracy at Stage 1; val top-1 + loss curve at Stage 2. Val loss alone is insufficient — also watch for overfitting gap (train loss >> val loss).

## Cloud Training — AWS Stack

| Layer | Service | Notes |
|-------|---------|-------|
| Compute | SageMaker Training Jobs | Managed, reproducible, supports spot |
| GPU | `ml.g4dn.xlarge` (T4) or `ml.g5.xlarge` (A10G) | Start with g4dn for Stage 1 |
| Storage | S3 | Dataset + model checkpoints |
| Container registry | ECR | Push training Docker image here |
| Cost savings | SageMaker Managed Spot Training | 60-90% savings, auto-checkpointing |

**Training container setup:**
1. Dockerfile with PyTorch + WandB + training script
2. Push image to ECR
3. Data uploaded to S3; passed to job via `SM_CHANNEL_TRAIN` env var
4. WandB API key injected via SageMaker Secrets Manager or env var

## Follow-up Experiments (Out of Scope for Now)

- GeM pooling vs. AdaptiveAvgPool2d (swap pooling layer after best head is found)
- ArcFace / CosFace loss instead of softmax (common in face recognition)
- Learning rate differential: lower LR for backbone, higher for new head layers

## Training Strategy

- **Starting point:** Best baseline model checkpoint (lowest val loss run)
- **Phase 1 — Head warmup:** Freeze backbone weights, train only the new MLP head for 3-5 epochs. Prevents the randomly initialized head from corrupting pretrained backbone features.
- **Phase 2 — End-to-end fine-tuning:** Unfreeze backbone, train everything with a lower LR for the backbone vs. the head (param groups).

## Next Steps

→ Create branch `mlp-head-experiment`  
→ `/workflows:plan` for implementation details
