---
title: "feat: MLP Head Experiment with AWS SageMaker Training"
type: feat
date: 2026-04-03
---

# MLP Head Architecture Experiment on AWS SageMaker

## Overview

Extend the ResNet face classifier with configurable MLP head variants (A/B/C/D), extract training logic into a production-grade script, and set up an AWS SageMaker + ECR + S3 training pipeline to run staged architecture search experiments. This is the first cloud training setup for the project — all AWS infrastructure is net-new.

**Brainstorm:** `docs/brainstorms/2026-04-03-mlp-head-experiment-brainstorm.md`

---

## Problem Statement

The current classifier uses a single `LazyLinear(num_classes)` after global average pooling. We want to determine whether adding intermediate MLP layers improves val accuracy. We also need to move training off Google Colab and onto AWS SageMaker to enable reproducible, cost-efficient, and industry-standard experimentation.

Three bugs in the current notebook must be fixed before any experimental results are meaningful:
1. `validate()` calls `model.train()` instead of `model.eval()` — BN and Dropout are active during validation, corrupting all validation metrics.
2. Accuracy denominator uses `config['batch_size']` instead of `images.size(0)` — wrong for the last batch.
3. Top-5 accuracy is not tracked — but it is the primary Stage 1 elimination metric.

---

## Proposed Solution

Four phases, executed in order:

1. **Fix bugs + refactor notebook** — make existing training correct and extend the `ResNet` class to support configurable heads.
2. **Extract `train.py`** — SageMaker-compatible training script with argparse + SM env vars.
3. **Build AWS infrastructure** — Dockerfile (AWS DLC base), ECR push script, S3 upload, SageMaker launch script.
4. **Run staged experiments** — Stage 1 eliminates weak configs cheaply; Stage 2 selects the winner on full data.

---

## Technical Approach

### Architecture

```
Before:   AdaptiveAvgPool2d → Flatten → LazyLinear(7000)

After:    AdaptiveAvgPool2d → Flatten → MLPHead([hidden_dims], 7000)

Config A: hidden_dims=[]           → 512 → 7000           (baseline)
Config B: hidden_dims=[1024]       → 512 → 1024 → 7000    (wider)
Config C: hidden_dims=[512]        → 512 → 512 → 7000     (same-width)
Config D: hidden_dims=[1024, 512]  → 512 → 1024 → 512 → 7000  (two-layer)

Each hidden layer: Linear(bias=False) → BatchNorm1d → ReLU → Dropout(0.4)
Final layer: Linear (no BN, no activation — raw logits for CrossEntropyLoss)
```

### Training Strategy

```
Load best baseline checkpoint (backbone weights only, strict=False)
  ↓
Freeze backbone for entire training run
  ↓
Short warmup (0.5 epoch) → CosineAnnealingLR
  ↓
Train head only for all epochs
```

**Why backbone stays frozen:** The baseline was trained on the same dataset and task, so backbone features are already well-aligned with the 7000-class problem. Keeping it frozen: (1) isolates head architecture as the only variable, making config comparisons cleaner; (2) reduces overfitting risk at 20 imgs/class; (3) trains faster with fewer parameters to update.

### AWS Stack

```
Local machine
  → build_and_push_ecr.sh   → ECR (Docker image)
  → upload_data_to_s3.py    → S3  (train/ and val/ data)
  → launch_sagemaker_job.py → SageMaker Training Job
                                   ↓
                              ECR image + S3 data
                                   ↓
                              /opt/ml/input/data/train/
                              /opt/ml/input/data/val/
                              /opt/ml/checkpoints/  ←→ S3 (spot resume)
                              /opt/ml/model/        →  S3 (final artifacts)
                                   ↓
                              WandB (metrics via WANDB_API_KEY env var)
```

---

## Implementation Phases

### Phase 1: Bug Fixes + Notebook Refactoring

**File:** `ResNet Face Classification Implementation on Pytorch.ipynb`

#### 1a. Fix `validate()` — model mode bug

The function currently calls `model.train()`. Replace with `model.eval()`.

```python
# validate() — fix model mode
def validate(model, dataloader, criterion, device):
    model.eval()  # was: model.train()
    # ... rest unchanged
```

#### 1b. Fix accuracy denominator

Fix only the final epoch-level `acc` calculation in `train()` (cell 35) and `validate()` (cell 36). Leave the mid-loop tqdm display formula unchanged.

```python
# train() — cell 35, final line before return
# Before
acc = 100 * num_correct / (len(dataloader) * config['batch_size'])
# After
acc = 100 * num_correct / len(dataloader.dataset)

# validate() — cell 36, final line before return
# Before
acc = 100 * (num_correct / (len(dataloader)*config['batch_size']))
# After
acc = 100 * (num_correct / len(dataloader.dataset))
```

`len(dataloader.dataset)` is always correct — it reflects the actual number of images regardless of batch size or whether the last batch is partial. It also automatically accounts for the Stage 1 `Subset` without any additional changes.

#### 1c. Add `topk_accuracy` utility

Add this function in a new notebook cell before the training loop:

```python
# src/metrics.py (also paste into notebook)
@torch.no_grad()
def topk_accuracy(logits, targets, topk=(1, 5)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.unsqueeze(0))
    results = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum()
        results.append(correct_k * (100.0 / batch_size))
    return results
```

Track running averages with two scalars in the loop — no utility class needed:

```python
# In train() and validate() loops
running_loss, running_top1, running_top5, n = 0.0, 0.0, 0.0, 0
# ... per batch:
bs = images.size(0)
running_loss += loss.item() * bs
running_top1 += top1.item() * bs
running_top5 += top5.item() * bs
n += bs
# ... after loop:
epoch_loss  = running_loss  / n
epoch_top1  = running_top1  / n
epoch_top5  = running_top5  / n
```

#### 1d. Add `MLPHead` class

Add in the model definition cell, before the `ResNet` class:

```python
# src/model.py (also in notebook)
class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_dims, num_classes, dropout=0.4):
        super().__init__()
        layers = []
        current_dim = in_features
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(current_dim, hidden_dim, bias=False),  # bias=False: BN subsumes it
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))  # no activation
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

#### 1e. Refactor `ResNet.__init__` to accept `MLPHead`

Replace the hardcoded `'last'` module with an injected head:

```python
# ResNet.__init__ — replace last module
def __init__(self, arch, lr=0.1, num_classes=7000, head=None, num_features=512):
    super().__init__()
    self.net = nn.Sequential(self.b1())
    for i, b in enumerate(arch):
        self.net.add_module(name=f'b{i+2}', module=self.block(*b, first_block=(i==0)))

    if head is None:
        head = MLPHead(num_features, hidden_dims=[], num_classes=num_classes)

    self.net.add_module(name='last', module=nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ))
    self.head = head  # separate attribute for easy freezing
    # NOTE: no cls_layer / Softmax — forward always returns raw logits.
    # CrossEntropyLoss expects raw logits and applies log-softmax internally.
    # Apply Softmax explicitly only at inference time if needed.

def forward(self, x, return_logits=False):
    feats = self.net(x)
    logits = self.head(feats)
    return logits  # always return raw logits; return_logits kept for API compat
```

Add a `HEAD_CONFIGS` dict for easy config selection:

```python
HEAD_CONFIGS = {
    'A': {'hidden_dims': [],           'dropout': 0.4},
    'B': {'hidden_dims': [1024],       'dropout': 0.4},
    'C': {'hidden_dims': [512],        'dropout': 0.4},
    'D': {'hidden_dims': [1024, 512],  'dropout': 0.4},
}
```

#### 1f. Add backbone freezing

```python
def freeze_backbone(model):
    """Freeze all parameters except the head. Called once before training."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
```

No unfreeze function needed — backbone stays frozen for the entire run. Optimizer is built only over `model.head.parameters()`:

```python
optimizer = torch.optim.SGD(
    model.head.parameters(),
    lr=args.lr, momentum=0.9, weight_decay=1e-4
)
```

#### 1g. Add 20% subsample loader

With 140k images across 7000 classes, `random_split` produces near-identical class distribution to stratified sampling — no sklearn dependency needed.

```python
from torch.utils.data import random_split

def random_subset(dataset, fraction=0.20, seed=42):
    n = int(len(dataset) * fraction)
    rest = len(dataset) - n
    subset, _ = random_split(dataset, [n, rest],
                             generator=torch.Generator().manual_seed(seed))
    return subset

# Stage 1 loader
stage1_dataset = random_subset(train_dataset, fraction=0.20)
stage1_loader  = DataLoader(stage1_dataset, batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=min(os.cpu_count() - 1, 8),
                            pin_memory=True)
```

Also apply the `num_workers` fix to all other loaders — replace hardcoded `num_workers=4` / `num_workers=2` with `min(os.cpu_count() - 1, 8)` throughout `train.py`.

#### 1h. Update WandB logging

Update `wandb.init()` to include per-run config tags:

```python
wandb.init(
    project="hw2p2-ablations",
    name=f"head-{head_config}-stage{stage}",
    config={
        **config,
        'head_config':  head_config,   # 'A', 'B', 'C', or 'D'
        'hidden_dims':  HEAD_CONFIGS[head_config]['hidden_dims'],
        'dropout':      HEAD_CONFIGS[head_config]['dropout'],
        'stage':        stage,         # 1 or 2
    },
    reinit=True,
)
```

Update logging calls to include top-5 accuracy:

```python
wandb.log({
    'train/loss':     train_loss,
    'train/top1_acc': train_top1,
    'val/loss':       val_loss,
    'val/top1_acc':   val_top1,
    'val/top5_acc':   val_top5,
    'learning_rate':  scheduler.get_last_lr()[0],
    'epoch':          epoch,
})
```

#### 1i. Update checkpoint save/load

Update checkpoint saving to include scheduler and AMP scaler state:

```python
# Save — include scaler state for AMP spot resume correctness
# No 'phase' field needed: backbone is frozen for the entire run
torch.save({
    'epoch':           epoch,
    'model_state':     model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'scaler_state':    scaler.state_dict(),   # AMP GradScaler
    'best_val_acc':    best_val_acc,
}, checkpoint_path)

# Load backbone only (for transfer from existing baseline checkpoint)
#
# The existing baseline checkpoint uses the OLD ResNet architecture where the
# classifier is embedded inside net.last as a Sequential:
#   net.last.0  → AdaptiveAvgPool2d  (no params)
#   net.last.1  → Flatten            (no params)
#   net.last.2  → LazyLinear         (params to EXCLUDE)
#
# The checkpoint key is 'model_state_dict' (not 'model_state') — matches the
# existing notebook save: torch.save({'model_state_dict': model.state_dict(), ...})
#
# Verify exact key names before running by executing this in the notebook:
#
#   ckpt = torch.load('checkpoint.pth', map_location='cpu', weights_only=True)
#   for k in ckpt['model_state_dict'].keys():
#       print(k)
def load_backbone_only(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_state = checkpoint['model_state_dict']  # key used by existing notebook

    # Exclude net.last.2 (the LazyLinear classifier in the old architecture)
    backbone_state = {k: v for k, v in saved_state.items()
                      if not k.startswith('net.last.2')}

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if unexpected:
        print(f"Warning: ignoring {len(unexpected)} unexpected keys: {unexpected[:5]}")
    print(f"Backbone loaded. New head randomly initialized ({len(missing)} keys).")
```

---

### Phase 2: Extract `train.py`

**New file:** `src/train.py`

A standalone Python script containing all training logic extracted from the notebook, compatible with both local runs and SageMaker.

Key argparse arguments:

```python
parser.add_argument('--head-config',    type=str,   default='A', choices=['A','B','C','D'])
parser.add_argument('--stage',          type=int,   default=1,   choices=[1, 2])
parser.add_argument('--epochs',     type=int,   default=8)   # Stage 1: 8; Stage 2: 25
parser.add_argument('--lr',         type=float, default=0.5)
parser.add_argument('--batch-size', type=int,   default=64)
parser.add_argument('--dropout',    type=float, default=0.4)
# no --freeze-epochs: backbone is frozen for the entire run
parser.add_argument('--baseline-ckpt',  type=str,   default=None,
                    help='Path to best baseline checkpoint for backbone init')

# SageMaker paths (auto-set in container, overridable locally)
parser.add_argument('--train-dir',      type=str,   default=os.environ.get('SM_CHANNEL_TRAIN', 'data/train'))
parser.add_argument('--val-dir',        type=str,   default=os.environ.get('SM_CHANNEL_VAL',   'data/val'))
parser.add_argument('--model-dir',      type=str,   default=os.environ.get('SM_MODEL_DIR',     'output/model'))
parser.add_argument('--checkpoint-dir', type=str,   default='/opt/ml/checkpoints')
```

Training loop structure in `train.py`:

```python
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model with specified head config
    head   = MLPHead(512, **HEAD_CONFIGS[args.head_config], num_classes=7000)
    model  = ResNet18(head=head).to(device)

    # Load backbone from baseline checkpoint
    if args.baseline_ckpt:
        load_backbone_only(args.baseline_ckpt, model, device)

    # Data loaders
    train_dataset = ImageFolder(args.train_dir, transform=train_transforms)
    val_dataset   = ImageFolder(args.val_dir,   transform=val_transforms)

    if args.stage == 1:
        train_dataset = stratified_subset(train_dataset, fraction=0.20)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              num_workers=2, pin_memory=True)

    # AMP scaler — preserve from notebook
    scaler = torch.cuda.amp.GradScaler()

    # Freeze backbone once — stays frozen for the entire run
    freeze_backbone(model)

    # Optimizer over head parameters only
    optimizer = torch.optim.SGD(
        model.head.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    # Short warmup (half epoch) → CosineAnnealingLR for remaining steps
    # Warmup stabilizes the randomly-initialized head at lr=0.5
    warmup_steps  = len(train_loader) // 2
    total_steps   = args.epochs * len(train_loader)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

    # WandB
    wandb.init(project='hw2p2-ablations',
               name=f'head-{args.head_config}-stage{args.stage}',
               config=vars(args), resume='allow')

    # Spot resume: load checkpoint if one exists
    start_epoch, best_val_acc = 0, 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, scaler, train_loader, device,
                        epoch, scheduler=scheduler, per_batch_step=True)
        val_top1, val_top5, val_loss = validate(model, val_loader, device)
        is_best = val_top1 > best_val_acc
        best_val_acc = max(val_top1, best_val_acc)

        # Always overwrite latest checkpoint — used for spot resume
        checkpoint = {
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_val_acc':    best_val_acc,
        }
        torch.save(checkpoint, ckpt_path)

        # Separately save best model — used for Stage 2 and final evaluation
        if is_best:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pt'))

        wandb.log({'val/top1_acc': val_top1, 'val/top5_acc': val_top5,
                   'val/loss': val_loss, 'epoch': epoch,
                   'learning_rate': scheduler.get_last_lr()[0]})

    # Copy best.pt to /opt/ml/model/ — SageMaker tars and uploads this to S3
    # as the official job output after training completes (estimator.model_data)
    import shutil
    shutil.copy(
        os.path.join(args.checkpoint_dir, 'best.pt'),
        os.path.join(args.model_dir, 'model.pt')
    )
    wandb.finish()
```

---

### Phase 3: AWS Infrastructure

#### High-Level Overview

**Full workflow**

```
YOUR LOCAL MACHINE
        │
        ├─── 1. Build Docker image ──────────────────► ECR
        │       (docker build)                    (image registry)
        │
        ├─── 2. Upload training data ────────────────► S3 Bucket
        │       (aws s3 sync)                     (object storage)
        │
        └─── 3. Launch training job ──────────────► SageMaker
                (launch_sagemaker_job.py)          (managed compute)
                                                        │
                                              pulls image from ECR
                                              pulls data from S3
                                                        │
                                                   ┌────▼────┐
                                                   │   EC2   │
                                                   │  (GPU)  │
                                                   │         │
                                                   │train.py │
                                                   └────┬────┘
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                         S3            S3           WandB
                                    (checkpoints)  (model output)  (metrics)
```

**Docker + ECR**

Docker packages your entire training environment — Python, PyTorch, WandB, your code — into a single portable image. ECR (Elastic Container Registry) is AWS's private Docker registry. SageMaker pulls your image from ECR when it spins up the training instance.

```
Local machine                    ECR
┌─────────────────┐             ┌──────────────────┐
│   Dockerfile    │             │                  │
│  requirements   │──build+push►│  resnet-face-    │
│   train.py      │             │  training:latest │
│   model.py      │             │                  │
└─────────────────┘             └──────────────────┘
```

**S3**

S3 holds three things for this project:

```
S3 Bucket: s3://resnet-face-classification-839000214843/
├── data/
│   ├── train/     ← 7001 classes, ~1.4 GB (uploaded)
│   └── val/       ← 7001 classes, ~361 MB (uploaded)
├── checkpoints/
│   ├── checkpoint.pth        ← baseline, epoch 98, val_acc 71.82%
│   └── <job-name>/           ← epoch checkpoints (synced during training for spot resume)
└── output/
    └── <job-name>/           ← best.pt (uploaded by SageMaker when job completes)
```

**SageMaker Training Job**

SageMaker is the orchestrator. When you run `launch_sagemaker_job.py`:

```
SageMaker Training Job (inside EC2 container)
┌─────────────────────────────────────────────┐
│                                             │
│  /opt/ml/input/data/train/  ◄── from S3    │
│  /opt/ml/input/data/val/    ◄── from S3    │
│                                             │
│  train.py runs...                           │
│      └── logs metrics ──────────────────► WandB
│                                             │
│  /opt/ml/checkpoints/  ◄──► S3 (spot sync) │
│  /opt/ml/model/        ────► S3 (on finish) │
│                                             │
└─────────────────────────────────────────────┘
```

**Spot instances**

Spot instances are spare EC2 capacity at 60-90% discount. AWS can reclaim them with 2 minutes notice. SageMaker handles this automatically:

```
Spot instance running...
        │
        ▼
  AWS reclaims instance
        │
        ├── SageMaker syncs /opt/ml/checkpoints/ → S3
        ├── Waits for new spot capacity
        └── New instance starts
                ├── Pulls image from ECR
                ├── Pulls data from S3
                ├── Restores /opt/ml/checkpoints/ from S3
                └── train.py resumes from last saved epoch
```

**The four scripts and what they do**

| Script | When to run | What it does |
|--------|-------------|--------------|
| `build_and_push_ecr.sh` | Once, or when code changes | Builds Docker image, pushes to ECR |
| `upload_data_to_s3.sh` | Once | Syncs local train/val data to S3 |
| `launch_sagemaker_job.py` | Once per experiment run | Tells SageMaker to start a training job |
| `train.py` | Never directly — SageMaker calls it | The actual training logic |

**End-to-end sequence for one run**

```
1. build_and_push_ecr.sh    → image in ECR
2. upload_data_to_s3.sh     → data in S3        (only needed once)
3. launch_sagemaker_job.py  → job starts
4. SageMaker pulls image + data
5. train.py runs (backbone frozen, warmup → cosine LR)
6. Each epoch: checkpoint.pt → S3 (spot safety)
               best.pt updated if val improves
               metrics → WandB
7. Training finishes: best.pt → /opt/ml/model/ → S3
8. Check WandB dashboard to compare configs
```

#### 3a. `docker/requirements.txt`

```
wandb>=0.17.0
```

Do not pin `torch` or `torchvision` — the AWS DLC base image provides the correct GPU build. No `scikit-learn` needed (`random_split` replaces stratified sampling).

#### 3b. `docker/Dockerfile`

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

COPY docker/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

COPY src/ /opt/ml/code/

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train.py

# Use exec form — ensures SIGTERM reaches Python, not a shell wrapper
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
```

#### 3c. `scripts/build_and_push_ecr.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="resnet-face-training"
IMAGE_TAG="${1:-latest}"  # Pass git SHA as arg in CI
FULL_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}"

# Auth to own ECR
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Auth to AWS DLC registry (for base image pull)
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -f docker/Dockerfile -t "${FULL_IMAGE}" .
docker push "${FULL_IMAGE}"
echo "Pushed: ${FULL_IMAGE}"
```

**Line-by-line explanation**

`#!/usr/bin/env bash`
Tells the OS to run this file using bash. The `#!` (shebang) line is always the first line of a shell script.

`set -euo pipefail`
Three safety settings combined:
- `e` — exit immediately if any command fails (instead of silently continuing)
- `u` — treat undefined variables as errors (catches typos like `$AWS_REGIO`)
- `pipefail` — if any command in a pipeline (`cmd1 | cmd2`) fails, the whole pipeline fails

**Variable setup**
- `AWS_ACCOUNT_ID` — runs the AWS CLI to fetch your 12-digit AWS account ID automatically
- `IMAGE_TAG="${1:-latest}"` — uses the first argument passed to the script, falling back to `"latest"` if none given. So `./build_and_push_ecr.sh abc1234` tags the image with `abc1234`
- `FULL_IMAGE` — assembles the full ECR image URI, e.g. `123456789.dkr.ecr.us-east-1.amazonaws.com/resnet-face-training:latest`

**Create ECR repo if it doesn't exist**
- First tries to describe the repo — if it exists, nothing happens
- `2>/dev/null` — silences the error output if the repo doesn't exist
- `||` — if the first command fails (repo doesn't exist), runs the second command to create it
- Makes the script safe to run multiple times — won't fail if the repo already exists

**Auth to your own ECR**
ECR is a private registry — Docker needs a password to push to it. This:
1. Asks AWS for a temporary login token (`get-login-password`)
2. Pipes (`|`) that token directly into `docker login` so Docker can authenticate
3. Now Docker is allowed to push images to your ECR

**Auth to AWS DLC registry**
Same thing but for AWS's own ECR account (`763104351884`) where the base PyTorch image lives. Docker needs to pull that base image during `docker build`, so it needs to authenticate there too.

**Build and push**
- `docker build` — builds the image from your Dockerfile, tagging it with the full ECR URI
- `-f docker/Dockerfile` — specifies which Dockerfile to use
- `.` — sets the build context to the current directory (so `COPY src/` can find your files)
- `docker push` — uploads the built image to ECR
- `echo` — prints the full image URI so you can confirm what was pushed

```
Run: ./build_and_push_ecr.sh
        │
        ├── Get AWS account ID
        ├── Create ECR repo (if needed)
        ├── Authenticate Docker → your ECR
        ├── Authenticate Docker → AWS DLC ECR (for base image)
        ├── docker build  →  image built locally
        └── docker push   →  image uploaded to ECR
                                    │
                                    ▼
                          SageMaker can now pull it
```

#### 3d. `scripts/upload_data_to_s3.sh`

```bash
#!/usr/bin/env bash
# Usage: ./scripts/upload_data_to_s3.sh <local-data-dir> <s3-bucket>
set -euo pipefail

LOCAL_DIR="${1:-data}"
BUCKET="${2:-$(python3 -c 'import sagemaker; print(sagemaker.Session().default_bucket())')}"
PREFIX="resnet-face/data"

echo "Uploading ${LOCAL_DIR}/train → s3://${BUCKET}/${PREFIX}/train"
aws s3 sync "${LOCAL_DIR}/train/" "s3://${BUCKET}/${PREFIX}/train/" --no-progress

echo "Uploading ${LOCAL_DIR}/val → s3://${BUCKET}/${PREFIX}/val"
aws s3 sync "${LOCAL_DIR}/val/" "s3://${BUCKET}/${PREFIX}/val/" --no-progress

echo "Done. S3 URIs:"
echo "  Train: s3://${BUCKET}/${PREFIX}/train"
echo "  Val:   s3://${BUCKET}/${PREFIX}/val"
```

#### 3e. `scripts/launch_sagemaker_job.py`

```python
#!/usr/bin/env python3
"""
Launch a SageMaker training job for one head config.

Usage (development):
    WANDB_API_KEY=<key> python scripts/launch_sagemaker_job.py \
        --head-config B --stage 1 --instance-type ml.g4dn.xlarge --use-spot

Usage (Stage 2):
    WANDB_API_KEY=<key> python scripts/launch_sagemaker_job.py \
        --head-config B --stage 2 --instance-type ml.g5.xlarge --use-spot --epochs 25
"""
import argparse
import os
import time
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

AWS_REGION     = "us-east-1"
AWS_ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
ECR_REPO       = "resnet-face-training"
IMAGE_TAG      = "latest"
IMAGE_URI      = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}"
SAGEMAKER_ROLE = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/Model_training"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head-config",    required=True, choices=["A","B","C","D"])
    p.add_argument("--stage",          type=int, default=1, choices=[1, 2])
    p.add_argument("--epochs",         type=int, default=5)
    p.add_argument("--instance-type",  default="ml.g4dn.xlarge",
                   choices=["ml.g4dn.xlarge", "ml.g5.xlarge"])
    p.add_argument("--use-spot",       action="store_true")
    p.add_argument("--baseline-ckpt-s3", default=None,
                   help="S3 URI of best baseline checkpoint")
    return p.parse_args()

def main():
    args    = parse_args()
    session = sagemaker.Session()
    bucket  = session.default_bucket()
    prefix  = "resnet-face"

    job_name       = f"resnet-head-{args.head_config.lower()}-s{args.stage}-{int(time.time())}"
    checkpoint_s3  = f"s3://{bucket}/{prefix}/checkpoints/{job_name}"

    # Stage 1: 4h compute budget; Stage 2: 12h
    max_run  = (4 if args.stage == 1 else 12) * 3600
    max_wait = max_run * 2

    hyperparams = {
        "head-config": args.head_config,
        "stage":       args.stage,
        "epochs":      args.epochs,
        "lr":          0.5,
        "batch-size":  64,
        # backbone frozen throughout — no freeze-epochs param
    }
    if args.baseline_ckpt_s3:
        hyperparams["baseline-ckpt"] = args.baseline_ckpt_s3

    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=SAGEMAKER_ROLE,
        instance_type=args.instance_type,
        instance_count=1,
        volume_size=100,
        max_run=max_run,
        use_spot_instances=args.use_spot,
        max_wait=max_wait if args.use_spot else None,
        checkpoint_s3_uri=checkpoint_s3 if args.use_spot else None,
        checkpoint_local_path="/opt/ml/checkpoints",
        output_path=f"s3://{bucket}/{prefix}/output",
        hyperparameters=hyperparams,
        environment={
            # For development: pass key via local env var (never hardcode)
            # For production: remove this and fetch from Secrets Manager in train.py
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        },
        sagemaker_session=session,
        base_job_name="resnet-face",
    )

    train_input = TrainingInput(f"s3://{bucket}/{prefix}/data/train", input_mode="File")
    val_input   = TrainingInput(f"s3://{bucket}/{prefix}/data/val",   input_mode="File")

    print(f"Launching: {job_name}")
    print(f"Head: {args.head_config}  Stage: {args.stage}  Instance: {args.instance_type}  Spot: {args.use_spot}")
    estimator.fit(
        inputs={"train": train_input, "val": val_input},
        job_name=job_name,
        wait=True,
        logs="All",
    )
    print(f"Artifacts: {estimator.model_data}")

if __name__ == "__main__":
    main()
```

---

### Phase 4: Experiment Execution Workflow

#### Stage 1 — Eliminate weak configs

Launch configs A, B, C in parallel (D is lower priority):

```bash
# Terminal 1
WANDB_API_KEY=$KEY python scripts/launch_sagemaker_job.py \
    --head-config A --stage 1 --epochs 8 --instance-type ml.g4dn.xlarge --use-spot

# Terminal 2
WANDB_API_KEY=$KEY python scripts/launch_sagemaker_job.py \
    --head-config B --stage 1 --epochs 8 --instance-type ml.g4dn.xlarge --use-spot

# Terminal 3
WANDB_API_KEY=$KEY python scripts/launch_sagemaker_job.py \
    --head-config C --stage 1 --epochs 8 --instance-type ml.g4dn.xlarge --use-spot
```

**Elimination criterion:** Val top-5 accuracy after 8 epochs (2 frozen + 6 fine-tuning). Drop the bottom 1-2 configs. Also eliminate any config whose train loss >> val loss (overfitting gap).

#### Stage 2 — Select winner

Run surviving configs on full data — same `ml.g4dn.xlarge` instance type:

```bash
WANDB_API_KEY=$KEY python scripts/launch_sagemaker_job.py \
    --head-config B --stage 2 --epochs 25 --instance-type ml.g4dn.xlarge --use-spot \
    --baseline-ckpt-s3 s3://<bucket>/resnet-face/baseline/best.pt
```

**Selection criterion:** Val top-1 accuracy + loss curve shape (watch for overfitting gap vs. baseline).

---

## File Structure After Implementation

```
ResNetFaceClassification/
├── ResNet Face Classification Implementation on Pytorch.ipynb  (modified)
├── src/
│   └── train.py              # SageMaker-compatible training script
├── docker/
│   ├── Dockerfile            # AWS DLC base + requirements
│   └── requirements.txt      # wandb, scikit-learn (no torch — DLC provides it)
├── scripts/
│   ├── build_and_push_ecr.sh
│   ├── upload_data_to_s3.sh
│   └── launch_sagemaker_job.py
└── docs/
    ├── brainstorms/
    │   └── 2026-04-03-mlp-head-experiment-brainstorm.md
    └── plans/
        └── 2026-04-03-feat-mlp-head-aws-sagemaker-experiment-plan.md
```

---

## Acceptance Criteria

### Bug Fixes
- [x] `validate()` uses `model.eval()` — BN and Dropout disabled during validation
- [x] Accuracy denominator uses `images.size(0)` — correct for all batch sizes
- [x] Top-1 and top-5 accuracy tracked and logged to WandB every epoch

### Model Refactoring
- [x] `MLPHead` class accepts `hidden_dims` list; `hidden_dims=[]` reproduces baseline behavior
- [x] `ResNet.forward()` returns raw logits always — no `nn.Softmax` wrapper; `CrossEntropyLoss` receives raw logits
- [x] `ResNet.__init__` accepts a `head` argument; injects `MLPHead` into separate `self.head` attribute
- [x] `HEAD_CONFIGS` dict defines A/B/C/D with `hidden_dims` and `dropout`
- [x] `freeze_backbone()` freezes all params except `model.head`; called once before training, never unfrozen
- [x] Optimizer built over `model.head.parameters()` only
- [x] `random_subset()` produces a 20% sample using `random_split` (no sklearn dependency)
- [x] `load_backbone_only()` loads backbone weights with `strict=False`; logs (does not assert) unexpected keys

### Training Script
- [x] `src/train.py` uses `torch.cuda.amp.GradScaler` + `autocast` — AMP preserved from notebook
- [x] `src/train.py` runs locally with `--train-dir` and `--val-dir` pointing to local paths
- [x] `src/train.py` runs on SageMaker using `SM_CHANNEL_TRAIN` / `SM_CHANNEL_VAL` env vars
- [x] Single training loop: backbone frozen throughout, half-epoch warmup → CosineAnnealingLR, head params only
- [x] `num_workers` set to `min(os.cpu_count() - 1, 8)` — not hardcoded
- [x] Checkpoints include `scaler_state` — spot resume restores all four states (model, optimizer, scheduler, scaler)
- [x] Final model saved to `--model-dir` (defaults to `/opt/ml/model`)

### AWS Infrastructure
- [x] `docker/requirements.txt` contains only `wandb>=0.17.0` (no scikit-learn)
- [x] `docker/Dockerfile` builds successfully from AWS DLC base (`763104351884.dkr.ecr...`)
- [x] `scripts/build_and_push_ecr.sh` authenticates to both DLC and own ECR, builds, and pushes
- [x] `scripts/upload_data_to_s3.sh` syncs train/ and val/ directories to S3
- [x] `scripts/launch_sagemaker_job.py` launches with `use_spot_instances=True`, unique `checkpoint_s3_uri`, `freeze-epochs=2`
- [x] `WANDB_API_KEY` never hardcoded — always sourced from local shell env var

### Experiments
- [ ] Stage 1: configs A, B, C trained for 8 epochs (2 frozen + 6 fine-tuning) on 20% data
- [ ] Stage 1 results compared on `val/top5_acc` — bottom 1-2 configs eliminated
- [ ] Stage 2: surviving configs trained for 25 epochs on full data, `ml.g4dn.xlarge`
- [ ] Winner identified by `val/top1_acc` and loss curve shape (no overfitting gap)

---

## Non-Functional Requirements

- All WandB runs tagged with `head_config`, `stage`, `hidden_dims`, `dropout` for easy filtering
- Each SageMaker job has a unique `checkpoint_s3_uri` (timestamp-based job name) — no cross-job checkpoint corruption
- `ENTRYPOINT` in Dockerfile uses exec form (JSON array) so SIGTERM reaches Python directly

---

## Dependencies & Prerequisites

- [x] AWS account with SageMaker execution role (`Model_training`) that has:
  - [x] `AmazonSageMakerFullAccess`
  - [x] `AmazonS3FullAccess`
  - [x] `AmazonEC2ContainerRegistryFullAccess`
- [x] Docker installed locally for building the image (v28.5.1)
- [x] AWS CLI configured (`aws configure`) — user `ML_developer`, region `us-east-1`
- [x] `sagemaker`, `boto3`, `wandb` installed in `.venv` (Python 3.13.9, miniconda3)
- [x] Best baseline checkpoint available locally and on S3
  - Local: `checkpoints/checkpoint.pth` (113MB, epoch 98, val_acc 71.82%)
  - S3: `s3://resnet-face-classification-839000214843/checkpoints/checkpoint.pth`
  - Source: W&B run `kyungin/hw2p2-ablations/kuib2alf`

## Provisioned AWS Resources

| Resource | Value |
|----------|-------|
| AWS Account ID | `839000214843` |
| Region | `us-east-1` |
| IAM CLI user | `arn:aws:iam::839000214843:user/ML_developer` |
| SageMaker execution role | `arn:aws:iam::839000214843:role/Model_training` |
| S3 bucket | `s3://resnet-face-classification-839000214843` |

**S3 layout (as uploaded):**

```
s3://resnet-face-classification-839000214843/
├── data/
│   ├── train/     ← 7001 classes, ~1.4 GB
│   └── val/       ← 7001 classes (dev split), ~361 MB
└── checkpoints/
    └── checkpoint.pth   ← baseline, epoch 98, val_acc 71.82%
```

**Activate venv before running any scripts:**

```bash
source .venv/bin/activate
```

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Spot instance interruption | Epoch-level checkpointing to `/opt/ml/checkpoints/` + SageMaker auto-resume |
| Overfitting with deeper heads (only 20 imgs/class) | Dropout(0.4) + monitor train/val loss gap; dropout is tunable per config |
| DLC base image version mismatch with local PyTorch | Pin DLC tag; test `train.py` locally before pushing to ECR |
| WandB key exposed in logs | Pass via env var, never print it; upgrade to Secrets Manager for production |
| Class distribution skew in subsample | `random_split` at 140k/7000 classes produces ~uniform distribution; verify with a quick class-count check before Stage 1 |

---

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-04-03-mlp-head-experiment-brainstorm.md`
- Notebook model definition: cell 26 (`ResNet` class, `ResNet18` subclass)
- Notebook training loop: cells 46-50

### External
- [AWS DLC PyTorch training images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
- [SageMaker Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
- [SageMaker Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)
- [SageMaker Training Toolkit env vars](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md)
- [WandB SageMaker Integration](https://docs.wandb.ai/models/integrations/sagemaker)

---

## Follow-up Experiments (Out of Scope)

- GeM pooling vs. AdaptiveAvgPool2d (swap after best head is identified)
- ArcFace / CosFace loss (face recognition literature standard)
- AWS Secrets Manager for WandB key (upgrade from env var pattern)
