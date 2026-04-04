"""
train.py — SageMaker-compatible training script for the MLP Head experiment.

Runs locally:
    python src/train.py --train-dir data/train --val-dir data/val \
        --head-config B --stage 1 --epochs 8 \
        --baseline-ckpt checkpoint.pth

Runs on SageMaker (paths injected via SM_CHANNEL_* env vars automatically).
"""

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import wandb

# ---------------------------------------------------------------------------
# Model — Residual block
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """Basic residual block used in ResNet18."""
    def __init__(self, num_channels, use_1x1conv=False, kernel_sizes=3, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_sizes, stride=strides,
                                   padding=(kernel_sizes - 1) // 2)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_sizes, stride=1,
                                   padding=(kernel_sizes - 1) // 2)
        self.conv3 = nn.LazyConv2d(num_channels, 1, stride=strides) if use_1x1conv else None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)


# ---------------------------------------------------------------------------
# Model — MLP head
# ---------------------------------------------------------------------------

HEAD_CONFIGS = {
    'A': {'hidden_dims': [],           'dropout': 0.4},  # baseline: 512 -> 7000
    'B': {'hidden_dims': [1024],       'dropout': 0.4},  # wider:    512 -> 1024 -> 7000
    'C': {'hidden_dims': [512],        'dropout': 0.4},  # same:     512 -> 512  -> 7000
    'D': {'hidden_dims': [1024, 512],  'dropout': 0.4},  # two-layer:512 -> 1024 -> 512 -> 7000
}


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
        layers.append(nn.Linear(current_dim, num_classes))  # no activation — raw logits
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Model — ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def __init__(self, arch, lr=0.1, num_classes=7000, head=None, num_features=512):
        super().__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(b[0], b[1], first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ))
        if head is None:
            head = MLPHead(num_features, hidden_dims=[], num_classes=num_classes)
        self.head = head

    def forward(self, x, return_feats=False):
        # return_feats kept for API compatibility — always returns raw logits
        return self.head(self.net(x))


class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=7000, head=None):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                         head=head, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model):
    """Freeze all parameters except model.head. Backbone stays frozen for the
    entire run — it was trained on the same dataset and task."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)


def load_backbone_only(checkpoint_path, model, device):
    """Load backbone weights from an existing baseline checkpoint.

    The baseline uses the OLD architecture where the classifier lives inside
    net.last.2 (LazyLinear). That key is excluded here so the new MLPHead is
    randomly initialised. Checkpoint key is 'model_state_dict'.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_state = checkpoint['model_state_dict']
    backbone_state = {k: v for k, v in saved_state.items()
                      if not k.startswith('net.last.2')}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if unexpected:
        print(f"Warning: ignoring {len(unexpected)} unexpected keys: {unexpected[:5]}")
    print(f"Backbone loaded. New head randomly initialized ({len(missing)} keys).")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def topk_accuracy(logits, targets, topk=(1, 5)):
    """Return top-k accuracy (%) for each k, as a list of scalar tensors."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()                           # (maxk, N)
    correct = pred.eq(targets.unsqueeze(0))   # (maxk, N)
    results = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum()
        results.append(correct_k * (100.0 / targets.size(0)))
    return results


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_transforms():
    mean = [0.4289, 0.3656, 0.3335]
    std  = [0.2511, 0.2274, 0.2122]

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, scale=(0.7, 1.3), shear=10),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        T.RandomGrayscale(),
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        T.ToTensor(),
    ])
    val_tf = T.Compose([T.ToTensor()])
    return train_tf, val_tf


def random_subset(dataset, fraction=0.20, seed=42):
    n = int(len(dataset) * fraction)
    rest = len(dataset) - n
    subset, _ = random_split(dataset, [n, rest],
                             generator=torch.Generator().manual_seed(seed))
    return subset


# ---------------------------------------------------------------------------
# Training + validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, scaler, loader, criterion, device,
                    epoch, scheduler):
    model.train()
    running_loss, running_top1, running_top5, n = 0.0, 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        running_top5 += top5.item() * bs
        n += bs

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # per-batch step

    epoch_loss = running_loss / n
    epoch_top1 = running_top1 / n
    epoch_top5 = running_top5 / n

    wandb.log({
        'train/loss':     epoch_loss,
        'train/top1_acc': epoch_top1,
        'train/top5_acc': epoch_top5,
        'learning_rate':  scheduler.get_last_lr()[0],
        'epoch':          epoch,
    })
    print(f"[Epoch {epoch}] train loss={epoch_loss:.4f}  top1={epoch_top1:.2f}%  top5={epoch_top5:.2f}%")
    return epoch_top1, epoch_loss, epoch_top5


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss, running_top1, running_top5, n = 0.0, 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            logits = model(images)
            loss = criterion(logits, labels)

        top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_top1 += top1.item() * bs
        running_top5 += top5.item() * bs
        n += bs

    epoch_loss = running_loss / n
    epoch_top1 = running_top1 / n
    epoch_top5 = running_top5 / n

    wandb.log({
        'val/loss':     epoch_loss,
        'val/top1_acc': epoch_top1,
        'val/top5_acc': epoch_top5,
        'epoch':        epoch,
    })
    print(f"[Epoch {epoch}]   val loss={epoch_loss:.4f}  top1={epoch_top1:.2f}%  top5={epoch_top5:.2f}%")
    return epoch_top1, epoch_loss, epoch_top5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='MLP Head experiment — ResNet face classifier')

    # Experiment config
    parser.add_argument('--head-config',   type=str,   default='A', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--stage',         type=int,   default=1,   choices=[1, 2])
    parser.add_argument('--epochs',        type=int,   default=8,   help='Stage 1: 8, Stage 2: 25')
    parser.add_argument('--lr',            type=float, default=0.5)
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--num-classes',   type=int,   default=7000)
    parser.add_argument('--baseline-ckpt', type=str,   default=None,
                        help='Path to best baseline checkpoint for backbone init')

    # Paths — SM env vars used as defaults so the same script runs locally + on SageMaker
    parser.add_argument('--train-dir',      type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data/train'))
    parser.add_argument('--val-dir',        type=str,
                        default=os.environ.get('SM_CHANNEL_VAL',   'data/val'))
    parser.add_argument('--model-dir',      type=str,
                        default=os.environ.get('SM_MODEL_DIR',     'output/model'))
    parser.add_argument('--checkpoint-dir', type=str,   default='/opt/ml/checkpoints')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Head: {args.head_config}  |  Stage: {args.stage}")

    # ── Model ────────────────────────────────────────────────────────────────
    head_cfg = HEAD_CONFIGS[args.head_config].copy()
    head_cfg['dropout'] = args.dropout  # allow CLI override
    head = MLPHead(512, num_classes=args.num_classes, **head_cfg)
    model = ResNet18(head=head, num_classes=args.num_classes).to(device)

    if args.baseline_ckpt:
        load_backbone_only(args.baseline_ckpt, model, device)

    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    train_dataset = ImageFolder(args.train_dir, transform=train_tf)
    val_dataset   = ImageFolder(args.val_dir,   transform=val_tf)

    if args.stage == 1:
        train_dataset = random_subset(train_dataset, fraction=0.20)
        print(f"Stage 1: using 20% subset — {len(train_dataset)} images")

    nw = min(os.cpu_count() - 1, 8)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    # ── Training setup ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.SGD(
        model.head.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    warmup_steps  = len(train_loader) // 2
    total_steps   = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )
    print(f"Warmup steps: {warmup_steps} | Total steps: {total_steps}")

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb.init(
        project='hw2p2-ablations',
        name=f'head-{args.head_config}-stage{args.stage}',
        config={
            'head_config':  args.head_config,
            'hidden_dims':  head_cfg['hidden_dims'],
            'dropout':      head_cfg['dropout'],
            'stage':        args.stage,
            'epochs':       args.epochs,
            'lr':           args.lr,
            'batch_size':   args.batch_size,
            'num_classes':  args.num_classes,
        },
        resume='allow',
    )

    # ── Spot resume ──────────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.model_dir,      exist_ok=True)

    start_epoch, best_val_acc = 0, 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        print(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, scaler, train_loader, criterion,
                        device, epoch, scheduler)
        val_top1, val_loss, val_top5 = validate(model, val_loader, criterion,
                                                 device, epoch)

        is_best = val_top1 > best_val_acc
        best_val_acc = max(val_top1, best_val_acc)

        # checkpoint.pt — overwritten every epoch for spot resume
        checkpoint = {
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_val_acc':    best_val_acc,
        }
        torch.save(checkpoint, ckpt_path)

        # best.pt — saved separately when val improves; used for Stage 2 + evaluation
        if is_best:
            best_path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"  ↑ New best val top1: {best_val_acc:.2f}%")

    # ── Final model export ───────────────────────────────────────────────────
    # SageMaker tars everything in --model-dir and uploads it to S3 as job output.
    best_path = os.path.join(args.checkpoint_dir, 'best.pt')
    shutil.copy(best_path, os.path.join(args.model_dir, 'model.pt'))
    print(f"Best model copied to {args.model_dir}/model.pt")

    wandb.finish()


if __name__ == '__main__':
    main()
