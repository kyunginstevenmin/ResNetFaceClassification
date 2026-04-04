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
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import wandb

from model import (HEAD_CONFIGS, MLPHead, ResNet18,
                   freeze_backbone, load_backbone_only, topk_accuracy)

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
    t0 = time.time()

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
    elapsed = time.time() - t0
    print(f"[Epoch {epoch}] train  loss={epoch_loss:.4f}  top1={epoch_top1:.2f}%  top5={epoch_top5:.2f}%  ({elapsed:.0f}s)")
    return epoch_top1, epoch_loss, epoch_top5


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss, running_top1, running_top5, n = 0.0, 0.0, 0.0, 0
    t0 = time.time()

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
    elapsed = time.time() - t0
    print(f"[Epoch {epoch}]   val  loss={epoch_loss:.4f}  top1={epoch_top1:.2f}%  top5={epoch_top5:.2f}%  ({elapsed:.0f}s)")
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
    parser.add_argument('--num-classes',   type=int,   default=7001)
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

    args, _ = parser.parse_known_args()  # ignore extra args SageMaker appends (channel names)
    return args


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Model ────────────────────────────────────────────────────────────────
    head_cfg = HEAD_CONFIGS[args.head_config].copy()
    head_cfg['dropout'] = args.dropout  # allow CLI override
    head = MLPHead(512, num_classes=args.num_classes, **head_cfg)
    model = ResNet18(head=head, num_classes=args.num_classes).to(device)

    if args.baseline_ckpt:
        load_backbone_only(args.baseline_ckpt, model, device)
        # Verify backbone loaded — sample the first conv weight mean
        # A freshly-initialized LazyConv2d has mean ≈ 0; a trained backbone will differ
        sample_w = next(p for n, p in model.named_parameters() if n == 'net.0.0.weight')
        print(f"Backbone weight check — net.0.0.weight mean: {sample_w.data.mean():.6f}")
    else:
        print("No baseline checkpoint provided — backbone randomly initialized")

    freeze_backbone(model)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    hidden = head_cfg['hidden_dims']
    arch   = f"512 → {' → '.join(str(d) for d in hidden)} → {args.num_classes}" if hidden \
             else f"512 → {args.num_classes}"

    print()
    print("=" * 50)
    print("  Training Config")
    print("=" * 50)
    print(f"  Head config:  {args.head_config}  ({arch})")
    print(f"  Stage:        {args.stage}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Dropout:      {head_cfg['dropout']}")
    print(f"  Device:       {device}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Frozen:       {frozen:,} params")
    print(f"  Checkpoint:   {args.baseline_ckpt or 'none'}")
    print("=" * 50)
    print()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    train_dataset = ImageFolder(args.train_dir, transform=train_tf)
    val_dataset   = ImageFolder(args.val_dir,   transform=val_tf)

    if args.stage == 1:
        train_dataset = random_subset(train_dataset, fraction=0.20)

    nw = min(os.cpu_count() - 1, 8)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Train dataset: {len(train_dataset):,} images" +
          (" (20% subset)" if args.stage == 1 else " (full)"))
    print(f"Val dataset:   {len(val_dataset):,} images")
    print(f"Batches/epoch: {len(train_loader)}")
    print(f"num_workers:   {nw}")
    print()

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
