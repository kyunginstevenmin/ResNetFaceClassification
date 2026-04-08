"""
train.py — SageMaker-compatible training script for the MLP Head experiment.

Runs locally:
    python src/train.py --train-dir data/train --val-dir data/dev \
        --head-config B --stage 1 --epochs 8 \
        --baseline-ckpt checkpoint.pth
        --checkpoint-dir output/checkpoints

Runs on SageMaker (paths injected via SM_CHANNEL_* env vars automatically).
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, random_split
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


def extract_features(backbone, dataset, device, num_copies, batch_size, num_workers):
    """Pre-extract 512-d backbone features N times with augmentation.

    Each pass through the dataset applies different random augmentations,
    producing num_copies * len(dataset) total samples in the returned TensorDataset.
    Backbone must already be frozen and in eval mode before calling.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_feats, all_labels = [], []
    backbone.eval()
    with torch.inference_mode():
        for copy_idx in range(num_copies):
            print(f"  Extracting features (copy {copy_idx + 1}/{num_copies})...", flush=True)
            for images, labels in loader:
                with torch.cuda.amp.autocast():
                    feats = backbone(images.to(device))
                all_feats.append(feats.float().cpu())
                all_labels.append(labels)
    features = torch.cat(all_feats)   # (num_copies * dataset_size, 512)
    labels   = torch.cat(all_labels)  # (num_copies * dataset_size,)
    print(f"  Done: {len(features):,} samples  "
          f"({len(dataset):,} images × {num_copies} copies,  "
          f"{features.nbytes / 1e9:.2f} GB in memory)", flush=True)
    return TensorDataset(features, labels)


# ---------------------------------------------------------------------------
# Training + validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, scaler, loader, criterion, device,
                    epoch, scheduler, use_features=False):
    model.train()
    running_loss, running_top1, running_top5, n = 0.0, 0.0, 0.0, 0
    t0 = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model.head(images) if use_features else model(images)
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
def validate(model, loader, criterion, device, epoch, use_features=False):
    model.eval()
    running_loss, running_top1, running_top5, n = 0.0, 0.0, 0.0, 0
    t0 = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            logits = model.head(images) if use_features else model(images)
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
    parser.add_argument('--num-aug-copies', type=int,  default=0,
                        help='Pre-extract backbone features N times with augmentation. 0 disables.')
    parser.add_argument('--num-classes',   type=int,   default=7001)
    parser.add_argument('--baseline-ckpt', type=str,
                        default=os.environ.get('SM_CHANNEL_CHECKPOINT', None),
                        help='Local path to baseline checkpoint. On SageMaker, set via checkpoint input channel.')

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
    # debugging statements:
    print(os.environ.get('SM_CHANNEL_CHECKPOINT', None), flush=True)
    print(args.baseline_ckpt, flush=True)
    # ── Model ────────────────────────────────────────────────────────────────
    head_cfg = HEAD_CONFIGS[args.head_config].copy()
    head_cfg['dropout'] = args.dropout  # allow CLI override
    head = MLPHead(512, num_classes=args.num_classes, **head_cfg)
    model = ResNet18(head=head, num_classes=args.num_classes).to(device)

    if args.baseline_ckpt:
        ckpt_path = args.baseline_ckpt
        # SM_CHANNEL_CHECKPOINT points to a directory — find the .pth file inside it
        if os.path.isdir(ckpt_path):
            pth_files = [f for f in os.listdir(ckpt_path) if f.endswith('.pth') or f.endswith('.pt')]
            if not pth_files:
                raise FileNotFoundError(f"No .pth or .pt file found in checkpoint dir: {ckpt_path}")
            ckpt_path = os.path.join(ckpt_path, pth_files[0])
            print(f"Checkpoint dir provided — using: {ckpt_path}")
        
        if args.stage == 2:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
            model.load_state_dict(state)
            print("Stage 2 — loaded full model (backbone + head) from Stage 1 checkpoint")
        else:
            load_backbone_only(ckpt_path, model, device)
        # Verify backbone loaded — sample the first conv weight mean
        # A freshly-initialized LazyConv2d has mean ≈ 0; a trained backbone will differ
        sample_w = next(p for n, p in model.named_parameters() if n == 'net.0.0.weight')
        print(f"Backbone weight check — net.0.0.weight mean: {sample_w.data.mean():.6f}")
    
    else:
        print("No baseline checkpoint provided — backbone randomly initialized")

    # Initialize lazy layers — batch size 2 required because BatchNorm1d errors on batch size 1
    _ = model(torch.zeros(2, 3, 224, 224, device=device))
    freeze_backbone(model)
    if torch.cuda.is_available():
        if args.num_aug_copies > 0:
            model.head = torch.compile(model.head)  # backbone not called during training; compile head only
        else:
            model = torch.compile(model)  # fuses kernels; first batch triggers compilation (~30-60s)
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
    print(f"  Aug copies:   {args.num_aug_copies if args.num_aug_copies > 0 else 'disabled (full pipeline)'}")
    print("=" * 50)
    print()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    train_dataset = ImageFolder(args.train_dir, transform=train_tf)
    val_dataset   = ImageFolder(args.val_dir,   transform=val_tf)

    # debugging:                       
    
    print(f"  train_dir: {args.train_dir}")
    print(f"  contents:  {os.listdir(args.train_dir)[:20]}")
    
    actual_classes = len(train_dataset.classes)              
    print(f"Actual classes in dataset: {actual_classes}")
    for c in train_dataset.classes:                                                       
        if not (c.startswith('n00')):                                  
            print(f"  Suspicious class: {repr(c)}")
                  
    assert actual_classes == args.num_classes, f"Mismatch: dataset has {actual_classes} classes, model expects {args.num_classes}"
    print()

    if args.stage == 1:
        train_dataset = random_subset(train_dataset, fraction=0.20)

    nw = min(os.cpu_count() - 1, 8)
    use_features = args.num_aug_copies > 0
    if use_features:
        print(f"Pre-extracting features ({args.num_aug_copies} augmented copies)...")
        train_dataset = extract_features(model.net, train_dataset, device,
                                         args.num_aug_copies, args.batch_size, nw)
        val_dataset   = extract_features(model.net, val_dataset, device,
                                         1, args.batch_size, nw)
        # Features are already in memory — no workers or pin_memory needed
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=nw, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=4)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=nw, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=4)

    print(f"Train dataset: {len(train_dataset):,} samples" +
          (" (20% subset)" if args.stage == 1 and not use_features else "") +
          (f" ({len(train_dataset) // args.num_aug_copies:,} images × {args.num_aug_copies} copies)" if use_features else ""))
    print(f"Val dataset:   {len(val_dataset):,} samples")
    print(f"Batches/epoch: {len(train_loader)}")
    print(f"num_workers:   {0 if use_features else nw}")
    

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
                        device, epoch, scheduler, use_features=use_features)
        val_top1, val_loss, val_top5 = validate(model, val_loader, criterion,
                                                 device, epoch, use_features=use_features)

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
