#!/usr/bin/env python3
"""
extract_and_upload_features.py — Pre-extract backbone features and upload to S3.

Runs the frozen ResNet18 backbone N times over the training set (each pass applies
different random augmentations) and once over the val set (no augmentation).
Saves 512-d feature tensors to S3 so training jobs can skip the backbone forward
pass entirely.

S3 output:
    s3://{bucket}/{features-prefix}/train_aug{N}.pt   shape: (N * num_train, 512)
    s3://{bucket}/{features-prefix}/val.pt            shape: (num_val, 512)

Usage:
    python scripts/extract_and_upload_features.py \
        --checkpoint checkpoints/baseline_best.pth \
        --train-dir data/train \
        --val-dir data/val \
        --num-aug-copies 5 \
        --bucket resnet-face-classification-839000214843 \
        --features-prefix data/features
"""

import argparse
import os
import sys
import tempfile
import time

import boto3
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, '/opt/ml/code/')
from model import HEAD_CONFIGS, MLPHead, ResNet18, load_backbone_only


def get_transforms():
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


def extract(backbone, dataset, device, num_copies, batch_size, num_workers, label):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_feats, all_labels = [], []
    backbone.eval()
    with torch.inference_mode():
        for copy_idx in range(num_copies):
            t0 = time.time()
            for images, labels in loader:
                with torch.cuda.amp.autocast():
                    feats = backbone(images.to(device))
                all_feats.append(feats.cpu())
                all_labels.append(labels)
            print(f"  [{label}] copy {copy_idx + 1}/{num_copies}  ({time.time() - t0:.0f}s)",
                  flush=True)

    features = torch.cat(all_feats)
    labels   = torch.cat(all_labels)
    print(f"  [{label}] {len(features):,} samples  "
          f"({len(dataset):,} images × {num_copies} copies,  "
          f"{features.nbytes / 1e9:.2f} GB)", flush=True)
    return features, labels


def upload(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  Uploading {size_mb:.0f} MB → s3://{bucket}/{s3_key}", flush=True)
    t0 = time.time()
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  Upload done ({time.time() - t0:.0f}s)", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',      required=True,
                   help='Path to baseline checkpoint .pth (must contain model_state_dict key)')
    p.add_argument('--train-dir',       required=True)
    p.add_argument('--val-dir',         required=True)
    p.add_argument('--num-aug-copies',  type=int, default=5)
    p.add_argument('--num-classes',     type=int, default=7001)
    p.add_argument('--batch-size',      type=int, default=256,
                   help='Can be larger than training batch size — no gradients computed')
    p.add_argument('--output-dir',      default=None,
                   help='Write tensors to this local dir instead of uploading to S3 '
                        '(used by SageMaker Processing — set to /opt/ml/processing/output/)')
    p.add_argument('--bucket',          default=None,
                   help='S3 bucket for direct upload (not needed when --output-dir is set)')
    p.add_argument('--features-prefix', default='data/features')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ── Backbone ─────────────────────────────────────────────────────────────
    head  = MLPHead(512, num_classes=args.num_classes, **HEAD_CONFIGS['A'])
    model = ResNet18(head=head, num_classes=args.num_classes).to(device)
    _ = model(torch.zeros(2, 3, 224, 224, device=device))  # initialize lazy layers
    load_backbone_only(args.checkpoint, model, device)
    backbone = model.net  # ResNet18 up to AdaptiveAvgPool + Flatten → (B, 512)
    print(f"Backbone loaded from: {args.checkpoint}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    train_dataset = ImageFolder(args.train_dir, transform=train_tf)
    val_dataset   = ImageFolder(args.val_dir,   transform=val_tf)
    print(f"Train: {len(train_dataset):,} images | Val: {len(val_dataset):,} images\n")

    nw = min(os.cpu_count() - 1, 8)

    # ── Extract ───────────────────────────────────────────────────────────────
    print(f"Extracting train features ({args.num_aug_copies} augmented copies)...")
    train_feats, train_labels = extract(backbone, train_dataset, device,
                                        args.num_aug_copies, args.batch_size, nw, 'train')

    print(f"\nExtracting val features (1 copy, no augmentation)...")
    val_feats, val_labels = extract(backbone, val_dataset, device,
                                    1, args.batch_size, nw, 'val')

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output_dir:
        # SageMaker Processing mode — write to local output dir; SM uploads to S3
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, f'train_aug{args.num_aug_copies}.pt')
        val_path   = os.path.join(args.output_dir, 'val.pt')
        print(f"\nSaving to {args.output_dir}...")
        torch.save({'features': train_feats, 'labels': train_labels,
                    'num_images': len(train_dataset), 'num_copies': args.num_aug_copies},
                   train_path)
        torch.save({'features': val_feats, 'labels': val_labels,
                    'num_images': len(val_dataset), 'num_copies': 1},
                   val_path)
        print(f"Done. SageMaker will upload output dir to S3.")
    else:
        # Direct upload mode — for local or EC2 use
        if not args.bucket:
            raise ValueError("--bucket is required when --output-dir is not set")
        train_key = f"{args.features_prefix}/train_aug{args.num_aug_copies}.pt"
        val_key   = f"{args.features_prefix}/val.pt"
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, 'train.pt')
            val_path   = os.path.join(tmpdir, 'val.pt')
            print(f"\nSaving locally...")
            torch.save({'features': train_feats, 'labels': train_labels,
                        'num_images': len(train_dataset), 'num_copies': args.num_aug_copies},
                       train_path)
            torch.save({'features': val_feats, 'labels': val_labels,
                        'num_images': len(val_dataset), 'num_copies': 1},
                       val_path)
            print("Uploading to S3...")
            upload(train_path, args.bucket, train_key)
            upload(val_path,   args.bucket, val_key)
        print(f"\nDone.")
        print(f"  s3://{args.bucket}/{train_key}")
        print(f"  s3://{args.bucket}/{val_key}")


if __name__ == '__main__':
    main()
