#!/usr/bin/env python3
"""
Download model.tar.gz from S3, extract model.pt, and test model loading and simple inference for sanity check.
Use small dev set (data/dev-small) for quick evaluation. For full evaluation, use the full validation set (data/dev).

Usage:
    python scripts/evaluate_s3_model.py \
        --model-s3 s3://resnet-face-classification-839000214843/resnet-face/output/resnet-face-20260404225850/output/model.tar.gz \
        --val-dir data/dev-small \
        --head-config A
"""

import argparse
import os
import sys
import tarfile
import tempfile

import boto3
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# Allow importing from src/ when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import HEAD_CONFIGS, MLPHead, ResNet18, topk_accuracy

AWS_REGION = "us-east-1"


def download_and_extract(s3_uri: str, dest_dir: str) -> str:
    """Download model.tar.gz from S3, extract, return path to model.pt."""
    # Parse s3://bucket/key
    without_scheme = s3_uri[len("s3://"):]
    bucket, key = without_scheme.split("/", 1)

    tar_path = os.path.join(dest_dir, "model.tar.gz")
    print(f"Downloading s3://{bucket}/{key} ...")
    boto3.client("s3", region_name=AWS_REGION).download_file(bucket, key, tar_path)

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir)

    model_path = os.path.join(dest_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.pt not found after extraction. Contents: {os.listdir(dest_dir)}"
        )
    return model_path


def build_model(head_config: str, num_classes: int, device: torch.device) -> ResNet18:
    head_cfg = HEAD_CONFIGS[head_config].copy()
    head = MLPHead(512, num_classes=num_classes, **head_cfg)
    model = ResNet18(head=head, num_classes=num_classes).to(device)
    # Initialize lazy layers
    _ = model(torch.zeros(2, 3, 224, 224, device=device))
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_top1, total_top5, n = 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
        bs = images.size(0)
        total_top1 += top1.item() * bs
        total_top5 += top5.item() * bs
        n += bs

    return total_top1 / n, total_top5 / n


def main():
    p = argparse.ArgumentParser(description="Evaluate a SageMaker model on the validation set")
    p.add_argument("--model-s3",    required=True,
                   help="S3 URI of model.tar.gz (e.g. s3://bucket/prefix/output/model.tar.gz)")
    p.add_argument("--val-dir",     default=os.environ.get("SM_CHANNEL_VAL", "data/val"),
                   help="Path to validation ImageFolder directory")
    p.add_argument("--head-config", default="A", choices=["A", "B", "C", "D"])
    p.add_argument("--num-classes", type=int, default=7001)
    p.add_argument("--batch-size",  type=int, default=128)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory() as tmp:
        model_path = download_and_extract(args.model_s3, tmp)

        print(f"Loading weights from {model_path} ...")
        model = build_model(args.head_config, args.num_classes, device)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        # model.pt is a full training checkpoint — pull just the model weights
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict)

    val_tf = T.Compose([T.ToTensor()])
    val_dataset = ImageFolder(args.val_dir, transform=val_tf)
    nw = min(os.cpu_count() - 1, 8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=nw,
                            pin_memory=(device.type == "cuda"))

    print(f"Val dataset: {len(val_dataset):,} images  |  {len(val_loader)} batches")
    print("Running inference ...")

    top1, top5 = evaluate(model, val_loader, device)

    print()
    print("=" * 40)
    print("  Evaluation Results")
    print("=" * 40)
    print(f"  Top-1 accuracy: {top1:.2f}%")
    print(f"  Top-5 accuracy: {top5:.2f}%")
    print(f"  Val images:     {len(val_dataset):,}")
    print(f"  Head config:    {args.head_config}")
    print("=" * 40)


if __name__ == "__main__":
    main()
