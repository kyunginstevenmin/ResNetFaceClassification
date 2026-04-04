#!/usr/bin/env python3
"""
Launch a SageMaker training job for one MLP head config.

WANDB_API_KEY must be set in the local environment — it is injected into
the training container as an env var. Never hardcode it.

Usage (Stage 1 — elimination, 20% data):
    WANDB_API_KEY=<key> python scripts/launch_sagemaker_job.py \
        --head-config B --stage 1 --epochs 8 \
        --instance-type ml.g4dn.xlarge --use-spot

Usage (Stage 2 — selection, full data):
    WANDB_API_KEY=<key> python scripts/launch_sagemaker_job.py \
        --head-config B --stage 2 --epochs 25 \
        --instance-type ml.g4dn.xlarge --use-spot \
        --baseline-ckpt-s3 s3://<bucket>/resnet-face/baseline/best.pt
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
SAGEMAKER_ROLE = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/SageMakerExecutionRole"


def parse_args():
    p = argparse.ArgumentParser(description="Launch a SageMaker training job")
    p.add_argument("--head-config",       required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--stage",             type=int, default=1, choices=[1, 2])
    p.add_argument("--epochs",            type=int, default=8,
                   help="Stage 1: 8 epochs, Stage 2: 25 epochs")
    p.add_argument("--lr",                type=float, default=0.5)
    p.add_argument("--batch-size",        type=int, default=64)
    p.add_argument("--instance-type",     default="ml.g4dn.xlarge",
                   choices=["ml.g4dn.xlarge", "ml.g5.xlarge"])
    p.add_argument("--use-spot",          action="store_true",
                   help="Use spot instances (60-90%% cheaper, requires checkpoint_s3_uri)")
    p.add_argument("--baseline-ckpt-s3", default=None,
                   help="S3 URI of best baseline checkpoint for backbone init")
    return p.parse_args()


def main():
    args    = parse_args()
    session = sagemaker.Session()
    bucket  = session.default_bucket()
    prefix  = "resnet-face"

    job_name      = f"resnet-head-{args.head_config.lower()}-s{args.stage}-{int(time.time())}"
    checkpoint_s3 = f"s3://{bucket}/{prefix}/checkpoints/{job_name}"

    # Compute budget: Stage 1 = 4h, Stage 2 = 12h; max_wait = 2x for spot queue time
    max_run  = (4 if args.stage == 1 else 12) * 3600
    max_wait = max_run * 2

    hyperparams = {
        "head-config": args.head_config,
        "stage":       args.stage,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "batch-size":  args.batch_size,
        # backbone frozen throughout — no freeze-epochs needed
    }
    if args.baseline_ckpt_s3:
        hyperparams["baseline-ckpt"] = args.baseline_ckpt_s3

    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=SAGEMAKER_ROLE,
        instance_type=args.instance_type,
        instance_count=1,
        volume_size=100,          # GB — holds dataset + checkpoints on instance
        max_run=max_run,
        use_spot_instances=args.use_spot,
        max_wait=max_wait if args.use_spot else None,
        checkpoint_s3_uri=checkpoint_s3 if args.use_spot else None,
        checkpoint_local_path="/opt/ml/checkpoints",
        output_path=f"s3://{bucket}/{prefix}/output",
        hyperparameters=hyperparams,
        environment={
            # Injected from local shell — never hardcoded in source
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        },
        sagemaker_session=session,
        base_job_name="resnet-face",
    )

    train_input = TrainingInput(f"s3://{bucket}/{prefix}/data/train", input_mode="File")
    val_input   = TrainingInput(f"s3://{bucket}/{prefix}/data/val",   input_mode="File")

    print(f"Launching: {job_name}")
    print(f"  Head:     {args.head_config}")
    print(f"  Stage:    {args.stage}  ({args.epochs} epochs)")
    print(f"  Instance: {args.instance_type}  Spot: {args.use_spot}")
    print(f"  Checkpoint S3: {checkpoint_s3}")

    estimator.fit(
        inputs={"train": train_input, "val": val_input},
        job_name=job_name,
        wait=True,
        logs="All",
    )
    print(f"\nDone. Artifacts: {estimator.model_data}")


if __name__ == "__main__":
    main()
