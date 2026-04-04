#!/usr/bin/env python3
"""
Launch a SageMaker training job for one MLP head config.

WANDB_API_KEY must be set in the local environment — it is injected into
the training container as an env var. Never hardcode it.

Usage (Stage 1 — elimination, 20% data):
    python scripts/launch_sagemaker_job.py \
--head-config A --stage 1 --epochs 8 --use-spot \
--bucket resnet-face-classification-839000214843 --prefix data \
--baseline-ckpt-s3 s3://resnet-face-classification-839000214843/checkpoints/checkpoint.pth

Usage (Stage 2 — selection, full data):
    python scripts/launch_sagemaker_job.py \
        --head-config B --stage 2 --epochs 25 \
        --instance-type ml.g4dn.xlarge --use-spot \
        --baseline-ckpt-s3 s3://<bucket>/resnet-face/baseline/best.pt
"""

import argparse
import os
import time
from dotenv import load_dotenv
load_dotenv()  # Load env vars from .env file if present

import boto3
import sagemaker
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import (
    InputData, Compute, CheckpointConfig,
    StoppingCondition, OutputDataConfig,
)
from sagemaker.core.helper.session_helper import Session, get_execution_role

AWS_REGION     = "us-east-1"
AWS_ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
ECR_REPO       = "resnet-face-training"
IMAGE_TAG      = "latest"
IMAGE_URI      = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}"
SAGEMAKER_ROLE = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/Model_training"

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
    p.add_argument("--bucket",           default=None,
                   help="S3 bucket name (defaults to SageMaker default bucket)")
    p.add_argument("--prefix",           default="resnet-face",
                   help="S3 key prefix scoping this project within the bucket")
    return p.parse_args()


def main():
    args    = parse_args()
    session = Session(boto_session=boto3.Session(region_name=AWS_REGION))
    bucket  = args.bucket or session.default_bucket()
    prefix  = args.prefix

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
    trainer = ModelTrainer(
        training_image=IMAGE_URI,
        role=SAGEMAKER_ROLE,
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=1,
            volume_size_in_gb=100,               # holds dataset + checkpoints on instance
            enable_managed_spot_training=args.use_spot or None,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=max_run,
            max_wait_time_in_seconds=max_wait if args.use_spot else None,
        ),
        checkpoint_config=CheckpointConfig(
            s3_uri=checkpoint_s3 if args.use_spot else None,
            local_path="/opt/ml/checkpoints",
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{bucket}/{prefix}/output",
        ),
        hyperparameters=hyperparams,
        environment={
            # Injected from local shell — never hardcoded in source
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "PYTHONUNBUFFERED": "1"
        },
        sagemaker_session=session,
        base_job_name="resnet-face",
    )

    train_input = InputData(channel_name="train", data_source=f"s3://{bucket}/{prefix}/train")
    val_input   = InputData(channel_name="val",   data_source=f"s3://{bucket}/{prefix}/val")

    input_channels = [train_input, val_input]
    if args.baseline_ckpt_s3:
        # Pass checkpoint as a channel — SageMaker downloads it to
        # /opt/ml/input/data/checkpoint/ before train.py starts.
        # train.py reads SM_CHANNEL_CHECKPOINT to get the local path.
        input_channels.append(
            InputData(channel_name="checkpoint", data_source=args.baseline_ckpt_s3)
        )

    print(f"Launching: {job_name}")
    print(f"  Head:     {args.head_config}")
    print(f"  Stage:    {args.stage}  ({args.epochs} epochs)")
    print(f"  Instance: {args.instance_type}  Spot: {args.use_spot}")
    print(f"  Baseline ckpt: {args.baseline_ckpt_s3 or 'none'}")

    trainer.train(
        input_data_config=input_channels,
        wait=True,
        logs=True,
    )
    print(f"\nDone. Job: {trainer._latest_training_job.training_job_name}")


if __name__ == "__main__":
    main()
