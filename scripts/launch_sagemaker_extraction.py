#!/usr/bin/env python3
"""
launch_sagemaker_extraction.py — Run feature extraction as a SageMaker Processing job.

Uses the existing resnet-face-training Docker image. SageMaker downloads train/val/checkpoint
from S3 into the container, runs extract_and_upload_features.py, then uploads the output
feature tensors back to S3 automatically.

S3 output:
    s3://{bucket}/{prefix}/features/train_aug{N}.pt
    s3://{bucket}/{prefix}/features/val.pt

Usage:
    python scripts/launch_sagemaker_extraction.py \
        --bucket $AWS_BUCKET \
        --checkpoint-s3 s3://$AWS_BUCKET/checkpoints/checkpoint.pth

Optional:
    --num-aug-copies 5          (default: 5)
    --instance-type ml.g4dn.xlarge
    --prefix data
"""

import argparse
import os
import time

# Must be set before any sagemaker imports — SageMakerClient is a singleton
# that reads this env var on first instantiation to determine which region to use.
os.environ.setdefault('SAGEMAKER_REGION', 'us-east-1')

import boto3
from dotenv import load_dotenv
load_dotenv()

import sagemaker
from sagemaker.core.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.core.shapes.shapes import ProcessingS3Input, ProcessingS3Output
from sagemaker.core.helper.session_helper import Session

AWS_REGION     = 'us-east-1'
boto3.setup_default_session(region_name=AWS_REGION)
AWS_ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']
ECR_REPO       = 'resnet-face-training'
IMAGE_TAG      = 'latest'
IMAGE_URI      = f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}'
SAGEMAKER_ROLE = f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/Model_training'
REPO_ROOT      = os.path.join(os.path.dirname(__file__), '..')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket',           default=os.environ.get('AWS_BUCKET'),
                   help='S3 bucket name (default: $AWS_BUCKET env var)')
    p.add_argument('--checkpoint-s3',    required=True,
                   help='S3 URI of baseline checkpoint .pth')
    p.add_argument('--prefix',           default='data')
    p.add_argument('--num-aug-copies',   type=int, default=5)
    p.add_argument('--instance-type',    default='ml.g4dn.xlarge',
                   choices=['ml.g4dn.xlarge', 'ml.g5.xlarge'])
    return p.parse_args()


def main():
    args    = parse_args()
    session = Session(boto_session=boto3.Session(region_name=AWS_REGION))
    bucket  = args.bucket
    prefix  = args.prefix

    features_s3 = f's3://{bucket}/{prefix}/features'
    job_name    = f'resnet-feature-extraction-{int(time.time())}'

    processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        command=['python3'],
        instance_type=args.instance_type,
        instance_count=1,
        role=SAGEMAKER_ROLE,
        volume_size_in_gb=30,       # checkpoint(~120MB) + data(~1.4GB) + features(~360MB)
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
        base_job_name='resnet-feature-extraction',
        env={'PYTHONUNBUFFERED': '1'},
    )


    # Parse checkpoint s3://bucket/key
    ckpt_s3_no_prefix = args.checkpoint_s3.replace('s3://', '')
    ckpt_bucket, ckpt_key = ckpt_s3_no_prefix.split('/', 1)
    ckpt_s3_dir = f's3://{ckpt_bucket}/{os.path.dirname(ckpt_key)}/'

    print(f"Launching: {job_name}")
    print(f"  Image:       {IMAGE_URI}")
    print(f"  Instance:    {args.instance_type}")
    print(f"  Aug copies:  {args.num_aug_copies}")
    print(f"  Checkpoint:  {args.checkpoint_s3}")
    print(f"  Output:      {features_s3}/")

    processor.run(
        code=os.path.join(REPO_ROOT, 'scripts', 'extract_and_upload_features.py'),
        inputs=[
            ProcessingInput(
                s3_input = ProcessingS3Input(s3_uri=f's3://{bucket}/{prefix}/train/', s3_data_type='S3Prefix', local_path='/opt/ml/processing/input/train/'),
                input_name='train',
            ),
            ProcessingInput(
                s3_input = ProcessingS3Input(s3_uri=f's3://{bucket}/{prefix}/val/', s3_data_type='S3Prefix', local_path='/opt/ml/processing/input/val/'),
                input_name='val',
            ),
            ProcessingInput(
                s3_input = ProcessingS3Input(s3_uri=ckpt_s3_dir, s3_data_type='S3Prefix', local_path='/opt/ml/processing/input/checkpoint/'),
                input_name='checkpoint',
            )
        ],
        outputs=[
            ProcessingOutput(
                s3_output = ProcessingS3Output(s3_uri=features_s3, local_path='/opt/ml/processing/output/', s3_upload_mode='EndOfJob'),
                output_name='features',
            ),
        ],
        arguments=[
            '--checkpoint',     f'/opt/ml/processing/input/checkpoint/{os.path.basename(ckpt_key)}',
            '--train-dir',      '/opt/ml/processing/input/train/',
            '--val-dir',        '/opt/ml/processing/input/val/',
            '--num-aug-copies', str(args.num_aug_copies),
            '--batch-size',     '256',
            '--output-dir',     '/opt/ml/processing/output/',
        ],
        # job_name=job_name,
        wait=True,
        logs=True,
    )

    print(f'\nDone.')
    print(f'  {features_s3}/train_aug{args.num_aug_copies}.pt')
    print(f'  {features_s3}/val.pt')


if __name__ == '__main__':
    main()
