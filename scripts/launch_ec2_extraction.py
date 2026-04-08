#!/usr/bin/env python3
"""
launch_ec2_extraction.py — Launch a spot GPU instance to pre-extract backbone features.

Uploads extract_and_upload_features.py + src/model.py to S3, then launches a
g4dn.xlarge spot instance with a user-data script that:
  1. Syncs train/val data from S3
  2. Downloads the checkpoint and scripts
  3. Runs extraction (5 augmented copies by default)
  4. Uploads feature tensors to S3
  5. Self-terminates

Prerequisites:
  - An EC2 instance profile with s3:GetObject, s3:PutObject, ec2:TerminateInstances
  - A security group (SSH optional — instance self-terminates, no login needed)
  - The baseline checkpoint already on S3

Usage:
    python scripts/launch_ec2_extraction.py \
        --checkpoint-s3 s3://resnet-face-classification-839000214843/checkpoints/checkpoint.pth \
        --instance-profile arn:aws:iam::839000214843:instance-profile/EC2FeatureExtraction\
        --bucket resnet-face-classification-839000214843

Optional:
    --num-aug-copies 5          (default: 5)
    --instance-type ml.g4dn.xlarge (default: g4dn.xlarge)
    --key-name my-key           (add if you want SSH access to monitor)
    --security-group-id sg-xxx  (default: uses default VPC security group)
    --prefix data               (S3 key prefix, default: data)
"""

import argparse
import base64
import os
import time

import boto3
from dotenv import load_dotenv
load_dotenv()

AWS_REGION = 'us-east-1'
REPO_ROOT  = os.path.join(os.path.dirname(__file__), '..')


def find_dlami(ec2):
    """Return the latest Deep Learning AMI (PyTorch, Ubuntu 22.04) in the region."""
    response = ec2.describe_images(
        Owners=['amazon'],
        Filters=[
            {'Name': 'name',   'Values': ['Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)*']},
            {'Name': 'state',  'Values': ['available']},
            {'Name': 'architecture', 'Values': ['x86_64']},
        ]
    )
    images = sorted(response['Images'], key=lambda x: x['CreationDate'], reverse=True)
    if not images:
        raise RuntimeError("No DLAMI found — check region or AMI name filter")
    ami = images[0]
    print(f"AMI: {ami['ImageId']}  ({ami['Name'][:70]})")
    return ami['ImageId']


def upload_scripts(bucket, prefix, s3):
    """Upload extraction script and model.py to S3 so the instance can download them."""
    scripts = {
        f"{prefix}/scripts/extract_and_upload_features.py":
            os.path.join(REPO_ROOT, 'scripts', 'extract_and_upload_features.py'),
        f"{prefix}/scripts/model.py":
            os.path.join(REPO_ROOT, 'src', 'model.py'),
    }
    for key, local_path in scripts.items():
        s3.upload_file(local_path, bucket, key)
        print(f"  Uploaded → s3://{bucket}/{key}")
    return list(scripts.keys())


def build_user_data(args, checkpoint_s3, bucket, prefix):
    """Build the bash script that runs on instance startup."""
    features_prefix = f"{prefix}/features"
    script_prefix   = f"{prefix}/scripts"

    # Parse s3://bucket/key into bucket + key for aws s3 cp
    ckpt_bucket, ckpt_key = checkpoint_s3.replace('s3://', '').split('/', 1)

    return f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/feature-extraction.log 2>&1

echo "=== Feature extraction starting ==="
date

# ── Download scripts ──────────────────────────────────────────────────────────
mkdir -p /home/ubuntu/src
aws s3 cp s3://{bucket}/{script_prefix}/extract_and_upload_features.py /home/ubuntu/
aws s3 cp s3://{bucket}/{script_prefix}/model.py /home/ubuntu/src/

# ── Download checkpoint ───────────────────────────────────────────────────────
aws s3 cp s3://{ckpt_bucket}/{ckpt_key} /home/ubuntu/checkpoint.pth

# ── Sync data from S3 ─────────────────────────────────────────────────────────
mkdir -p /home/ubuntu/data/train /home/ubuntu/data/val
echo "Syncing train data..."
aws s3 sync s3://{bucket}/{prefix}/train/ /home/ubuntu/data/train/ --quiet
echo "Syncing val data..."
aws s3 sync s3://{bucket}/{prefix}/val/   /home/ubuntu/data/val/   --quiet
echo "Data sync complete."

# ── Run extraction ────────────────────────────────────────────────────────────
# DLAMI pytorch env has PyTorch + torchvision + CUDA pre-installed
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

pip install -q boto3

python /home/ubuntu/extract_and_upload_features.py \
    --checkpoint   /home/ubuntu/checkpoint.pth \
    --train-dir    /home/ubuntu/data/train \
    --val-dir      /home/ubuntu/data/val \
    --num-aug-copies {args.num_aug_copies} \
    --batch-size   256 \
    --bucket       {bucket} \
    --features-prefix {features_prefix}

echo "=== Extraction complete ==="
date

# ── Self-terminate ────────────────────────────────────────────────────────────
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {AWS_REGION}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-s3',    required=True,
                   help='S3 URI of baseline checkpoint, e.g. s3://bucket/path/best.pth')
    p.add_argument('--instance-profile', required=True,
                   help='ARN of EC2 instance profile with S3 + ec2:TerminateInstances permissions')
    p.add_argument('--bucket',           required=True)
    p.add_argument('--prefix',           default='data')
    p.add_argument('--num-aug-copies',   type=int, default=5)
    p.add_argument('--instance-type',    default='g4dn.xlarge',
                   choices=['g4dn.xlarge', 'g4dn.2xlarge', 'g5.xlarge'])
    p.add_argument('--key-name',         default=None,
                   help='EC2 key pair name for SSH access (optional)')
    p.add_argument('--security-group-id', default=None,
                   help='Security group ID (default: uses default VPC security group)')
    return p.parse_args()


def main():
    args = parse_args()

    ec2 = boto3.client('ec2', region_name=AWS_REGION)
    s3  = boto3.client('s3',  region_name=AWS_REGION)

    print("Uploading scripts to S3...")
    upload_scripts(args.bucket, args.prefix, s3)

    print("\nFinding latest DLAMI...")
    ami_id = find_dlami(ec2)

    user_data = build_user_data(args, args.checkpoint_s3, args.bucket, args.prefix)
    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    launch_spec = {
        'ImageId':      ami_id,
        'InstanceType': args.instance_type,
        'UserData':     user_data_b64,
        'IamInstanceProfile': {'Arn': args.instance_profile},
        'BlockDeviceMappings': [{
            'DeviceName': '/dev/sda1',
            'Ebs': {'VolumeSize': 30, 'VolumeType': 'gp3'},  # OS + data + features
        }],
        'TagSpecifications': [{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'resnet-feature-extraction'}],
        }],
    }
    if args.key_name:
        launch_spec['KeyName'] = args.key_name
    if args.security_group_id:
        launch_spec['SecurityGroupIds'] = [args.security_group_id]

    print(f"\nLaunching spot {args.instance_type}...")
    response = ec2.run_instances(
        **launch_spec,
        MinCount=1,
        MaxCount=1,
        InstanceMarketOptions={
            'MarketType': 'spot',
            'SpotOptions': {
                'SpotInstanceType': 'one-time',
                'InstanceInterruptionBehavior': 'terminate',
            },
        },
    )

    instance_id = response['Instances'][0]['InstanceId']
    print(f"\nInstance launched: {instance_id}")
    print(f"  Type:    {args.instance_type}  (spot)")
    print(f"  Copies:  {args.num_aug_copies} augmented passes")
    print(f"  Output:  s3://{args.bucket}/{args.prefix}/features/")
    print(f"\nMonitor logs (once instance is running, ~2 min):")
    print(f"  aws ssm start-session --target {instance_id} --region {AWS_REGION}")
    print(f"  # then: tail -f /var/log/feature-extraction.log")
    print(f"\nInstance will self-terminate when extraction completes.")


if __name__ == '__main__':
    main()
