#!/usr/bin/env python3
"""
create_ec2_instance_profile.py — Create IAM role + instance profile for EC2 feature extraction.

Creates:
  - IAM role:             EC2FeatureExtraction
  - Inline policy:        S3 read/write on your bucket + ec2:TerminateInstances
  - Managed policy:       AmazonSSMManagedInstanceCore (for SSM Session Manager / log tailing)
  - Instance profile:     EC2FeatureExtraction (same name, attached to role)

Usage:
    python scripts/create_ec2_instance_profile.py \
        --bucket resnet-face-classification-839000214843
"""

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError

ROLE_NAME    = 'EC2FeatureExtraction'
PROFILE_NAME = 'EC2FeatureExtraction'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket', required=True, help='S3 bucket the instance needs to access')
    return p.parse_args()


def main():
    args = parse_args()
    iam  = boto3.client('iam')

    # ── 1. Create role ────────────────────────────────────────────────────────
    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    })

    try:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=trust_policy,
            Description='Allows EC2 feature extraction instances to access S3 and self-terminate',
        )
        role_arn = role['Role']['Arn']
        print(f"Created role: {role_arn}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
            print(f"Role already exists: {role_arn}")
        else:
            raise

    # ── 2. Attach inline policy (S3 + ec2:TerminateInstances) ────────────────
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3Access",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{args.bucket}",
                    f"arn:aws:s3:::{args.bucket}/*",
                ]
            },
            {
                "Sid": "SelfTerminate",
                "Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "ec2:ResourceTag/Name": "resnet-feature-extraction"
                    }
                }
            }
        ]
    })

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName='FeatureExtractionPolicy',
        PolicyDocument=policy,
    )
    print("Attached inline policy (S3 + self-terminate)")

    # ── 3. Attach SSM managed policy (enables Session Manager / log tailing) ──
    ssm_policy_arn = 'arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore'
    try:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=ssm_policy_arn)
        print("Attached AmazonSSMManagedInstanceCore (SSM Session Manager)")
    except ClientError as e:
        if 'already attached' in str(e).lower():
            print("AmazonSSMManagedInstanceCore already attached")
        else:
            raise

    # ── 4. Create instance profile and add role ───────────────────────────────
    try:
        profile = iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        profile_arn = profile['InstanceProfile']['Arn']
        print(f"Created instance profile: {profile_arn}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            profile_arn = iam.get_instance_profile(
                InstanceProfileName=PROFILE_NAME)['InstanceProfile']['Arn']
            print(f"Instance profile already exists: {profile_arn}")
        else:
            raise

    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=PROFILE_NAME,
            RoleName=ROLE_NAME,
        )
        print(f"Added role {ROLE_NAME} to instance profile")
    except ClientError as e:
        if 'already associated' in str(e).lower() or 'LimitExceeded' in str(e):
            print(f"Role already associated with instance profile")
        else:
            raise

    print(f"\nDone. Use this ARN in launch_ec2_extraction.py:")
    print(f"  --instance-profile {profile_arn}")


if __name__ == '__main__':
    main()
