#!/usr/bin/env bash
# Build the training Docker image and push it to ECR.
#
# Usage:
#   ./scripts/build_and_push_ecr.sh            # tags image as "latest"
#   ./scripts/build_and_push_ecr.sh abc1234    # tags image with git SHA (recommended in CI)
#
# Run from the repo root so that COPY src/ in the Dockerfile can find src/.
set -euo pipefail

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="resnet-face-training"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Create ECR repo if it doesn't exist
# aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null \
#   || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}"

# Auth Docker to your own ECR (for push)
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Auth Docker to AWS DLC registry (for base image pull during docker build)
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 -f docker/Dockerfile -t "${FULL_IMAGE}" .
docker push "${FULL_IMAGE}"
echo "Pushed: ${FULL_IMAGE}"
