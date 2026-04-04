#!/usr/bin/env bash
# Upload training and validation data to S3.
# Only needs to be run once (or when the dataset changes).
#
# Usage:
#   ./scripts/upload_data_to_s3.sh                        # uses ./data and default SageMaker bucket
#   ./scripts/upload_data_to_s3.sh data my-bucket-name   # explicit local dir + bucket
set -euo pipefail

LOCAL_DIR="${1:-data}"
BUCKET="${2:-$(python3 -c 'import sagemaker; print(sagemaker.Session().default_bucket())')}"
PREFIX="resnet-face/data"

echo "Uploading ${LOCAL_DIR}/train → s3://${BUCKET}/${PREFIX}/train"
aws s3 sync "${LOCAL_DIR}/train/" "s3://${BUCKET}/${PREFIX}/train/" --no-progress

echo "Uploading ${LOCAL_DIR}/val → s3://${BUCKET}/${PREFIX}/val"
aws s3 sync "${LOCAL_DIR}/val/" "s3://${BUCKET}/${PREFIX}/val/" --no-progress

echo "Done. S3 URIs:"
echo "  Train: s3://${BUCKET}/${PREFIX}/train"
echo "  Val:   s3://${BUCKET}/${PREFIX}/val"
