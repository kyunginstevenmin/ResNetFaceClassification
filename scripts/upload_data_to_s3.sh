#!/usr/bin/env bash
# Upload training and validation data to S3.
# Only needs to be run once (or when the dataset changes).
#
# Usage:
#   ./scripts/upload_data_to_s3.sh                        # uses ./data and default SageMaker bucket
#   ./scripts/upload_data_to_s3.sh data $AWS_BUCKET   # explicit local dir + bucket
set -euo pipefail

LOCAL_DIR="${1:-data}"
BUCKET="${2:-$(python3 -c 'import sagemaker; print(sagemaker.Session().default_bucket())')}"
PREFIX="data"

echo "Uploading ${LOCAL_DIR}/train-small → s3://${BUCKET}/${PREFIX}/train-small"
aws s3 sync "${LOCAL_DIR}/train-small/" "s3://${BUCKET}/${PREFIX}/train-small/" --no-progress

echo "Uploading ${LOCAL_DIR}/dev-small → s3://${BUCKET}/${PREFIX}/val-small"
aws s3 sync "${LOCAL_DIR}/dev-small/" "s3://${BUCKET}/${PREFIX}/val-small/" --no-progress

echo "Uploading ${LOCAL_DIR}/test → s3://${BUCKET}/${PREFIX}/test"
aws s3 sync "${LOCAL_DIR}/test/" "s3://${BUCKET}/${PREFIX}/test/" --no-progress

echo "Done. S3 URIs:"
echo "  Train: s3://${BUCKET}/${PREFIX}/train-small"
echo "  Val:   s3://${BUCKET}/${PREFIX}/val-small"
echo "  Test:  s3://${BUCKET}/${PREFIX}/test"
