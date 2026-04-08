#!/usr/bin/env python3
"""
Estimate the cost of a full SageMaker training job by extrapolating
from a completed Phase 1 (short-epoch) run.

Usage:
    python scripts/estimate_cost.py \
        --phase1-job resnet-face-20260404225850 \
        --phase1-epochs 8 \
        --full-epochs 25 \
        --instance-type ml.g4dn.xlarge \
        --use-spot
"""

import argparse
import boto3

AWS_REGION = "us-east-1"

# On-demand prices (us-east-1, as of 2026-04)
ONDEMAND_RATES = {
    "ml.g4dn.xlarge":  0.736,
    "ml.g5.xlarge":    1.408,
    "ml.g4dn.2xlarge": 1.235,
    "ml.g5.2xlarge":   2.048,
}

# Spot discount factor (approximate — real price varies)
SPOT_DISCOUNT = 0.35


def fetch_job_duration_seconds(job_name: str) -> float:
    """Return wall-clock seconds for a completed SageMaker training job."""
    client = boto3.client("sagemaker", region_name=AWS_REGION)
    resp = client.describe_training_job(TrainingJobName=job_name)

    start = resp["TrainingStartTime"]
    end   = resp["TrainingEndTime"]
    return (end - start).total_seconds()


def main():
    p = argparse.ArgumentParser(description="Estimate SageMaker training cost")
    p.add_argument("--phase1-job",      required=True,
                   help="Name of the completed Phase 1 job to extrapolate from")
    p.add_argument("--phase1-epochs",   type=int, required=True,
                   help="Number of epochs run in Phase 1")
    p.add_argument("--full-epochs",     type=int, required=True,
                   help="Number of epochs planned for the full run")
    p.add_argument("--instance-type",   default="ml.g4dn.xlarge",
                   choices=list(ONDEMAND_RATES))
    p.add_argument("--use-spot",        action="store_true")
    args = p.parse_args()

    # ── Pull actual Phase 1 duration ─────────────────────────────────────────
    print(f"Fetching job stats for: {args.phase1_job} ...")
    phase1_seconds = fetch_job_duration_seconds(args.phase1_job)
    seconds_per_epoch = phase1_seconds / args.phase1_epochs

    # ── Extrapolate ──────────────────────────────────────────────────────────
    estimated_seconds = seconds_per_epoch * args.full_epochs
    estimated_hours   = estimated_seconds / 3600

    # ── Cost ─────────────────────────────────────────────────────────────────
    ondemand_rate = ONDEMAND_RATES[args.instance_type]
    spot_rate     = ondemand_rate * SPOT_DISCOUNT

    ondemand_cost = ondemand_rate * estimated_hours
    spot_cost     = spot_rate     * estimated_hours

    active_rate = spot_rate if args.use_spot else ondemand_rate
    active_cost = spot_cost if args.use_spot else ondemand_cost

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 48)
    print("  Cost Estimate")
    print("=" * 48)
    print(f"  Phase 1 job:        {args.phase1_job}")
    print(f"  Phase 1 duration:   {phase1_seconds/60:.1f} min  ({args.phase1_epochs} epochs)")
    print(f"  Time per epoch:     {seconds_per_epoch:.1f}s")
    print()
    print(f"  Full run epochs:    {args.full_epochs}")
    print(f"  Estimated duration: {estimated_hours:.2f}h  ({estimated_seconds/60:.0f} min)")
    print(f"  Instance:           {args.instance_type}")
    print()
    print(f"  On-demand rate:     ${ondemand_rate:.3f}/hr  →  ${ondemand_cost:.2f}")
    print(f"  Spot rate (~35%):   ${spot_rate:.3f}/hr  →  ${spot_cost:.2f}")
    print()
    print(f"  Selected ({'spot' if args.use_spot else 'on-demand'}):       ${active_cost:.2f}")
    print("=" * 48)


if __name__ == "__main__":
    main()
