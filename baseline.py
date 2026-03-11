#!/usr/bin/env python3
import json
import logging
import math
import boto3
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

LOG_FILE = "app.log"


class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key

    def load(self) -> dict:
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            baseline = json.loads(response["Body"].read())
            logger.info("Loaded baseline from s3://%s/%s", self.bucket, self.baseline_key)
            return baseline
        except s3.exceptions.NoSuchKey:
            logger.info("No existing baseline found in S3, starting fresh.")
            return {}
        except Exception as e:
            logger.error("Failed to load baseline from S3: %s", e)
            return {}

    def save(self, baseline: dict):
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )
            logger.info("Baseline saved to s3://%s/%s", self.bucket, self.baseline_key)
        except Exception as e:
            logger.error("Failed to save baseline to S3: %s", e)

        # Sync the log file to S3 alongside every baseline save
        try:
            s3.upload_file(LOG_FILE, self.bucket, "logs/app.log")
            logger.info("Log file synced to s3://%s/logs/app.log", self.bucket)
        except Exception as e:
            logger.error("Failed to sync log file to S3: %s", e)

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}

            state = baseline[channel]

            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state
            logger.info(
                "Baseline updated for channel '%s': count=%d, mean=%.4f, std=%.4f",
                channel, state["count"], state["mean"], state["std"],
            )
        except Exception as e:
            logger.error("Failed to update baseline for channel '%s': %s", channel, e)

        return baseline

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        return baseline.get(channel)
