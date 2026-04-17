"""Run manifests and on-disk persistence of trained ensembles.

Every experiment is identified by a ``run_id`` (timestamp + short hash) and
is fully reproducible from the stored manifest + seeds.

Layout on disk::

    runs/<run_id>/
        manifest.yaml         # config, seeds, data checksums
        pipeline.json         # FeaturePipeline state
        plan.npz              # ShardPlan (shard_of_row + indices)
        learners/
            shard_00.joblib
            shard_01.joblib
            ...
        metrics/
            baseline.json
            after_unlearn_<n>.json
        events.log            # append-only audit trail
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    salt = hashlib.sha1(f"{stamp}-{time.time_ns()}".encode()).hexdigest()[:6]
    return f"{prefix}_{stamp}_{salt}"


class RunRegistry:
    """Manage artefacts for a single experiment run."""

    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.root = Path(root)
        self.run_id = run_id or new_run_id()
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "learners").mkdir(exist_ok=True)
        (self.dir / "metrics").mkdir(exist_ok=True)
        self._event_log = self.dir / "events.log"

    # ---- manifest -----------------------------------------------------------

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        with open(self.dir / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

    def read_manifest(self) -> dict[str, Any]:
        path = self.dir / "manifest.yaml"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ---- pipeline -----------------------------------------------------------

    def save_pipeline(self, state: dict) -> None:
        with open(self.dir / "pipeline.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_pipeline(self) -> dict:
        with open(self.dir / "pipeline.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- shard plan ---------------------------------------------------------

    def save_plan(self, shard_of_row: np.ndarray, n_shards: int) -> None:
        np.savez(self.dir / "plan.npz", shard_of_row=shard_of_row, n_shards=np.array([n_shards]))

    def load_plan(self) -> tuple[np.ndarray, int]:
        data = np.load(self.dir / "plan.npz")
        return data["shard_of_row"], int(data["n_shards"][0])

    # ---- learners -----------------------------------------------------------

    def save_learner(self, shard_id: int, learner_class: str, state: dict) -> Path:
        path = self.dir / "learners" / f"shard_{shard_id:02d}.joblib"
        joblib.dump({"class": learner_class, "state": state}, path)
        return path

    def load_learner(self, shard_id: int) -> dict:
        path = self.dir / "learners" / f"shard_{shard_id:02d}.joblib"
        return joblib.load(path)

    # ---- metrics / audit ----------------------------------------------------

    def save_metrics(self, name: str, payload: dict) -> Path:
        path = self.dir / "metrics" / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        return path

    def log_event(self, event: str, payload: dict | None = None) -> None:
        line = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload or {},
        }
        with open(self._event_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        logger.info("[%s] %s %s", self.run_id, event, payload or {})


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
