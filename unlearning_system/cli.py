#!/usr/bin/env python3
"""CLI for the Digital Forgetting prototype.

Commands
--------
    train      — обучить SISA-ансамбль на датасете
    evaluate   — оценить качество обученного ансамбля
    unlearn    — удалить данные по ID или фильтру, пересчитать метрики
    report     — вывести накопленные метрики из папки runs/

Примеры
-------
    python cli.py train      --config configs/default.yaml \\
                             --data ../synthetic_data_generator/data/synthetic_dataset.csv
    python cli.py unlearn    --run runs/run_latest --user-ids USR-000001 USR-000002
    python cli.py unlearn    --run runs/run_latest --filter "consent_given == False"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pipeline import ExperimentPipeline
from ml.registry import RunRegistry, file_sha1
from ml.unlearning import UnlearnRequest

logger = logging.getLogger("unlearning.cli")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


# ---- train -----------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    _setup_logging()
    pipe = ExperimentPipeline.from_config(args.config)
    pipe.load_dataset(args.data)
    t0 = time.time()
    pipe.fit()
    train_time = time.time() - t0
    report = pipe.evaluate()

    registry = RunRegistry(Path(args.run_root), run_id=args.run_id)
    registry.write_manifest({
        "run_id": registry.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "data_path": str(args.data),
        "data_sha1": file_sha1(Path(args.data)),
        "train_rows": int(len(pipe.df_train_)),
        "test_rows": int(len(pipe.df_test_)),
        "n_features": int(pipe.X_train_.shape[1]),
        "n_shards": pipe.plan_.n_shards,
        "shard_sizes": pipe.plan_.sizes(),
        "wall_time_train_sec": train_time,
        "backend": pipe.config["model"]["backend"],
    })
    registry.save_pipeline(pipe.pipeline_.state_dict())
    registry.save_plan(pipe.plan_.shard_of_row, pipe.plan_.n_shards)
    for shard_id, learner in enumerate(pipe.ensemble_.learners):
        if learner is None:
            continue
        registry.save_learner(shard_id, learner.__class__.__name__, learner.state_dict())
    registry.save_metrics("baseline", report.as_dict())
    registry.log_event("train", {"duration_sec": train_time,
                                 "shard_sizes": pipe.plan_.sizes()})

    print(f"Run ID: {registry.run_id}")
    print(f"Training wall time: {train_time:.2f}s")
    _print_report("Baseline", report)
    print(f"Saved to: {registry.dir}")
    return 0


# ---- evaluate --------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    _setup_logging()
    run_dir = Path(args.run)
    manifest_path = run_dir / "manifest.yaml"
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        return 1

    pipe = ExperimentPipeline.from_config(args.config)
    pipe.load_dataset(args.data)
    pipe.fit()  # re-fit for now; certified load is a TODO extension
    report = pipe.evaluate()
    _print_report("Evaluation", report)
    return 0


# ---- unlearn ---------------------------------------------------------------

def cmd_unlearn(args: argparse.Namespace) -> int:
    _setup_logging()
    pipe = ExperimentPipeline.from_config(args.config)
    pipe.load_dataset(args.data)
    pipe.fit()

    request = UnlearnRequest(
        user_ids=list(args.user_ids) if args.user_ids else None,
        filter_query=args.filter,
        reason=args.reason,
    )
    if request.is_empty():
        logger.error("Nothing to unlearn: specify --user-ids and/or --filter.")
        return 2

    result = pipe.unlearn(request)

    registry = RunRegistry(Path(args.run_root), run_id=args.run_id)
    registry.write_manifest({
        "run_id": registry.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "data_path": str(args.data),
        "data_sha1": file_sha1(Path(args.data)),
        "unlearn_request": {
            "user_ids": request.user_ids,
            "filter_query": request.filter_query,
            "reason": request.reason,
        },
    })
    registry.save_metrics("before_unlearn", result.metrics_before.as_dict())
    registry.save_metrics("after_unlearn", result.metrics_after.as_dict())
    registry.save_metrics("delta", result.delta)
    registry.log_event("unlearn", {
        "deleted_rows": result.deleted_rows,
        "affected_shards": result.affected_shards,
        "wall_time_sec": result.wall_time_sec,
        "prediction_disagreement": result.prediction_disagreement,
    })

    _print_report("Before unlearn", result.metrics_before)
    _print_report("After unlearn", result.metrics_after)
    print()
    print("=== Delta (after − before) ===")
    for k, v in result.delta.items():
        print(f"  Δ{k:10s}: {v:+.4f}")
    print(f"Deleted rows          : {result.deleted_rows}")
    print(f"Affected shards       : {result.affected_shards}")
    print(f"Prediction disagree.  : {result.prediction_disagreement:.4f}")
    print(f"Retrain wall time     : {result.retrain_report.total_wall_time_sec:.3f}s")
    print(f"Full cycle wall time  : {result.wall_time_sec:.3f}s")
    print(f"Run saved to          : {registry.dir}")
    return 0


# ---- report ----------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    _setup_logging()
    run_dir = Path(args.run)
    metric_files = sorted((run_dir / "metrics").glob("*.json"))
    if not metric_files:
        print(f"No metric files found in {run_dir}/metrics", file=sys.stderr)
        return 1
    for mf in metric_files:
        with open(mf, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"--- {mf.stem} ---")
        for k, v in payload.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")
        print()
    return 0


# ---- printing helpers ------------------------------------------------------

def _print_report(title: str, report) -> None:
    print()
    print(f"=== {title} ===")
    print(f"  accuracy : {report.accuracy:.4f}")
    print(f"  precision: {report.precision:.4f}")
    print(f"  recall   : {report.recall:.4f}")
    print(f"  F1       : {report.f1:.4f}")
    print(f"  AUC      : {report.auc:.4f}")
    print(f"  MSE      : {report.mse:.4f}")
    print(f"  pos rate : true={report.positive_rate_true:.3f} "
          f"pred={report.positive_rate_pred:.3f}")


# ---- argparse --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unlearning",
        description="CLI для Machine Unlearning — HSE MIEM, проект 2039",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Обучить ансамбль SISA")
    pt.add_argument("--config", required=True)
    pt.add_argument("--data", required=True)
    pt.add_argument("--run-root", default="runs")
    pt.add_argument("--run-id", default=None)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", help="Оценить обученный ансамбль")
    pe.add_argument("--config", required=True)
    pe.add_argument("--data", required=True)
    pe.add_argument("--run", required=True)
    pe.set_defaults(func=cmd_evaluate)

    pu = sub.add_parser("unlearn", help="Удалить данные и пересчитать метрики")
    pu.add_argument("--config", required=True)
    pu.add_argument("--data", required=True)
    pu.add_argument("--user-ids", nargs="*", default=None,
                    help="Список user_id для удаления, через пробел")
    pu.add_argument("--filter", default=None,
                    help="Pandas .query() выражение, например \"consent_given == False\"")
    pu.add_argument("--reason", default="user_request")
    pu.add_argument("--run-root", default="runs")
    pu.add_argument("--run-id", default=None)
    pu.set_defaults(func=cmd_unlearn)

    pr = sub.add_parser("report", help="Показать метрики предыдущего запуска")
    pr.add_argument("--run", required=True)
    pr.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
