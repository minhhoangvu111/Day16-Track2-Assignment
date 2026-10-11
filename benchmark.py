#!/usr/bin/env python3
"""CPU LightGBM benchmark for the Kaggle Credit Card Fraud dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50
N_ESTIMATORS = 400
LATENCY_REPEATS = 300
BATCH_SIZE = 1_000
BATCH_REPEATS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "ml-benchmark" / "creditcard.csv",
        help="Path to creditcard.csv (default: ~/ml-benchmark/creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="JSON output path (default: ./benchmark_result.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    load_start = time.perf_counter()
    frame = pd.read_csv(args.data)
    data_load_time_s = time.perf_counter() - load_start

    if "Class" not in frame.columns:
        raise ValueError("Expected target column 'Class' in the dataset")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; handle them before training")

    y = frame.pop("Class").astype("int8")
    X = frame.astype("float32")
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary labels 0 and 1; got {sorted(y.unique())}")

    split_start = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    split_time_s = time.perf_counter() - split_start

    model_params = {
        "objective": "binary",
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": SEED,
        "n_jobs": min(2, os.cpu_count() or 1),
        "verbosity": -1,
    }
    model = LGBMClassifier(**model_params)
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_s = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    prediction_time_s = time.perf_counter() - predict_start
    predictions = (probabilities >= THRESHOLD).astype("int8")

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    # Warm up prediction code before measuring latency.
    one_row = X_test.iloc[[0]]
    batch = X_test.iloc[: min(BATCH_SIZE, len(X_test))]
    model.predict_proba(one_row)
    model.predict_proba(batch)

    single_start = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        model.predict_proba(one_row)
    single_total_s = time.perf_counter() - single_start
    single_latency_ms = single_total_s * 1_000 / LATENCY_REPEATS

    actual_batch_size = len(batch)
    batch_start = time.perf_counter()
    for _ in range(BATCH_REPEATS):
        model.predict_proba(batch)
    batch_total_s = time.perf_counter() - batch_start
    batch_throughput_rows_per_s = (
        actual_batch_size * BATCH_REPEATS / batch_total_s
    )

    result = {
        "dataset": {
            "path": str(args.data),
            "rows": int(len(frame)),
            "features": int(X.shape[1]),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "stratified_split": True,
            "threshold": THRESHOLD,
            "threshold_note": "Fixed probability threshold; predictions are positive when P(Class=1) >= 0.5.",
            "early_stopping": False,
            "best_iteration": None,
            "model_parameters": model_params,
        },
        "timings": {
            "data_load_s": float(data_load_time_s),
            "split_s": float(split_time_s),
            "training_s": float(training_time_s),
            "test_prediction_s": float(prediction_time_s),
            "single_row_latency_ms": float(single_latency_ms),
            "single_row_latency_repeats": LATENCY_REPEATS,
            "batch_size": int(actual_batch_size),
            "batch_repeats": BATCH_REPEATS,
            "batch_total_s": float(batch_total_s),
            "batch_throughput_rows_per_s": float(batch_throughput_rows_per_s),
        },
        "metrics": metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=== LightGBM CPU Benchmark ===")
    print(f"Dataset: {args.data} ({len(frame):,} rows, {X.shape[1]} features)")
    print(f"Split: stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed={SEED}")
    print(f"Training time: {training_time_s:.3f} s ({N_ESTIMATORS} estimators)")
    print(f"Test prediction time: {prediction_time_s:.3f} s")
    print(f"Threshold: {THRESHOLD:.2f}; best iteration: not applicable (no early stopping)")
    print(
        "Metrics: "
        + ", ".join(f"{name}={metrics[name]:.6f}" for name in ("roc_auc", "accuracy", "f1", "precision", "recall"))
    )
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Single-row latency: {single_latency_ms:.4f} ms (mean of {LATENCY_REPEATS})")
    print(
        f"Batch throughput: {batch_throughput_rows_per_s:,.2f} rows/s "
        f"({actual_batch_size} rows x {BATCH_REPEATS})"
    )
    print(f"JSON result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
