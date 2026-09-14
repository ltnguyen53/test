"""CLI wrapper mỏng cho stage `evaluate`. Không chứa logic — logic nằm ở
evaluation/evaluate.py. Dùng resolve_cache_dir() (phiên 5.1) để tìm đúng
thư mục cache, cùng cách run_train_phase2.py (phiên 5.2) đã làm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_action_mlops.config.loader import load_config
from video_action_mlops.evaluation.evaluate import evaluate_temporal_onnx
from video_action_mlops.features.extract_embeddings import resolve_cache_dir


def run(
    config_path: str,
    checkpoint_phase1: str,
    temporal_onnx_path: str,
    labels_csv: str,
    processed_root: str,
    metrics_out: str,
) -> None:
    cfg = load_config(config_path)
    processed_dir = resolve_cache_dir(cfg, checkpoint_phase1, processed_root)

    metrics = evaluate_temporal_onnx(temporal_onnx_path, processed_dir, labels_csv)

    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[evaluate] {metrics}")
    print(f"[evaluate] metrics -> {metrics_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal.onnx trên tập val")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-phase1", default="models/phase1_checkpoint.pt")
    parser.add_argument("--temporal-onnx", default="models/temporal.onnx")
    parser.add_argument("--labels-csv", default="data/raw/labels.csv")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--metrics", default="reports/evaluate_metrics.json")
    args = parser.parse_args()
    run(
        args.config,
        args.checkpoint_phase1,
        args.temporal_onnx,
        args.labels_csv,
        args.processed_root,
        args.metrics,
    )


if __name__ == "__main__":
    main()
