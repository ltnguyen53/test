"""CLI wrapper mỏng cho stage `train_phase2`. Không chứa logic — logic nằm
ở training/phase2_trainer.py. Dùng resolve_cache_dir() (features/
extract_embeddings.py, phiên 5.1) để tìm ĐÚNG thư mục cache đã ghi, không
tự đoán đường dẫn.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from video_action_mlops.config.loader import load_config
from video_action_mlops.data.datasets import Phase2SequenceDataset
from video_action_mlops.features.extract_embeddings import resolve_cache_dir
from video_action_mlops.training.callbacks import (
    configure_mlflow,
    get_or_create_parent_run,
    log_phase_result,
    register_model_at_parent,
)
from video_action_mlops.training.phase2_trainer import train_phase2


def run(
    config_path: str,
    checkpoint_phase1: str,
    labels_csv: str,
    processed_root: str,
    checkpoint_out: str,
    metrics_out: str,
    resume_state_path: str | None = None,
) -> None:
    cfg = load_config(config_path)
    processed_dir = resolve_cache_dir(cfg, checkpoint_phase1, processed_root)
    train_dataset = Phase2SequenceDataset(
        processed_dir=processed_dir, labels_csv=labels_csv, split="train"
    )
    val_dataset = Phase2SequenceDataset(
        processed_dir=processed_dir, labels_csv=labels_csv, split="val"
    )

    result = train_phase2(
        cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=checkpoint_out,
        resume_state_path=resume_state_path,
    )

    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result["final"], f, indent=2, ensure_ascii=False)

    print(f"[train_phase2] đọc cache từ -> {processed_dir}")
    print(f"[train_phase2] checkpoint -> {checkpoint_out}")
    print(f"[train_phase2] metrics -> {metrics_out}: {result['final']}")

    # MLflow: get_or_create_parent_run() ở đây sẽ ĐỌC LẠI run_id đã ghi bởi
    # run_train_phase1.py (file reports/mlflow_parent_run_id.txt) — KHÔNG
    # tạo run cha mới, đúng yêu cầu "cả cặp checkpoint dưới 1 run cha" (mục
    # 3.2). Nếu file đó chưa tồn tại (ai đó chạy train_phase2 riêng lẻ mà
    # bỏ qua train_phase1), hàm sẽ tự tạo 1 run cha mới — TỰ NÓ THÀNH 1 CẶP
    # LẺ, không sai kỹ thuật nhưng mất ý nghĩa "cặp" — cần cảnh giác.
    configure_mlflow()
    parent_run_id = get_or_create_parent_run(cfg)
    log_phase_result(
        phase_name="phase2",
        parent_run_id=parent_run_id,
        params={
            "input_dim": cfg.phase2.input_dim,
            "num_heads": cfg.phase2.num_heads,
            "epochs": cfg.phase2.epochs,
            "learning_rate": cfg.phase2.learning_rate,
        },
        history=result["history"],
        checkpoint_path=checkpoint_out,
    )

    # ĐĂNG KÝ vào Model Registry Ở ĐÂY — sau train_phase2 (luôn chạy SAU
    # train_phase1 trong dvc.yaml), tức là điểm sớm nhất cả CẶP checkpoint
    # đã log xong dưới CÙNG 1 run cha (mục 3.2). Đăng ký sớm hơn (sau
    # phase1) sẽ tạo version registry chỉ có 1 nửa cặp — sai đúng điều mục
    # 3.2 cảnh báo tránh.
    version = register_model_at_parent(parent_run_id, model_name=cfg.project.name)
    print(f"[train_phase2] đăng ký Model Registry: {cfg.project.name} version {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phase 2 (temporal aggregator)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-phase1", default="models/phase1_checkpoint.pt")
    parser.add_argument("--labels-csv", default="data/raw/labels.csv")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--checkpoint-out", default="models/phase2_checkpoint.pt")
    parser.add_argument("--metrics", default="reports/phase2_metrics.json")
    parser.add_argument(
        "--resume-state",
        default=str(Path(os.environ.get("RESUME_DIR", "models")) / "phase2_resume_state.pt"),
        help=(
            "Duong dan file resume state (checkpoint resume - phien 13.2), "
            "cung thiet ke voi --resume-state cua run_train_phase1.py."
        ),
    )
    args = parser.parse_args()
    run(
        args.config,
        args.checkpoint_phase1,
        args.labels_csv,
        args.processed_root,
        args.checkpoint_out,
        args.metrics,
        resume_state_path=args.resume_state,
    )


if __name__ == "__main__":
    main()
