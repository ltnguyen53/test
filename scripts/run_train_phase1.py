"""CLI wrapper mỏng cho stage `train_phase1`: load config, dựng
Phase1FrameDataset từ data/interim/ + labels.csv, gọi train_phase1(), ghi
reports/phase1_metrics.json. Không chứa logic training — logic nằm ở
training/phase1_trainer.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from video_action_mlops.config.loader import load_config
from video_action_mlops.data.datasets import Phase1FrameDataset
from video_action_mlops.training.callbacks import (
    configure_mlflow,
    get_or_create_parent_run,
    log_phase_result,
)
from video_action_mlops.training.phase1_trainer import train_phase1


def run(
    config_path: str,
    interim_dir: str,
    labels_csv: str,
    checkpoint_path: str,
    metrics_path: str,
    resume_state_path: str | None = None,
) -> None:
    cfg = load_config(config_path)
    train_dataset = Phase1FrameDataset(
        interim_dir=interim_dir, labels_csv=labels_csv, split="train"
    )
    val_dataset = Phase1FrameDataset(interim_dir=interim_dir, labels_csv=labels_csv, split="val")

    result = train_phase1(
        cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=checkpoint_path,
        resume_state_path=resume_state_path,
    )

    metrics_out = Path(metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(result["final"], f, indent=2, ensure_ascii=False)

    print(f"[train_phase1] checkpoint -> {checkpoint_path}")
    print(f"[train_phase1] metrics -> {metrics_path}: {result['final']}")

    # MLflow: LUÔN mở/tái sử dụng run cha ở đây (lần train_phase1 chạy TRƯỚC
    # train_phase2 theo đúng thứ tự dvc.yaml) — training/phase1_trainer.py
    # (phiên 4.3) không hề biết mlflow tồn tại, xem docstring callbacks.py.
    configure_mlflow()
    parent_run_id = get_or_create_parent_run(cfg)
    log_phase_result(
        phase_name="phase1",
        parent_run_id=parent_run_id,
        params={
            "embed_dim_out": cfg.phase1.embed_dim_out,
            "epochs": cfg.phase1.epochs,
            "learning_rate": cfg.phase1.learning_rate,
        },
        history=result["history"],
        checkpoint_path=checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phase 1 (spatial backbone)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--labels-csv", default="data/raw/labels.csv")
    parser.add_argument("--checkpoint", default="models/phase1_checkpoint.pt")
    parser.add_argument("--metrics", default="reports/phase1_metrics.json")
    parser.add_argument(
        "--resume-state",
        default=str(Path(os.environ.get("RESUME_DIR", "models")) / "phase1_resume_state.pt"),
        help=(
            "Duong dan file resume state (checkpoint resume - phien 13.2). "
            "Neu file da ton tai (tu lan chay truoc bi ngat), tu resume tiep "
            "thay vi train lai tu dau. Mac dinh doc bien moi truong RESUME_DIR "
            "neu co (vd tro vao Google Drive tren Colab - phien 13.3)."
        ),
    )
    args = parser.parse_args()
    run(
        args.config,
        args.interim_dir,
        args.labels_csv,
        args.checkpoint,
        args.metrics,
        resume_state_path=args.resume_state,
    )


if __name__ == "__main__":
    main()
