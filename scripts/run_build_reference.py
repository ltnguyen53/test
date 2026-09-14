"""CLI wrapper mỏng cho monitoring/build_reference.py. Không chứa logic —
logic nằm ở src/ (vá lại vi phạm quy ước đã phát hiện ở phiên 12.2: file
gốc từng có argparse/main() ngay trong src/, sai quy ước từ phiên 4.2)."""

from __future__ import annotations

import argparse

from video_action_mlops.monitoring.build_reference import build_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference window cho drift report")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--spatial-onnx", default="models/spatial.onnx")
    parser.add_argument("--temporal-onnx", default="models/temporal.onnx")
    parser.add_argument("--output", default="reports/reference_stats.csv")
    args = parser.parse_args()
    build_reference(args.config, args.raw_dir, args.spatial_onnx, args.temporal_onnx, args.output)


if __name__ == "__main__":
    main()
