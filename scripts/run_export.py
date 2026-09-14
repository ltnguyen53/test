"""CLI wrapper mỏng cho stage `export`. Export + verify parity NGAY trong
cùng 1 script — không tách 2 script riêng, vì "export xong mà chưa verify"
không phải trạng thái hợp lệ để dừng lại (mục 3.3/verify_parity.py, phiên 7.1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_action_mlops.config.loader import load_config
from video_action_mlops.export.to_onnx import export_spatial_onnx, export_temporal_onnx
from video_action_mlops.export.verify_parity import verify_spatial_parity, verify_temporal_parity


def run(
    config_path: str,
    checkpoint_phase1: str,
    checkpoint_phase2: str,
    spatial_onnx_out: str,
    temporal_onnx_out: str,
    report_out: str,
) -> None:
    cfg = load_config(config_path)

    export_spatial_onnx(cfg, checkpoint_phase1, spatial_onnx_out)
    export_temporal_onnx(cfg, checkpoint_phase2, temporal_onnx_out)

    # Raise ParityError ngay nếu lệch quá atol — dừng cả dvc stage, không để
    # file .onnx sai lọt qua sang stage evaluate.
    spatial_diff = verify_spatial_parity(cfg, checkpoint_phase1, spatial_onnx_out)
    temporal_diff = verify_temporal_parity(cfg, checkpoint_phase2, temporal_onnx_out)

    report = {"spatial_max_diff": spatial_diff, "temporal_max_diff": temporal_diff}
    report_path = Path(report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[export] spatial max_diff={spatial_diff:.2e}")
    print(f"[export] temporal max_diff={temporal_diff:.2e}")
    print(f"[export] report -> {report_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX + verify parity (phase1+phase2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-phase1", default="models/phase1_checkpoint.pt")
    parser.add_argument("--checkpoint-phase2", default="models/phase2_checkpoint.pt")
    parser.add_argument("--spatial-onnx-out", default="models/spatial.onnx")
    parser.add_argument("--temporal-onnx-out", default="models/temporal.onnx")
    parser.add_argument("--report", default="reports/export_parity.json")
    args = parser.parse_args()
    run(
        args.config,
        args.checkpoint_phase1,
        args.checkpoint_phase2,
        args.spatial_onnx_out,
        args.temporal_onnx_out,
        args.report,
    )


if __name__ == "__main__":
    main()
