"""CLI wrapper mỏng cho monitoring/drift_report.py. Không chứa logic —
logic nằm ở src/ (vá vi phạm quy ước phát hiện ở phiên 12.2)."""

from __future__ import annotations

import argparse
import os

from video_action_mlops.monitoring.drift_report import build_drift_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh drift report (Evidently) so val vs recent")
    parser.add_argument("--reference-csv", default="reports/reference_stats.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--recent-limit", type=int, default=500)
    args = parser.parse_args()

    # Check DATABASE_URL SAU khi parse_args() -- de --help van chay duoc
    # ngay ca khi chua set bien moi truong nay (bug that: ban truoc kiem
    # tra truoc argparse, --help cung crash vi thieu DATABASE_URL).
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Thiếu biến môi trường DATABASE_URL.")

    out_path = build_drift_report(
        reference_csv=args.reference_csv,
        database_url=database_url,
        output_dir=args.output_dir,
        recent_limit=args.recent_limit,
    )
    print(f"[drift_report] -> {out_path}")


if __name__ == "__main__":
    main()
