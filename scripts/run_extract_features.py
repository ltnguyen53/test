"""CLI wrapper mỏng cho stage `extract_features`. Không chứa logic — logic
nằm ở features/extract_embeddings.py."""

from __future__ import annotations

import argparse

from video_action_mlops.config.loader import load_config
from video_action_mlops.features.extract_embeddings import extract_embeddings


def run(config_path: str, checkpoint_path: str, interim_dir: str, processed_root: str) -> None:
    cfg = load_config(config_path)
    out_dir = extract_embeddings(cfg, checkpoint_path, interim_dir, processed_root)
    print(f"[extract_features] cache dir -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract phase1 embeddings (tầng cache thứ 2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="models/phase1_checkpoint.pt")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-root", default="data/processed")
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.interim_dir, args.processed_root)


if __name__ == "__main__":
    main()
