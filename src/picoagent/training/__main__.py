"""CLI entry point: python -m picoagent.training --help."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import TrainingConfig
from .data import prepare_dataset, verify_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audited local SFT. No GPU provisioning or automatic benchmark runs.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze verified train/dev JSONL inputs; excludes lockbox and authored examples")
    prepare.add_argument("--train", required=True)
    prepare.add_argument("--dev", required=True)
    prepare.add_argument("--output-dir", required=True)
    validate = commands.add_parser("validate", help="Recheck hashes, admission and split separation without ML dependencies")
    validate.add_argument("--manifest", required=True)
    train = commands.add_parser("train", help="Run full SFT or distinctly labeled QLoRA on your already configured machine")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", default=None, help="Verified checkpoint in the original run directory; immutable config must match")
    smoke = commands.add_parser("smoke", help="Optional offline CPU random-model pipeline check; not agent evaluation")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda", "xla"), default="cpu")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = {"manifest": str(prepare_dataset(args.train, args.dev, args.output_dir))}
    elif args.command == "validate":
        manifest, rows = verify_dataset(args.manifest)
        result = {"verified": True, "records": {split: len(records) for split, records in rows.items()}, "lockbox_used": False}
    elif args.command == "train":
        from .train import run_training
        result = run_training(TrainingConfig.load(args.config), resume_from_checkpoint=args.resume)
    else:
        from .smoke import run_smoke
        result = run_smoke(args.output_dir, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
