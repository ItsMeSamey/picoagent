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
    tokens = commands.add_parser("prepare-tokens", help="CPU-only strict verification and exact token preparation; does not approve its output")
    tokens.add_argument("--config", required=True)
    tokens.add_argument("--output-dir", required=True)
    tokens.add_argument("--tokenizer-path", help="Pinned local tokenizer assets; useful for offline CPU preparation")
    validate = commands.add_parser("validate", help="Recheck hashes, admission and split separation without ML dependencies")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--allow-native-teacher-observed", action="store_true",
                          help="Explicitly allow audited native teacher snapshots; not arbitrary native execution")
    validate.add_argument("--allow-artificial-action-plans", action="store_true")
    train = commands.add_parser("train", help="Run full SFT or distinctly labeled QLoRA on your already configured machine")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", default=None, help="Verified checkpoint in the original run directory; immutable config must match")
    train.add_argument("--finalize-only", action="store_true", help="With --resume at the exact final checkpoint, evaluate/export with zero optimizer updates")
    train.add_argument("--segment-steps", type=int, help="Operational maximum optimizer updates before a sealed-checkpoint pause; does not alter the configured full-run schedule")
    train.add_argument("--continue-through-checkpoints", action="store_true", help="Keep ordinary/timed checkpoints and train to the exact --segment-steps cap; default pauses at the first sealed checkpoint")
    train.add_argument("--output-budget-bytes", type=int, help="Fail-closed bound for current outputs plus one checkpoint, final model, and margin; requires --segment-steps")
    train.add_argument("--durability-timeout-seconds", type=float, help="Wait after each sealed checkpoint for a verified off-runtime backup acknowledgement; timeout stops training; required on every resume once enabled for a new run")
    train.add_argument("--output-budget-root", help="Directory whose complete saved-output tree is included in --output-budget-bytes")
    smoke = commands.add_parser("smoke", help="Optional offline CPU random-model pipeline check; not agent evaluation")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda", "xla"), default="cpu")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = {"manifest": str(prepare_dataset(args.train, args.dev, args.output_dir))}
    elif args.command == "prepare-tokens":
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from .prepared import build_prepared_dataset
        from .data import sha256_file
        path = build_prepared_dataset(TrainingConfig.load(args.config), args.output_dir,
                                      tokenizer_path=args.tokenizer_path)
        result = {"manifest": str(path), "sha256": sha256_file(path),
                  "production_approved": False, "next_step": "independent CPU rebuild/equality review, then pin approved digest in a new run config"}
    elif args.command == "validate":
        manifest, rows = verify_dataset(args.manifest,
                                         allow_native_teacher=args.allow_native_teacher_observed,
                                         allow_artificial_action_plans=args.allow_artificial_action_plans)
        result = {"verified": True, "records": {split: len(records) for split, records in rows.items()}, "lockbox_used": False}
    elif args.command == "train":
        from .train import run_training
        result = run_training(TrainingConfig.load(args.config), resume_from_checkpoint=args.resume,
                              segment_steps=args.segment_steps, output_budget_bytes=args.output_budget_bytes,
                              output_budget_root=args.output_budget_root,
                              continue_through_checkpoints=args.continue_through_checkpoints,
                              durability_timeout_seconds=args.durability_timeout_seconds,
                              finalize_only=args.finalize_only)
    else:
        from .smoke import run_smoke
        result = run_smoke(args.output_dir, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
