#!/usr/bin/env python3
"""Plan and explicitly apply bounded, run-scoped checkpoint archive retention.

Only generated checkpoint directories may be removed. Datasets, source snapshots,
receipts, annotations, manifests and the deletion audit log are never candidates.
The operator must authorize the exact plan before apply; merely making a plan
performs no deletion. This does not prune the runtime or allocate any compute.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from picoagent.training.provenance import verify_checkpoint  # noqa: E402
from picoagent.training.retention import _exclusive_lock, _hash_tree  # noqa: E402
from picoagent.training.data import canonical_json, sha256_bytes, sha256_file  # noqa: E402

NAME = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
SCHEMA = "picoagent.durable_retention_plan.v1"


def plan_retention(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("archive root must be a real existing run directory")
    root = root.resolve()
    manifest = root / "run_manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("archive requires a regular run manifest")
    run_hash = sha256_file(manifest)
    entries, best = [], None
    for path in sorted((p for p in root.iterdir() if NAME.fullmatch(p.name)),
                       key=lambda p: int(p.name.split("-")[1])):
        hashes = _hash_tree(path)
        verify_checkpoint(path, run_hash)
        if any(Path(name).suffix in {".jsonl", ".ndjson"} or "trace" in name.lower()
               for name in hashes):
            raise ValueError("checkpoint contains trajectory data; refusing retention")
        receipt = root / "receipts" / (path.name + ".json")
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError("archive checkpoint lacks its transfer receipt")
        receipt_data = json.loads(receipt.read_text())
        if (receipt_data.get("checkpoint") != path.name
                or receipt_data.get("checkpoint_manifest_sha256") != hashes["checkpoint_manifest.json"]
                or receipt_data.get("run_manifest_sha256") != run_hash
                or receipt_data.get("off_runtime_attested") is not True):
            raise ValueError("archive receipt does not bind verified checkpoint")
        state = json.loads((path / "trainer_state.json").read_text())
        step = int(path.name.split("-")[1])
        if state.get("global_step") != step:
            raise ValueError("checkpoint name and saved training step differ")
        losses = [row["eval_loss"] for row in state.get("log_history", [])
                  if row.get("step") == step and type(row.get("eval_loss")) in {int, float}
                  and math.isfinite(row["eval_loss"])]
        loss = min(losses) if losses else None
        if loss is not None and (best is None or (loss, step) < best[:2]):
            best = (loss, step, path.name)
        entries.append({"name": path.name, "checkpoint_manifest_sha256": hashes["checkpoint_manifest.json"],
                        "tree_sha256": sha256_bytes(canonical_json(hashes).encode()),
                        "receipt_sha256": sha256_file(receipt), "bytes": sum((path / n).stat().st_size for n in hashes),
                        "eval_loss_at_save": loss})
    keep = {row["name"] for row in entries[-2:]}
    if best:
        keep.add(best[2])
    delete = [row for row in entries if row["name"] not in keep]
    return {"schema": SCHEMA, "root": str(root), "run_manifest_sha256": run_hash,
            "policy": "latest_two_plus_best_dev_loss", "checkpoints": entries,
            "keep": sorted(keep), "delete": delete,
            "bytes_reclaimed": sum(row["bytes"] for row in delete),
            "irreversible": True, "preserve": ["datasets", "source", "receipts", "traces", "manifests"]}


def apply_retention(plan: dict, *, confirm_delete_old_checkpoints: bool = False) -> list[str]:
    if not confirm_delete_old_checkpoints:
        raise ValueError("explicit approval of generated-checkpoint deletion is required")
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported retention plan")
    root = Path(plan["root"])
    removed = []
    with _exclusive_lock(root):
        if plan_retention(root) != plan:
            raise ValueError("archive changed since retention plan; review a fresh plan")
        log = root / "checkpoint_retention.jsonl"
        if log.is_symlink():
            raise ValueError("retention log cannot be a symlink")
        digest = sha256_bytes(canonical_json(plan).encode())
        import os
        def record(event):
            with log.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        record({"event": "approved_retention_started", "plan_sha256": digest, "plan": plan})
        for entry in plan["delete"]:
            target = root / entry["name"]
            hashes = _hash_tree(target)
            verify_checkpoint(target, plan["run_manifest_sha256"])
            if sha256_bytes(canonical_json(hashes).encode()) != entry["tree_sha256"]:
                raise ValueError("checkpoint changed before deletion; stop and inspect")
            shutil.rmtree(target)
            removed.append(entry["name"])
            record({"event": "checkpoint_removed", "name": entry["name"],
                    "plan_sha256": digest, "checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"]})
        record({"event": "approved_retention_finished", "plan_sha256": digest, "removed": removed})
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("plan")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--confirm-delete-old-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        plan = plan_retention(args.root)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"delete": [row["name"] for row in plan["delete"]],
                          "keep": plan["keep"], "bytes_reclaimed": plan["bytes_reclaimed"]}))
    else:
        print(json.dumps({"removed": apply_retention(json.loads(args.plan.read_text()),
              confirm_delete_old_checkpoints=args.confirm_delete_old_checkpoints)}))


if __name__ == "__main__":
    main()
