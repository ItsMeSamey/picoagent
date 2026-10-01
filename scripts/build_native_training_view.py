#!/usr/bin/env python3
"""Rebuild the reviewed compact-plan view without running commands or a model.

Uses an already sealed native collection and cached, revision-pinned tokenizer.
No external dataset download, teacher sampling or accelerator allocation occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from picoagent.data.artificial_plans import build_plan_snapshot  # noqa: E402
from picoagent.data.audit import file_hash  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/native-training-collection-v1/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/native-training-plans-v1"))
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    review = json.loads(Path("data/native-source-reviews/artificial-cli-plans-v2.json").read_text())
    identity = review["tokenizer_identity"]
    asset = hf_hub_download(identity["model_id"], "tokenizer.json", revision=identity["revision"],
                            local_files_only=True, token=False)
    if file_hash(asset) != identity["tokenizer_json_sha256"]:
        raise ValueError("cached tokenizer differs from independently reviewed tokenizer bytes")
    tokenizer = AutoTokenizer.from_pretrained(str(Path(asset).parent),
                    use_fast=True, trust_remote_code=False, local_files_only=True)
    source = Path("data/artificial-reasoning-cli-v2")
    notes = [json.loads(line) for line in (source / "train.notes.jsonl").read_text().splitlines()]
    templates = {}
    for line in (source / "templates.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["template_id"] in templates:
            raise ValueError("duplicate literal template")
        templates[item["template_id"]] = item
    path = build_plan_snapshot(args.base, notes, templates, args.output,
                note_counter=lambda value: len(tokenizer.encode(value, add_special_tokens=False)),
                tokenizer_identity=identity, template_review=review)
    manifest = json.loads(path.read_text())
    print(json.dumps({"manifest": str(path), "sha256": file_hash(path),
                      "counts": manifest["counts"],
                      "annotated_train_tasks": manifest["annotated_train_tasks"]}, sort_keys=True))


if __name__ == "__main__":
    main()
