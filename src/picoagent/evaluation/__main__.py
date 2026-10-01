"""Evaluate an actual policy in all three modes; no teacher fallback."""
import argparse
import json
from pathlib import Path

from picoagent.data.audit import read_jsonl
from picoagent.data.schema import content_hash
from picoagent.evaluation import evaluate
from picoagent.harness.sandbox import ContainerSandbox
from picoagent.inference import HFPolicy
from picoagent.training.provenance import tree_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime", choices=["docker", "podman"])
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()
    tasks = read_jsonl(args.tasks)
    # Fail before expensive weight loading if the isolated runtime is absent.
    with ContainerSandbox(image=args.image, runtime=args.runtime) as sandbox:
        sandbox.probe()
    path = Path(args.model)
    checkpoint = ("sha256:" + content_hash(tree_hashes(path)) if path.is_dir()
                  else f"{args.model}@{args.revision}")
    policy = HFPolicy(args.model, revision=args.revision,
                      max_new_tokens=args.max_new_tokens, seed=args.seed)
    report = evaluate(tasks, policy=policy, checkpoint=checkpoint,
                      archive_root=args.output, image=args.image, runtime=args.runtime,
                      max_tokens=args.context, reserve_tokens=args.max_new_tokens,
                      max_steps=args.max_steps, unlock_test=args.unlock_test)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
