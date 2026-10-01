"""python -m picoagent.data {generate,validate,collect,export,verify-attempt}."""
from __future__ import annotations

import argparse

from .audit import read_jsonl, verify_attempt, verify_curriculum, write_curriculum
from .collector import collect_task, export_attempts
from .generators import generate_tasks
from .schema import canonical_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate original specs and unexecuted authored examples")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--seeds-per-family", type=int, default=4)
    generate.add_argument("--seed-start", type=int, default=0)
    generate.add_argument("--holdout-seeds-per-family", type=int, help="Separate dev/test count per family")
    tool_generate = sub.add_parser("generate-tools", help="Generate separate installed-help and original module/source tasks")
    tool_generate.add_argument("--output-dir", required=True)
    tool_generate.add_argument("--seeds-per-family", type=int, default=8)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    collect = sub.add_parser("collect", help="Collect actual scripted-teacher environment traces; requires Docker/Podman")
    collect.add_argument("--tasks", required=True)
    collect.add_argument("--archive-dir", required=True)
    collect.add_argument("--image", default="python:3.11-slim")
    collect.add_argument("--runtime", choices=("docker", "podman"))
    collect.add_argument("--max-steps", type=int, default=16)
    collect.add_argument("--limit", type=int)
    collect.add_argument("--include-test", action="store_true", help="Explicitly unlock test-family collection; never feed these to training")
    export = sub.add_parser("export")
    export.add_argument("--archive-dir", required=True)
    export.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify-attempt")
    verify.add_argument("--attempt-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        tasks = generate_tasks(seeds_per_family=args.seeds_per_family, seed_start=args.seed_start, holdout_seeds_per_family=args.holdout_seeds_per_family)
        manifest = write_curriculum(args.output_dir, tasks, configuration={"seeds_per_family": args.seeds_per_family, "seed_start": args.seed_start, "holdout_seeds_per_family": args.holdout_seeds_per_family})
        print(canonical_json({"manifest": str(manifest), "tasks": len(tasks), "verified_traces": 0}))
    elif args.command == "generate-tools":
        from .tool_curriculum import TOOL_SPLIT_POLICY, generate_tool_tasks
        tasks = generate_tool_tasks(seeds_per_family=args.seeds_per_family)
        manifest = write_curriculum(args.output_dir, tasks, configuration={"track": "installed_help_and_original_api_v1", "seeds_per_family": args.seeds_per_family}, split_policy=TOOL_SPLIT_POLICY)
        print(canonical_json({"manifest": str(manifest), "tasks": len(tasks), "verified_traces": 0}))
    elif args.command == "validate":
        print(canonical_json(verify_curriculum(args.manifest)))
    elif args.command == "collect":
        tasks = read_jsonl(args.tasks)
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be positive")
            tasks = tasks[:args.limit]
        if not tasks:
            parser.error("task file is empty")
        if any(task.get("split") == "test" for task in tasks) and not args.include_test:
            parser.error("test-family collection requires --include-test; test data is never eligible for SFT")
        failures = 0
        for task in tasks:
            result = collect_task(task, args.archive_dir, image=args.image, runtime=args.runtime, max_steps=args.max_steps)
            trace = result["trace"]
            print(canonical_json({"task_id": task["task_id"], "status": trace["status"], "attempt_path": result["attempt_path"], "verification": trace["verification"]}), flush=True)
            failures += trace["status"] != "success"
            if trace["provenance"]["execution"] == "unexecuted":
                # No point spawning every remaining episode against an unavailable
                # runtime. The one preserved error is explicit and command fails.
                return 2
        return 1 if failures else 0
    elif args.command == "export":
        print(canonical_json(export_attempts(args.archive_dir, args.output_dir)))
    else:
        print(canonical_json(verify_attempt(args.attempt_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
