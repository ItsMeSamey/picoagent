"""Run a locally saved or pinned HF policy with the shared tool harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--image", required=True, help="Operator-built sandbox image (prefer digest)")
    parser.add_argument("--runtime", choices=["docker", "podman"])
    parser.add_argument("--search-url", help="Optional SearXNG JSON endpoint")
    parser.add_argument("--output", default="runs/inference")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    from picoagent.harness import AgentHarness, ContainerSandbox, ContextManager, KnowledgeStore, ToolRegistry
    from picoagent.harness.search import SearXNGSearch
    from picoagent.inference import HFPolicy

    episode = Path(args.output) / uuid.uuid4().hex
    episode.mkdir(parents=True, exist_ok=False)
    # Fail closed before loading expensive weights if isolation is unavailable.
    with ContainerSandbox(image=args.image, runtime=args.runtime) as sandbox:
        policy = HFPolicy(args.model, revision=args.revision, max_new_tokens=args.max_new_tokens)
        tools = ToolRegistry(sandbox, KnowledgeStore(episode / "knowledge.json"),
                             SearXNGSearch(args.search_url) if args.search_url else None)
        context = ContextManager(
            policy, max_tokens=args.context, reserve_tokens=args.max_new_tokens,
            token_counter=lambda messages: policy.count_tokens(messages, tools.schemas),
        )
        harness = AgentHarness(policy, tools, context=context, max_steps=args.max_steps,
                               trace_path=episode / "events.jsonl")
        result = harness.run(args.prompt)
        (episode / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(result.final or f"Stopped: {result.stop_reason}; {result.error or ''}")
        print(f"Trace: {episode}")


if __name__ == "__main__":
    main()
