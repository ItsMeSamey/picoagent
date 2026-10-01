"""Genuine rollout collection through the shared fail-closed container harness.

No model-produced command is run by this module on the host. A scripted teacher
is available to bootstrap original tool demonstrations; its identity is explicit.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Callable

from .audit import AttemptArchive, read_jsonl, verify_attempt, write_new_json
from .generators import GENERATOR_VERSION, SYSTEM_PROMPT
from .oracles import check_task_result
from .schema import canonical_json, content_hash, validate_task, validate_trace


class LocalCorpusSearch:
    """Deterministic retrieval over one episode's original fixture corpus only."""
    def __init__(self, documents: list[dict[str, str]]):
        self.documents = copy.deepcopy(documents)

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 10:
            raise ValueError("invalid local search query/limit")
        tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
        ranked = []
        for doc in self.documents:
            words = set(re.findall(r"[a-z0-9_]+", (doc["title"] + " " + doc["content"]).lower()))
            score = len(tokens & words)
            if score:
                ranked.append((-score, doc["id"], doc))
        ranked.sort(key=lambda row: (row[0], row[1]))
        return {"query": query, "results": [{"id": doc["id"], "title": doc["title"], "url": "local://docs/" + doc["id"],
                                               "content": doc["content"]} for _, _, doc in ranked[:limit]],
                "source": "original_fixture_corpus", "untrusted": True}


class ScriptedTeacher:
    """Execute a hand-authored solution plan; never synthesize a tool reply.

    This is a procedural teacher, not a learned policy or a benchmark score. The
    actual harness supplies every tool result. Any failed step stops the plan.
    """
    def __init__(self, task: dict[str, Any]):
        self.plan = copy.deepcopy(task["reference"]["plan"])
        self.final = task["reference"]["final"]
        self.domain = task["domain"]
        self.family = task["family"]
        self.prompt = task["prompt"]
        self.step = 0

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        if messages and messages[-1]["role"] == "tool":
            result = json.loads(messages[-1]["content"])
            if ("error" in result or result.get("exit_code", 0) != 0 or result.get("timed_out") or result.get("truncated")):
                return {"role": "assistant", "content": "The tool step failed; the task was not completed."}
        observations = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if self.step >= len(self.plan):
            return {"role": "assistant", "content": self._final_from_observations(observations)}
        action = copy.deepcopy(self.plan[self.step])
        if action.get("derive_local_api_from_observation"):
            module = re.search(r"Original module (local_api_[a-f0-9]+)\.py", self.prompt).group(1)
            documentation = observations[-1]["stdout"]
            kind = action["derive_local_api_from_observation"]
            if kind == "affine":
                function = re.search(r"API callable: ([a-zA-Z0-9_]+)", documentation).group(1)
                expression = f"[{module}.{function}(value) for value in values]"
            elif kind == "index":
                origin = int(re.search(r"positions are ([01])-based", documentation).group(1))
                expression = f"[{module}.lookup(values, {origin + 2})]"
            else:
                convention = re.search(r"stop is (inclusive|exclusive)", documentation).group(1)
                endpoint = 4 if convention == "inclusive" else 5
                expression = f"{module}.select(values, 2, {endpoint})"
            action["arguments"]["code"] = f'import json\nimport {module}\nvalues=json.load(open("input/values.json"))\nprint(json.dumps({{"values":{expression}}}))'
        if action.get("derive_value_from_observations"):
            action["arguments"]["value"] = self._kv_value(observations)
        if self.domain == "docs" and self.step == 1:
            doc_id = observations[-1]["results"][0]["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_]+", doc_id):
                raise ValueError("unexpected fixture document ID")
            action["arguments"]["command"] = f"cat docs/{doc_id}.md"
        if self.family == "search.two_hop_lookup" and self.step == 1:
            content = observations[-1]["results"][0]["content"]
            action["arguments"]["query"] = re.search(r"follows protocol ([a-zA-Z0-9_]+)", content).group(1)
        self.step += 1
        return {"role": "assistant", "content": "", "tool_calls": [{"id": f"teacher_call_{self.step}", "type": "function",
                "function": {"name": action["name"], "arguments": canonical_json(action["arguments"])}}]}


    def _kv_value(self, observations: list[dict]) -> int:
        values = [row["value"] for row in observations if "key" in row and "value" in row]
        if self.family == "kv.copy_value":
            return values[-1]
        if self.family == "kv.counter_update":
            increment = int(re.search(r", add ([0-9]+),", self.prompt).group(1))
            return values[-1] + increment
        return sum(values[-2:])

    def _final_from_observations(self, observations: list[dict]) -> str:
        # No fixture-dependent final answer is copied from the private oracle.
        if self.domain in {"instruction", "math"}:
            return self.final  # The complete problem is in the student prompt.
        if self.domain in {"bash", "python"}:
            return observations[-1]["stdout"].strip()
        if self.domain == "visualization":
            return "Wrote output/chart.svg."
        if self.domain == "kv":
            key = {"kv.copy_value": "copied", "kv.counter_update": "count", "kv.merge_values": "total"}[self.family]
            return canonical_json({key: self._kv_value(observations)})
        if self.domain == "docs":
            manual = observations[-1]["stdout"]
            if self.family == "docs.flag_lookup":
                tool = re.search(r"documentation for ([a-zA-Z0-9_]+)", self.prompt).group(1)
                count = re.search(r"for ([0-9]+) workers", self.prompt).group(1)
                workers = re.search(r"(--[a-z-]+) N sets worker count", manual).group(1)
                output = re.search(r"(--[a-z-]+) json selects JSON output", manual).group(1)
                preview = re.search(r"(--[a-z-]+) previews work", manual).group(1)
                return canonical_json({"argv": [tool, workers, count, output, "json", preview]})
            if self.family == "docs.default_override":
                workers = int(re.search(r"Override workers to ([0-9]+)", self.prompt).group(1))
                retries = int(re.search(r"retries=([0-9]+)", manual).group(1))
                return canonical_json({"workers": workers, "format": "json", "retries": retries})
            recipe = re.search(r"mandatory order: ([a-z, ]+)\.", manual).group(1)
            return canonical_json({"steps": recipe.split(", ")})
        station = re.search(r"station_[a-f0-9]+", self.prompt).group(0)
        docs = {doc["id"]: doc for row in observations for doc in row.get("results", [])}
        primary = next(doc for doc in docs.values() if station in doc["content"])
        if self.family == "search.two_hop_lookup":
            second_id = re.search(r"follows protocol ([a-zA-Z0-9_]+)", primary["content"]).group(1)
            signal = re.search(r"signal to ([a-z]+)", docs[second_id]["content"]).group(1)
            return canonical_json({"signal": signal, "sources": [primary["id"], second_id]})
        if self.family == "search.exception_lookup":
            signal = re.search(r"use ([a-z]+)\.", primary["content"]).group(1)
        else:
            signal = re.search(r" is ([a-z]+)\.", primary["content"]).group(1)
        return canonical_json({"signal": signal, "source": primary["id"]})


def _full_transcript(prompt: str, events: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    for event in events:
        if event.get("type") == "assistant":
            messages.append(event["message"])
        elif event.get("type") == "tool_execution":
            messages.append({"role": "tool", "name": event["name"], "tool_call_id": event["tool_call_id"], "content": canonical_json(event["result"])})
    return messages


def collect_task(task: dict[str, Any], archive_root: str | Path, *, model: Callable | None = None,
                 teacher_name: str | None = None, image: str = "python:3.11-slim", runtime: str | None = None,
                 max_steps: int = 16, context_max_tokens: int | None = 4096,
                 context_reserve_tokens: int = 512, token_counter: Callable | None = None) -> dict[str, Any]:
    """Preserve one complete attempt, whether success, error, or unexecuted.

    A supplied student model receives only canonical messages and tool schemas,
    never the oracle/reference object. There is intentionally no local fallback.
    Exceptions are archived and returned as error traces; KeyboardInterrupt and
    process death leave an inspectable unfinished attempt directory/event log.
    """
    from picoagent.harness.agent import AgentHarness
    from picoagent.harness.context import ContextManager
    from picoagent.harness.knowledge import KnowledgeStore
    from picoagent.harness.sandbox import ContainerSandbox
    from picoagent.harness.tools import ToolRegistry

    validate_task(task)
    if task["provenance"].get("generator_version") != GENERATOR_VERSION:
        raise ValueError("task generator version is excluded or unsupported; regenerate the admitted curriculum")
    teacher = teacher_name or ("scripted_procedural_v1" if model is None else "provided_model")
    archive = AttemptArchive(archive_root, task, teacher=teacher)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]
    raw: dict[str, Any] = {"task_id": task["task_id"], "teacher": teacher, "result": None, "probe": None, "artifacts": {}, "kv": {}, "error": None}
    verification: dict[str, Any] = {"passed": False, "failures": ["attempt did not reach oracle"], "checks": []}
    provenance = {**task["provenance"], "execution": "unexecuted", "teacher": teacher, "context_compaction_enabled": False}
    status, events = "error", []
    callback = model if model is not None else ScriptedTeacher(task)

    def observed_model(model_messages: list[dict], tools: list[dict]) -> dict:
        # Journal before validation so malformed/rejected model output is retained.
        archive.event("model_request", {"messages": model_messages, "tools": tools})
        result = callback(model_messages, tools)
        archive.event("model_response", result)
        return result

    backend = None
    try:
        backend = ContainerSandbox(image=image, runtime=runtime)
        backend.seed_files(task["environment"]["files"])
        raw["probe"] = backend.probe().to_dict()
        metadata = backend.runtime_metadata()
        if metadata.get("backend") != "container" or metadata.get("runtime") not in {"docker", "podman"} or not metadata.get("container_id"):
            raise RuntimeError("runtime returned no trustworthy container lifecycle receipt")
        provenance.update(execution="verified_environment", runtime={"backend": metadata["runtime"], "container_id": metadata["container_id"], "image": metadata["image"]})
        archive.event("runtime_probe", raw["probe"])
        knowledge = KnowledgeStore(archive.path / "knowledge.json")
        for key, value in task["environment"]["kv"].items():
            knowledge.set(key, value)
        registry = ToolRegistry(backend, knowledge, search_client=LocalCorpusSearch(task["environment"]["docs"]))
        context = None
        if model is not None and context_max_tokens is not None:
            counter = token_counter
            if counter is None and hasattr(model, "count_tokens"):
                def counter(rows):
                    return model.count_tokens(rows, registry.schemas)
            if counter is None:
                raise ValueError("model collection with compaction requires the policy tokenizer; pass token_counter or explicitly context_max_tokens=None")
            context = ContextManager(observed_model, max_tokens=context_max_tokens, reserve_tokens=context_reserve_tokens, token_counter=counter)
        provenance["context_compaction_enabled"] = context is not None
        provenance["context_budget"] = {"max_tokens": context_max_tokens, "reserve_tokens": context_reserve_tokens} if context else None
        harness = AgentHarness(observed_model, registry, context=context, max_steps=max_steps, trace_path=archive.path / "harness.jsonl", system_prompt=SYSTEM_PROMPT)
        result = harness.run(task["prompt"])
        raw["result"] = result.to_dict()
        events = result.events
        messages = _full_transcript(task["prompt"], events)
        raw["effective_messages"] = result.messages
        provenance["accepted_compactions"] = sum(event.get("type") == "compaction" and event.get("accepted") is True for event in events)
        artifact_path = task["oracle"].get("artifact_path")
        if artifact_path:
            try:
                raw["artifacts"][artifact_path] = backend.read_file(artifact_path)
            except (ValueError, RuntimeError, OSError) as exc:
                raw["artifact_error"] = f"{type(exc).__name__}: {exc}"
        raw["kv"] = knowledge.list()
        verification = check_task_result(task, result.final, artifacts=raw["artifacts"], kv=raw["kv"])
        if result.stop_reason != "final":
            verification["passed"] = False
            verification["failures"].append("harness stopped without final answer: " + result.stop_reason)
        status = "success" if verification["passed"] else "failed"
    except Exception as exc:
        raw["error"] = {"type": type(exc).__name__, "message": str(exc)}
        archive.event("exception", raw["error"])
        verification = {"passed": False, "checks": [], "failures": [f"{type(exc).__name__}: {exc}"]}
        # Preserve valid partial messages after a provider or dispatch exception.
        journal = archive.path / "harness.jsonl"
        if journal.exists():
            events = read_jsonl(journal)
            messages = _full_transcript(task["prompt"], events)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                raw["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                archive.event("cleanup_error", raw["cleanup_error"])
    raw["events"] = events
    trace = {"schema_version": task["schema_version"], "trace_id": "attempt:" + archive.attempt_id,
             "task_id": task["task_id"], "family": task["family"], "template_id": task["template_id"], "split": task["split"],
             "task_sha256": content_hash(task), "status": status, "provenance": provenance, "messages": messages,
             "effective_messages": raw.get("effective_messages", messages),
             "tools": registry.schemas if "registry" in locals() else [],
             "tool_events": [event for event in events if event.get("type") == "tool_execution"],
             "model_events": [event for event in events if event.get("type") in {"assistant", "compaction"}],
             "verification": verification}
    trace_path = archive.finalize(raw, trace)
    return {"trace": trace, "trace_path": str(trace_path), "attempt_path": str(archive.path)}


def export_attempts(archive_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Export every valid trace plus a deterministic success-only admission view.

    Retries are all preserved. Exactly one successful attempt per task (smallest
    immutable attempt ID) enters the admission file to avoid overweighting retries.
    Failed/unexecuted traces never enter admitted train/dev/test files.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    all_traces, incomplete = [], []
    for directory in sorted(Path(archive_root).iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "manifest.json").exists():
            incomplete.append(directory.name)
            continue
        verify_attempt(directory)
        if not (directory / "trace.json").exists():
            incomplete.append(directory.name)
            continue
        trace = json.loads((directory / "trace.json").read_text(encoding="utf-8"))
        validate_trace(trace)
        all_traces.append(trace)
    admitted: dict[str, dict[str, Any]] = {}
    for trace in all_traces:
        if trace["status"] == "success" and trace["provenance"]["execution"] == "verified_environment" and trace["verification"]["passed"]:
            admitted.setdefault(trace["task_id"], trace)
    files = {"all_attempts.jsonl": all_traces}
    files.update({f"{split}.jsonl": sorted([t for t in admitted.values() if t["split"] == split], key=lambda row: row["task_id"]) for split in ("train", "dev", "test")})
    for filename, rows in files.items():
        with (output / filename).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
    summary = {"attempts": len(all_traces), "admitted": len(admitted), "incomplete_or_invalid_attempts": incomplete,
               "counts": {filename: len(rows) for filename, rows in files.items()}, "deduplication": "lexicographically_first_successful_attempt_id_per_task"}
    write_new_json(output / "export.json", summary)
    return summary
