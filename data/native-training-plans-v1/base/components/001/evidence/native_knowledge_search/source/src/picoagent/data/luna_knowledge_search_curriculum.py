"""Original train/dev tasks for the real knowledge and local-search tools.

The task families exercise in-process KnowledgeStore persistence and
Collector.LocalCorpusSearch over task-owned fixtures. They do not call a live
search service, execute learner code, or draw from public task corpora.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .generators import GENERATOR_VERSION
from .schema import SCHEMA_VERSION, canonical_json, content_hash, validate_task


TRACK = "luna-native-knowledge-search-v1"
SOURCE_ID = "native_knowledge_search"
FAMILY_SPLIT_POLICY = {
    "kv.note_write_verify": "train",
    "kv.prefix_total": "train",
    "kv.discover_copy": "train",
    "kv.delete_one": "train",
    "kv.counter_update": "dev",
    "kv.prefix_cleanup": "dev",
    "search.fact_lookup": "train",
    "search.ranked_titles": "train",
    "search.two_hop": "train",
    "search.no_match": "train",
    "search.catalog_detail": "dev",
    "search.active_summary": "dev",
}

TRAIN_FAMILIES = tuple(sorted(f for f, split in FAMILY_SPLIT_POLICY.items() if split == "train"))
DEV_FAMILIES = tuple(sorted(f for f, split in FAMILY_SPLIT_POLICY.items() if split == "dev"))
_KEY_SAFE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")


def _rng(family: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{TRACK}:{family}:{seed}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _token(family: str, seed: int) -> str:
    return hashlib.sha256(f"{TRACK}:{family}:{seed}".encode("utf-8")).hexdigest()[:10]


def _key(value: str) -> str:
    if not _KEY_SAFE.fullmatch(value):
        raise ValueError(f"generated unsafe key: {value}")
    return value


def _task_shell(family: str, seed: int, prompt: str, env: dict[str, Any], expected: str,
                kv_expected: dict[str, Any] | None, plan: list[dict[str, Any]]) -> dict[str, Any]:
    split = FAMILY_SPLIT_POLICY[family]
    oracle: dict[str, Any] = {"kind": "text_exact", "expected": expected}
    if kv_expected is not None:
        oracle["kv_expected"] = copy.deepcopy(kv_expected)
    task = {
        "schema_version": SCHEMA_VERSION,
        "task_id": f"native-knowledge-search-v1:{family}:{seed:08d}",
        "family": family,
        "template_id": f"{TRACK}.{family}",
        "domain": family.split(".", 1)[0],
        "split": split,
        "seed": seed,
        "prompt": prompt,
        "environment": env,
        "oracle": oracle,
        "reference": {"plan": plan, "final": expected},
        "provenance": {
            "source": "original_procedural",
            "benchmark": False,
            "generator_version": GENERATOR_VERSION,
            "curriculum_track": TRACK,
            "origin": "Original task-scoped KV and authored local-document fixtures",
            "source_id": SOURCE_ID,
        },
    }
    task["input_sha256"] = content_hash({"prompt": prompt, "environment": env})
    validate_task(task)
    return task


def _make_kv_note(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("kv.note_write_verify", seed)
    key = _key(f"notes/{token}/checkpoint")
    owner = rng.choice(["Mira", "Owen", "Tari", "Niko", "Luz", "Ari"])
    checkpoint = rng.randint(13, 87)
    payload = {"owner": owner, "checkpoint": checkpoint,
               "labels": sorted(rng.sample(["amber", "birch", "coral", "drift", "ember", "fern"], 2))}
    prompt = (f"Workflow: kv.note_write_verify. Store the exact JSON payload {canonical_json(payload)} "
              f"at key {key}, then read it back to verify. Reply in one short English sentence naming "
              f"the key, owner, and checkpoint.")
    expected = f"Saved and verified note {key}: owner {owner}, checkpoint {checkpoint}."
    plan = [
        {"tool": "knowledge", "arguments": {"operation": "set", "key": key, "value": payload}},
        {"tool": "knowledge", "arguments": {"operation": "get", "key": key}},
        {"final_from": "observed_get_result"},
    ]
    return _task_shell("kv.note_write_verify", seed, prompt, {"files": {}, "kv": {}, "docs": []},
                       expected, {key: payload}, plan)


def _make_kv_prefix_total(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("kv.prefix_total", seed)
    prefix = _key(f"ledger/{token}/")
    target = _key(prefix + "rollup")
    entries = {prefix + f"item{i}": {"amount": rng.randint(3, 90), "label": f"line-{i}"}
               for i in range(4)}
    env_kv = {**entries, f"unrelated/{token}/keep": {"amount": 999, "label": "outside"}}
    total = sum(value["amount"] for value in entries.values())
    summary = {"amount_total": total, "source_count": len(entries)}
    prompt = (f"Workflow: kv.prefix_total. List only notes under prefix {prefix}; total their amount fields, "
              f"store {{\"amount_total\":number,\"source_count\":number}} at {target}, then read it back. "
              "Reply in one short English sentence with the total and note count.")
    expected = f"Stored total {total} from {len(entries)} notes at {target}."
    plan = [
        {"tool": "knowledge", "arguments": {"operation": "list", "prefix": prefix}},
        {"derive": "sum_amounts_from_observed_items", "tool": "knowledge", "operation": "set", "key": target},
        {"tool": "knowledge", "arguments": {"operation": "get", "key": target}},
        {"final_from": "observed_get_result"},
    ]
    return _task_shell("kv.prefix_total", seed, prompt, {"files": {}, "kv": env_kv, "docs": []},
                       expected, {target: summary}, plan)


def _make_kv_discover_copy(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("kv.discover_copy", seed)
    prefix = _key(f"archive/{token}/")
    source_primary = _key(prefix + "primary")
    source_backup = _key(prefix + "backup")
    target = _key(f"restored/{token}/selected")
    primary = {"role": "primary", "tag": rng.choice(["north", "violet", "spruce", "delta"]),
               "points": rng.randint(21, 75), "verified": True}
    backup = {"role": "backup", "tag": rng.choice(["east", "copper", "willow", "sigma"]),
              "points": rng.randint(1, 20), "verified": False}
    env_kv = {source_backup: backup, f"other/{token}/private": {"role": "primary", "tag": "outside", "points": 999},
              source_primary: primary}
    prompt = (f"Workflow: kv.discover_copy. List only notes under {prefix}; identify the note whose role is primary, "
              f"copy its complete JSON value to {target}, and read it back. Ignore notes outside that prefix. "
              "Reply in one short English sentence naming the copied tag and points.")
    expected = f"Copied {primary['tag']} ({primary['points']} points) to {target}."
    plan = [
        {"tool": "knowledge", "arguments": {"operation": "list", "prefix": prefix}},
        {"derive": "copy_observed_primary_object", "tool": "knowledge", "operation": "set", "key": target},
        {"tool": "knowledge", "arguments": {"operation": "get", "key": target}},
        {"final_from": "observed_get_result"},
    ]
    return _task_shell("kv.discover_copy", seed, prompt, {"files": {}, "kv": env_kv, "docs": []},
                       expected, {target: primary}, plan)


def _make_kv_delete_one(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("kv.delete_one", seed)
    prefix = _key(f"cache/{token}/")
    target = _key(prefix + "stale")
    env_kv = {target: {"stamp": rng.randint(100, 999), "kind": "stale"},
              _key(prefix + "keep-a"): {"kind": "current", "value": rng.randint(1, 30)},
              _key(prefix + "keep-b"): {"kind": "current", "value": rng.randint(31, 60)},
              _key(f"outside/{token}/keep"): {"kind": "current", "value": 777}}
    expected = f"Removed the requested note: yes; 2 notes remain under {prefix}."
    prompt = (f"Workflow: kv.delete_one. Delete exactly key {target}, then list prefix {prefix} to verify only the "
              "two current notes remain. Leave all other keys unchanged. Reply in one short English sentence stating "
              "whether deletion succeeded and how many notes remain under the prefix.")
    final_kv = {key: value for key, value in env_kv.items() if key != target}
    plan = [
        {"tool": "knowledge", "arguments": {"operation": "delete", "key": target}},
        {"tool": "knowledge", "arguments": {"operation": "list", "prefix": prefix}},
        {"final_from": "observed_delete_and_list_results"},
    ]
    return _task_shell("kv.delete_one", seed, prompt, {"files": {}, "kv": env_kv, "docs": []}, expected, final_kv, plan)


def _make_kv_counter_update(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("kv.counter_update", seed)
    key = _key(f"counter/{token}/visits")
    base = rng.randint(5, 54)
    delta = rng.randint(2, 13)
    initial = {"count": base, "unit": "visits", "label": "west hall"}
    updated = {**initial, "count": base + delta}
    prompt = (f"Workflow: kv.counter_update. Read key {key}, add {delta} to its count, preserve its other fields, "
              "store the updated object at the same key, and read it back. Reply in one short English sentence with the new count.")
    expected = f"Updated {key} to {base + delta}."
    plan = [
        {"tool": "knowledge", "arguments": {"operation": "get", "key": key}},
        {"derive": "add_prompt_delta_to_observed_count", "tool": "knowledge", "operation": "set", "key": key},
        {"tool": "knowledge", "arguments": {"operation": "get", "key": key}},
        {"final_from": "observed_get_result"},
    ]
    return _task_shell("kv.counter_update", seed, prompt, {"files": {}, "kv": {key: initial}, "docs": []},
                       expected, {key: updated}, plan)


def _make_kv_prefix_cleanup(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("kv.prefix_cleanup", seed)
    prefix = _key(f"queue/{token}/")
    env_kv = {}
    for index in range(5):
        state = "stale" if index in {1, 3} else "active"
        env_kv[_key(prefix + f"slot{index}")] = {"status": state, "size": rng.randint(1, 40)}
    env_kv[_key(f"unrelated/{token}/hold")] = {"status": "stale", "size": 999}
    final_kv = {key: value for key, value in env_kv.items() if not (key.startswith(prefix) and value["status"] == "stale")}
    prompt = (f"Workflow: kv.prefix_cleanup. List notes under {prefix}; delete only entries marked status stale, "
              "then list the prefix again. Do not touch notes outside it. Reply in one short English sentence with the number removed and active notes remaining.")
    expected = "Removed 2 stale notes; 3 active notes remain."
    plan = [
        {"tool": "knowledge", "arguments": {"operation": "list", "prefix": prefix}},
        {"derive": "delete_each_observed_stale_key"},
        {"tool": "knowledge", "arguments": {"operation": "list", "prefix": prefix}},
        {"final_from": "observed_list_results"},
    ]
    return _task_shell("kv.prefix_cleanup", seed, prompt, {"files": {}, "kv": env_kv, "docs": []}, expected, final_kv, plan)


def _docs_fact(seed: int, token: str, family: str) -> tuple[list[dict[str, str]], str, str]:
    rng = _rng(family, seed)
    group = rng.choice(["cedar", "harbor", "orchid", "summit", "willow"])
    relay = rng.choice(["amber", "birch", "cobalt", "dune", "ember"])
    hour = rng.randint(6, 18)
    minute = rng.choice([0, 10, 20, 30, 40, 50])
    clock = f"{hour:02d}:{minute:02d}"
    docs = [
        {"id": f"relay_{token}", "title": f"{group.title()} relay departure card",
         "content": f"For group {group}, relay {relay} departs at {clock}. Contact the desk before loading."},
        {"id": f"dispatch_{token}", "title": "General dispatch checklist",
         "content": "Confirm the manifest, label each crate, and close the loading gate after departure."},
        {"id": f"hours_{token}", "title": f"{group.title()} desk opening hours",
         "content": "The local desk opens at 08:30 and closes at 16:30 on weekdays."},
        {"id": f"relay_note_{token}", "title": "Relay terminology",
         "content": f"A relay is a handoff between teams; it does not specify a clock time for group {group}."},
        {"id": f"contacts_{token}", "title": "Contact roster",
         "content": f"The loading contact for {group} is {rng.choice(['Jules', 'Noor', 'Pavel', 'Rina'])}."},
    ]
    query = f"relay departure group {group}"
    return docs, query, f"{relay}|{clock}"


def _make_search_fact(seed: int, token: str) -> dict[str, Any]:
    docs, query, fact = _docs_fact(seed, token, "search.fact_lookup")
    relay, clock = fact.split("|")
    prompt = (f"Workflow: search.fact_lookup. Search the local documentation for this query: {query}. "
              "From the matching card, report the relay and its departure time, and cite the document ID. "
              "Reply in one short English sentence.")
    doc_id = f"relay_{token}"
    expected = f"The {relay} relay departs at {clock}, according to {doc_id}."
    plan = [
        {"tool": "search", "arguments": {"query": query, "limit": 5}},
        {"final_from": "observed_result_content_and_source_id"},
    ]
    return _task_shell("search.fact_lookup", seed, prompt, {"files": {}, "kv": {}, "docs": docs}, expected, None, plan)


def _make_search_ranked(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("search.ranked_titles", seed)
    topic = rng.choice(["coastal storage", "winter survey", "orchard route", "museum transfer", "ridge census"])
    docs = [
        {"id": f"rank_a_{token}", "title": f"{topic.title()} complete field plan",
         "content": f"The {topic} plan covers all three phases and the equipment list."},
        {"id": f"rank_b_{token}", "title": f"{topic.title()} daily notes",
         "content": f"Daily notes for the {topic} team include phase summaries."},
        {"id": f"rank_c_{token}", "title": "Equipment return ledger",
         "content": f"Record equipment after each phase of the {topic} work."},
        {"id": f"rank_d_{token}", "title": f"{topic.split()[0].title()} map index",
         "content": f"An index of nearby {topic.split()[0]} paths."},
        {"id": f"rank_e_{token}", "title": "Archive retention guide",
         "content": "Keep signed copies in the archive for one season."},
    ]
    query = f"{topic} phase equipment"
    ranked = _fixture_search(docs, query, 2)
    titles = [result["title"] for result in ranked]
    prompt = (f"Workflow: search.ranked_titles. Search local notes for query: {query}. Request the two best matches "
              "in the tool's returned order, then report their titles in that order. Reply in one short English sentence.")
    expected = f"The two best matches are ‘{titles[0]}’ and ‘{titles[1]}’, in that order."
    plan = [{"tool": "search", "arguments": {"query": query, "limit": 2}},
            {"final_from": "observed_result_titles_in_order"}]
    return _task_shell("search.ranked_titles", seed, prompt, {"files": {}, "kv": {}, "docs": docs}, expected, None, plan)


def _make_search_two_hop(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("search.two_hop", seed)
    site = rng.choice(["station", "depot", "greenhouse", "pier", "archive"])
    channel = f"protocol_{token}"
    signal = rng.choice(["steady", "yellow", "quiet", "ready", "clear"])
    docs = [
        {"id": f"incident_{token}", "title": f"{site.title()} incident index",
         "content": f"The current {site} incident brief points to protocol {channel} for its final signal."},
        {"id": channel, "title": f"{site.title()} signal protocol",
         "content": f"Detailed procedure for {channel}: the signal to the crew is {signal}. Confirm only after reading."},
        {"id": f"other_{token}", "title": "Old protocol list",
         "content": "An archived route uses signal retired and should not be used for current incidents."},
        {"id": f"general_{token}", "title": f"{site.title()} safety notes",
         "content": "Keep the walkways clear and log any changes at shift end."},
    ]
    query = f"current {site} incident brief"
    prompt = (f"Workflow: search.two_hop. Search local documentation for {query}, follow the protocol identifier "
              "named in the returned incident brief with a second search, then report the signal and both source IDs. "
              "Reply in one short English sentence.")
    expected = f"The incident signal is {signal}, recorded in incident_{token} and {channel}."
    plan = [{"tool": "search", "arguments": {"query": query, "limit": 5}},
            {"derive": "search_protocol_named_in_observed_result"},
            {"final_from": "observed_second_result_and_source_ids"}]
    return _task_shell("search.two_hop", seed, prompt, {"files": {}, "kv": {}, "docs": docs}, expected, None, plan)


def _make_search_no_match(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("search.no_match", seed)
    code = f"qzx{token}"
    docs = [
        {"id": f"shelf_{token}", "title": "Shelf map",
         "content": f"Shelf codes for the north room include {rng.choice(['birch', 'copper', 'linen'])}."},
        {"id": f"shift_{token}", "title": "Shift handover",
         "content": "Record any open items in the handover notebook."},
    ]
    query = f"missing code {code}"
    prompt = (f"Workflow: search.no_match. Search the local corpus for exact code {code}. If no document is returned, "
              "say that plainly; do not infer a value from unrelated notes. Reply in one short English sentence.")
    expected = f"No local document matched ‘{query}’."
    plan = [{"tool": "search", "arguments": {"query": query, "limit": 5}},
            {"final_from": "observed_empty_result"}]
    return _task_shell("search.no_match", seed, prompt, {"files": {}, "kv": {}, "docs": docs}, expected, None, plan)


def _make_search_catalog(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("search.catalog_detail", seed)
    asset = f"asset_{token}"
    technician = rng.choice(["Lina", "Marco", "Sana", "Tomas", "Yuki"])
    docs = [
        {"id": f"catalog_{token}", "title": f"Equipment assignment {asset}",
         "content": f"Asset {asset} is assigned to technician {technician} in the current roster."},
        {"id": f"asset_other_{token}", "title": "Equipment naming rules",
         "content": "Asset codes identify the item, while technician names identify the responsible person."},
        {"id": f"roster_{token}", "title": "General roster",
         "content": f"The {rng.choice(['east', 'west', 'central'])} team checks equipment every Friday."},
    ]
    query = f"asset {asset} technician assignment"
    prompt = (f"Workflow: search.catalog_detail. Search the local catalog for asset {asset}, find who is assigned to it, "
              "and cite the returned document ID. Reply in one short English sentence.")
    expected = f"Asset {asset} is assigned to technician {technician}, according to catalog_{token}."
    plan = [{"tool": "search", "arguments": {"query": query, "limit": 5}},
            {"final_from": "observed_result_content_and_source_id"}]
    return _task_shell("search.catalog_detail", seed, prompt, {"files": {}, "kv": {}, "docs": docs}, expected, None, plan)


def _make_search_active_summary(seed: int, token: str) -> dict[str, Any]:
    rng = _rng("search.active_summary", seed)
    project = rng.choice(["bluebird", "cairn", "delta", "lumen", "meadow"])
    docs = []
    records = []
    for i in range(4):
        record_id = f"record_{token}_{i}"
        score = rng.randint(12, 97)
        status = "active" if i != 2 else "paused"
        records.append((record_id, score, status))
        docs.append({"id": f"doc_{token}_{i}", "title": f"{project.title()} inventory record {i}",
                     "content": f"Project {project} inventory record {record_id} has score {score} and status {status}."})
    # Prevent an accidental tie for the maximum among active rows.
    active = [row for row in records if row[2] == "active"]
    winner = max(active, key=lambda row: (row[1], row[0]))
    query = f"project {project} inventory record score"
    prompt = (f"Workflow: search.active_summary. Search local records for project {project} inventory; among the returned "
              "active records, report the one with the highest score and its score. Reply in one short English sentence.")
    expected = f"Among matching active records, {winner[0]} has the highest score ({winner[1]})."
    plan = [{"tool": "search", "arguments": {"query": query, "limit": 6}},
            {"derive": "choose_highest_score_from_observed_active_results"}]
    return _task_shell("search.active_summary", seed, prompt, {"files": {}, "kv": {}, "docs": docs}, expected, None, plan)


def generate_knowledge_search_task(family: str, seed: int = 0) -> dict[str, Any]:
    if family not in FAMILY_SPLIT_POLICY or type(seed) is not int or seed < 0:
        raise ValueError("invalid family or seed")
    token = _token(family, seed)
    makers = {
        "kv.note_write_verify": _make_kv_note,
        "kv.prefix_total": _make_kv_prefix_total,
        "kv.discover_copy": _make_kv_discover_copy,
        "kv.delete_one": _make_kv_delete_one,
        "kv.counter_update": _make_kv_counter_update,
        "kv.prefix_cleanup": _make_kv_prefix_cleanup,
        "search.fact_lookup": _make_search_fact,
        "search.ranked_titles": _make_search_ranked,
        "search.two_hop": _make_search_two_hop,
        "search.no_match": _make_search_no_match,
        "search.catalog_detail": _make_search_catalog,
        "search.active_summary": _make_search_active_summary,
    }
    task = makers[family](seed, token)
    return task


def generate_knowledge_search_tasks(*, train_seeds_per_family: int = 16,
                                    dev_seeds_per_family: int = 8) -> list[dict[str, Any]]:
    if train_seeds_per_family < 1 or dev_seeds_per_family < 1:
        raise ValueError("seed counts must be positive")
    tasks = []
    for family in sorted(FAMILY_SPLIT_POLICY):
        count = train_seeds_per_family if FAMILY_SPLIT_POLICY[family] == "train" else dev_seeds_per_family
        tasks.extend(generate_knowledge_search_task(family, seed) for seed in range(count))
    return tasks


def authored_unexecuted_candidate(task: dict[str, Any]) -> dict[str, Any]:
    """Return an unexecuted original plan sidecar, never a fabricated transcript."""
    return {
        "schema": "picoagent.unexecuted_candidate.v1",
        "candidate_id": f"procedural:{task['task_id']}",
        "task_id": task["task_id"],
        "task_sha256": content_hash(task),
        "family": task["family"],
        "template_id": task["template_id"],
        "split": task["split"],
        "author": "reviewed_procedural_plan",
        "model": None,
        "status": "unexecuted_candidate_plan",
        "execution": "unexecuted",
        "plan": copy.deepcopy(task["reference"]["plan"]),
        "receipts": [],
        "training_eligible": False,
        "note": "Plan metadata only; actual host-function results must come from ToolRegistry dispatch.",
    }


class KnowledgeSearchTeacher:
    """Reviewed deterministic callback that observes only transcript messages.

    The visible ``Workflow`` label selects an action pattern. Values used in
    follow-up calls and every final answer are read from the prompt or actual
    tool-result messages; the callback receives no task, oracle, or reference.
    """

    def __init__(self) -> None:
        self.call_index = 0
        self.cleanup_keys: list[str] = []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        del tools
        user = next(message["content"] for message in messages if message.get("role") == "user")
        match = re.search(r"Workflow: ([a-z_.]+)\.", user)
        if not match:
            raise ValueError("visible prompt does not specify a supported workflow")
        family = match.group(1)
        observations = [json.loads(message["content"]) for message in messages if message.get("role") == "tool"]
        if observations and ("error" in observations[-1] or observations[-1].get("exit_code", 0) != 0):
            return {"role": "assistant", "content": "I could not finish because the tool reported an error."}

        def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.call_index += 1
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"ks_call_{self.call_index:04d}", "type": "function",
                "function": {"name": name, "arguments": canonical_json(arguments)},
            }]}

        n = len(observations)
        if family == "kv.note_write_verify":
            key_match = re.search(r" at key ([A-Za-z0-9_./-]+), then read it back", user)
            payload_text = user.split("Store the exact JSON payload ", 1)[1].split(" at key ", 1)[0]
            payload = json.loads(payload_text)
            key = key_match.group(1)
            if n == 0:
                return call("knowledge", {"operation": "set", "key": key, "value": payload})
            if n == 1:
                return call("knowledge", {"operation": "get", "key": key})
            value = observations[-1].get("value") or {}
            return {"role": "assistant", "content": f"Saved and verified note {key}: owner {value['owner']}, checkpoint {value['checkpoint']}."}

        if family == "kv.prefix_total":
            prefix = re.search(r"under prefix ([A-Za-z0-9_./-]+);", user).group(1)
            target = re.search(r"at ([A-Za-z0-9_./-]+), then read it back", user).group(1)
            if n == 0:
                return call("knowledge", {"operation": "list", "prefix": prefix})
            if n == 1:
                items = observations[-1].get("items", {})
                summary = {"amount_total": sum(value["amount"] for value in items.values()),
                           "source_count": len(items)}
                return call("knowledge", {"operation": "set", "key": target, "value": summary})
            if n == 2:
                return call("knowledge", {"operation": "get", "key": target})
            summary = observations[-1].get("value") or {}
            return {"role": "assistant", "content": f"Stored total {summary['amount_total']} from {summary['source_count']} notes at {target}."}

        if family == "kv.discover_copy":
            prefix = re.search(r"under ([A-Za-z0-9_./-]+);", user).group(1)
            target = re.search(r"copy its complete JSON value to ([A-Za-z0-9_./-]+),", user).group(1)
            if n == 0:
                return call("knowledge", {"operation": "list", "prefix": prefix})
            if n == 1:
                items = observations[-1].get("items", {})
                value = next(row for key, row in sorted(items.items()) if key.startswith(prefix) and row.get("role") == "primary")
                return call("knowledge", {"operation": "set", "key": target, "value": value})
            if n == 2:
                return call("knowledge", {"operation": "get", "key": target})
            value = observations[-1].get("value") or {}
            return {"role": "assistant", "content": f"Copied {value['tag']} ({value['points']} points) to {target}."}

        if family == "kv.delete_one":
            target = re.search(r"Delete exactly key ([A-Za-z0-9_./-]+),", user).group(1)
            prefix = re.search(r"list prefix ([A-Za-z0-9_./-]+) to", user).group(1)
            if n == 0:
                return call("knowledge", {"operation": "delete", "key": target})
            if n == 1:
                return call("knowledge", {"operation": "list", "prefix": prefix})
            remaining = len(observations[-1].get("items", {}))
            deleted = observations[0].get("deleted") is True
            return {"role": "assistant", "content": f"Removed the requested note: {'yes' if deleted else 'no'}; {remaining} notes remain under {prefix}."}

        if family == "kv.counter_update":
            key = re.search(r"Read key ([A-Za-z0-9_./-]+),", user).group(1)
            delta = int(re.search(r"add ([0-9]+) to its count", user).group(1))
            if n == 0:
                return call("knowledge", {"operation": "get", "key": key})
            if n == 1:
                value = observations[-1].get("value") or {}
                updated = {**value, "count": value["count"] + delta}
                return call("knowledge", {"operation": "set", "key": key, "value": updated})
            if n == 2:
                return call("knowledge", {"operation": "get", "key": key})
            value = observations[-1].get("value") or {}
            return {"role": "assistant", "content": f"Updated {key} to {value['count']}."}

        if family == "kv.prefix_cleanup":
            prefix = re.search(r"under ([A-Za-z0-9_./-]+);", user).group(1)
            if n == 0:
                return call("knowledge", {"operation": "list", "prefix": prefix})
            if n == 1:
                items = observations[-1].get("items", {})
                self.cleanup_keys = sorted(key for key, value in items.items() if value.get("status") == "stale")
                if not self.cleanup_keys:
                    return call("knowledge", {"operation": "list", "prefix": prefix})
                return call("knowledge", {"operation": "delete", "key": self.cleanup_keys[0]})
            deleted_so_far = n - 1
            if deleted_so_far < len(self.cleanup_keys):
                return call("knowledge", {"operation": "delete", "key": self.cleanup_keys[deleted_so_far]})
            if deleted_so_far == len(self.cleanup_keys):
                return call("knowledge", {"operation": "list", "prefix": prefix})
            items = observations[-1].get("items", {})
            live = sum(value.get("status") == "active" for value in items.values())
            return {"role": "assistant", "content": f"Removed {len(self.cleanup_keys)} stale notes; {live} active notes remain."}

        if family == "search.fact_lookup":
            query = re.search(r"this query: (.+?)\. From", user).group(1)
            if n == 0:
                return call("search", {"query": query, "limit": 5})
            results = observations[-1].get("results", [])
            if not results:
                return {"role": "assistant", "content": "No matching local note was found."}
            result = results[0]
            relay, clock = re.search(r"relay (\w+) departs at ([0-9:]+)", result["content"]).groups()
            return {"role": "assistant", "content": f"The {relay} relay departs at {clock}, according to {result['id']}."}

        if family == "search.ranked_titles":
            query = re.search(r"query: (.+?)\. Request", user).group(1)
            if n == 0:
                return call("search", {"query": query, "limit": 2})
            results = observations[-1].get("results", [])
            if len(results) < 2:
                return {"role": "assistant", "content": "Fewer than two local matches were returned."}
            return {"role": "assistant", "content": f"The two best matches are ‘{results[0]['title']}’ and ‘{results[1]['title']}’, in that order."}

        if family == "search.two_hop":
            site = re.search(r"for current ([a-z]+) incident", user).group(1)
            if n == 0:
                return call("search", {"query": f"current {site} incident brief", "limit": 5})
            if n == 1:
                results = observations[-1].get("results", [])
                if not results:
                    return {"role": "assistant", "content": "No incident brief was returned."}
                channel = re.search(r"protocol ([A-Za-z0-9_]+)", results[0]["content"]).group(1)
                return call("search", {"query": channel + " detailed procedure", "limit": 5})
            results = observations[-1].get("results", [])
            if not results:
                return {"role": "assistant", "content": "No referenced protocol was returned."}
            signal = re.search(r"signal to the crew is ([a-z]+)", results[0]["content"]).group(1)
            first = observations[0]["results"][0]["id"]
            return {"role": "assistant", "content": f"The incident signal is {signal}, recorded in {first} and {results[0]['id']}."}

        if family == "search.no_match":
            code = re.search(r"exact code ([A-Za-z0-9_]+)", user).group(1)
            query = "missing code " + code
            if n == 0:
                return call("search", {"query": query, "limit": 5})
            results = observations[-1].get("results", [])
            if results:
                return {"role": "assistant", "content": f"A local document matched ‘{query}’."}
            return {"role": "assistant", "content": f"No local document matched ‘{query}’."}

        if family == "search.catalog_detail":
            asset = re.search(r"Search the local catalog for asset ([A-Za-z0-9_]+),", user).group(1)
            if n == 0:
                return call("search", {"query": f"asset {asset} technician assignment", "limit": 5})
            results = observations[-1].get("results", [])
            if not results:
                return {"role": "assistant", "content": f"No catalog entry was returned for asset {asset}."}
            person = re.search(r"assigned to technician ([A-Za-z]+)", results[0]["content"]).group(1)
            return {"role": "assistant", "content": f"Asset {asset} is assigned to technician {person}, according to {results[0]['id']}."}

        if family == "search.active_summary":
            project = re.search(r"for project ([a-z]+) inventory", user).group(1)
            if n == 0:
                return call("search", {"query": f"project {project} inventory record score", "limit": 6})
            parsed = []
            for result in observations[-1].get("results", []):
                match = re.search(r"record ([A-Za-z0-9_]+) has score ([0-9]+) and status ([a-z]+)", result["content"])
                if match and match.group(3) == "active":
                    parsed.append((int(match.group(2)), match.group(1)))
            if not parsed:
                return {"role": "assistant", "content": "No active matching record was returned."}
            score, record_id = max(parsed)
            return {"role": "assistant", "content": f"Among matching active records, {record_id} has the highest score ({score})."}

        raise ValueError(f"unsupported visible workflow: {family}")


def _task_fixtures(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compute expected final text and full post-run KV state from task inputs only."""
    family = task["family"]
    env = task["environment"]
    prompt = task["prompt"]
    final_kv = copy.deepcopy(env.get("kv", {}))
    if family == "kv.note_write_verify":
        m = re.search(r"Store the exact JSON payload (\{.*?\}) at key ([^, ]+),", prompt)
        # The generated prompt uses canonical JSON with no spaces, and the key follows it.
        if not m:
            raise ValueError("could not derive note write input from prompt")
        payload, key = json.loads(m.group(1)), m.group(2)
        final_kv[key] = payload
        return f"Saved and verified note {key}: owner {payload['owner']}, checkpoint {payload['checkpoint']}.", final_kv
    if family == "kv.prefix_total":
        prefix = re.search(r"prefix ([A-Za-z0-9_./-]+);", prompt).group(1)
        target = re.search(r"store \{.*?\} at ([A-Za-z0-9_./-]+),", prompt).group(1)
        rows = [value for key, value in env["kv"].items() if key.startswith(prefix)]
        total = sum(row["amount"] for row in rows)
        final_kv[target] = {"amount_total": total, "source_count": len(rows)}
        return f"Stored total {total} from {len(rows)} notes at {target}.", final_kv
    if family == "kv.discover_copy":
        prefix = re.search(r"under ([A-Za-z0-9_./-]+);", prompt).group(1)
        target = re.search(r"copy its complete JSON value to ([A-Za-z0-9_./-]+),", prompt).group(1)
        row = next(value for key, value in env["kv"].items() if key.startswith(prefix) and value.get("role") == "primary")
        final_kv[target] = copy.deepcopy(row)
        return f"Copied {row['tag']} ({row['points']} points) to {target}.", final_kv
    if family == "kv.delete_one":
        target = re.search(r"Delete exactly key ([A-Za-z0-9_./-]+),", prompt).group(1)
        prefix = re.search(r"list prefix ([A-Za-z0-9_./-]+) to", prompt).group(1)
        existed = target in final_kv
        final_kv.pop(target, None)
        remaining = sum(key.startswith(prefix) for key in final_kv)
        return f"Removed the requested note: {'yes' if existed else 'no'}; {remaining} notes remain under {prefix}.", final_kv
    if family == "kv.counter_update":
        key = re.search(r"Read key ([A-Za-z0-9_./-]+),", prompt).group(1)
        delta = int(re.search(r"add ([0-9]+) to its count", prompt).group(1))
        final_kv[key] = {**final_kv[key], "count": final_kv[key]["count"] + delta}
        return f"Updated {key} to {final_kv[key]['count']}.", final_kv
    if family == "kv.prefix_cleanup":
        prefix = re.search(r"under ([A-Za-z0-9_./-]+);", prompt).group(1)
        removed = [key for key, value in final_kv.items() if key.startswith(prefix) and value.get("status") == "stale"]
        for key in removed:
            final_kv.pop(key)
        remaining = sum(key.startswith(prefix) and value.get("status") == "active" for key, value in final_kv.items())
        return f"Removed {len(removed)} stale notes; {remaining} active notes remain.", final_kv

    docs = env["docs"]
    if family == "search.fact_lookup":
        query = re.search(r"query: (.+?)\.", prompt).group(1)
        result = _fixture_search(docs, query, 5)[0]
        group = re.search(r"For group (\w+), relay (\w+) departs at ([0-9:]+)", result["content"])
        return f"The {group.group(2)} relay departs at {group.group(3)}, according to {result['id']}.", final_kv
    if family == "search.ranked_titles":
        query = re.search(r"query: (.+?)\. Request", prompt).group(1)
        results = _fixture_search(docs, query, 2)
        return f"The two best matches are ‘{results[0]['title']}’ and ‘{results[1]['title']}’, in that order.", final_kv
    if family == "search.two_hop":
        site = re.search(r"for (?:current )?([a-z]+) incident", prompt).group(1)
        query = f"current {site} incident brief"
        first = _fixture_search(docs, query, 5)[0]
        channel = re.search(r"protocol ([A-Za-z0-9_]+)", first["content"]).group(1)
        second = _fixture_search(docs, channel + " detailed procedure", 5)[0]
        signal = re.search(r"signal to the crew is ([a-z]+)", second["content"]).group(1)
        return f"The incident signal is {signal}, recorded in {first['id']} and {second['id']}.", final_kv
    if family == "search.no_match":
        query = re.search(r"exact code ([A-Za-z0-9_]+)", prompt).group(1)
        query_text = "missing code " + query
        if _fixture_search(docs, query_text, 5):
            raise ValueError("no-match fixture unexpectedly retrieves a document")
        return f"No local document matched ‘{query_text}’." , final_kv
    if family == "search.catalog_detail":
        asset = re.search(r"asset ([A-Za-z0-9_]+),", prompt).group(1)
        query = f"asset {asset} technician assignment"
        result = _fixture_search(docs, query, 5)[0]
        person = re.search(r"is assigned to technician ([A-Za-z]+)", result["content"]).group(1)
        return f"Asset {asset} is assigned to technician {person}, according to {result['id']}.", final_kv
    if family == "search.active_summary":
        project = re.search(r"for project ([a-z]+) inventory", prompt).group(1)
        rows = _fixture_search(docs, f"project {project} inventory record score", 6)
        parsed = []
        for result in rows:
            match = re.search(r"record ([A-Za-z0-9_]+) has score ([0-9]+) and status ([a-z]+)", result["content"])
            if match and match.group(3) == "active":
                parsed.append((int(match.group(2)), match.group(1)))
        score, record_id = max(parsed)
        return f"Among matching active records, {record_id} has the highest score ({score}).", final_kv
    raise ValueError(f"no fixture oracle for {family}")


def _fixture_search(docs: list[dict[str, str]], query: str, limit: int) -> list[dict[str, str]]:
    """Independent pure retrieval oracle; no LocalCorpusSearch instance is used."""
    tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
    ranked = []
    for doc in docs:
        words = set(re.findall(r"[a-z0-9_]+", (doc["title"] + " " + doc["content"]).lower()))
        score = len(tokens & words)
        if score:
            ranked.append((-score, doc["id"], doc))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return [{"id": doc["id"], "title": doc["title"], "url": "local://docs/" + doc["id"], "content": doc["content"]}
            for _, _, doc in ranked[:limit]]


def independent_knowledge_search_oracle(task: dict[str, Any], final: str | None,
                                        kv: dict[str, Any] | None) -> dict[str, Any]:
    """Recompute from visible prompt and fixtures; never read reference or oracle."""
    try:
        expected_final, expected_kv = _task_fixtures(task)
    except Exception as exc:
        return {"passed": False, "failures": [f"oracle could not derive fixture answer: {type(exc).__name__}: {exc}"],
                "expected_final": None, "expected_kv": None, "method": "independent_fixture_recomputation"}
    failures = []
    if final != expected_final:
        failures.append("final answer differs from fixture-derived result")
    if kv != expected_kv:
        failures.append("post-run knowledge store differs from fixture-derived state")
    return {"passed": not failures, "failures": failures, "expected_final": expected_final,
            "expected_kv": expected_kv, "method": "independent_fixture_recomputation"}


def verify_knowledge_search_curriculum(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [task["task_id"] for task in tasks]
    inputs = [task["input_sha256"] for task in tasks]
    if len(ids) != len(set(ids)) or len(inputs) != len(set(inputs)):
        raise ValueError("duplicate task IDs or inputs")
    family_splits: dict[str, set[str]] = {}
    for task in tasks:
        validate_task(task)
        family_splits.setdefault(task["family"], set()).add(task["split"])
        if task["family"] not in FAMILY_SPLIT_POLICY or FAMILY_SPLIT_POLICY[task["family"]] != task["split"]:
            raise ValueError("family split policy mismatch")
        expected_final, expected_state = _task_fixtures(task)
        if expected_final != task["oracle"].get("expected") or expected_final != task["reference"].get("final"):
            raise ValueError(f"stored answer/reference mismatch for {task['task_id']}")
        for key, value in task["oracle"].get("kv_expected", {}).items():
            if expected_state.get(key) != value:
                raise ValueError(f"stored KV postcondition mismatch for {task['task_id']}:{key}")
        independent = independent_knowledge_search_oracle(task, expected_final, expected_state)
        if not independent["passed"]:
            raise ValueError(f"fixture oracle regression for {task['task_id']}: {independent['failures']}")
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise ValueError("family crosses data splits")
    return {"passed": True, "counts": {split: sum(task["split"] == split for task in tasks) for split in ("train", "dev", "test")},
            "families": {split: sorted(family for family, splits in family_splits.items() if split in splits)
                         for split in ("train", "dev", "test")},
            "family_count": len(family_splits), "checks": ["unique_task_ids", "unique_inputs", "disjoint_families", "independent_fixtures"]}


def write_knowledge_search_curriculum(output_dir: str | Path, *, train_seeds_per_family: int = 16,
                                      dev_seeds_per_family: int = 8) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    tasks = generate_knowledge_search_tasks(train_seeds_per_family=train_seeds_per_family,
                                            dev_seeds_per_family=dev_seeds_per_family)
    report = verify_knowledge_search_curriculum(tasks)
    output.mkdir(parents=True)
    (output / "candidates").mkdir()
    paths = {"train": output / "train.tasks.jsonl", "dev": output / "dev.tasks.jsonl"}
    file_info = {}
    for split, path in paths.items():
        rows = [task for task in tasks if task["split"] == split]
        raw = "".join(canonical_json(task) + "\n" for task in rows).encode("utf-8")
        path.write_bytes(raw)
        file_info[path.name] = {"records": len(rows), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    candidates = [authored_unexecuted_candidate(task) for task in tasks]
    candidate_bytes = "".join(canonical_json(row) + "\n" for row in candidates).encode("utf-8")
    candidate_path = output / "candidates" / "unexecuted_plans.jsonl"
    candidate_path.write_bytes(candidate_bytes)
    module_path = Path(__file__).resolve()
    manifest = {
        "schema": "picoagent.luna_knowledge_search_curriculum.v1",
        "track": TRACK,
        "source_id": SOURCE_ID,
        "execution": "unexecuted",
        "training_eligible": False,
        "configuration": {"train_seeds_per_family": train_seeds_per_family,
                           "dev_seeds_per_family": dev_seeds_per_family,
                           "test_families_generated": False},
        "family_split_policy": FAMILY_SPLIT_POLICY,
        "family_split_policy_sha256": content_hash(FAMILY_SPLIT_POLICY),
        "counts": report["counts"],
        "families": report["families"],
        "file_hashes": file_info,
        "candidate_sidecar": {"path": "candidates/unexecuted_plans.jsonl", "records": len(candidates),
                              "bytes": len(candidate_bytes), "sha256": hashlib.sha256(candidate_bytes).hexdigest()},
        "generator_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "audit": report,
    }
    (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return output / "manifest.json"
