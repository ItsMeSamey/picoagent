"""Native producer contract tests; no simulated process output is used."""
import base64
import copy
import gzip
import json
from pathlib import Path

import pytest

from picoagent.data.audit import file_hash, read_jsonl
from picoagent.data.compaction_curriculum import generate_compaction_task
from picoagent.data.native_compaction_curriculum import (
    Journal, MODES, PILOT_BASES, ReviewedCatRegistry, independent_oracle_check,
    _write_evidence, scaling_bases,
)
from picoagent.data.schema import canonical_json, content_hash, _validate_model_event_replay


def test_independent_oracle_ignores_private_expected_answer():
    for family, seed in PILOT_BASES:
        task = generate_compaction_task(family, seed)
        final = canonical_json(task["oracle"]["expected"])
        task["oracle"]["expected"] = {"future_hidden_oracle": "DO_NOT_READ"}
        task["reference"] = {"final": "DO_NOT_READ"}
        assert independent_oracle_check(task, final)["passed"]
        assert not independent_oracle_check(task, '{"result":"invented"}')["passed"]


@pytest.mark.parametrize("command", [
    "cat -- /etc/passwd", "cat -- records/0000000000000000.json; echo unsafe",
    "cat -- records/0000000000000000.json\necho unsafe", "python -c 'print(1)'",
    "cat -- records/0000000000000000.json > copied.txt", "cat -- records/../secret",
])
def test_unreviewed_commands_are_journaled_then_rejected_without_execution(tmp_path, command):
    task = generate_compaction_task("compaction.running_balance", 0)
    journal = Journal(tmp_path / "journal.jsonl")
    registry = ReviewedCatRegistry(task, tmp_path, journal, {})
    arguments = canonical_json({"command": command})
    registry.pending_call = {"id": "record_0", "type": "function",
                             "function": {"name": "bash", "arguments": arguments}}
    with pytest.raises(ValueError, match="reviewed cat"):
        registry.dispatch("bash", arguments)
    events = read_jsonl(journal.path)
    assert [row["kind"] for row in events] == ["tool_requested", "tool_rejected"]
    assert not registry.receipts


def test_native_pilot_has_balanced_modes_without_new_task_identity():
    assert len(PILOT_BASES) == 6
    modes = {"train": [], "dev": []}
    ids = set()
    for index, (family, seed) in enumerate(PILOT_BASES):
        task = generate_compaction_task(family, seed)
        ids.add(task["task_id"])
        modes[task["split"]].append(MODES[index % 3])
    assert len(ids) == 6
    assert all(set(values) == set(MODES) for values in modes.values())


def test_journal_preserves_failure_and_binds_sequence(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.write("requested", {"id": "a"})
    journal.write("rejected", {"reason": "no execution"})
    previous = "0" * 64
    for index, row in enumerate(read_jsonl(journal.path)):
        digest = row.pop("sha256")
        assert row["previous_sha256"] == previous
        assert row["sequence"] == index and digest == content_hash(row)
        previous = digest
    assert previous == journal.previous


def test_compressed_journal_members_are_readable_and_hash_bound_after_each_write(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl.gz")
    for index in range(3):
        journal.write("unexecuted_contract_test", {"index": index})
        with gzip.open(journal.path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == index + 1
        assert rows[-1]["sha256"] == journal.previous
    value = {"execution": "unexecuted", "test": "byte-preserving compression"}
    target = _write_evidence(tmp_path / "raw.json", value, compressed=True)
    assert not (tmp_path / "raw.json").exists()
    assert gzip.decompress(target.read_bytes()) == (canonical_json(value) + "\n").encode()
    with pytest.raises(FileExistsError):
        _write_evidence(tmp_path / "raw.json", value, compressed=True)


def test_fixed_scale_has_216_bases_and_balanced_first_modes_within_families():
    from collections import Counter, defaultdict
    bases = scaling_bases()
    assert len(bases) == len(set(bases)) == 216
    by_family = defaultdict(Counter)
    split_counts = Counter()
    for family, seed in bases:
        task = generate_compaction_task(family, seed)
        split_counts[task["split"]] += 1
        by_family[family][MODES[seed % 3]] += 1
    assert split_counts == {"train": 192, "dev": 24}
    for counts in by_family.values():
        assert set(counts) == set(MODES)
        assert len(set(counts.values())) == 1


def test_actual_pilot_retains_raw_native_receipts_and_exact_mode_events():
    """Read-only regression over real produced artifacts; never manufacture evidence."""
    path = Path(__file__).resolve().parents[1] / "data/native-compaction-v1/observations.jsonl"
    if not path.exists():
        pytest.skip("actual native pilot has not yet been produced")
    rows = read_jsonl(path)
    assert len(rows) == 18 and len({row["base_task_id"] for row in rows}) == 6
    assert all(row["status"] == "observed_success" for row in rows)
    assert {row["mode"] for row in rows} == set(MODES)
    assert all(row["task"]["split"] in {"train", "dev"} for row in rows)
    for raw in rows:
        assert raw["sft_admissible"] is False
        assert raw["teacher_model"] is None
        assert raw["teacher_decision_mode"] == "reviewed_procedural_replay"
        assert raw["task_id"] == raw["base_task_id"] == raw["task"]["task_id"]
        _validate_model_event_replay(raw)
        assert raw["fixture_sha256_before"] == raw["fixture_sha256_after"]
        assert raw["independent_oracle"]["passed"]
        assert any(event["type"] == "compaction" for event in raw["model_events"])
        assert len(raw["receipts"]) == len(raw["tool_events"]) == 16
        for receipt, event in zip(raw["receipts"], raw["tool_events"]):
            assert receipt["result"] == event["result"]
            assert receipt["tool_call_id"] == event["tool_call_id"]
            assert "container_id" not in receipt["result"]
            assert receipt["argv"][-2:] == ["-c", json.loads(receipt["arguments"])["command"]]
            for stream in ("stdout", "stderr", "stdin"):
                import hashlib
                value = base64.b64decode(receipt[stream + "_b64"], validate=True)
                assert hashlib.sha256(value).hexdigest() == receipt[stream + "_sha256"]
                assert len(value) == receipt[stream + "_bytes"]
                if stream != "stdin":
                    assert value.decode("utf-8", errors="replace") == receipt["result"][stream]
        journal_path = path.parent / raw["journal"]["path"]
        assert file_hash(journal_path) == raw["journal"]["file_sha256"]
        mutated = copy.deepcopy(raw["task"])
        mutated["oracle"]["expected"] = None
        assert independent_oracle_check(mutated, raw["final"])["passed"]
