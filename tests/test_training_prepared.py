"""Prepared-token equivalence and hostile-artifact checks; no model or tools run."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from picoagent.training import prepared, prepared_approvals
from picoagent.training.config import TrainingConfig
from picoagent.training.data import canonical_json, prepare_dataset, verify_dataset
from picoagent.training.encoding import encode_records


@pytest.fixture
def source(tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from picoagent.harness.protocol import render_messages

    rows = {"train": [], "dev": []}
    for split, number in (("train", 2), ("train", 3), ("dev", 5)):
        call = {
            "id": f"call-{number}",
            "type": "function",
            "function": {
                "name": "python",
                "arguments": canonical_json({"code": f"print({number}+1)"}),
            },
        }
        messages = [
            {"role": "user", "content": f"Compute {number}+1"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": str(number + 1)},
            {"role": "assistant", "content": str(number + 1)},
        ]
        row = {
            "trace_id": f"trace-{number}",
            "task_id": f"task-{number}",
            "family": split,
            "template_id": split + "-template",
            "split": split,
            "messages": messages,
            "tools": [],
            "provenance": {"source": "pipeline_smoke"},
        }
        if split == "train":
            row["model_events"] = [
                {"type": "assistant", "input_messages": messages[:1], "message": messages[1]},
                {"type": "assistant", "input_messages": messages[:3], "message": messages[3]},
            ]
        rows[split].append(row)
    paths = {}
    for split, values in rows.items():
        paths[split] = tmp_path / (split + ".jsonl")
        paths[split].write_text("".join(canonical_json(row) + "\n" for row in values))
    manifest = prepare_dataset(paths["train"], paths["dev"], tmp_path / "source", smoke_only=True)
    core = Tokenizer(models.BPE(unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    core.train_from_iterator(
        [render_messages(row["messages"], tools=[]) for values in rows.values() for row in values],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["[UNK]", "[PAD]", "[EOS]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    model = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model)
    config = TrainingConfig(
        model_id=str(model),
        model_revision=None,
        dataset_manifest=str(manifest),
        output_dir=str(tmp_path / "run"),
        smoke_test=True,
        device="cpu",
        max_seq_length=1024,
    )
    return config


def build(source, tmp_path, name="prepared"):
    path = prepared.build_prepared_dataset(source, tmp_path / name, shard_bytes=4096)
    return dataclasses.replace(
        source, prepared_manifest=str(path), prepared_manifest_sha256=prepared.sha256_file(path)
    )


def repin(config, mutate):
    path = Path(config.prepared_manifest)
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.chmod(0o600)
    path.write_text(canonical_json(manifest) + "\n")
    return dataclasses.replace(config, prepared_manifest_sha256=prepared.sha256_file(path))


def test_every_array_order_mask_matches_production_and_independent_rebuild(source, tmp_path):
    config = build(source, tmp_path)
    loaded = prepared.load_prepared_dataset(config)
    _, rows = verify_dataset(source.dataset_manifest, allow_smoke=True)
    tokenizer = prepared._tokenizer(source)
    for split in ("train", "dev"):
        expected, stats = encode_records(rows[split], tokenizer, 1024)
        assert list(loaded.datasets[split]) == expected
        assert loaded.stats[split] == stats
        assert loaded.datasets[split][-1] == expected[-1]
        assert loaded.datasets[split][:2] == expected[:2]
    second = build(source, tmp_path, "independent")
    assert config.prepared_manifest_sha256 == second.prepared_manifest_sha256
    first_root, second_root = (
        Path(config.prepared_manifest).parent,
        Path(second.prepared_manifest).parent,
    )
    assert {
        str(p.relative_to(first_root)): p.read_bytes() for p in first_root.rglob("*") if p.is_file()
    } == {
        str(p.relative_to(second_root)): p.read_bytes()
        for p in second_root.rglob("*")
        if p.is_file()
    }


def test_fast_load_does_not_call_semantic_verifier_or_encoder(source, tmp_path, monkeypatch):
    config = build(source, tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("expensive preparation repeated")

    monkeypatch.setattr(prepared, "verify_dataset", forbidden)
    monkeypatch.setattr(prepared, "encode_trace", forbidden)
    monkeypatch.setattr(prepared, "event_examples", forbidden)
    loaded = prepared.load_prepared_dataset(config)
    assert len(loaded.datasets["train"]) == 4


def test_original_source_byte_corruption_fails(source, tmp_path):
    config = build(source, tmp_path)
    original = Path(source.dataset_manifest).parent / "train.jsonl"
    original.chmod(0o600)
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw source evidence"):
        prepared.load_prepared_dataset(config)


def test_manifest_pin_and_raw_token_byte_corruption_fail(source, tmp_path):
    config = build(source, tmp_path)
    with pytest.raises(ValueError, match="explicit pin"):
        prepared.load_prepared_dataset(
            dataclasses.replace(config, prepared_manifest_sha256="0" * 64)
        )
    manifest = json.loads(Path(config.prepared_manifest).read_text())
    shard = Path(config.prepared_manifest).parent / manifest["splits"]["train"]["shards"][0]["data"]
    shard.chmod(0o600)
    shard.write_bytes(shard.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="file bytes changed"):
        prepared.load_prepared_dataset(config)


@pytest.mark.parametrize(
    "kind",
    [
        "path",
        "extra",
        "symlink",
        "duplicate_json",
        "packages",
        "encoder",
        "tokenizer",
        "length",
        "source_manifest",
    ],
)
def test_malformed_and_stale_artifacts_fail(source, tmp_path, kind):
    config = build(source, tmp_path)
    root = Path(config.prepared_manifest).parent
    if kind == "path":
        config = repin(
            config,
            lambda m: m["files"].update(
                {"../outside": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}
            ),
        )
    elif kind == "extra":
        (root / "extra.txt").write_text("not listed")
    elif kind == "symlink":
        (root / "extra-link").symlink_to(tmp_path / "train.jsonl")
    elif kind == "duplicate_json":
        path = Path(config.prepared_manifest)
        raw = path.read_text()
        path.chmod(0o600)
        path.write_text('{"schema":"duplicate",' + raw[1:])
        config = dataclasses.replace(config, prepared_manifest_sha256=prepared.sha256_file(path))
    elif kind == "packages":
        config = repin(config, lambda m: m["transform"]["packages"].update({"tokenizers": "0.bad"}))
    elif kind == "encoder":
        config = repin(
            config, lambda m: m["transform"]["source"].update({"training/encoding.py": "0" * 64})
        )
    elif kind == "tokenizer":
        config = repin(config, lambda m: m["tokenizer"].update({"pad_token_id": 999999}))
    elif kind == "length":
        config = dataclasses.replace(config, max_seq_length=2048)
    else:
        config = repin(config, lambda m: m.update({"source_manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError):
        prepared.load_prepared_dataset(config)


def test_rehashed_malformed_arrays_rejected(source, tmp_path):
    config = build(source, tmp_path)
    root = Path(config.prepared_manifest).parent

    def mutate(manifest):
        descriptor = manifest["splits"]["train"]["shards"][0]
        path = root / descriptor["data"]
        packed = bytearray(gzip.decompress(path.read_bytes()))
        n = prepared.struct.unpack_from("<I", packed, len(prepared.MAGIC))[0]
        # Change first attention mask entry and recompute all public hashes.
        prepared.struct.pack_into("<i", packed, len(prepared.MAGIC) + 4 + 4 * n, 7)
        logical = bytes(packed)
        raw = prepared._compress(logical)
        path.chmod(0o600)
        path.write_bytes(raw)
        manifest["files"][descriptor["data"]] = {
            "bytes": len(raw),
            "sha256": prepared.sha256_bytes(raw),
        }
        descriptor["logical_sha256"] = prepared.sha256_bytes(logical)
        indexpath = root / descriptor["index"]
        entries = [
            json.loads(line) for line in gzip.decompress(indexpath.read_bytes()).splitlines()
        ]
        entries[0]["encoded_sha256"] = prepared.sha256_bytes(
            logical[len(prepared.MAGIC) : len(prepared.MAGIC) + 4 + 12 * n]
        )
        indexraw = ("".join(canonical_json(row) + "\n" for row in entries)).encode()
        compressed = prepared._compress(indexraw)
        indexpath.chmod(0o600)
        indexpath.write_bytes(compressed)
        descriptor["index_logical_bytes"] = len(indexraw)
        descriptor["index_logical_sha256"] = prepared.sha256_bytes(indexraw)
        manifest["files"][descriptor["index"]] = {
            "bytes": len(compressed),
            "sha256": prepared.sha256_bytes(compressed),
        }

    config = repin(config, mutate)
    with pytest.raises(ValueError, match="attention mask"):
        prepared.load_prepared_dataset(config)


def test_production_requires_independent_approval_even_with_self_hash(
    source, tmp_path, monkeypatch
):
    config = build(source, tmp_path)
    production = dataclasses.replace(
        config, model_id="org/model", model_revision="a" * 40, smoke_test=False
    )
    monkeypatch.setattr(prepared_approvals, "APPROVED_PREPARED_MANIFESTS", frozenset())
    with pytest.raises(ValueError, match="independent production approval"):
        prepared.load_prepared_dataset(production)


def test_config_requires_valid_pair_and_eval_fields(source):
    with pytest.raises(ValueError, match="together"):
        dataclasses.replace(source, prepared_manifest="cache.json")
    with pytest.raises(ValueError, match="SHA256"):
        dataclasses.replace(
            source, prepared_manifest="cache.json", prepared_manifest_sha256="untrusted"
        )
    with pytest.raises(ValueError, match="eval_steps"):
        dataclasses.replace(source, eval_steps=True)
    with pytest.raises(ValueError, match="boolean"):
        dataclasses.replace(source, checkpoint_before_eval="true")


def test_exact_snapshot_copy_and_relocation(source, tmp_path):
    config = build(source, tmp_path)
    loaded = prepared.load_prepared_dataset(config)
    prepared.copy_prepared_snapshots(
        loaded,
        source_destination=tmp_path / "run-data",
        prepared_destination=tmp_path / "run-tokens",
    )
    moved = dataclasses.replace(
        config,
        dataset_manifest=str(tmp_path / "run-data/manifest.json"),
        prepared_manifest=str(tmp_path / "run-tokens/manifest.json"),
    )
    actual = prepared.load_prepared_dataset(moved)
    assert actual.identity == loaded.identity
    assert list(actual.datasets["train"]) == list(loaded.datasets["train"])
    assert (tmp_path / "run-data/manifest.json").read_bytes() == Path(
        source.dataset_manifest
    ).read_bytes()


def test_actual_training_tokenizer_settings_are_bound(source, tmp_path):
    config = build(source, tmp_path)
    loaded = prepared.load_prepared_dataset(config)
    tokenizer = prepared._tokenizer(source)
    prepared.validate_training_tokenizer(loaded, tokenizer)
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="training tokenizer"):
        prepared.validate_training_tokenizer(loaded, tokenizer)


def test_trainer_prepared_admission_happens_before_device_and_skips_raw_verifier(
    source, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from picoagent.training import train

    config = build(source, tmp_path)
    calls = []
    original = prepared.load_prepared_dataset

    def load(config):
        calls.append("prepared")
        return original(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw verifier repeated")

    def stop_before_device(*args, **kwargs):
        calls.append("device_boundary")
        raise RuntimeError("intentional stop before device allocation")

    monkeypatch.setattr(prepared, "load_prepared_dataset", load)
    monkeypatch.setattr(train, "verify_dataset", forbidden)
    monkeypatch.setattr(train, "resolve_device", stop_before_device)
    with pytest.raises(RuntimeError, match="intentional stop"):
        train.run_training(config)
    assert calls == ["prepared", "device_boundary"]


def test_compaction_event_gaps_masks_and_dataset_immutability(source, tmp_path):
    """Accepted summaries train only their output; rejected events keep index gaps."""
    _, rows = verify_dataset(source.dataset_manifest, allow_smoke=True)
    for row in rows["train"]:
        row["model_events"][1:1] = [
            {"type": "compaction", "accepted": False},
            {
                "type": "compaction",
                "accepted": True,
                "summary_request": [{"role": "user", "content": "Summarize the tool result"}],
                "summary_response": {"role": "assistant", "content": "The calculation succeeded"},
            },
        ]
    paths = {}
    for split, values in rows.items():
        paths[split] = tmp_path / f"compaction-{split}.jsonl"
        paths[split].write_text("".join(canonical_json(row) + "\n" for row in values))
    manifest = prepare_dataset(
        paths["train"], paths["dev"], tmp_path / "compaction-source", smoke_only=True
    )
    source = dataclasses.replace(source, dataset_manifest=str(manifest))
    config = build(source, tmp_path)
    loaded = prepared.load_prepared_dataset(config)
    expected, stats = encode_records(rows["train"], prepared._tokenizer(source), 1024)
    assert list(loaded.datasets["train"]) == expected
    assert loaded.stats["train"] == stats
    indices = []
    for shard in loaded.manifest["splits"]["train"]["shards"]:
        data = gzip.decompress((loaded.path.parent / shard["index"]).read_bytes())
        indices.extend(json.loads(line) for line in data.splitlines())
    assert [item["event_index"] for item in indices] == [0, 2, 3, 0, 2, 3]
    assert all(item["last_only"] for item in indices)
    returned = loaded.datasets["train"][0]
    returned["input_ids"][0] = -1
    assert loaded.datasets["train"][0] == expected[0]


def test_bounded_gzip_decode_rejects_expansion():
    compressed = prepared._compress(b"x" * 2048)
    with pytest.raises(ValueError, match="logical shard exceeds"):
        prepared._decompress(compressed, 1024)


def test_builder_never_overwrites_existing_output(source, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "evidence"
    marker.write_bytes(b"keep original")
    with pytest.raises(FileExistsError):
        prepared.build_prepared_dataset(source, existing)
    assert marker.read_bytes() == b"keep original"


@pytest.mark.parametrize("phase", ["admission", "encoding"])
def test_source_identity_is_frozen_before_admission_and_through_encoding(
    source, tmp_path, monkeypatch, phase
):
    initial = prepared._identity()
    changed = False

    def identity():
        return {**initial, "protocol": "concurrently-changed"} if changed else initial

    function_name = "verify_dataset" if phase == "admission" else "encode_trace"
    original = getattr(prepared, function_name)

    def change_during_phase(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        changed = True
        return result

    monkeypatch.setattr(prepared, "_identity", identity)
    monkeypatch.setattr(prepared, function_name, change_during_phase)
    output = tmp_path / "changing-source"
    with pytest.raises(ValueError, match="transform implementation changed"):
        prepared.build_prepared_dataset(source, output)
    assert not (output / "manifest.json").exists()
    if phase == "admission":
        assert not output.exists()


def test_transform_identity_binds_complete_python_tree_except_approval_registry(tmp_path, monkeypatch):
    package = tmp_path / "picoagent"
    training = package / "training"
    training.mkdir(parents=True)
    (training / "prepared.py").write_text("# prepared source\n")
    approval = training / "prepared_approvals.py"
    approval.write_text("# reviewed registry, excluded to avoid hash cycle\n")
    context = package / "harness/context.py"
    context.parent.mkdir()
    context.write_text("# admission dependency\n")
    monkeypatch.setattr(prepared, "__file__", str(training / "prepared.py"))
    initial = prepared._identity()
    assert set(initial["source"]) == {"training/prepared.py", "harness/context.py"}
    approval.write_text("# changed approval registry\n")
    assert prepared._identity() == initial
    context.write_text("# changed admission dependency\n")
    assert prepared._identity() != initial
    context.write_text("# admission dependency\n")
    added = training / "new_dependency.py"
    added.write_text("# newly imported implementation\n")
    assert prepared._identity() != initial
    added.unlink()
    assert prepared._identity() == initial
    context.unlink()
    assert prepared._identity() != initial


def test_independent_stdlib_auditor_reconstructs_complete_raw_stream(source, tmp_path, monkeypatch):
    import importlib.util
    from picoagent.training.encoding import event_examples

    config = build(source, tmp_path)
    _, rows = verify_dataset(source.dataset_manifest, allow_smoke=True)
    reference = {"manifest_sha256": prepared.sha256_file(source.dataset_manifest), "encoded": {}}
    for split, records in rows.items():
        encoded, stats = encode_records(records, prepared._tokenizer(source), 1024)
        identifiers = [example["trace_id"] for row in records for example, _ in event_examples(row)]
        stream = hashlib.sha256()
        for identifier, item in zip(identifiers, encoded):
            stream.update((canonical_json({"trace_id": identifier, **item}) + "\n").encode())
        reference["encoded"][split] = {
            **stats.as_dict(), "tasks": len(records), "encoded_stream_sha256": stream.hexdigest()
        }
    script = Path(__file__).resolve().parents[1] / "scripts/audit_prepared_equivalence.py"
    spec = importlib.util.spec_from_file_location("independent_prepared_auditor", script)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)

    def forbidden(*args, **kwargs):
        raise AssertionError("independent auditor must not use the prepared decoder")

    monkeypatch.setattr(prepared, "_unpack", forbidden)
    _, result = auditor.decode_streams(Path(config.prepared_manifest), reference)
    assert result["streams"]["train"]["examples"] == 4
    assert result["streams"]["dev"]["examples"] == 1
    reference["encoded"]["train"]["encoded_stream_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw/prepared arrays disagree"):
        auditor.decode_streams(Path(config.prepared_manifest), reference)


def test_raw_training_admission_still_uses_original_verifier(source, monkeypatch):
    pytest.importorskip("torch")
    from picoagent.training import train

    calls = []
    original = train.verify_dataset

    def verify(*args, **kwargs):
        calls.append("raw")
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw path must not use prepared admission")

    def stop_before_device(*args, **kwargs):
        calls.append("device_boundary")
        raise RuntimeError("intentional stop before device allocation")

    monkeypatch.setattr(train, "verify_dataset", verify)
    monkeypatch.setattr(prepared, "load_prepared_dataset", forbidden)
    monkeypatch.setattr(train, "resolve_device", stop_before_device)
    with pytest.raises(RuntimeError, match="intentional stop"):
        train.run_training(source)
    assert calls == ["raw", "device_boundary"]


@pytest.mark.skipif(
    os.environ.get("PICOAGENT_RUN_ML_TESTS") != "1",
    reason="optional tiny CPU prepared/raw/resume training equivalence",
)
def test_prepared_training_resume_matches_raw_weights_and_scheduler(tmp_path):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from picoagent.training.smoke import run_smoke
    from picoagent.training.train import run_training

    reference = run_smoke(
        tmp_path / "raw", device="cpu", dropout=0.1, train_records=3, max_steps=4
    )
    source = TrainingConfig.load(tmp_path / "raw/smoke-config.json")
    path = prepared.build_prepared_dataset(source, tmp_path / "tokens")
    config = dataclasses.replace(
        source,
        output_dir=str(tmp_path / "prepared-run"),
        prepared_manifest=str(path),
        prepared_manifest_sha256=prepared.sha256_file(path),
    )
    paused = run_training(config, segment_steps=1)
    assert paused["status"] == "paused"
    assert paused["global_step"] == 1
    resumed = run_training(
        config, resume_from_checkpoint=str(Path(config.output_dir) / "checkpoint-1")
    )
    assert resumed["status"] == "completed"
    assert resumed["global_step"] == 4
    expected = load_file(str(Path(reference["artifact"]) / "model.safetensors"))
    actual = load_file(str(Path(resumed["artifact"]) / "model.safetensors"))
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in actual)
    def equal_state(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_state(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert type(left) is type(right) and len(left) == len(right)
            for x, y in zip(left, right):
                equal_state(x, y)
        else:
            assert left == right

    # These are trusted state files created by this test, not artifact inputs.
    for filename in ("scheduler.pt", "optimizer.pt", "rng_state.pth"):
        equal_state(
            torch.load(tmp_path / "raw/run/checkpoint-4" / filename, weights_only=False),
            torch.load(tmp_path / "prepared-run/checkpoint-4" / filename, weights_only=False),
        )
    assert (tmp_path / "prepared-run/prepared_snapshot/manifest.json").read_bytes() == path.read_bytes()
