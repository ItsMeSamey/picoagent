"""Runtime pruning binds exact derived exports to durable acknowledgements."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import checkpoint_sync as sync  # noqa: E402
from colab_run import collect  # noqa: E402
from picoagent.training.evaluation import EVALUATION_SCHEMA  # noqa: E402
from picoagent.training.provenance import checkpoint_evidence  # noqa: E402


def make_checkpoint(run, step):
    checkpoint = run / f"checkpoint-{step}"
    checkpoint.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(f"{name}:{step}".encode())
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": step, "log_history": [{"step": step, "eval_loss": 1.0 / step}],
    }))
    checkpoint_evidence(checkpoint, sync.sha256(run / "run_manifest.json"))
    return checkpoint


def save_evaluation(run, step=1):
    path = run / "evaluations" / f"checkpoint-{step}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "schema": EVALUATION_SCHEMA, "checkpoint": f"checkpoint-{step}", "global_step": step,
        "checkpoint_manifest_sha256": sync.sha256(run / f"checkpoint-{step}/checkpoint_manifest.json"),
        "metrics": {"eval_loss": 0.001},
    }))
    return path


def acknowledge(run, exports):
    acknowledged = {}
    for checkpoint in sorted(run.glob("checkpoint-*")):
        bundle = sync.pack_checkpoint(run, checkpoint.name, exports, chunk_bytes=97)
        evaluation = sync.evaluation_record(bundle)
        acknowledged[checkpoint.name] = {
            "checkpoint_manifest_sha256": sync.sha256(checkpoint / "checkpoint_manifest.json"),
            "run_manifest_sha256": sync.sha256(run / "run_manifest.json"),
            "transfer_manifest_sha256": sync.sha256(bundle / sync.BUNDLE_MANIFEST),
            "evaluation_sha256": evaluation["sha256"] if evaluation else None,
        }
    return acknowledged


@pytest.fixture
def trees(tmp_path):
    run, exports = tmp_path / "run", tmp_path / "exports"
    run.mkdir()
    (run / "run_manifest.json").write_text('{"run":"runtime-retention"}')
    for step in range(1, 6):
        make_checkpoint(run, step)
    acknowledged = acknowledge(run, exports)
    return run, exports, acknowledged


def prune(trees):
    return sync.prune_runtime_exports(*trees, best_checkpoint="checkpoint-5")


def assert_untouched(trees):
    run, exports, _ = trees
    for step in range(1, 6):
        assert (run / f"checkpoint-{step}").is_dir()
        assert (exports / f"checkpoint-{step}").exists()


def rewrite_manifest(trees, transform, step=3):
    _, exports, acknowledged = trees
    name = f"checkpoint-{step}"
    path = exports / name / sync.BUNDLE_MANIFEST
    manifest = json.loads(path.read_text())
    transform(manifest)
    path.write_text(json.dumps(manifest))
    acknowledged[name]["transfer_manifest_sha256"] = sync.sha256(path)


def test_prunes_only_corresponding_acknowledged_exports_and_keeps_unrelated(trees):
    run, exports, _ = trees
    for root in (run, exports):
        (root / "traces").mkdir()
        (root / "traces/raw.jsonl").write_text("preserve\n")
        (root / ".incoming").mkdir()
        (root / ".incoming/download.partial").write_text("partial")
        (root / "controller.log").write_text("logs")
    (exports / "checkpoint-99").mkdir()
    (exports / "checkpoint-99/unrelated").write_text("not acknowledged")
    result = prune(trees)
    assert result == {"pruned_runtime": ["checkpoint-1", "checkpoint-2", "checkpoint-3"],
                      "pending_evaluations": []}
    for root in (run, exports):
        assert (root / "checkpoint-4").is_dir()
        assert (root / "checkpoint-5").is_dir()
        assert (root / "traces/raw.jsonl").read_text() == "preserve\n"
        assert (root / ".incoming/download.partial").read_text() == "partial"
        assert (root / "controller.log").read_text() == "logs"
    assert (exports / "checkpoint-99/unrelated").read_text() == "not acknowledged"


@pytest.mark.parametrize("key", ["checkpoint_manifest_sha256", "transfer_manifest_sha256", "run_manifest_sha256"])
def test_acknowledgement_hash_mismatch_aborts_whole_plan(trees, key):
    trees[2]["checkpoint-3"][key] = "0" * 64
    with pytest.raises(ValueError):
        prune(trees)
    assert_untouched(trees)


@pytest.mark.parametrize("identity", ["checkpoint", "run"])
def test_export_manifest_identity_mismatch_even_with_acknowledged_digest(trees, identity):
    def mutate(manifest):
        if identity == "checkpoint":
            manifest["checkpoint"] = "checkpoint-2"
            manifest["files"]["checkpoint-2/checkpoint_manifest.json"] = manifest["files"]["checkpoint-3/checkpoint_manifest.json"]
        else:
            manifest["files"]["run_manifest.json"]["sha256"] = "0" * 64
    rewrite_manifest(trees, mutate)
    with pytest.raises(ValueError):
        prune(trees)
    assert_untouched(trees)


def test_exact_manifest_bytes_not_merely_equivalent_json(trees):
    manifest = trees[1] / "checkpoint-3" / sync.BUNDLE_MANIFEST
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="transfer manifest"):
        prune(trees)
    assert_untouched(trees)


def test_corrupted_chunk_blocks_all_deletion(trees):
    chunk = next((trees[1] / "checkpoint-3/chunks").iterdir())
    chunk.write_bytes(b"x" * chunk.stat().st_size)
    with pytest.raises(ValueError, match="chunk changed"):
        prune(trees)
    assert_untouched(trees)


def test_corrupted_checkpoint_payload_blocks_all_deletion(trees):
    payload = trees[0] / "checkpoint-3/model.safetensors"
    payload.write_bytes(b"x" * payload.stat().st_size)
    with pytest.raises(ValueError, match="integrity"):
        prune(trees)
    assert_untouched(trees)


def test_transfer_file_identity_must_match_acknowledged_checkpoint_seal(trees):
    rewrite_manifest(trees, lambda manifest: manifest["files"]["checkpoint-3/model.safetensors"].update({
        "sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="file identities"):
        prune(trees)
    assert_untouched(trees)


@pytest.mark.parametrize("relative", ["trajectory.jsonl", "events.ndjson", "trace.log", "controller.log",
                                      "chunks/interrupted.partial", ".incoming/cache", "empty/"])
def test_unexpected_bundle_entries_are_never_swept_up(trees, relative):
    path = trees[1] / "checkpoint-3" / relative
    path.parent.mkdir(exist_ok=True)
    if relative.endswith("/"):
        path.mkdir()
    else:
        path.write_text("preserve")
    with pytest.raises(ValueError):
        prune(trees)
    assert path.exists()
    assert_untouched(trees)


@pytest.mark.parametrize("relative", [".git/untracked", "empty/", "trace.jsonl"])
def test_checkpoint_inventory_does_not_ignore_hidden_or_empty_entries(trees, relative):
    path = trees[0] / "checkpoint-3" / relative
    path.parent.mkdir(exist_ok=True)
    if relative.endswith("/"):
        path.mkdir()
    else:
        path.write_text("preserve")
    with pytest.raises(ValueError):
        prune(trees)
    assert_untouched(trees)


def test_trace_hidden_behind_content_address_is_not_deleted(trees):
    rewrite_manifest(trees, lambda manifest: manifest["files"].update({
        "source_snapshot/trace.jsonl": manifest["files"]["run_manifest.json"],
    }))
    with pytest.raises(ValueError, match="trace"):
        prune(trees)
    assert_untouched(trees)


@pytest.mark.parametrize("root_index,relative", [(0, "checkpoint-3/pipe"), (1, "checkpoint-3/chunks/pipe")])
def test_special_files_fail_before_hashing_or_pruning(trees, root_index, relative):
    path = trees[root_index] / relative
    os.mkfifo(path)
    with pytest.raises(ValueError, match="special"):
        prune(trees)
    assert path.exists()
    assert_untouched(trees)


@pytest.mark.parametrize("root_index", [0, 1])
def test_symlink_candidate_does_not_delete_target(trees, tmp_path, root_index):
    path = trees[root_index] / "checkpoint-3"
    moved = tmp_path / "unrelated"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prune(trees)
    assert moved.is_dir()
    assert_untouched(trees)


def test_symlink_chunk_is_never_followed(trees, tmp_path):
    chunk = next((trees[1] / "checkpoint-3/chunks").iterdir())
    moved = tmp_path / "unrelated"
    chunk.rename(moved)
    chunk.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        prune(trees)
    assert moved.is_file()
    assert_untouched(trees)


@pytest.mark.parametrize("root_index", [0, 1])
def test_symlink_ancestor_is_rejected(trees, tmp_path, root_index):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    arguments = list(trees)
    arguments[root_index] = alias / trees[root_index].name
    with pytest.raises(ValueError, match="symlink"):
        prune(tuple(arguments))
    assert_untouched(trees)


def test_missing_bundle_or_file_collision_preserves_runtime_checkpoint(trees):
    bundle = trees[1] / "checkpoint-3"
    bundle.rename(trees[1] / "unrelated")
    with pytest.raises(FileNotFoundError):
        prune(trees)
    bundle.write_text("unrelated file")
    with pytest.raises(ValueError, match="directory"):
        prune(trees)
    assert (trees[0] / "checkpoint-1").is_dir()
    assert (trees[0] / "checkpoint-3").is_dir()
    assert bundle.read_text() == "unrelated file"


def test_late_evaluation_defers_every_candidate_and_export(trees):
    save_evaluation(trees[0])
    assert prune(trees) == {"pruned_runtime": [], "pending_evaluations": ["checkpoint-1"]}
    assert_untouched(trees)


@pytest.mark.parametrize("arrival_call", [3, 4])
def test_late_evaluation_during_plan_or_boundary_keeps_deferral(trees, monkeypatch, arrival_call):
    original = sync._runtime_prune_pair
    calls = 0
    def validate(*args):
        nonlocal calls
        result = original(*args)
        calls += 1
        if calls == arrival_call:
            save_evaluation(trees[0])
        return result
    monkeypatch.setattr(sync, "_runtime_prune_pair", validate)
    assert prune(trees) == {"pruned_runtime": [], "pending_evaluations": ["checkpoint-1"]}
    assert_untouched(trees)


def test_acknowledged_content_addressed_evaluation_can_be_pruned(trees):
    run, exports, acknowledged = trees
    save_evaluation(run, 3)
    acknowledged.update(acknowledge(run, exports))
    result = sync.prune_runtime_exports(run, exports, acknowledged, best_checkpoint="checkpoint-1")
    assert result["pruned_runtime"] == ["checkpoint-2", "checkpoint-3"]
    assert (run / "evaluations/checkpoint-3.json").is_file()
    assert not (exports / "checkpoint-3").exists()
    assert (exports / "checkpoint-1").is_dir()


def test_new_save_and_bundle_during_planning_are_not_added_to_deletions(trees, monkeypatch):
    original = sync._runtime_prune_pair
    saved = False
    def validate(*args):
        nonlocal saved
        result = original(*args)
        if not saved:
            saved = True
            make_checkpoint(trees[0], 6)
            sync.pack_checkpoint(trees[0], "checkpoint-6", trees[1])
        return result
    monkeypatch.setattr(sync, "_runtime_prune_pair", validate)
    assert prune(trees)["pruned_runtime"] == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
    assert (trees[0] / "checkpoint-6").is_dir()
    assert (trees[1] / "checkpoint-6").is_dir()


def test_byte_identical_replacement_bundle_is_not_the_planned_bundle(trees, monkeypatch):
    original = sync._runtime_prune_pair
    calls = 0
    def validate(*args):
        nonlocal calls
        calls += 1
        if calls == 4:
            path = trees[1] / "checkpoint-1"
            path.rename(trees[1] / "original")
            shutil.copytree(trees[1] / "original", path)
        return original(*args)
    monkeypatch.setattr(sync, "_runtime_prune_pair", validate)
    with pytest.raises(ValueError, match="changed before deletion"):
        prune(trees)
    assert_untouched(trees)


def test_export_rechecked_after_original_checkpoint_removal(trees, monkeypatch):
    original = sync.shutil.rmtree
    def remove(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path == trees[0] / "checkpoint-1":
            (trees[1] / "checkpoint-1/trace.jsonl").write_text("racing evidence")
    monkeypatch.setattr(sync.shutil, "rmtree", remove)
    with pytest.raises(ValueError, match="traces"):
        prune(trees)
    assert not (trees[0] / "checkpoint-1").exists()
    assert (trees[1] / "checkpoint-1/trace.jsonl").read_text() == "racing evidence"
    assert (trees[0] / "checkpoint-2").is_dir()


class FakeColab:
    def __init__(self):
        self.downloads = []

    def execute(self, code):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, "fake-colab.py", "exec"), {})
        return json.loads(stream.getvalue().split("PICOAGENT_RESULT=")[-1])

    def transfer(self):
        downloads = self.downloads
        class AbsoluteLocal(sync.LocalTransfer):
            def download(self, remote, local):
                downloads.append(remote)
                super().download("/" + remote.lstrip("/"), local)
        return AbsoluteLocal()


def collect_args(trees, tmp_path):
    return (str(Path(__file__).resolve().parents[1]), str(trees[0]), str(trees[1]),
            tmp_path / "durable", True)


def test_repeated_collection_keeps_retained_bundles_unchanged_without_repacking(trees, tmp_path):
    client = FakeColab()
    args = collect_args(trees, tmp_path)
    assert collect(client, *args, prune=True)["pruned_runtime"] == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
    before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
              for path in trees[1].rglob("*") if path.is_file()}
    client.downloads.clear()
    assert collect(client, *args, prune=True)["pruned_runtime"] == []
    after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
             for path in trees[1].rglob("*") if path.is_file()}
    assert before == after
    assert client.downloads and all(path.endswith(sync.BUNDLE_MANIFEST) for path in client.downloads)
    assert all((tmp_path / "durable" / f"checkpoint-{step}").is_dir() for step in range(1, 6))


@pytest.mark.parametrize("collision", ["same", "ancestor", "descendant", "parent_traversal"])
def test_collect_rejects_overlapping_roots_before_mutation(trees, tmp_path, collision):
    args = list(collect_args(trees, tmp_path))
    run = trees[0]
    args[2] = str({"same": run, "ancestor": run.parent, "descendant": run / "exports",
                   "parent_traversal": run / ".." / "exports"}[collision])
    before = set(run.rglob("*"))
    with pytest.raises(ValueError):
        collect(FakeColab(), *args, prune=True)
    assert set(run.rglob("*")) == before
    assert not (tmp_path / "durable").exists()
    assert_untouched(trees)


def test_new_save_between_collection_and_prune_is_unacknowledged_and_preserved(trees, tmp_path):
    class RacingColab(FakeColab):
        calls = 0
        def execute(self, code):
            self.calls += 1
            if self.calls == 2:
                make_checkpoint(trees[0], 6)
                sync.pack_checkpoint(trees[0], "checkpoint-6", trees[1])
            return super().execute(code)
    result = collect(RacingColab(), *collect_args(trees, tmp_path), prune=True)
    assert result["pruned_runtime"] == ["checkpoint-1", "checkpoint-2", "checkpoint-3", "checkpoint-4"]
    assert "checkpoint-6" not in result["verified_checkpoints"]
    assert (trees[0] / "checkpoint-6").is_dir()
    assert (trees[1] / "checkpoint-6").is_dir()


@pytest.mark.parametrize("collision", ["bundle_symlink", "chunks_symlink", "fifo"])
def test_collect_preflights_all_export_trees_before_packing(trees, tmp_path, collision):
    _, exports, _ = trees
    # The earlier missing bundle would be repacked if validation were per-bundle.
    shutil.rmtree(exports / "checkpoint-1")
    bundle = exports / "checkpoint-3"
    (bundle / sync.BUNDLE_MANIFEST).unlink()
    outside = tmp_path / "unrelated"
    if collision == "bundle_symlink":
        bundle.rename(outside)
        bundle.symlink_to(outside, target_is_directory=True)
    elif collision == "chunks_symlink":
        (bundle / "chunks").rename(outside)
        (bundle / "chunks").symlink_to(outside, target_is_directory=True)
    else:
        outside.mkdir()
        os.mkfifo(bundle / "chunks/pipe")
    (outside / "preserve.log").write_text("outside evidence")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in outside.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="symlinks and special files"):
        collect(FakeColab(), *collect_args(trees, tmp_path), prune=True)
    assert not (exports / "checkpoint-1").exists()
    assert not (tmp_path / "durable").exists()
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in outside.rglob("*") if path.is_file()}
    assert all((trees[0] / f"checkpoint-{step}").is_dir() for step in range(1, 6))


def test_collection_preflight_allows_regular_partial_files_without_removing_them(trees, tmp_path):
    bundle = trees[1] / "checkpoint-3"
    (bundle / sync.BUNDLE_MANIFEST).unlink()
    partial = bundle / "chunks/interrupted.partial"
    partial.write_text("retryable partial cache")
    before = partial.stat()
    result = collect(FakeColab(), *collect_args(trees, tmp_path), prune=False)
    assert len(result["verified_checkpoints"]) == 5
    assert partial.read_text() == "retryable partial cache"
    assert (partial.stat().st_ino, partial.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
