import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("picoagent_kaggle_job", SCRIPTS / "kaggle_job.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from source_staging import extract_source_archive, file_hash  # noqa: E402


def put(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def source(tmp_path, *, native_opt_in=True):
    root = tmp_path / "source"
    put(root, "src/picoagent/fixture.py", b"def fixture(): return 'native SFT only'\n")
    put(root, ".env", b"SECRET=must-not-upload")
    train = put(root, "data/native/train.jsonl", b'{"split":"train"}\n')
    dev = put(root, "data/native/dev.jsonl", b'{"split":"dev"}\n')
    files = {}
    for path in (train, dev):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    snapshot = {
        "schema": "picoagent.native_teacher.dataset.v1",
        "lockbox_used": False,
        "splits": {"train": {"path": "train.jsonl", "records": 1}, "dev": {"path": "dev.jsonl", "records": 1}},
        "files": files,
    }
    put(root, "data/native/manifest.json", json.dumps(snapshot, sort_keys=True).encode() + b"\n")
    config = {
        "model_id": "HuggingFaceTB/SmolLM2-360M",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "dataset_manifest": "data/native/manifest.json",
        "output_dir": "runs/native-sft",
        "training_mode": "full",
        "smoke_test": False,
        "allow_native_teacher_observed": native_opt_in,
    }
    put(root, "configs/native.json", json.dumps(config, sort_keys=True).encode() + b"\n")
    return root


def test_inline_mode_is_small_private_fixture_smoke_only(tmp_path):
    root = source(tmp_path)
    destination = tmp_path / "kernel"
    module.build(root, destination, "owner", "picoagent-smoke")
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == []
    program = (destination / "main.py").read_text()
    compile(program, "main.py", "exec")
    assert "base64.b64decode" in program
    assert "process.wait(timeout=60)" in program
    assert "signal.SIGKILL" in program
    assert len(program.encode()) < 100_000
    receipt = json.loads((destination / "source-receipt.json").read_text())
    assert ".env" not in receipt["manifest"]["files"]
    package_receipt = json.loads((destination / "kernel-package-receipt.json").read_text())
    assert package_receipt["provider_calls_made"] is False
    assert package_receipt["kernel_privacy"].startswith("requested_private_structurally_only")


def test_private_input_dataset_is_chunked_pinned_and_not_embedded(tmp_path):
    root = source(tmp_path)
    kernel = tmp_path / "kernel"
    dataset = tmp_path / "private-input-package"
    module.build(
        root, kernel, "kernel-owner", "picoagent-native-sft", "configs/native.json",
        input_dataset="dataset-owner/native-private-bundle", data_output=dataset,
        dataset_manifest="data/native/manifest.json", dataset_license="unknown", chunk_bytes=128,
        segment_steps=5,
    )
    metadata = json.loads((kernel / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["dataset-owner/native-private-bundle"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    program = (kernel / "main.py").read_text()
    compile(program, "main.py", "exec")
    assert len(program.encode()) < 100_000
    assert "base64.b64decode" not in program
    assert "EXPECTED_TRANSFER_MANIFEST_SHA256" in program
    assert "EXPECTED_STAGING_HELPER_SHA256" in program
    assert "extract_source_archive" in program
    assert "native_sft_only" in program
    assert "smoke_test" not in program
    assert '"--segment-steps", \'5\'' in program
    assert '"--output-budget-bytes", \'20000000000\'' in program
    kernel_receipt = json.loads((kernel / "kernel-package-receipt.json").read_text())
    assert kernel_receipt["kernel_main_sha256"] == hashlib.sha256(program.encode()).hexdigest()
    assert kernel_receipt["staging_helper_sha256"] in program
    assert kernel_receipt["segment_steps"] == 5
    assert kernel_receipt["output_budget_bytes"] == module.KAGGLE_OUTPUT_BUDGET_BYTES
    assert kernel_receipt["requested_accelerator"] == "NvidiaTeslaT4"

    transfer_path = dataset / "transfer_manifest.json"
    transfer_raw = transfer_path.read_bytes()
    assert kernel_receipt["input_transfer_manifest_sha256"] == hashlib.sha256(transfer_raw).hexdigest()
    transfer = json.loads(transfer_raw)
    assert transfer["dataset_handle"] == "dataset-owner/native-private-bundle"
    assert transfer["bundle"]["schema"] == "picoagent.source-transfer.v1"
    bundle = transfer["bundle"]
    assert max(chunk["bytes"] for chunk in bundle["chunks"]) <= 128
    assert all(chunk["bytes"] <= module.KAGGLE_CHUNK_BYTES_LIMIT for chunk in bundle["chunks"])
    assert transfer["source_manifest_path"] == "data/native/manifest.json"
    manifest_meta = json.loads((dataset / "dataset-metadata.json").read_text())
    assert manifest_meta["id"] == "dataset-owner/native-private-bundle"
    assert manifest_meta["licenses"] == [{"name": "unknown"}]
    assert all(license_row["name"] != "CC0-1.0" for license_row in manifest_meta["licenses"])
    receipt = json.loads((dataset / "package-receipt.json").read_text())
    assert receipt["provider_calls_made"] is False
    assert "not provider-verified" in receipt["dataset_visibility"]
    assert receipt["transfer_manifest_sha256"] == hashlib.sha256(transfer_raw).hexdigest()

    archive_path = tmp_path / "reconstructed-source.tar.gz"
    digest = hashlib.sha256()
    size = 0
    with archive_path.open("xb") as output:
        for chunk in bundle["chunks"]:
            name = transfer["chunk_paths_by_sha256"][chunk["sha256"]]
            path = dataset / name
            assert path.stat().st_size == chunk["bytes"]
            assert file_hash(path) == chunk["sha256"]
            with path.open("rb") as source_stream:
                while block := source_stream.read(31):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
    assert size == bundle["archive"]["bytes"]
    assert digest.hexdigest() == bundle["archive"]["sha256"]
    materialized = tmp_path / "materialized"
    result = extract_source_archive(archive_path, str(materialized), bundle)
    assert result["verified"] is True
    assert (materialized / "configs/native.json").is_file()
    assert (materialized / "data/native/train.jsonl").read_bytes() == (root / "data/native/train.jsonl").read_bytes()
    assert not (materialized / ".env").exists()


def test_private_dataset_needs_explicit_native_sft_and_license(tmp_path):
    root = source(tmp_path, native_opt_in=False)
    with pytest.raises(ValueError, match="allow native_teacher_observed"):
        module.build(root, tmp_path / "kernel", "owner", "kernel", "configs/native.json",
                     input_dataset="owner/private-data", data_output=tmp_path / "dataset",
                     dataset_manifest="data/native/manifest.json", dataset_license="unknown", segment_steps=5)
    root = source(tmp_path / "missing-opt-in", native_opt_in=True)
    with pytest.raises(ValueError, match="requires --data-output"):
        module.build(root, tmp_path / "kernel-2", "owner", "kernel-2", "configs/native.json",
                     input_dataset="owner/private-data", dataset_manifest="data/native/manifest.json")
    with pytest.raises(ValueError, match="requires a positive --segment-steps"):
        module.build(root, tmp_path / "kernel-3", "owner", "kernel-3", "configs/native.json",
                     input_dataset="owner/private-data", data_output=tmp_path / "dataset-3",
                     dataset_manifest="data/native/manifest.json", dataset_license="unknown")
    with pytest.raises(ValueError, match="'other' license requires"):
        module._dataset_metadata("owner/private-data", "private-data", "other", None)


def test_artificial_plan_wrapper_manifest_uses_exact_train_dev_counts(tmp_path):
    root = source(tmp_path)
    wrapper_source = Path(__file__).resolve().parents[1] / "data/cli-compact-plan-view-v1/manifest.json"
    actual_wrapper = json.loads(wrapper_source.read_text())
    wrapper_path = put(root, "data/artificial/manifest.json", wrapper_source.read_bytes())
    config = json.loads((root / "configs/native.json").read_text())
    config.update({
        "dataset_manifest": "data/artificial/manifest.json",
        "allow_artificial_action_plans": True,
    })
    config_path = put(root, "configs/artificial.json", json.dumps(config).encode() + b"\n")

    _, _, selected, validated = module._validate_native_sft_inputs(root, "configs/artificial.json", "data/artificial/manifest.json")
    assert selected == "data/artificial/manifest.json"
    assert validated["schema"] == "picoagent.artificial_action_plan.dataset.v1"
    assert set(validated["counts"]) == {"train", "dev"}
    assert validated["counts"] == actual_wrapper["counts"]
    assert config_path.is_file() and wrapper_path.is_file()

    config["allow_artificial_action_plans"] = False
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="allow_artificial_action_plans opt-in"):
        module._validate_native_sft_inputs(root, "configs/artificial.json", "data/artificial/manifest.json")


@pytest.mark.parametrize("extra_split", ["test", "holdout"])
def test_native_manifest_rejects_any_extra_split_key(tmp_path, extra_split):
    root = source(tmp_path)
    manifest_path = root / "data/native/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"][extra_split] = {"path": "extra.jsonl", "records": 1}
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="exactly train and dev"):
        module._validate_native_sft_inputs(root, "configs/native.json", "data/native/manifest.json")


@pytest.mark.parametrize("extra_split", ["test", "validation"])
def test_artificial_counts_reject_any_extra_split_key(tmp_path, extra_split):
    root = source(tmp_path)
    wrapper_source = Path(__file__).resolve().parents[1] / "data/cli-compact-plan-view-v1/manifest.json"
    manifest = json.loads(wrapper_source.read_text())
    manifest["counts"][extra_split] = 0
    put(root, "data/artificial/manifest.json", json.dumps(manifest).encode() + b"\n")
    config = json.loads((root / "configs/native.json").read_text())
    config.update({"dataset_manifest": "data/artificial/manifest.json", "allow_artificial_action_plans": True})
    put(root, "configs/artificial.json", json.dumps(config).encode() + b"\n")
    with pytest.raises(ValueError, match="exactly train and dev counts"):
        module._validate_native_sft_inputs(root, "configs/artificial.json", "data/artificial/manifest.json")


def test_inline_archive_limit_rejects_large_payload_and_smoke_over_60(tmp_path):
    root = source(tmp_path)
    put(root, "src/picoagent/random.bin", os.urandom(48_000))
    destination = tmp_path / "oversized-kernel"
    with pytest.raises(ValueError, match="exceeding the 1024-byte fixture limit"):
        module.build(root, destination, "owner", "too-large", inline_archive_max_bytes=1024)
    assert not destination.exists()
    with pytest.raises(ValueError, match="60 seconds"):
        module.build(root, tmp_path / "bad-timeout", "owner", "timeout", smoke_timeout_seconds=61)
    with pytest.raises(ValueError, match="Training configs require private-input-dataset"):
        module.build(root, tmp_path / "inline-training", "owner", "inline-train", "configs/native.json")


def test_builder_rejects_existing_destination_and_unsafe_dataset_handle(tmp_path):
    root = source(tmp_path)
    destination = tmp_path / "kernel"
    module.build(root, destination, "owner", "first")
    with pytest.raises(FileExistsError):
        module.build(root, destination, "owner", "first")
    with pytest.raises(ValueError, match="simple Kaggle identifier"):
        module.build(root, tmp_path / "bad-handle", "owner", "bad", "configs/native.json",
                     input_dataset="../private", data_output=tmp_path / "dataset",
                     dataset_manifest="data/native/manifest.json", dataset_license="unknown", segment_steps=5)
