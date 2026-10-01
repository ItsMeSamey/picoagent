import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("picoagent_kaggle_job", SCRIPTS / "kaggle_job.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tmp_path):
    root = tmp_path / "source"
    (root / "src").mkdir(parents=True)
    (root / "src" / "fixture.py").write_text("print('fixture')\n")
    (root / ".env").write_text("SECRET=must-not-upload")
    return root


def test_private_gpu_smoke_builder(tmp_path):
    destination = tmp_path / "job"
    module.build(source(tmp_path), destination, "owner", "picoagent-smoke")
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    program = (destination / "main.py").read_text()
    compile(program, "main.py", "exec")
    assert 'refusing silent CPU fallback' in program
    receipt = json.loads((destination / "source-receipt.json").read_text())
    assert ".env" not in receipt["manifest"]["files"]


def test_training_output_is_in_saved_kaggle_directory(tmp_path):
    destination = tmp_path / "job"
    module.build(source(tmp_path), destination, "owner", "picoagent-train", "configs/train.json")
    program = (destination / "main.py").read_text()
    compile(program, "main.py", "exec")
    assert 'training_config["output_dir"] = "/kaggle/working/picoagent-training"' in program


def test_builder_rejects_existing_job_and_bad_owner(tmp_path):
    root = source(tmp_path)
    with pytest.raises(ValueError):
        module.build(root, tmp_path / "bad", "../owner", "slug")
    destination = tmp_path / "job"
    module.build(root, destination, "owner", "slug")
    with pytest.raises(FileExistsError):
        module.build(root, destination, "owner", "slug")
