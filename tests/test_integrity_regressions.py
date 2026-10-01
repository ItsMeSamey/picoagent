"""Regression checks for trusted helper isolation and pinned adapter inference.

ML libraries and container execution are mocked: these are boundary-contract
tests, not claims of a real model rollout or a Docker/Podman integration run.
"""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from picoagent.harness.sandbox import ContainerSandbox, ExecutionResult, SandboxLimits
from picoagent.harness.tools import ToolRegistry
from picoagent.inference import HFPolicy


def test_artifact_reader_uses_isolated_python_before_importing_trusted_helpers():
    sandbox = object.__new__(ContainerSandbox)
    sandbox.limits = SandboxLimits()
    observed = []

    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        return ExecutionResult(stdout="<svg />", stderr="", exit_code=0)

    sandbox.run = run
    assert sandbox.read_file("output/chart.svg") == "<svg />"
    argv, kwargs = observed[0]
    # Without -I, /workspace/pathlib.py can print a forged artifact and exit
    # before the trusted reader ever opens output/chart.svg.
    assert argv[:3] == ["python", "-I", "-c"]
    assert "import json,pathlib,sys" in argv[3]
    assert kwargs["model_generated"] is False
    assert json.loads(kwargs["stdin"])["path"] == "output/chart.svg"


def test_runtime_probe_uses_isolated_python():
    sandbox = object.__new__(ContainerSandbox)
    observed = []

    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        return ExecutionResult(
            stdout="picoagent-sandbox-ready\n", stderr="", exit_code=0,
            container_id="c" * 64,
        )

    sandbox.run = run
    assert sandbox.probe().container_id == "c" * 64
    assert observed[0][0][:3] == ["python", "-I", "-c"]
    assert observed[0][1]["model_generated"] is False


@pytest.mark.parametrize("name,arguments", [
    ("python", {"code": "print(1)"}),
    ("write_file", {"path": "output/x.txt", "content": "safe fixture"}),
])
def test_tool_python_helpers_ignore_workspace_module_shadowing(name, arguments):
    observed = []

    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        return ExecutionResult(stdout="", stderr="", exit_code=0)

    backend = SimpleNamespace(allows_model_code=True, run=run)
    result = ToolRegistry(backend, None).dispatch(name, arguments)
    assert result["exit_code"] == 0
    assert observed[0][0][:3] == ["python", "-I", "-c"]
    assert observed[0][1]["model_generated"] is True


@pytest.fixture
def fake_ml(monkeypatch):
    calls = {"tokenizer": [], "base": [], "adapter": [], "seed": []}

    class Model:
        def __init__(self):
            self.device = None
            self.evaluating = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluating = True
            return self

    base_model, adapted_model = Model(), Model()
    tokenizer = object()

    def load_tokenizer(*args, **kwargs):
        calls["tokenizer"].append((args, kwargs))
        return tokenizer

    def load_base(*args, **kwargs):
        calls["base"].append((args, kwargs))
        return base_model

    def load_adapter(*args, **kwargs):
        calls["adapter"].append((args, kwargs))
        return adapted_model

    torch = types.ModuleType("torch")
    torch.float32 = "mock-fp32"
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_base)
    transformers.set_seed = calls["seed"].append
    peft = types.ModuleType("peft")
    peft.PeftModel = SimpleNamespace(from_pretrained=load_adapter)
    for name, module in (("torch", torch), ("transformers", transformers), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        calls=calls, tokenizer=tokenizer, base=base_model, adapted=adapted_model,
    )


def test_local_adapter_loads_explicit_pinned_base_before_attaching(tmp_path, fake_ml):
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "example/base", "revision": None,
    }))
    revision = "a" * 40
    (tmp_path / "picoagent_inference.json").write_text(json.dumps({
        "mode": "qlora", "base_model_id": "example/base",
        "base_model_revision": revision,
    }))

    policy = HFPolicy(str(tmp_path), device="cpu")

    # Loading the adapter path with AutoModelForCausalLM would let Transformers
    # re-resolve the base model at its default branch instead of this exact SHA.
    assert fake_ml.calls["base"] == [(("example/base",), {
        "revision": revision, "torch_dtype": "mock-fp32", "trust_remote_code": False,
    })]
    assert fake_ml.calls["adapter"] == [((fake_ml.base, str(tmp_path)), {})]
    assert policy.model is fake_ml.adapted
    assert policy.model.device == "cpu"
    assert policy.model.evaluating
    assert policy.tokenizer is fake_ml.tokenizer


def test_local_adapter_without_pin_metadata_fails_before_loading_base(tmp_path, fake_ml):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned.*metadata"):
        HFPolicy(str(tmp_path), device="cpu")
    assert not fake_ml.calls["base"]
    assert not fake_ml.calls["adapter"]


@pytest.mark.parametrize("mode,revision", [
    ("qlora", "main"), ("qlora", ""), ("full", "a" * 40),
])
def test_local_adapter_rejects_unpinned_or_wrong_mode_metadata(tmp_path, fake_ml, mode, revision):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "picoagent_inference.json").write_text(json.dumps({
        "mode": mode, "base_model_id": "example/base", "base_model_revision": revision,
    }))
    with pytest.raises(ValueError, match="exact pinned revision"):
        HFPolicy(str(tmp_path), device="cpu")
    assert not fake_ml.calls["base"]
    assert not fake_ml.calls["adapter"]
