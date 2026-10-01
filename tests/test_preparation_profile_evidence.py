"""Validate preserved CPU profile evidence without running ML or trace tools."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/validation"


def normalized_tree(source):
    tree = ast.parse(source)

    class SplitImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return [ast.Import(names=[alias]) for alias in node.names]

    return ast.dump(SplitImports().visit(tree), include_attributes=False)


def test_readable_profiler_preserves_executed_logic_and_compiles():
    original = (DOCS / "profile_cpu_preparation.executed.py.txt").read_text()
    readable = (DOCS / "profile_cpu_preparation.py").read_text()
    compile(original, "executed-profiler", "exec")
    compile(readable, "readable-profiler", "exec")
    assert normalized_tree(original) == normalized_tree(readable)
    report = json.loads((DOCS / "20261001-cpu-preparation-profile.json").read_text())
    assert hashlib.sha256(original.encode()).hexdigest() == report["profiler_sha256"]


def test_profile_counts_and_recommendation_distinguish_estimates():
    report = json.loads((DOCS / "20261001-cpu-preparation-profile.json").read_text())
    recommendation = json.loads((DOCS / "20261001-cpu-preparation-recommendation.json").read_text())
    assert report["head"] == "c2f82ade626f1fc6501885315c507572fbd25bb7"
    assert report["manifest_sha256"] == "e5774693bc42815ab44858754303b3970c7e34f26a1140425dccdd5853ccfbd9"
    assert recommendation["head"] == report["head"]
    assert recommendation["data_manifest_sha256"] == report["manifest_sha256"]
    assert {split: values["examples"] for split, values in report["encoded"].items()} == {"train": 48702, "dev": 1203}
    assert sum(row["total_tokens"] for row in report["encoded"].values()) == 53929387
    assert sum(row["assistant_tokens"] for row in report["encoded"].values()) == 6146621
    assert max(row["maximum_length"] for row in report["encoded"].values()) == 3445
    assert recommendation["measured"]["verify_seconds"] == report["strict_verify_seconds"]
    evaluation = recommendation["checkpoint_evaluation"]
    assert evaluation["estimated_dev_evaluation_seconds"] == evaluation["parent_reported_dev_examples"] / evaluation["parent_reported_dev_examples_per_second"]
    assert "no active run modifications" in recommendation["scope"]
