"""Original documentation-execution tasks, a separate unexecuted expansion track.

These use real installed help/pydoc output or original module source, not a
fictional manual returned as if an installed program had executed it.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

from .generators import GENERATOR_VERSION
from .schema import SCHEMA_VERSION, canonical_json, content_hash, validate_task

TOOL_SPLIT_POLICY = {
    "tool_docs.sort_reverse": "train",
    "tool_docs.pydoc_affine": "train",
    "tool_docs.cut_fields": "dev",
    "tool_docs.source_index": "dev",
    "tool_docs.uniq_counts": "test",
    "tool_docs.pydoc_window": "test",
}


def generate_tool_task(family: str, seed: int) -> dict[str, Any]:
    if family not in TOOL_SPLIT_POLICY or type(seed) is not int or seed < 0:
        raise ValueError("unknown tool documentation family or invalid seed")
    token = hashlib.sha256(f"tool-docs-v1:{family}:{seed}".encode()).hexdigest()[:8]
    rng = random.Random(int(token, 16))
    env: dict[str, Any] = {"files": {}, "kv": {}, "docs": []}
    plan: list[dict[str, Any]] = []
    oracle: dict[str, Any] = {"kind": "text_exact"}
    domain = "bash"
    if family == "tool_docs.sort_reverse":
        numbers = [rng.randint(-30, 40) for _ in range(18)]
        env["files"]["input/numbers.txt"] = "\n".join(map(str, numbers)) + "\n"
        prompt = (f"CLI task {token}. Read sort --help to choose the correct options. Sort input/numbers.txt by numeric value, "
                  "largest first, removing duplicates. Return only the resulting newline-separated integers.")
        oracle["expected"] = "\n".join(map(str, sorted(set(numbers), reverse=True)))
        plan = [{"name": "bash", "arguments": {"command": "sort --help"}},
                {"name": "bash", "arguments": {"command": "sort -nr -u input/numbers.txt"}}]
    elif family == "tool_docs.cut_fields":
        rows = [(f"row{i}", f"shade{rng.randint(1,9)}", str(rng.randint(10,99))) for i in range(7)]
        env["files"]["input/records.txt"] = "\n".join(";".join(row) for row in rows) + "\n"
        prompt = (f"CLI task {token}. Read cut --help. Extract only the second semicolon-delimited field from input/records.txt, "
                  "preserving line order. Return only one extracted field per line.")
        oracle["expected"] = "\n".join(row[1] for row in rows)
        plan = [{"name": "bash", "arguments": {"command": "cut --help"}},
                {"name": "bash", "arguments": {"command": "cut -d ';' -f 2 input/records.txt"}}]
    elif family == "tool_docs.uniq_counts":
        values = [rng.choice(["birch", "cedar", "elm", "fir"]) for _ in range(18)]
        env["files"]["input/labels.txt"] = "\n".join(values) + "\n"
        prompt = (f"CLI task {token}. Read uniq --help. Sort input/labels.txt alphabetically, then use uniq to print each unique "
                  "label prefixed by its count. Return exactly the command's output, with only outer whitespace optional.")
        oracle["expected"] = "\n".join(f"{values.count(value):7} {value}" for value in sorted(set(values))).strip()
        plan = [{"name": "bash", "arguments": {"command": "uniq --help"}},
                {"name": "bash", "arguments": {"command": "sort input/labels.txt | uniq -c"}}]
    else:
        domain = "python"
        module = "local_api_" + token
        numbers = [rng.randint(-15, 20) for _ in range(8)]
        env["files"]["input/values.json"] = canonical_json(numbers)
        if family == "tool_docs.pydoc_affine":
            multiplier, offset = rng.randint(2, 6), rng.randint(-5, 8)
            before = rng.choice([True, False])
            expression = f"(value + {offset}) * {multiplier}" if before else f"value * {multiplier} + {offset}"
            semantics = f"Add {offset} to the value, then multiply by {multiplier}." if before else f"Multiply the value by {multiplier}, then add {offset}."
            function_name = "convert_" + token[:4]
            source = f'"""Original local numeric transform API.\nAPI callable: {function_name}.\n{semantics}\n"""\n\ndef {function_name}(value):\n    """{semantics}"""\n    return {expression}\n'
            expected = [(value + offset) * multiplier if before else value * multiplier + offset for value in numbers]
            action = "affine"
            request = f"Read this module's documentation using python -m pydoc {module}. Discover the documented conversion callable and use it on every input value, preserving order"
            read = f"python -m pydoc {module}"
        elif family == "tool_docs.source_index":
            base = rng.choice([0, 1])
            source = (f'"""Original indexed selection API. Public positions are {base}-based.\n"""\n\ndef lookup(values, position):\n'
                      f'    """Return the entry at {base}-based position; no negative positions."""\n'
                      f'    if position < {base}:\n        raise ValueError("position below origin")\n    return values[position - {base}]\n')
            expected = [numbers[2]]
            action = "index"
            request = f"Read {module}.py source to learn its public indexing convention. Use its documented selection API to return the third input element in original order, adjusting the position argument to that convention"
            read = f"cat {module}.py"
        else:
            inclusive = rng.choice([True, False])
            stop_description = "inclusive" if inclusive else "exclusive"
            extra = 1 if inclusive else 0
            source = (f'"""Original range selection API. Indices are zero-based; stop is {stop_description}.\n"""\n\ndef select(values, start, stop):\n'
                      f'    """Select start through stop, with {stop_description} stop."""\n    return values[start:stop + {extra}]\n')
            expected = numbers[2:5]
            action = "window"
            request = f"Read python -m pydoc {module} to learn its endpoint convention. Use its documented selection API to return the third through fifth input elements inclusive, adjusting both arguments as needed"
            read = f"python -m pydoc {module}"
        env["files"][module + ".py"] = source
        prompt = (f"Local API task {token}. Original module {module}.py and input/values.json are in the workspace. "
                  f"{request}. Return only JSON with key values holding the resulting list.")
        oracle = {"kind": "json_exact", "expected": {"values": expected}}
        plan = [{"name": "bash", "arguments": {"command": read}},
                {"name": "python", "arguments": {}, "derive_local_api_from_observation": action}]
    final = canonical_json(oracle["expected"]) if oracle["kind"] == "json_exact" else oracle["expected"]
    task = {"schema_version": SCHEMA_VERSION, "task_id": f"{family}:{seed:08d}", "family": family, "template_id": family + ".v1",
            "domain": domain, "split": TOOL_SPLIT_POLICY[family], "seed": seed, "prompt": prompt,
            "environment": env, "oracle": oracle, "reference": {"plan": plan, "final": final},
            "provenance": {"source": "original_procedural", "benchmark": False, "generator_version": GENERATOR_VERSION,
                           "curriculum_track": "installed_help_and_original_api_v1", "origin": "Original tasks using installed help or original source; no benchmark text"}}
    task["input_sha256"] = content_hash({"prompt": prompt, "environment": env})
    validate_task(task)
    return task


def generate_tool_tasks(*, seeds_per_family: int = 8) -> list[dict[str, Any]]:
    if seeds_per_family < 1:
        raise ValueError("seeds_per_family must be positive")
    return [generate_tool_task(family, seed) for family in sorted(TOOL_SPLIT_POLICY) for seed in range(seeds_per_family)]
