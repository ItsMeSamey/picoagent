"""Trusted, pure checks. Never execute generated code or shell commands here."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from typing import Any

from .schema import canonical_json


def check_task_result(task: dict[str, Any], final_content: str, *, artifacts: dict[str, str] | None = None,
                      kv: dict[str, Any] | None = None) -> dict[str, Any]:
    oracle = task["oracle"]
    failures: list[str] = []
    kind = oracle["kind"]
    expected = oracle["expected"]
    if kind == "json_exact":
        try:
            result = json.loads(final_content)
            # Canonical comparison distinguishes True from 1 and rejects nonfinite values.
            if canonical_json(result) != canonical_json(expected):
                failures.append("JSON answer differs from expected value")
        except (ValueError, TypeError):
            failures.append("final answer must be exactly one JSON value without markdown")
    elif kind == "text_exact":
        if final_content.strip() != expected:
            failures.append("text answer differs from expected value")
    elif kind == "svg":
        path = oracle.get("artifact_path")
        svg = (artifacts or {}).get(path, "") if path else final_content
        if not svg:
            failures.append("required SVG artifact is missing")
        else:
            failures.extend(_check_svg(svg, expected))
    else:
        failures.append("unknown oracle kind")
    for key, value in oracle.get("kv_expected", {}).items():
        if kv is None or key not in kv or canonical_json(kv[key]) != canonical_json(value):
            failures.append(f"KV postcondition mismatch: {key}")
    return {"passed": not failures, "checks": [kind] + (["kv_postconditions"] if "kv_expected" in oracle else []),
            "failures": failures, "oracle_version": "1"}


def _check_svg(svg: str, expected: dict[str, Any]) -> list[str]:
    if len(svg) > 1_000_000 or "<!DOCTYPE" in svg.upper() or "<!ENTITY" in svg.upper():
        return ["SVG contains forbidden DTD/entity or exceeds size bound"]
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return ["invalid SVG XML"]
    def local(tag):
        return tag.rsplit("}", 1)[-1]
    if local(root.tag) != "svg":
        return ["artifact root must be svg"]
    failures: list[str] = []
    titles = ["".join(node.itertext()) for node in root.iter() if local(node.tag) == "title"]
    if titles != [expected["title"]]:
        failures.append("SVG must have the exact accessible title")
    for node in root.iter():
        if local(node.tag) in {"script", "foreignObject", "image", "use"} or any(
                key.lower().startswith("on") or key.rsplit("}", 1)[-1] == "href" for key in node.attrib):
            failures.append("SVG contains active content or an external reference")
            break
    rects = [node for node in root.iter() if local(node.tag) == "rect" and node.get("data-label")]
    if len(rects) != len(expected["bars"]):
        failures.append("wrong number of labeled bars")
    found = {node.get("data-label"): node for node in rects}
    for label, width in expected["bars"].items():
        try:
            actual = float(found[label].attrib["width"])
            if not math.isfinite(actual) or actual != width:
                failures.append(f"incorrect bar width: {label}")
        except (KeyError, ValueError):
            failures.append(f"missing/nonnumeric bar width: {label}")
    if "order" in expected:
        try:
            positions = [float(found[label].attrib["y"]) for label in expected["order"]]
            if not all(math.isfinite(y) for y in positions) or not all(a < b for a, b in zip(positions, positions[1:])):
                failures.append("incorrect vertical bar order")
        except (KeyError, ValueError):
            failures.append("missing/nonnumeric bar position")
    return failures
