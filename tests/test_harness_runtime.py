import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from picoagent.harness import ContainerSandbox, KnowledgeStore, SandboxLimits, SandboxUnavailable, SearXNGSearch, ToolRegistry, TrustedLocalSandbox
from picoagent.harness.sandbox import safe_relative_path


class RuntimeTests(unittest.TestCase):
    def test_no_runtime_fails_closed(self):
        with patch("picoagent.harness.sandbox.shutil.which", return_value=None):
            with self.assertRaises(SandboxUnavailable):
                ContainerSandbox()

    def test_container_arguments_are_hardened_and_receipts_initially_empty(self):
        with tempfile.TemporaryDirectory() as root, patch("picoagent.harness.sandbox.shutil.which", return_value="/usr/bin/docker"):
            with ContainerSandbox(root, runtime="docker") as sandbox:
                command = sandbox.command(["bash", "-c", "echo test"], name="test")
                for flag in ["--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pull=never", "--cidfile", "--memory", "--pids-limit", "--cpus", "--user"]:
                    self.assertIn(flag, command)
                self.assertNotIn("--privileged", command)
                self.assertEqual(command[-3:], ["bash", "-c", "echo test"])
                self.assertIsNone(sandbox.runtime_metadata()["container_id"])
                sandbox.seed_files({"docs/reference.txt": "example"})
                self.assertEqual((sandbox.workspace / "docs/reference.txt").read_text(), "example")
                sandbox._executed = True
                with self.assertRaises(RuntimeError):
                    sandbox.seed_files({"late": "bad"})

    def test_fixture_paths_are_safe(self):
        for path in ["/etc/passwd", "../escape", "x/../../escape", "", ".", "a\\b", "bad\x00name"]:
            with self.assertRaises(ValueError):
                safe_relative_path(path)

    def test_local_helper_rejects_all_model_code(self):
        with tempfile.TemporaryDirectory() as root:
            backend = TrustedLocalSandbox(root)
            with self.assertRaises(PermissionError):
                backend.run([sys.executable, "-c", "print(1)"])
            tools = ToolRegistry(backend, KnowledgeStore(Path(root) / "knowledge.json"))
            result = tools.dispatch("bash", {"command": "echo should-not-run"})
            self.assertEqual(result["error"], "PermissionError")
            self.assertEqual(tools.dispatch("python", {"code": "print(1)"})["error"], "PermissionError")
            self.assertEqual(tools.dispatch("write_file", {"path": "x", "content": "x"})["error"], "PermissionError")

    def test_hand_authored_local_fixture_output_and_timeout_limits(self):
        with tempfile.TemporaryDirectory() as root:
            backend = TrustedLocalSandbox(root, limits=SandboxLimits(timeout_seconds=0.3, max_output_bytes=128))
            result = backend.run([sys.executable, "-c", "print('a'*5000)"], model_generated=False)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(result.truncated)
            self.assertLessEqual(len(result.stdout.encode()) + len(result.stderr.encode()), 128)
            self.assertEqual(result.backend, "trusted-local")
            result = backend.run([sys.executable, "-c", "import time; time.sleep(3)"], model_generated=False)
            self.assertTrue(result.timed_out)
            self.assertEqual(result.exit_code, 124)

    def test_knowledge_persistence_limits_and_copying(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "knowledge.json"
            store = KnowledgeStore(path, max_items=2, max_bytes=100)
            value = {"done": [1]}
            store.set("task:a", value)
            value["done"].append(2)
            self.assertEqual(KnowledgeStore(path).get("task:a"), {"done": [1]})
            store.set("task:b", "ok")
            with self.assertRaises(ValueError):
                store.set("c", "overflow")
            with self.assertRaises(ValueError):
                store.set("task:b", "x" * 200)
            self.assertEqual(list(store.list("task:")), ["task:a", "task:b"])
            self.assertTrue(store.delete("task:a"))
            self.assertFalse(store.delete("task:a"))

    def test_tools_validate_shapes_without_execution(self):
        with tempfile.TemporaryDirectory() as root:
            tools = ToolRegistry(None, KnowledgeStore(Path(root) / "kv.json"))
            for name, arguments in [("unknown", {}), ("bash", {}), ("bash", {"command": "a", "timeout": True}), ("bash", {"command": "a", "extra": 1}), ("knowledge", {"operation": "set", "key": "x"}), ("search", {"query": "q", "limit": 20})]:
                self.assertIn("error", tools.dispatch(name, arguments))
            self.assertEqual(tools.dispatch("knowledge", {"operation": "set", "key": "x", "value": {"a": 1}}), {"stored": "x"})
            self.assertEqual(tools.dispatch("knowledge", {"operation": "get", "key": "x"})["value"], {"a": 1})

    def test_searxng_http_request_uses_mock_and_bounds_response(self):
        class Opener:
            def open(self, request, timeout):
                self.request, self.timeout = request, timeout
                return io.BytesIO(json.dumps({"results": [{"title": "API", "url": "https://example.org/doc", "content": "docs"}]}).encode())
        opener = Opener()
        client = SearXNGSearch("https://search.example/search?categories=it", opener=opener)
        result = client.search("test & query", 1)
        self.assertTrue(result["untrusted"])
        self.assertEqual(result["results"][0]["title"], "API")
        self.assertIn("q=test+%26+query", opener.request.full_url)
        self.assertIn("format=json", opener.request.full_url)
        with self.assertRaises(ValueError):
            SearXNGSearch("https://user:password@example.org")
        with self.assertRaises(ValueError):
            SearXNGSearch("https://example.org", opener=opener, max_response_bytes=1).search("q")
