#!/usr/bin/env python3
"""Explicit, bounded CPU-only transport experiment; never inspect auth contents.

Run only with authorization for the single named standard CPU session and
synthetic 1 MiB transfer. No accelerator, high-memory, TLS, proxy or auth changes.
Only sanitized whitelisted receipts are persisted. Raw CLI output stays in RAM.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
CLI = WORKSPACE / ".bin/colab-safe"
PYTHON = WORKSPACE / ".tools/google-colab-cli/bin/python"
SESSION = "picoagent-transport-cpu-v1"
EVIDENCE = Path(__file__).with_name("20261001-colab-cpu-timeout60.json")
OVERALL_SECONDS = 840
REQUEST_TIMEOUT = "60"
SENTINEL = "PICOAGENT_TRANSPORT_PROBE="
ERROR_TYPES = ("AttributeError", "WebSocketConnectionClosedException", "ConnectionError",
               "ReadTimeout", "ConnectTimeout", "TimeoutError", "SSLError", "HTTPError", "ProxyError")


def main():
    started = time.monotonic()
    env = {**os.environ, "HOME": str(WORKSPACE / ".colab-home"), "REQUEST_TIMEOUT": REQUEST_TIMEOUT}
    report = {"schema": "picoagent.colab_cpu_transport_probe.v1", "session": SESSION,
              "requested_hardware": "CPU", "requested_shape": "Standard", "request_timeout_seconds": 60,
              "overall_deadline_seconds": OVERALL_SECONDS, "accelerator_requested": False,
              "security_or_auth_configuration_changed": False, "raw_cli_output_persisted": False,
              "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "operations": []}
    if EVIDENCE.exists():
        raise RuntimeError("Refusing to overwrite prior probe evidence")
    def save():
        temporary = EVIDENCE.with_suffix(".partial")
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, EVIDENCE)
    def run(label, args, *, timeout=130, closing=False):
        reserve = 0 if closing else 80
        remaining = OVERALL_SECONDS - (time.monotonic() - started) - reserve
        if remaining < 3:
            item = {"operation": label, "attempted": False, "reason": "overall_deadline_reserve"}
            report["operations"].append(item)
            save()
            return item, ""
        launched = time.monotonic()
        process = subprocess.Popen([str(CLI), *args], env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=min(timeout, remaining))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
        finally:
            # Reap only this invocation's possible leaked client threads/processes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        combined = stdout + "\n" + stderr
        item = {"operation": label, "attempted": True, "exit_code": process.returncode,
                "timed_out": timed_out, "wall_seconds": round(time.monotonic() - launched, 3),
                "error_types": [name for name in ERROR_TYPES if name in combined],
                "diagnostic_frames": re.findall(r"(?m)^\[picoagent (?:transport|cleanup|cli)\] ([A-Za-z0-9_.-]+:[0-9]+ [A-Za-z0-9_]+)$", combined),
                "allocation_refused": "Allocation refused" in combined,
                "session_ready": "[colab] Session READY." in stdout,
                "session_terminated": "[colab] Session terminated." in stdout,
                "session_not_found": f"Session '{SESSION}' not found" in stdout}
        receipts = []
        for line in stdout.splitlines():
            if line.startswith(SENTINEL):
                try:
                    receipt = json.loads(line[len(SENTINEL):])
                    allowed = {"probe", "value", "bytes", "sha256", "accelerator_devices_present"}
                    if set(receipt) <= allowed:
                        receipts.append(receipt)
                except (ValueError, TypeError):
                    pass
        item["receipts"] = receipts
        report["operations"].append(item)
        save()
        print(json.dumps(item, sort_keys=True), flush=True)
        return item, stdout
    created_or_uncertain = False
    try:
        versions = subprocess.run([str(PYTHON), "-c", "import importlib.metadata as m,json;from jupyter_kernel_client.constants import REQUEST_TIMEOUT;from jupyter_kernel_client.wsclient import KernelWebSocketClient;import inspect;print(json.dumps({'versions':{n:m.version(n) for n in ['google-colab-cli','jupyter-kernel-client','websocket-client']},'request_timeout':REQUEST_TIMEOUT,'websocket_constructor_timeout':inspect.signature(KernelWebSocketClient.__init__).parameters['timeout'].default}))"], env=env, capture_output=True, text=True, timeout=30, check=True)
        report["local_client_metadata"] = json.loads(versions.stdout)
        assert report["local_client_metadata"]["request_timeout"] == 60
        assert report["local_client_metadata"]["websocket_constructor_timeout"] == 60
        package = WORKSPACE / ".tools/google-colab-cli/lib/python3.12/site-packages"
        report["source_sha256"] = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in ["jupyter_kernel_client/constants.py", "jupyter_kernel_client/wsclient.py", "websocket/_app.py", "colab_cli/runtime.py", "colab_cli/commands/execution.py"]}
        save()
        preflight, output = run("sessions_before", ["sessions"], timeout=45)
        if preflight.get("exit_code") != 0 or f"[{SESSION}]" in output:
            report["aborted"] = "session inventory unavailable or requested name already existed"
            return
        # No GPU, TPU or high-memory flag: official resolve_runtime_options maps
        # this exact command to DEFAULT/NONE with standard memory.
        created_or_uncertain = True
        creation, _ = run("allocate_standard_cpu", ["new", "--session", SESSION], timeout=60)
        if creation.get("exit_code") != 0 or not creation["session_ready"]:
            report["aborted"] = "allocation not confirmed; no allocation retry"
            return
        status, output = run("verify_hardware", ["status", "--session", SESSION], timeout=45)
        matching = [line for line in output.splitlines() if line.startswith(f"[{SESSION}]")]
        hardware_ok = bool(matching) and all("| Hardware: CPU | Shape: Standard | Variant: DEFAULT" in line for line in matching)
        report["hardware_verified_cpu_standard"] = hardware_ok
        if status.get("exit_code") != 0 or not hardware_ok:
            report["aborted"] = "CPU/standard status not verified"
            return
        with tempfile.TemporaryDirectory(prefix="picoagent-transport-probe-", dir=WORKSPACE) as directory:
            temporary = Path(directory)
            def execute(index):
                path = temporary / f"probe-{index}.py"
                path.write_text("import glob,json\n" + "print(" + repr(SENTINEL) + "+json.dumps({'probe':" + str(index) + ",'value':sum(range(101)),'accelerator_devices_present':bool(glob.glob('/dev/nvidia[0-9]*')+glob.glob('/dev/accel[0-9]*'))}))\n")
                item, _ = run(f"execute_{index}", ["exec", "--session", SESSION, "--timeout", "30", "--file", str(path)])
                item["validated"] = item.get("exit_code") == 0 and item.get("receipts") == [{"probe": index, "value": 5050, "accelerator_devices_present": False}]
                save()
                return item["validated"]
            results = [execute(index) for index in range(1, 4)]
            payload = temporary / "payload.bin"
            payload.write_bytes(bytes(range(256)) * 4096)
            expected = hashlib.sha256(payload.read_bytes()).hexdigest()
            report["transfer_fixture"] = {"bytes": payload.stat().st_size, "sha256": expected,
                                           "kind": "synthetic repeating byte values 0..255"}
            remote = "content/picoagent-transport-probe-1mib.bin"
            upload, _ = run("upload_1mib", ["upload", "--session", SESSION, str(payload), remote], timeout=90)
            downloaded = temporary / "readback.bin"
            if upload.get("exit_code") == 0:
                download, _ = run("download_1mib", ["download", "--session", SESSION, remote, str(downloaded)], timeout=90)
                report["roundtrip_verified"] = download.get("exit_code") == 0 and downloaded.is_file() and downloaded.stat().st_size == 1048576 and hashlib.sha256(downloaded.read_bytes()).hexdigest() == expected
            else:
                report["roundtrip_verified"] = False
            # Repeat WebSocket connections after REST activity if any first call
            # succeeded; three consecutive failures already refute a reliable fix.
            if any(results):
                results.extend(execute(index) for index in range(4, 6))
            report["small_execute_results"] = results
            report["repeated_execute_verified"] = len(results) >= 3 and all(results)
            report["timeout60_demonstrably_reliable_on_this_cpu_probe"] = report["repeated_execute_verified"] and report["roundtrip_verified"]
            save()
    except Exception as error:
        report["probe_error_type"] = type(error).__name__
    finally:
        # Save operation evidence before deallocating exactly this CPU session.
        save()
        if created_or_uncertain:
            stopped, _ = run("stop_owned_cpu_session", ["stop", "--session", SESSION], timeout=60, closing=True)
            report["stop_confirmed"] = stopped.get("exit_code") == 0 and stopped.get("session_terminated", False)
            inventory, output = run("sessions_after", ["sessions"], timeout=30, closing=True)
            report["owned_session_absent_after_stop"] = inventory.get("exit_code") == 0 and f"[{SESSION}]" not in output
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        print(json.dumps({key: report.get(key) for key in ["aborted", "probe_error_type", "small_execute_results", "roundtrip_verified", "timeout60_demonstrably_reliable_on_this_cpu_probe", "stop_confirmed", "owned_session_absent_after_stop", "elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
