"""Fail-closed container execution. Local execution is ONLY for trusted fixtures.

This is a defense-in-depth baseline, not a guarantee against kernel/container
escapes. Run Docker/Podman rootless on a disposable worker for hostile workloads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
import json
import math
import re
from pathlib import Path, PurePosixPath
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Mapping, Sequence
import uuid


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 15.0
    max_output_bytes: int = 32_768
    memory_mb: int = 256
    cpus: float = 1.0
    pids: int = 64
    tmpfs_mb: int = 64
    max_file_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("all sandbox limits must be positive")


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    truncated: bool = False
    backend: str = "container"
    duration_seconds: float = 0.0
    runtime: str | None = None
    container_id: str | None = None
    image: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def safe_relative_path(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ValueError("path must be a nonempty POSIX-relative path")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or path.startswith("-") or relative == PurePosixPath("."):
        raise ValueError("path must remain inside task workspace")
    return relative


def _bounded_process(argv: Sequence[str], *, cwd: Path, stdin: str, timeout: float, max_output: int, env: dict[str, str] | None = None) -> ExecutionResult:
    """Drain both pipes continuously while retaining only a bounded byte total."""
    started = time.monotonic()
    # File-backed stdin avoids a pipe deadlock when a child emits before reading.
    with tempfile.TemporaryFile() as input_file:
        input_file.write(stdin.encode("utf-8"))
        input_file.seek(0)
        process = subprocess.Popen(list(argv), cwd=cwd, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
        selector = selectors.DefaultSelector()
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        assert process.stdout is not None and process.stderr is not None
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        total = 0
        truncated = False
        timed_out = False
        try:
            while selector.get_map():
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                for key, _ in selector.select(min(0.05, max(0.0, timeout - (time.monotonic() - started)))):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    available = max(0, max_output - total)
                    buffers[key.data].extend(data[:available])
                    total += min(available, len(data))
                    truncated |= len(data) > available
            if timed_out:
                process.wait(timeout=5)
            else:
                try:
                    process.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        return ExecutionResult(
            stdout=buffers["stdout"].decode("utf-8", errors="replace"),
            stderr=buffers["stderr"].decode("utf-8", errors="replace"),
            exit_code=124 if timed_out else process.returncode,
            timed_out=timed_out, truncated=truncated,
            duration_seconds=round(time.monotonic() - started, 6),
        )


class ContainerSandbox:
    """One fresh host directory mounted at /workspace, never the host home.

    Images must be installed explicitly ahead of time: --pull=never is mandatory.
    Config (image/runtime/resource limits) is operator-owned, not model input.
    """
    backend_name = "container"
    allows_model_code = True

    def __init__(self, task_root: str | Path | None = None, *, image: str = "python:3.11-slim", runtime: str | None = None, limits: SandboxLimits | None = None):
        if runtime is not None and runtime not in {"docker", "podman"}:
            raise ValueError("runtime must be docker or podman")
        selected = runtime or next((name for name in ("podman", "docker") if shutil.which(name)), None)
        executable = shutil.which(selected) if selected else None
        if not executable:
            raise SandboxUnavailable("Docker or Podman is required; refusing to execute model code on the host")
        if not image or image.startswith("-") or any(char.isspace() for char in image):
            raise ValueError("invalid container image")
        self.runtime = executable
        self.runtime_name = selected
        self._last_container_id: str | None = None
        self.image = image
        self.limits = limits or SandboxLimits()
        root = Path(task_root).resolve() if task_root is not None else None
        if root:
            root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(tempfile.mkdtemp(prefix="picoagent-task-", dir=root))
        self._closed = False
        self._executed = False
        # Runtime receives no API keys, model credentials, proxy credentials, or
        # inherited DOCKER_HOST. No runtime config folder is mounted in the image.
        self._client_home = Path(tempfile.mkdtemp(prefix="picoagent-runtime-"))
        self._runtime_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self._client_home)}
        if selected == "podman":
            self._runtime_env["XDG_DATA_HOME"] = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))
        if "XDG_RUNTIME_DIR" in os.environ:
            self._runtime_env["XDG_RUNTIME_DIR"] = os.environ["XDG_RUNTIME_DIR"]

    def seed_files(self, files: Mapping[str, str]) -> None:
        """Operator fixtures only, before any untrusted code has been executed."""
        if self._executed or self._closed:
            raise RuntimeError("seed files before execution, using trusted fixtures only")
        for path, content in files.items():
            relative = safe_relative_path(path)
            if not isinstance(content, str) or len(content.encode()) > self.limits.max_file_bytes:
                raise ValueError("invalid or oversized fixture file")
            destination = self.workspace.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ValueError("fixture paths must be unique")
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(content)

    def command(self, argv: Sequence[str], *, name: str) -> list[str]:
        """Expose launch arguments for auditing and tests; no shell interpolation."""
        flags = [self.runtime, "run", "--rm", "--name", name, "--cidfile", str(self._client_home / (name + ".cid")), "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit", str(self.limits.pids), "--memory", f"{self.limits.memory_mb}m", "--memory-swap", f"{self.limits.memory_mb}m", "--cpus", str(self.limits.cpus), "--ulimit", f"fsize={self.limits.max_file_bytes}:{self.limits.max_file_bytes}", "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/workspace", "--mount", f"type=bind,src={self.workspace},dst=/workspace", "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.limits.tmpfs_mb}m", "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1", "--no-healthcheck", "--entrypoint", "", "-i", self.image]
        return flags + list(argv)

    def run(self, argv: Sequence[str], *, stdin: str = "", timeout: float | None = None, model_generated: bool = True) -> ExecutionResult:
        if self._closed:
            raise RuntimeError("sandbox is closed")
        if not argv or not all(isinstance(value, str) for value in argv):
            raise ValueError("argv must be a nonempty list of strings")
        effective_timeout = min(self.limits.timeout_seconds if timeout is None else timeout, self.limits.timeout_seconds)
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise ValueError("timeout must be positive")
        self._executed = True
        name = "picoagent-" + uuid.uuid4().hex
        try:
            result = _bounded_process(self.command(["timeout", "--signal=KILL", str(effective_timeout) + "s", *argv], name=name), cwd=self.workspace, stdin=stdin, timeout=effective_timeout, max_output=self.limits.max_output_bytes, env=self._runtime_env)
            cidfile = self._client_home / (name + ".cid")
            raw_id = cidfile.read_text().strip() if cidfile.exists() else ""
            self._last_container_id = raw_id if re.fullmatch(r"[0-9a-f]{64}", raw_id) else None
            return replace(result, runtime=self.runtime_name, container_id=self._last_container_id, image=self.image)
        finally:
            # Killing the client alone can leave its container alive. Always ask
            # the runtime to remove it, including after a timeout or interruption.
            try:
                subprocess.run([self.runtime, "rm", "--force", name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False, env=self._runtime_env)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise SandboxUnavailable("container cleanup could not be confirmed; internal deadline remains active") from error

    def runtime_metadata(self) -> dict:
        return {"backend": "container", "runtime": self.runtime_name, "container_id": self._last_container_id, "image": self.image}

    def probe(self) -> ExecutionResult:
        """A real lifecycle probe; unavailable daemon/image is an explicit failure."""
        result = self.run(["python", "-I", "-c", "print('picoagent-sandbox-ready')"], model_generated=False)
        if result.exit_code != 0 or result.timed_out or not result.container_id:
            raise SandboxUnavailable("container probe failed: " + (result.stderr or result.stdout)[:1000])
        return result

    def read_file(self, path: str, *, max_bytes: int = 32_768) -> str:
        """Read artifacts through the container, never following host symlinks."""
        safe_relative_path(path)
        if not 0 < max_bytes <= self.limits.max_output_bytes:
            raise ValueError("read limit must fit sandbox output limit")
        source = """import json,pathlib,sys
p=json.load(sys.stdin)
root=pathlib.Path('/workspace').resolve()
path=(root/p['path']).resolve()
if root not in path.parents: raise ValueError('path escapes workspace')
with path.open('rb') as f: data=f.read(p['max_bytes']+1)
if len(data)>p['max_bytes']: raise ValueError('artifact exceeds read limit')
sys.stdout.buffer.write(data)
"""
        result = self.run(["python", "-I", "-c", source], stdin=json.dumps({"path": path, "max_bytes": max_bytes}), model_generated=False)
        if result.exit_code != 0 or result.timed_out or result.truncated:
            raise RuntimeError("artifact read failed: " + result.stderr[:1000])
        return result.stdout

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            shutil.rmtree(self.workspace)
            shutil.rmtree(self._client_home)

    def __enter__(self) -> "ContainerSandbox":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class TrustedLocalSandbox:
    """Explicit developer-test helper, NEVER a fallback or model-code backend.

    It has no isolation. run(..., model_generated=False) is permitted only for
    hand-authored tests. ToolRegistry refuses this backend even if supplied.
    """
    backend_name = "trusted-local"
    allows_model_code = False

    def __init__(self, workspace: str | Path, *, limits: SandboxLimits | None = None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.limits = limits or SandboxLimits()

    def run(self, argv: Sequence[str], *, stdin: str = "", timeout: float | None = None, model_generated: bool = True) -> ExecutionResult:
        if model_generated:
            raise PermissionError("TrustedLocalSandbox cannot run model-generated code")
        result = _bounded_process(argv, cwd=self.workspace, stdin=stdin, timeout=min(self.limits.timeout_seconds if timeout is None else timeout, self.limits.timeout_seconds), max_output=self.limits.max_output_bytes, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.workspace)})
        return ExecutionResult(**{**result.to_dict(), "backend": self.backend_name})
