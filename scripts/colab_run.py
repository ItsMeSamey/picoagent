#!/usr/bin/env python3
"""Operate only an explicitly named, already allocated Colab session.

No allocation, stopping, payment, authentication, keepalive, or quota workarounds.
Run this controller on a machine separate from the transient training runtime.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from checkpoint_sync import CLITransfer, pack_checkpoint, pull_checkpoint, run_cli, sha256, upload_bundle

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = {"src", "scripts", "configs", "docs", "tests", "container", "data", ".github"}
SOURCE_FILES = {"README.md", "LICENSE", "SECURITY.md", "pyproject.toml", "uv.lock", "requirements-cpu.lock.txt", "requirements-tpu.txt", ".gitignore", ".env.example"}
EXCLUDED = {".git", ".venv", "venv", ".config", ".aws", ".ssh", ".codex", ".agents", ".colab-home", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
SECRET_NAME = re.compile(r"(^\.env($|\.)|credentials|oauth|(^|[-_])secrets?([_.-]|$)|(^|[-_])tokens?([_.-]|$)|\.pem$|\.key$)", re.I)
PUBLIC_DATA_METADATA = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                        "token_validation.json", "cli-10k-token-audit.json", "cli-10k-token-audit-source.py"}
OUTPUT_SENTINEL = "PICOAGENT_RESULT="


def archive_source(root: Path, output: Path) -> dict:
    """Archive all approved code/data roots, with per-file hashes and no symlinks.

    Denylisted names are a backstop, not secret detection. Review archive contents
    before uploading; never place real secrets in source/config/data files.
    """
    root = root.resolve()
    files = []
    # Honor repository exclusions (for example raw working copies already
    # preserved in byte-exact bounded archives), including untracked but
    # non-ignored source/data. Non-git fixture/source directories still work.
    inventory = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                               cwd=root, capture_output=True, text=True)
    included = set(inventory.stdout.split("\0")) if inventory.returncode == 0 else None
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if included is not None and relative.as_posix() not in included:
            continue
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if relative.parts[0] not in SOURCE_ROOTS and relative.as_posix() not in SOURCE_FILES:
            continue
        public_metadata = relative.parts[0] == "data" and path.name in PUBLIC_DATA_METADATA
        checked_parts = relative.parts[:-1] if public_metadata else relative.parts
        if any(SECRET_NAME.search(part) for part in checked_parts) and relative.as_posix() != ".env.example":
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing source symlink: {relative}")
        if path.is_file() and path.resolve() != output.resolve():
            files.append(path)
    manifest = {"schema": "picoagent.source-archive.v1", "files": {
        path.relative_to(root).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in files}}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    manifest["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=root, capture_output=True, text=True)
    manifest["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
        payload = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
        info = tarfile.TarInfo("SOURCE_MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return {"archive": str(output), "sha256": sha256(output), "manifest": manifest}


class Colab:
    def __init__(self, session: str, executable: str = "colab", timeout: int = 1800):
        if not session.strip() or session.startswith("-"):
            raise ValueError("An explicit valid --session is required")
        self.session, self.executable, self.timeout = session, executable, timeout

    def command(self, name: str, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
        if name not in {"status", "exec", "upload", "download"}:
            raise ValueError("Controller cannot allocate, stop, or modify account/session settings")
        return run_cli([self.executable, name, "--session", self.session, *args],
                       capture=capture, timeout=self.timeout)

    def execute(self, code: str) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as script:
            script.write(code)
            script.flush()
            result = self.command("exec", "--timeout", str(max(1, self.timeout - 10)), "--file", script.name, capture=True)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(OUTPUT_SENTINEL):
                return json.loads(line[len(OUTPUT_SENTINEL):])
        raise RuntimeError(f"Colab command did not return a result: {result.stdout[-1500:]} {result.stderr[-1500:]}")

    def transfer(self) -> CLITransfer:
        prefix = [self.executable, "download", "--session", self.session]
        return CLITransfer([*prefix, "{remote}", "{local}"],
                           [self.executable, "upload", "--session", self.session, "{local}", "{remote}"],
                           timeout=self.timeout)


def remote_prelude(project: str) -> str:
    return ("import json, sys\n"
            f"sys.path.insert(0, {json.dumps(project + '/scripts')})\n"
            f"sys.path.insert(0, {json.dumps(project + '/src')})\n")


def status(client: Colab, project: str, run_dir: str) -> dict:
    return client.execute(remote_prelude(project) + f"""
from pathlib import Path
run = Path({json.dumps(run_dir)})
status_file = run / 'run_status.json'
job_file = Path({json.dumps(project)}) / '.picoagent-job.json'
result = {{'job_status': json.loads(job_file.read_text()) if job_file.exists() else None,
          'run_status': json.loads(status_file.read_text()) if status_file.exists() else None,
          'checkpoints': sorted(p.name for p in run.glob('checkpoint-*') if (p / 'checkpoint_manifest.json').exists())}}
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps(result))
""")


def stage(client: Colab, root: Path, project: str, archive_path: Path) -> dict:
    result = archive_source(root, archive_path)
    remote_archive = f"/content/picoagent-source-{result['sha256'][:16]}.tar.gz"
    client.command("upload", str(archive_path), remote_archive.lstrip("/"))
    return client.execute(f"""
import hashlib, json, pathlib, tarfile
archive_path = pathlib.Path({json.dumps(remote_archive)})
target = pathlib.Path({json.dumps(project)})
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == {json.dumps(result['sha256'])}, 'Archive hash mismatch'
if target.exists() and any(target.iterdir()):
    raise RuntimeError('Destination exists; choose a new project directory rather than overwrite a run')
target.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive_path) as archive:
    for member in archive:
        relative = pathlib.PurePosixPath(member.name)
        if relative.is_absolute() or '..' in relative.parts or not member.isfile():
            raise ValueError('Unsafe archive entry')
        path = target / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source, path.open('xb') as destination:
            import shutil
            shutil.copyfileobj(source, destination)
manifest = json.loads((target / 'SOURCE_MANIFEST.json').read_text())
for relative, record in manifest['files'].items():
    assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == record['sha256'], relative
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps({{'project': str(target), 'source_files': len(manifest['files']), 'verified': True}}))
""")


def start(client: Colab, project: str, command: list[str]) -> dict:
    if not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("--command-json must be a nonempty JSON argv array")
    # The child supervisor records terminal status even if the controller disconnects.
    supervisor = """import json, os, pathlib, subprocess, sys, time
project, command = sys.argv[1], json.loads(sys.argv[2])
state = pathlib.Path(project) / '.picoagent-job.json'
def save(payload):
    tmp = state.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, state)
save({'status': 'running', 'pid': os.getpid(), 'command': command, 'started_at': time.time()})
try:
    result = subprocess.run(command, cwd=project, check=False)
    save({'status': 'completed' if result.returncode == 0 else 'failed', 'returncode': result.returncode, 'finished_at': time.time()})
except BaseException as exc:
    save({'status': 'failed', 'error': str(exc), 'finished_at': time.time()})
    raise
"""
    return client.execute(f"""
import json, os, pathlib, subprocess, sys
project = pathlib.Path({json.dumps(project)})
state = project / '.picoagent-job.json'
if state.exists():
    old = json.loads(state.read_text())
    if old.get('status') in {{'running', 'starting'}}:
        raise RuntimeError('A recorded job is already running. Inspect it before restarting; no automatic duplicate launch.')
lock = project / '.picoagent-launch.lock'
with lock.open('x') as guard:
    guard.write('launch in progress')
try:
    # Reserve before spawn so an uncertain/repeated request cannot race the child.
    state.write_text(json.dumps({{'status': 'starting', 'command': {repr(command)}}}))
    log = (project / 'controller-job.log').open('ab', buffering=0)
    try:
        child = subprocess.Popen([sys.executable, '-c', {json.dumps(supervisor)}, str(project), {json.dumps(json.dumps(command))}],
                                 cwd=project, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    except BaseException:
        state.write_text(json.dumps({{'status': 'failed', 'error': 'Supervisor launch failed'}}))
        raise
    print({json.dumps(OUTPUT_SENTINEL)} + json.dumps({{'supervisor_pid': child.pid, 'log': str(project / 'controller-job.log')}}))
finally:
    lock.unlink()

""")


def collect(client: Colab, project: str, run_dir: str, export_root: str,
            destination: Path, off_runtime: bool, prune: bool = False) -> dict:
    if not off_runtime:
        raise ValueError("--off-runtime is required: a Colab-local copy is not a backup")
    result = client.execute(remote_prelude(project) + f"""
from pathlib import Path
from checkpoint_sync import pack_checkpoint, sha256, BUNDLE_MANIFEST, CHECKPOINT
run, exports = Path({json.dumps(run_dir)}), Path({json.dumps(export_root)})
checkpoints = sorted((p for p in run.glob('checkpoint-*') if CHECKPOINT.fullmatch(p.name) and (p / 'checkpoint_manifest.json').is_file()), key=lambda p: int(p.name.split('-')[1]))
result = {{'exports': [], 'best_checkpoint': None}}
best_loss = float('inf')
for checkpoint in checkpoints:
    bundle = pack_checkpoint(run, checkpoint.name, exports)
    result['exports'].append({{'name': checkpoint.name, 'path': str(bundle), 'sha256': sha256(bundle / BUNDLE_MANIFEST)}})
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
    for row in trainer.get('log_history', []):
        if row.get('step') == int(checkpoint.name.split('-')[1]) and isinstance(row.get('eval_loss'), (int, float)) and row['eval_loss'] < best_loss:
            best_loss, result['best_checkpoint'] = row['eval_loss'], checkpoint.name
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps(result))
""")
    acknowledged = {}
    for bundle in result["exports"]:
        final = pull_checkpoint(bundle["path"].lstrip("/"), destination, client.transfer(),
                                off_runtime=off_runtime, expected_manifest_sha256=bundle["sha256"])
        acknowledged[final.name] = sha256(final / "checkpoint_manifest.json")
    pruned = []
    if prune and acknowledged:
        # Remote deletion is limited to checkpoint dirs just verified on the controller.
        # Retain current latest two and best; new saves racing collection are untouched.
        remote_result = client.execute(remote_prelude(project) + f"""
import shutil
from pathlib import Path
from checkpoint_sync import CHECKPOINT, sha256
from picoagent.training.provenance import verify_checkpoint
run = Path({json.dumps(run_dir)})
acknowledged = {json.dumps(acknowledged)}
best = {repr(result['best_checkpoint'])}
checkpoints = sorted((p for p in run.glob('checkpoint-*') if CHECKPOINT.fullmatch(p.name)), key=lambda p: int(p.name.split('-')[1]))
keep = set(p.name for p in checkpoints[-2:]) | ({{best}} if best else set())
run_hash = sha256(run / 'run_manifest.json')
removed = []
for checkpoint in checkpoints:
    if checkpoint.name in keep or checkpoint.name not in acknowledged:
        continue
    verify_checkpoint(checkpoint, run_hash)
    if sha256(checkpoint / 'checkpoint_manifest.json') != acknowledged[checkpoint.name]:
        raise ValueError('Checkpoint changed after durable acknowledgement')
    if any(p.suffix in {{'.jsonl', '.ndjson'}} or 'trace' in p.name.lower() for p in checkpoint.rglob('*')):
        raise ValueError('Refusing to remove a checkpoint directory containing traces')
    shutil.rmtree(checkpoint)
    export = Path({json.dumps(export_root)}) / checkpoint.name
    if export.is_dir() and not export.is_symlink():
        shutil.rmtree(export)
    removed.append(checkpoint.name)
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps({{'pruned_runtime': removed}}))
""")
        pruned = remote_result["pruned_runtime"]
    return {"verified_checkpoints": list(acknowledged), "destination": str(destination),
            "best_checkpoint": result["best_checkpoint"], "pruned_runtime": pruned}



def restore(client: Colab, project: str, source: Path, checkpoint: str,
            run_dir: str, local_export: Path, remote_export: str) -> dict:
    bundle = pack_checkpoint(source, checkpoint, local_export)
    remote = remote_export.rstrip("/") + "/" + checkpoint
    client.execute(f"""
import json
from pathlib import Path
root = Path({json.dumps(remote)})
if (root / 'transfer_manifest.json').exists():
    raise RuntimeError('Restore export prefix already committed; choose a fresh --export-root')
(root / 'chunks').mkdir(parents=True, exist_ok=True)
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps({{'ready': True}}))
""")
    digest = upload_bundle(bundle, remote.lstrip("/"), client.transfer())
    return client.execute(remote_prelude(project) + f"""
from pathlib import Path
from checkpoint_sync import LocalTransfer, pull_checkpoint
from picoagent.training.retention import _exclusive_lock
run = Path({json.dumps(run_dir)})
run.mkdir(parents=True, exist_ok=True)
with _exclusive_lock(run):
    restored = pull_checkpoint({json.dumps(remote)}, run, LocalTransfer(), off_runtime=False,
                               restore=True, expected_manifest_sha256={json.dumps(digest)})
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps({{'restored_checkpoint': str(restored), 'durable_backup': False}}))
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--colab", default="colab")
    parser.add_argument("--project", default="/content/picoagent")
    parser.add_argument("--timeout", type=int, default=1800)
    sub = parser.add_subparsers(dest="action", required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--source", type=Path, default=PROJECT)
    staging.add_argument("--archive", type=Path, required=True)
    launch = sub.add_parser("start")
    launch.add_argument("--command-json", required=True)
    recovery = sub.add_parser("restore")
    recovery.add_argument("--source", type=Path, required=True)
    recovery.add_argument("--checkpoint", required=True)
    recovery.add_argument("--run-dir", required=True)
    recovery.add_argument("--local-export", type=Path, required=True)
    recovery.add_argument("--export-root", required=True)
    check = sub.add_parser("status")
    check.add_argument("--run-dir", required=True)
    for name in ("collect", "watch"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--export-root", default="/content/picoagent-checkpoint-exports")
        command.add_argument("--destination", required=True, type=Path)
        command.add_argument("--off-runtime", action="store_true", required=True)
        command.add_argument("--prune", action="store_true", help="Keep latest two plus best, after host verification")
        if name == "watch":
            command.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    client = Colab(args.session, args.colab, args.timeout)
    if args.action == "stage":
        result = stage(client, args.source, args.project, args.archive)
    elif args.action == "start":
        result = start(client, args.project, json.loads(args.command_json))
    elif args.action == "status":
        result = status(client, args.project, args.run_dir)
    elif args.action == "restore":
        result = restore(client, args.project, args.source, args.checkpoint, args.run_dir,
                         args.local_export, args.export_root)
    else:
        args.destination.mkdir(parents=True, exist_ok=True)
        from picoagent.training.retention import _exclusive_lock
        with _exclusive_lock(args.destination):
            while True:
                result = collect(client, args.project, args.run_dir, args.export_root,
                                 args.destination, args.off_runtime, args.prune)
                print(json.dumps(result), flush=True)
                if args.action == "collect":
                    return
                current = status(client, args.project, args.run_dir)
                states = {(current.get(key) or {}).get("status") for key in ("run_status", "job_status")}
                if states.intersection({"completed", "failed"}):
                    print(json.dumps(current), flush=True)
                    return
                # Transport failures exit visibly. Restart watch to resume verified chunks.
                # Never provision a replacement or retry an account quota denial.
                time.sleep(max(10, args.interval))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
