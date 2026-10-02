#!/usr/bin/env python3
"""Operate only an explicitly named, already allocated Colab session.

No allocation, stopping, payment, authentication, keepalive, or quota workarounds.
Run this controller on a machine separate from the transient training runtime.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import json
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from checkpoint_sync import (CLITransfer, evaluation_record, pack_checkpoint, pull_checkpoint,
                             run_cli, sha256, upload_bundle)
from source_staging import (DEFAULT_CHUNK_BYTES, integrity_record, matches, no_symlink_path,
                            pack_archive, relative_path)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = {"src", "scripts", "configs", "docs", "tests", "container", "data", ".github"}
SOURCE_FILES = {"README.md", "THIRD_PARTY_NOTICES.md", "LICENSE", "SECURITY.md", "pyproject.toml", "uv.lock", "requirements-cpu.lock.txt", "requirements-tpu.txt", ".gitignore", ".env.example"}
EXCLUDED = {".git", ".venv", "venv", ".config", ".aws", ".ssh", ".codex", ".agents", ".colab-home", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
SECRET_NAME = re.compile(r"(^\.env($|\.)|credentials|oauth|(^|[-_])secrets?([_.-]|$)|(^|[-_])tokens?([_.-]|$)|\.pem$|\.key$)", re.I)
PUBLIC_DATA_METADATA = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                        "token_validation.json", "cli-10k-token-audit.json", "cli-10k-token-audit-source.py"}
OUTPUT_SENTINEL = "PICOAGENT_RESULT="


def permitted_source(relative: Path) -> bool:
    if any(part in EXCLUDED for part in relative.parts):
        return False
    if relative.parts[0] not in SOURCE_ROOTS and relative.as_posix() not in SOURCE_FILES:
        return False
    public_metadata = relative.parts[0] == "data" and relative.name in PUBLIC_DATA_METADATA
    checked_parts = relative.parts[:-1] if public_metadata else relative.parts
    return (relative.as_posix() == ".env.example"
            or not any(SECRET_NAME.search(part) for part in checked_parts))


def selected_dataset_files(root: Path, manifest_path: Path) -> tuple[dict, dict]:
    """Select a self-contained hash-listed snapshot, bypassing git-ignore only."""
    manifest_path = no_symlink_path(manifest_path if manifest_path.is_absolute() else root / manifest_path)
    if not manifest_path.is_relative_to(root) or manifest_path.relative_to(root).parts[0] != "data":
        raise ValueError("Selected dataset manifest must be inside the source data directory")
    if not permitted_source(manifest_path.relative_to(root)):
        raise ValueError("Required dataset manifest would be removed by secret filtering")
    if not manifest_path.is_file() or manifest_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Selected dataset manifest is missing or exceeds bounds")
    before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise ValueError("Selected dataset requires a nonempty self-contained files hash map")
    files = {}
    for name, info in listed.items():
        path = no_symlink_path(manifest_path.parent / relative_path(name))
        relative = path.relative_to(root)
        if not permitted_source(relative):
            raise ValueError("Required dataset file would be removed by secret filtering")
        integrity_record(info)
        if not matches(path, info):
            raise ValueError("Required dataset file is missing or failed integrity verification")
        files[relative.as_posix()] = {"sha256": info["sha256"], "bytes": info["bytes"]}
    if sha256(manifest_path) != before:
        raise ValueError("Selected dataset manifest changed during verification")
    name = manifest_path.relative_to(root).as_posix()
    files[name] = {"sha256": before, "bytes": manifest_path.stat().st_size}
    return files, {"path": name, "sha256": before}


def archive_source(root: Path, output: Path, dataset_manifest: Path | None = None) -> dict:
    """Archive code and either legacy eligible data or one explicit sealed snapshot.

    Name filtering is a backstop, not content secret detection. Explicit dataset
    files cannot silently disappear because of filters or git-ignore rules.
    """
    root, output = no_symlink_path(root), no_symlink_path(output)
    if output.is_relative_to(root) and permitted_source(output.relative_to(root)):
        raise ValueError("Archive output must be outside approved source/data roots")
    selected, selection = selected_dataset_files(root, dataset_manifest) if dataset_manifest is not None else ({}, None)
    inventory = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                               cwd=root, capture_output=True, text=True)
    included = set(inventory.stdout.split("\0")) if inventory.returncode == 0 else None
    chunk_root = Path(str(output) + ".chunks")
    files = dict(selected)
    # Prune unselected data before walking it: historical raw working copies may
    # contain millions of files and are not needed for a selected snapshot.
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED
                         and not (parent == root and (name not in SOURCE_ROOTS or (selection and name == "data")))
                         and parent / name != chunk_root)
        for name in dirs:
            path = parent / name
            if path.is_symlink() and permitted_source(path.relative_to(root)):
                raise ValueError("Refusing source symlink")
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(root)
            if (path == output or not permitted_source(relative)
                    or (included is not None and relative.as_posix() not in included)):
                continue
            no_symlink_path(path)
            if not path.is_file():
                raise ValueError("Source entry is not a regular file")
            files[relative.as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"schema": "picoagent.source-archive.v1", "files": dict(sorted(files.items()))}
    if selection:
        manifest["selected_dataset"] = selection
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    manifest["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    status_result = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                                   cwd=root, capture_output=True, text=True)
    # Exclude staging outputs from this informational flag, including on a retry
    # of a clean checkout with --archive at its top level.
    statuses = iter(status_result.stdout.split("\0"))
    dirty = False
    for entry in statuses:
        if not entry:
            continue
        names = [entry[3:]]
        if "R" in entry[:2] or "C" in entry[:2]:
            names.append(next(statuses, ""))
        for name in names:
            candidate = root / name
            if name and candidate != output and not candidate.is_relative_to(chunk_root):
                dirty = True
    manifest["git_dirty"] = dirty if status_result.returncode == 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".source-", delete=False) as raw:
            temporary = Path(raw.name)
            # Both gzip and tar timestamps are fixed so retries reuse the chunks.
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    for name, expected in manifest["files"].items():
                        path = no_symlink_path(root / name)
                        info = archive.gettarinfo(str(path), arcname=name)
                        if not info.isfile() or info.size != expected["bytes"]:
                            raise ValueError("Source changed while archiving")
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        digest = hashlib.sha256()
                        class HashReader:
                            def __init__(self, stream):
                                self.stream = stream
                            def read(self, size):
                                block = self.stream.read(size)
                                digest.update(block)
                                return block
                        with path.open("rb") as handle:
                            archive.addfile(info, HashReader(handle))
                            if handle.read(1) or digest.hexdigest() != expected["sha256"]:
                                raise ValueError("Source changed while archiving")
                    payload = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
                    info = tarfile.TarInfo("SOURCE_MANIFEST.json")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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
        raise RuntimeError("Colab command did not return a valid result; remote output suppressed")

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
upload_file = run / 'checkpoint_upload_status.json'
result = {{'job_status': json.loads(job_file.read_text()) if job_file.exists() else None,
          'run_status': json.loads(status_file.read_text()) if status_file.exists() else None,
          'checkpoint_upload_status': json.loads(upload_file.read_text()) if upload_file.exists() else None,
          'checkpoints': sorted(p.name for p in run.glob('checkpoint-*') if (p / 'checkpoint_manifest.json').exists())}}
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps(result))
""")


def staging_call(client: Colab, operation: str, *args) -> dict:
    if operation not in {"probe_chunks", "commit_chunk", "materialize_source"}:
        raise ValueError("Unknown source staging operation")
    helper = Path(__file__).with_name("source_staging.py").read_text()
    code = helper + "\ntry:\n    result = " + operation + "(*" + repr(args) + ")\n"
    code += "except Exception as error:\n    result = {'staging_error': type(error).__name__}\n"
    code += "print(" + repr(OUTPUT_SENTINEL) + " + json.dumps(result))\n"
    result = client.execute(code)
    if "staging_error" in result:
        raise RuntimeError("Remote source staging failed: " + result["staging_error"])
    return result


def stage(client: Colab, root: Path, project: str, archive_path: Path, *,
          dataset_manifest: Path | None = None, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
          transfer_root: str = "/content/picoagent-source-uploads") -> dict:
    result = archive_source(root, archive_path, dataset_manifest)
    chunks, bundle = pack_archive(archive_path, result["manifest"], chunk_bytes)
    remote_root = Path(transfer_root)
    if not remote_root.is_absolute() or ".." in remote_root.parts:
        raise ValueError("Source transfer root must be an absolute path")
    remote = str(remote_root / bundle["archive"]["sha256"])
    verified = set(staging_call(client, "probe_chunks", remote, bundle)["verified_chunks"])
    uploaded = 0
    for chunk in bundle["chunks"]:
        digest = chunk["sha256"]
        if digest in verified:
            continue
        local = chunks / digest
        if not matches(local, chunk):
            raise ValueError("Local source chunk failed verification")
        # The CLI base64-loads just this bounded chunk, never the whole tarball.
        client.command("upload", str(local), (remote + "/" + digest + ".partial").lstrip("/"), capture=True)
        receipt = staging_call(client, "commit_chunk", remote, chunk)
        if receipt.get("verified_chunk") != digest:
            raise ValueError("Source upload acknowledgement mismatch")
        verified.add(digest)
        uploaded += 1
    final = staging_call(client, "materialize_source", remote, project, bundle)
    return {**final, "archive_sha256": result["sha256"], "uploaded_chunks": uploaded,
            "unique_chunks": len(verified), "chunk_bytes": chunk_bytes,
            "selected_dataset": result["manifest"].get("selected_dataset")}


def start(client: Colab, project: str, command: list[str], *, wait_for_completion: bool = False) -> dict:
    """Launch once; optionally keep this execution awaiting the actual workload.

    An ambiguous execute timeout must be inspected, never automatically retried.
    The supervisor owns job status even if this foreground execution disconnects.
    """
    if type(wait_for_completion) is not bool:
        raise ValueError("wait_for_completion must be a boolean")
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
# Exit outside the exception handler so SystemExit does not overwrite job state.
sys.exit(result.returncode)
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
    # Another launcher may have reserved/spawned between our initial status read
    # and acquisition. Recheck while holding the exclusive launch lock.
    if state.exists():
        old = json.loads(state.read_text())
        if old.get('status') in {{'running', 'starting'}}:
            raise RuntimeError('A recorded job is already running. Inspect it before restarting; no automatic duplicate launch.')
    # Reserve before spawn so an uncertain/repeated request cannot race the child.
    state.write_text(json.dumps({{'status': 'starting', 'command': {repr(command)}}}))
    log = (project / 'controller-job.log').open('ab', buffering=0)
    try:
        child = subprocess.Popen([sys.executable, '-c', {json.dumps(supervisor)}, str(project), {json.dumps(json.dumps(command))}],
                                 cwd=project, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    except BaseException:
        state.write_text(json.dumps({{'status': 'failed', 'error': 'Supervisor launch failed'}}))
        raise
    result = {{'supervisor_pid': child.pid, 'log': str(project / 'controller-job.log')}}
finally:
    lock.unlink()
# Release the launch reservation before waiting on the real supervised workload.
# This does not poll, reconnect, retry, or perform unrelated keepalive activity.
if {wait_for_completion!r}:
    result['returncode'] = child.wait()
    result['waited_for_completion'] = True
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps(result))
""")


def validate_publication_options(publish_repository, approve_public_checkpoints,
                                 prune_local_published_cache=False, upload_workers=1):
    if type(upload_workers) is not int or not 1 <= upload_workers <= 4:
        raise ValueError("upload_workers must be an integer from 1 to 4")
    if upload_workers != 1 and not publish_repository:
        raise ValueError("--upload-workers requires --publish-repository")
    if prune_local_published_cache and not publish_repository:
        raise ValueError("--prune-local-published-cache requires --publish-repository and --approve-public-checkpoints")
    if publish_repository is not None:
        from github_checkpoint_release import REPO
        if not approve_public_checkpoints:
            raise ValueError("--publish-repository requires --approve-public-checkpoints")
        if not REPO.fullmatch(publish_repository) or publish_repository.split("/")[1] in {".", ".."}:
            raise ValueError("Expected explicit GitHub OWNER/REPO")


def publish_collected_checkpoint(client, project, run_dir, destination, bundle, repository, *, upload_workers=1):
    """Persist exact release evidence before acknowledging the trainer's barrier.

    A controller-lifetime cache skips repeated public reads only after a confirmed
    runtime acknowledgement still matches the exact locally preserved evidence.
    Restarted controllers and uncertain acknowledgements reverify remote assets.
    """
    from github_checkpoint_release import digest_json, make_plan, upload_checkpoint
    from picoagent.training.durability import acknowledgement_from_receipt
    from picoagent.training.provenance import write_json
    from picoagent.training.data import canonical_json

    validate_publication_options(repository, True, upload_workers=upload_workers)
    manifest = no_symlink_path(destination / ".incoming" / bundle["name"] / bundle["sha256"]
                               / "transfer_manifest.json")
    if sha256(manifest) != bundle["sha256"]:
        raise ValueError("Cached transfer manifest differs from verified download")
    plan = make_plan(destination, manifest, repository)
    evidence = no_symlink_path(destination / "release-backups" / bundle["name"] / bundle["sha256"])
    evidence.mkdir(parents=True, exist_ok=True)
    # Persist newly created ancestor entries as well as the deepest evidence
    # directory; write_json below fsyncs each plan/receipt and its own parent.
    from checkpoint_sync import sync_directory
    for directory in (evidence, *evidence.parents):
        sync_directory(directory)
        if directory == destination.resolve():
            break

    def preserve(name, payload):
        path = no_symlink_path(evidence / name)
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError("Existing release evidence disagrees with verified identity")
        else:
            write_json(path, payload, exclusive=True)

    preserve("plan.json", plan)
    plan_digest = digest_json(plan)
    cache = getattr(client, "_confirmed_release_receipts", {})
    receipt_path = no_symlink_path(evidence / "receipt.json")
    if plan_digest in cache and receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if digest_json(saved) == cache[plan_digest]:
            saved_ack = acknowledgement_from_receipt(saved)
            ack_digest = hashlib.sha256((canonical_json(saved_ack) + "\n").encode()).hexdigest()
            if bundle.get("durability_ack_sha256") == ack_digest:
                return str(receipt_path)
    receipt = upload_checkpoint(destination, manifest, plan,
                                approved_plan_sha256=digest_json(plan), publish=True,
                                **({"upload_workers": upload_workers} if upload_workers != 1 else {}))
    if receipt.get("identity") != plan["identity"] or receipt.get("plan_sha256") != digest_json(plan):
        raise ValueError("Release receipt differs from approved checkpoint identity")
    if (receipt.get("tag") != plan["tag"]
            or {name: {"sha256": record.get("sha256"), "bytes": record.get("bytes")}
                for name, record in receipt.get("assets", {}).items()} != plan["assets"]):
        raise ValueError("Release receipt inventory differs from approved plan")
    acknowledgement = acknowledgement_from_receipt(receipt)
    preserve("receipt.json", receipt)
    remote = client.execute(remote_prelude(project) + f"""
from pathlib import Path
from source_staging import no_symlink_path
from picoagent.training.provenance import write_json
acknowledgement = {repr(acknowledgement)}
target = no_symlink_path(Path({json.dumps(run_dir)}) / 'durability' / {repr(bundle['name'] + '.json')})
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    if json.loads(target.read_text()) != acknowledgement:
        raise ValueError('Existing durability acknowledgement identity differs')
else:
    write_json(target, acknowledgement, exclusive=True)
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps({{'durability_acknowledged': {repr(bundle['name'])}}}))
""")
    if remote.get("durability_acknowledged") != bundle["name"]:
        raise RuntimeError("Runtime durability acknowledgement was not confirmed")
    cache[plan_digest] = digest_json(receipt)
    client._confirmed_release_receipts = cache
    return str(receipt_path)


def collect(client: Colab, project: str, run_dir: str, export_root: str,
            destination: Path, off_runtime: bool, prune: bool = False, *,
            publish_repository: str | None = None, approve_public_checkpoints: bool = False,
            prune_local_published_cache: bool = False, upload_workers: int = 1,
            prune_to_latest_published: bool = False) -> dict:
    validate_publication_options(publish_repository, approve_public_checkpoints, prune_local_published_cache, upload_workers)
    if prune_to_latest_published and (not prune or publish_repository is None):
        raise ValueError('--prune-to-latest-published requires --prune and --publish-repository')
    if not off_runtime:
        raise ValueError("--off-runtime is required: a Colab-local copy is not a backup")
    result = client.execute(remote_prelude(project) + f"""
from pathlib import Path
from checkpoint_sync import pack_checkpoint, evaluation_record, sha256, BUNDLE_MANIFEST, CHECKPOINT, runtime_export_roots
from checkpoint_sync import preflight_export_trees
from picoagent.training.evaluation import checkpoint_eval_loss
from source_staging import no_symlink_path
run, exports = runtime_export_roots(Path({json.dumps(run_dir)}), Path({json.dumps(export_root)}))
checkpoints = sorted((p for p in run.glob('checkpoint-*') if CHECKPOINT.fullmatch(p.name) and (p / 'checkpoint_manifest.json').is_file()), key=lambda p: int(p.name.split('-')[1]))
preflight_export_trees(exports, [checkpoint.name for checkpoint in checkpoints])
result = {{'exports': [], 'best_checkpoint': None}}
best_loss = float('inf')
for checkpoint in checkpoints:
    loss = checkpoint_eval_loss(run, checkpoint.name)
    bundle = pack_checkpoint(run, checkpoint.name, exports)
    evaluation = evaluation_record(bundle)
    result['exports'].append({{'name': checkpoint.name, 'path': str(bundle), 'sha256': sha256(bundle / BUNDLE_MANIFEST),
                              'evaluation_sha256': evaluation['sha256'] if evaluation else None}})
    if {publish_repository is not None!r}:
        ack = no_symlink_path(run / 'durability' / (checkpoint.name + '.json'))
        result['exports'][-1]['durability_ack_sha256'] = (sha256(ack) if ack.is_file()
                                                        and ack.stat().st_size <= 2 * 1024**2 else None)
    if evaluation is not None:
        loss = evaluation['eval_loss']
    if loss is not None and loss < best_loss:
        best_loss, result['best_checkpoint'] = loss, checkpoint.name
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps(result))
""")
    acknowledged = {}
    release_receipts = {}
    pruned_local = []
    newest = max((int(item['name'].split('-')[1]) for item in result['exports']), default=-1)
    for bundle in result["exports"]:
        if prune_local_published_cache and int(bundle['name'].split('-')[1]) < newest:
            from local_checkpoint_cache import skip_published
            saved = skip_published(client, destination, bundle, publish_repository)
            if saved is not None:
                receipt_path, identity = saved
                release_receipts[bundle['name']] = receipt_path
                acknowledged[bundle['name']] = {
                    key: identity[key] for key in ('checkpoint_manifest_sha256',
                                                  'run_manifest_sha256', 'transfer_manifest_sha256')}
                acknowledged[bundle['name']]['evaluation_sha256'] = bundle['evaluation_sha256']
                continue
        final = pull_checkpoint(bundle["path"].lstrip("/"), destination, client.transfer(),
                                off_runtime=off_runtime, expected_manifest_sha256=bundle["sha256"],
                                expected_evaluation_sha256=bundle["evaluation_sha256"])
        if final.name != bundle["name"]:
            raise ValueError("Collected checkpoint identity differs from advertised export")
        if publish_repository is not None:
            release_receipts[final.name] = publish_collected_checkpoint(
                client, project, run_dir, destination, bundle, publish_repository,
                **({"upload_workers": upload_workers} if upload_workers != 1 else {}))
        if prune_local_published_cache:
            from local_checkpoint_cache import prune_published
            pruned_local.extend(prune_published(destination, bundle, publish_repository, client))
        acknowledged[final.name] = {
            "checkpoint_manifest_sha256": sha256(final / "checkpoint_manifest.json"),
            "run_manifest_sha256": sha256(destination / "run_manifest.json"),
            "transfer_manifest_sha256": bundle["sha256"],
            "evaluation_sha256": bundle["evaluation_sha256"],
        }
    pruned = []
    pending_evaluations = []
    if prune and acknowledged:
        # Remote deletion is limited to checkpoint dirs just verified on the controller.
        # Retain current latest two and best; new saves racing collection are untouched.
        remote_result = client.execute(remote_prelude(project) + f"""
from pathlib import Path
from checkpoint_sync import prune_runtime_exports
result = prune_runtime_exports(Path({json.dumps(run_dir)}), Path({json.dumps(export_root)}),
                               {repr(acknowledged)}, {repr(result['best_checkpoint'])},
                               latest_published_repository={repr(publish_repository if prune_to_latest_published else None)})
print({json.dumps(OUTPUT_SENTINEL)} + json.dumps(result))
""")
        pruned = remote_result["pruned_runtime"]
        pending_evaluations = remote_result["pending_evaluations"]
    return {"verified_checkpoints": list(acknowledged), "destination": str(destination),
            "best_checkpoint": result["best_checkpoint"], "pruned_runtime": pruned,
            "pending_evaluations": pending_evaluations,
            **({"pruned_local_published_cache": pruned_local} if prune_local_published_cache else {}),
            **({"release_receipts": release_receipts} if publish_repository is not None else {})}



def restore(client: Colab, project: str, source: Path, checkpoint: str,
            run_dir: str, local_export: Path, remote_export: str) -> dict:
    bundle = pack_checkpoint(source, checkpoint, local_export)
    evaluation = evaluation_record(bundle)
    evaluation_digest = evaluation["sha256"] if evaluation else None
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
                               restore=True, expected_manifest_sha256={json.dumps(digest)},
                               expected_evaluation_sha256={evaluation_digest!r})
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
    staging.add_argument("--dataset-manifest", type=Path, help="Include only this self-contained snapshot under data/")
    staging.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    staging.add_argument("--transfer-root", default="/content/picoagent-source-uploads")
    launch = sub.add_parser("start")
    launch.add_argument("--command-json", required=True)
    launch.add_argument("--wait-for-completion", action="store_true",
                        help="Keep this execution waiting on the actual training supervisor; never retry an ambiguous timeout")
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
        command.add_argument("--prune-local-published-cache", action="store_true",
                             help="Approve eviction of older locally verified public checkpoint payloads; retain latest")
        command.add_argument("--publish-repository", help="Publish exact verified checkpoints to public OWNER/REPO")
        command.add_argument("--approve-public-checkpoints", action="store_true",
                             help="Approve public disclosure of checkpoints and their pinned run/source data")
        command.add_argument("--upload-workers", type=int, choices=range(1, 5), default=1,
                             help="Bounded GitHub upload/read-back concurrency (default: serial)")
        if name == "watch":
            command.add_argument("--stop-file", type=Path,
                                 help="Stop controller after a completed collection if this local file exists")
            command.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.action in {"collect", "watch"}:
        validate_publication_options(args.publish_repository, args.approve_public_checkpoints, args.prune_local_published_cache, args.upload_workers)
    client = Colab(args.session, args.colab, args.timeout)
    if args.action == "stage":
        result = stage(client, args.source, args.project, args.archive,
                       dataset_manifest=args.dataset_manifest, chunk_bytes=args.chunk_bytes,
                       transfer_root=args.transfer_root)
    elif args.action == "start":
        result = start(client, args.project, json.loads(args.command_json),
                       **({"wait_for_completion": True} if args.wait_for_completion else {}))
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
                                 args.destination, args.off_runtime, args.prune,
                                 publish_repository=args.publish_repository,
                                 approve_public_checkpoints=args.approve_public_checkpoints,
                                 prune_local_published_cache=args.prune_local_published_cache,
                                 **({"upload_workers": args.upload_workers} if args.upload_workers != 1 else {}))
                print(json.dumps(result), flush=True)
                if args.action == "collect":
                    return
                if args.stop_file is not None and args.stop_file.exists():
                    print(json.dumps({"controller_stopped": True, "reason": "stop_file",
                                      "collection_completed": True, "stop_file": str(args.stop_file)}), flush=True)
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
    try:
        main()
    except Exception as error:
        from colab_safe_cli import report_error
        report_error("controller", error)
        raise SystemExit(1) from None
