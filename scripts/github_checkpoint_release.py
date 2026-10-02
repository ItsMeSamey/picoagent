#!/usr/bin/env python3
"""Explicitly approved, no-clobber GitHub release backup of a sealed checkpoint.

`plan` is local-only. `upload` requires the exact approved plan hash; publication
also requires --publish. Credentials stay in the controller's official gh CLI.
No checkpoint, release, asset or tag is deleted or replaced. Temporary transfer
files alone are cleaned up. Public restore uses no GitHub credentials.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from checkpoint_sync import (BUNDLE_MANIFEST, DIGEST, checkpoint_name, pull_checkpoint,
                             reject_symlinks, run_cli, safe_relative, sha256, validate_manifest)
from colab_safe_cli import report_error
from picoagent.training.data import canonical_json
from picoagent.training.provenance import verify_checkpoint, write_json
from picoagent.training.retention import _exclusive_lock

SCHEMA = "picoagent.github-checkpoint-plan.v1"
MAX_ASSETS = 1000
ASSET_LIMIT = 2 * 1024**3  # GitHub requires strictly less than 2 GiB.
METADATA_LIMIT = 16 * 1024**2
PARALLEL_CHUNK_LIMIT = 32 * 1024**2
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_json(path):
    if path.is_symlink() or path.stat().st_size > METADATA_LIMIT:
        raise ValueError("Unsafe or oversized metadata file")
    return json.loads(path.read_text())


def tag_for(identity):
    return (f"checkpoint-s{identity['source_tree_sha256']}-r{identity['run_manifest_sha256']}"
            f"-{identity['checkpoint']}-m{identity['transfer_manifest_sha256'][:16]}")


def asset_records(manifest, manifest_size, manifest_hash):
    assets = {}
    for info in manifest["files"].values():
        for chunk in info["chunks"]:
            assets[f"chunk-{chunk['sha256']}.bin"] = {"sha256": chunk["sha256"], "bytes": chunk["bytes"]}
    assets[BUNDLE_MANIFEST] = {"sha256": manifest_hash, "bytes": manifest_size}
    if len(assets) > MAX_ASSETS or any(not 0 < item["bytes"] < ASSET_LIMIT for item in assets.values()):
        raise ValueError("Release exceeds GitHub asset count or size limits")
    return dict(sorted(assets.items()))


def verify_source(run_dir, run, manifest):
    code = run.get("code", {})
    files = code.get("files")
    source = run.get("identity", {}).get("source_tree_sha256")
    if (not isinstance(files, dict) or not files or not DIGEST.fullmatch(str(source))
            or digest_json(files) != source or code.get("tree_sha256") != source):
        raise ValueError("Run source identity is incomplete or inconsistent")
    for relative, digest in files.items():
        safe_relative(relative)
        snapshot = f"source_snapshot/{relative}"
        if manifest["files"].get(snapshot, {}).get("sha256") != digest:
            raise ValueError("Transfer omits or changes an exact source snapshot")
        if sha256(run_dir / snapshot) != digest:
            raise ValueError("Source snapshot integrity mismatch")
    return source


def make_plan(run_dir, manifest_path, repository):
    """Verify every local source/file/chunk before describing public data scope."""
    if not REPO.fullmatch(repository) or repository.split("/")[1] in {".", ".."}:
        raise ValueError("Expected explicit GitHub OWNER/REPO")
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    name = checkpoint_name(manifest["checkpoint"])
    for relative, info in manifest["files"].items():
        path = run_dir / safe_relative(relative)
        if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
            raise ValueError("Transfer source must contain regular files without symlinks")
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError("Local file disagrees with pinned transfer manifest")
        with path.open("rb") as handle:
            for chunk in info["chunks"]:
                digest = hashlib.sha256()
                left = chunk["bytes"]
                while left:
                    data = handle.read(min(left, 1024**2))
                    if not data:
                        raise ValueError("Local chunk is truncated")
                    digest.update(data)
                    left -= len(data)
                if digest.hexdigest() != chunk["sha256"]:
                    raise ValueError("Local chunk integrity mismatch")
    reject_symlinks(run_dir / name)
    run_hash = sha256(run_dir / "run_manifest.json")
    verify_checkpoint(run_dir / name, run_hash)
    expected = {str(path.relative_to(run_dir)) for path in (run_dir / name).rglob("*") if path.is_file()}
    actual = {relative for relative in manifest["files"] if relative.startswith(name + "/")}
    if expected != actual:
        raise ValueError("Transfer does not cover the exact checkpoint tree")
    state = read_json(run_dir / name / "trainer_state.json")
    if type(state.get("global_step")) is not int or state["global_step"] != int(name.split("-")[1]):
        raise ValueError("Checkpoint step mismatch")
    run = read_json(run_dir / "run_manifest.json")
    if run.get("schema") != "picoagent.training.run.v1":
        raise ValueError("Unknown training run identity")
    source = verify_source(run_dir, run, manifest)
    commit = run["code"].get("git", {}).get("commit")
    if not COMMIT.fullmatch(str(commit)):
        raise ValueError("An exact original source Git commit is required")
    identity = {"repository": repository, "visibility": "public", "checkpoint": name,
                "run_manifest_sha256": run_hash, "source_tree_sha256": source,
                "source_commit": commit, "transfer_manifest_sha256": sha256(manifest_path),
                "checkpoint_manifest_sha256": sha256(run_dir / name / "checkpoint_manifest.json")}
    return {"schema": SCHEMA, "identity": identity, "tag": tag_for(identity),
            "assets": asset_records(manifest, manifest_path.stat().st_size, sha256(manifest_path))}


def validate_plan(plan):
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unknown release plan")
    identity = plan["identity"]
    checkpoint_name(identity["checkpoint"])
    if (not REPO.fullmatch(identity["repository"]) or identity.get("visibility") != "public"
            or identity["repository"].split("/")[1] in {".", ".."}
            or not COMMIT.fullmatch(identity["source_commit"]) or plan["tag"] != tag_for(identity)):
        raise ValueError("Invalid pinned release identity")
    for key in ("run_manifest_sha256", "source_tree_sha256", "transfer_manifest_sha256", "checkpoint_manifest_sha256"):
        if not DIGEST.fullmatch(identity[key]):
            raise ValueError("Invalid pinned identity digest")
    assets = plan["assets"]
    if not assets or len(assets) > MAX_ASSETS or BUNDLE_MANIFEST not in assets:
        raise ValueError("Invalid release asset inventory")
    for name, record in assets.items():
        if (not DIGEST.fullmatch(record["sha256"]) or type(record["bytes"]) is not int
                or not 0 < record["bytes"] < ASSET_LIMIT
                or (name != BUNDLE_MANIFEST and name != f"chunk-{record['sha256']}.bin")):
            raise ValueError("Invalid pinned release asset")
    if assets[BUNDLE_MANIFEST]["sha256"] != identity["transfer_manifest_sha256"]:
        raise ValueError("Release manifest pin mismatch")


def readback(stream, record, output=None):
    digest, count = hashlib.sha256(), 0
    while data := stream.read(1024**2):
        count += len(data)
        if count > record["bytes"]:
            raise ValueError("Read-back exceeds pinned asset length")
        digest.update(data)
        if output is not None:
            output.write(data)
    if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Read-back asset integrity mismatch")


class GitHub:
    """Only the official controller CLI handles authenticated GitHub requests."""
    def __init__(self, repository):
        self.repository = repository
        self.prefix = f"repos/{repository}"

    def call(self, *args):
        return run_cli(["gh", *args], timeout=600, capture=True).stdout

    def api(self, path):
        return json.loads(self.call("api", "--hostname", "github.com", "--method", "GET", path))

    def pages(self, path):
        result = []
        for page in range(1, 102):
            batch = self.api(f"{path}?per_page=100&page={page}")
            if not isinstance(batch, list):
                raise ValueError("Invalid GitHub inventory")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise ValueError("GitHub inventory exceeded bounded pagination")

    def release(self, tag):
        matches = [item for item in self.pages(f"{self.prefix}/releases") if item["tag_name"] == tag]
        if len(matches) > 1:
            raise ValueError("Duplicate release tag")
        return matches[0] if matches else None

    def assets(self, release_id):
        return self.pages(f"{self.prefix}/releases/{release_id}/assets")

    def check_tag(self, tag, commit, *, required=False):
        refs = self.api(f"{self.prefix}/git/matching-refs/tags/{quote(tag, safe='')}")
        found = False
        for ref in refs:
            if ref["ref"] == f"refs/tags/{tag}":
                found = True
                if ref["object"].get("type") != "commit" or ref["object"].get("sha") != commit:
                    raise ValueError("Existing tag does not point to the exact source commit")
        if required and not found:
            raise ValueError("Published source tag was not confirmed")

    def create(self, plan, body):
        self.call("release", "create", plan["tag"], "--repo", f"https://github.com/{self.repository}",
                  "--draft", "--prerelease", "--latest=false", "--target", plan["identity"]["source_commit"],
                  "--title", plan["tag"], "--notes", body)

    def upload(self, tag, path):
        self.call("release", "upload", tag, str(path), "--repo", f"https://github.com/{self.repository}")

    def publish(self, tag):
        self.call("release", "edit", tag, "--repo", f"https://github.com/{self.repository}", "--draft=false", "--latest=false")

    @contextmanager
    def stream(self, asset):
        argv = ["gh", "api", "--hostname", "github.com", "--method", "GET",
                f"{self.prefix}/releases/assets/{asset['id']}", "-H", "Accept: application/octet-stream"]
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        def stop():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        timer = threading.Timer(600, stop)
        timer.start()
        try:
            yield process.stdout
            if process.wait(timeout=5):
                raise RuntimeError("GitHub read-back failed")
        finally:
            timer.cancel()
            stop()
            process.stdout.close()
            process.wait()


def asset_inventory(client, release, plan):
    assets = {}
    for asset in client.assets(release["id"]):
        name = asset["name"]
        if name in assets or name not in plan["assets"]:
            raise ValueError("Duplicate or unexpected release asset collision")
        record = plan["assets"][name]
        if (asset.get("state") != "uploaded" or asset.get("size") != record["bytes"]
                or type(asset.get("id")) is not int or asset["id"] <= 0
                or (asset.get("digest") is not None and asset["digest"] != f"sha256:{record['sha256']}")):
            raise ValueError("Release asset size, state or API digest collision")
        assets[name] = asset
    if BUNDLE_MANIFEST in assets and set(assets) != set(plan["assets"]):
        raise ValueError("Completion manifest exists without every pinned asset")
    return assets


def check_release(release, plan, body):
    if (release["tag_name"] != plan["tag"] or release.get("body") != body
            or release.get("target_commitish") != plan["identity"]["source_commit"]
            or type(release.get("draft")) is not bool):
        raise ValueError("Existing release identity collision")


def asset_identity(assets):
    # download_count and other observation metadata may change on read-back.
    return {name: {key: asset.get(key) for key in ("id", "name", "size", "state", "digest")}
            for name, asset in assets.items()}


def run_batch(names, operation, workers):
    """Join every worker before returning/raising, including on interruption.

    Batches contain at most workers items, so no unbounded queued work or disk
    materialization can accumulate. Running CLI operations retain their timeout.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        try:
            for name in names:
                futures.append(executor.submit(operation, name))
            return [future.result() for future in as_completed(futures)]
        except BaseException:
            for future in futures:
                future.cancel()
            raise  # Executor drains running operations before unwinding cleanup.


def upload_checkpoint(run_dir, manifest_path, plan, *, approved_plan_sha256, publish=False,
                      client=None, public_opener=None, upload_workers=1):
    """Call only after approval of this exact repository/public data scope.

    Parallel clients must support concurrent upload/stream calls. Inventory and
    release mutations stay on the caller thread. Default behavior remains serial.
    """
    if type(upload_workers) is not int or not 1 <= upload_workers <= 4:
        raise ValueError("upload_workers must be an integer from 1 to 4")
    validate_plan(plan)
    if upload_workers > 1 and any(record["bytes"] > PARALLEL_CHUNK_LIMIT
                                  for record in plan["assets"].values()):
        raise ValueError("Parallel upload requires assets no larger than 32 MiB")
    if digest_json(plan) != approved_plan_sha256:
        raise ValueError("Exact approved plan SHA256 is required before any GitHub access")
    if make_plan(run_dir, manifest_path, plan["identity"]["repository"]) != plan:
        raise ValueError("Local checkpoint changed after release approval")
    client = client or GitHub(plan["identity"]["repository"])
    if client.api(client.prefix).get("private") is not False:
        raise ValueError("Approved public repository visibility mismatch")
    body = canonical_json({"schema": SCHEMA, "plan_sha256": digest_json(plan), "identity": plan["identity"]})
    client.check_tag(plan["tag"], plan["identity"]["source_commit"])
    release = client.release(plan["tag"])
    if release is None:
        client.create(plan, body)
        release = client.release(plan["tag"])
        if release is None:
            raise RuntimeError("Draft creation was not confirmed; inspect before retry")
    check_release(release, plan, body)
    assets = asset_inventory(client, release, plan)
    if not release["draft"] and set(assets) != set(plan["assets"]):
        raise ValueError("Published release is incomplete; refusing mutation")
    manifest = read_json(manifest_path)
    locations = {}
    for relative, info in manifest["files"].items():
        offset = 0
        for chunk in info["chunks"]:
            locations.setdefault(f"chunk-{chunk['sha256']}.bin", (run_dir / relative, offset))
            offset += chunk["bytes"]
    locations[BUNDLE_MANIFEST] = (manifest_path, 0)
    ordered = sorted(name for name in locations if name != BUNDLE_MANIFEST) + [BUNDLE_MANIFEST]
    verified = {}
    with tempfile.TemporaryDirectory(prefix="checkpoint-release-") as directory:
        def upload_one(name):
            record = plan["assets"][name]
            path = Path(directory) / name
            source, offset = locations[name]
            try:
                with source.open("rb") as input_file, path.open("xb") as output:
                    input_file.seek(offset)
                    left = record["bytes"]
                    while left:
                        data = input_file.read(min(left, 1024**2))
                        if not data:
                            raise ValueError("Upload source became truncated")
                        output.write(data)
                        left -= len(data)
                if sha256(path) != record["sha256"]:
                    raise ValueError("Upload source changed")
                client.upload(plan["tag"], path)  # Never --clobber.
            finally:
                path.unlink(missing_ok=True)  # Only this invocation's temporary chunk.

        def verify_one(name):
            record = plan["assets"][name]
            with client.stream(assets[name]) as stream:
                readback(stream, record)
            return name, {"id": assets[name]["id"], **record}

        if upload_workers > 1:
            chunks = ordered[:-1]
            for start in range(0, len(chunks), upload_workers):
                batch = chunks[start:start + upload_workers]
                missing = [name for name in batch if name not in assets]
                previous = asset_identity(assets)
                run_batch(missing, upload_one, upload_workers)
                # No sibling upload is still in progress when we inspect state.
                assets = asset_inventory(client, release, plan)
                if set(assets) != set(previous) | set(missing):
                    raise RuntimeError("Uploaded asset inventory was not confirmed; inspect before retry")
                if asset_identity({name: assets[name] for name in previous}) != previous:
                    raise ValueError("Release assets changed during upload")
                verified.update(run_batch(batch, verify_one, upload_workers))
            ordered = [BUNDLE_MANIFEST]
        for name in ordered:
            if name not in assets:
                previous = asset_identity(assets)
                upload_one(name)
                assets = asset_inventory(client, release, plan)
                if upload_workers > 1 and (
                        set(assets) != set(previous) | {name}
                        or asset_identity({key: assets[key] for key in previous if key in assets}) != previous):
                    raise ValueError("Release assets changed during manifest upload")
                if name not in assets:
                    raise RuntimeError("Uploaded asset was not confirmed; inspect before retry")
            key, record = verify_one(name)
            verified[key] = record
    latest = client.release(plan["tag"])
    check_release(latest, plan, body)
    if asset_identity(asset_inventory(client, latest, plan)) != asset_identity(assets):
        raise ValueError("Release assets changed during verification")
    if publish and latest["draft"]:
        client.check_tag(plan["tag"], plan["identity"]["source_commit"])
        client.publish(plan["tag"])
        latest = client.release(plan["tag"])
        check_release(latest, plan, body)
        if latest["draft"] or asset_identity(asset_inventory(client, latest, plan)) != asset_identity(assets):
            raise RuntimeError("Publication identity was not confirmed")
    if not latest["draft"]:
        client.check_tag(plan["tag"], plan["identity"]["source_commit"], required=True)
        if client.api(client.prefix).get("private") is not False:
            raise ValueError("Published public visibility was not confirmed")
        with tempfile.TemporaryDirectory(prefix="checkpoint-public-check-") as directory:
            PublicTransfer(plan, public_opener).download("release/" + BUNDLE_MANIFEST,
                                                        Path(directory) / BUNDLE_MANIFEST)
    return {"schema": "picoagent.github-release-receipt.v1" if not latest["draft"] else "picoagent.github-draft-staging.v1",
            "published": not latest["draft"], "plan_sha256": digest_json(plan), "identity": plan["identity"],
            "tag": plan["tag"], "release_id": latest["id"], "assets": verified,
            "independent_readback_verified": True}


class GitHubRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}:
            raise ValueError("Unexpected public asset redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicTransfer:
    def __init__(self, plan, opener=None):
        validate_plan(plan)
        self.plan = plan
        self.opener = opener or build_opener(GitHubRedirects())

    def download(self, remote, local):
        relative = remote.removeprefix("release/")
        if relative == BUNDLE_MANIFEST:
            name = BUNDLE_MANIFEST
        elif relative.startswith("chunks/") and DIGEST.fullmatch(relative[7:]):
            name = f"chunk-{relative[7:]}.bin"
        else:
            raise ValueError("Unexpected restore asset path")
        record = self.plan["assets"].get(name)
        if record is None:
            raise ValueError("Restore asset is outside pinned release inventory")
        url = (f"https://github.com/{self.plan['identity']['repository']}/releases/download/"
               f"{quote(self.plan['tag'], safe='')}/{name}")
        request = Request(url, headers={"Accept": "application/octet-stream"})
        with self.opener.open(request, timeout=60) as response, Path(local).open("xb") as output:
            readback(response, record, output)


def restore_checkpoint(plan, destination, *, expected_plan_sha256, opener=None):
    validate_plan(plan)
    if digest_json(plan) != expected_plan_sha256:
        raise ValueError("Exact trusted plan SHA256 is required before public restore")
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("Restore destination must not contain symlinks")
    destination.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(destination), tempfile.TemporaryDirectory(prefix="checkpoint-public-manifest-") as directory:
        transfer = PublicTransfer(plan, opener)
        manifest_path = Path(directory) / BUNDLE_MANIFEST
        transfer.download("release/" + BUNDLE_MANIFEST, manifest_path)
        manifest = read_json(manifest_path)
        validate_manifest(manifest)
        if (manifest["checkpoint"] != plan["identity"]["checkpoint"]
                or asset_records(manifest, manifest_path.stat().st_size, sha256(manifest_path)) != plan["assets"]
                or manifest["files"]["run_manifest.json"]["sha256"] != plan["identity"]["run_manifest_sha256"]
                or manifest["files"][f"{manifest['checkpoint']}/checkpoint_manifest.json"]["sha256"] != plan["identity"]["checkpoint_manifest_sha256"]):
            raise ValueError("Pinned restore identity does not match transfer manifest")
        result = pull_checkpoint("release", destination, transfer, off_runtime=False,
                                 restore=True, expected_manifest_sha256=plan["identity"]["transfer_manifest_sha256"])
        if make_plan(destination, manifest_path, plan["identity"]["repository"]) != plan:
            raise ValueError("Restored source/run/checkpoint identity differs from approved plan")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_cmd = commands.add_parser("plan", help="Local-only integrity and publication scope review")
    upload = commands.add_parser("upload", help="Requires prior approval of exact public data scope")
    restore = commands.add_parser("restore", help="Anonymous restore using a pinned local plan")
    for command in (plan_cmd, upload):
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--transfer-manifest", type=Path, required=True)
    plan_cmd.add_argument("--repository", required=True)
    plan_cmd.add_argument("--output", type=Path, required=True)
    for command in (upload, restore):
        command.add_argument("--plan", type=Path, required=True)
    upload.add_argument("--approved-plan-sha256", required=True)
    upload.add_argument("--publish", action="store_true", help="Only after explicit public publication approval")
    upload.add_argument("--upload-workers", type=int, choices=range(1, 5), default=1,
                        help="Bounded upload/read-back workers (default: serial; parallel assets <=32 MiB)")
    upload.add_argument("--receipt", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(args.run_dir, args.transfer_manifest, args.repository)
        write_json(args.output, plan, exclusive=True)
        print(json.dumps({"plan_sha256": digest_json(plan), "tag": plan["tag"], "assets": len(plan["assets"])}))
    elif args.command == "upload":
        if args.receipt.exists() or args.receipt.is_symlink():
            raise ValueError("Receipt output must be new")
        with _exclusive_lock(args.run_dir):
            result = upload_checkpoint(args.run_dir, args.transfer_manifest, read_json(args.plan),
                                       approved_plan_sha256=args.approved_plan_sha256, publish=args.publish,
                                       upload_workers=args.upload_workers)
            write_json(args.receipt, result, exclusive=True)
        print(json.dumps({"published": result["published"], "receipt": str(args.receipt)}))
    else:
        print(restore_checkpoint(read_json(args.plan), args.destination,
                                 expected_plan_sha256=args.expected_plan_sha256))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        report_error("github-checkpoint-release", error)
        raise SystemExit(1) from None
