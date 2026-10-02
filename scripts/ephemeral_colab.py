"""External controller: vendor auth unchanged, runtime credentials memory-only.

No CLI state/history/log setup, no implicit runtime allocation or termination.
Descriptor files contain only allowlisted nonsecret reconnect identifiers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import logging
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import uuid

from colab_safe_cli import sanitized_controller_error

FIELDS = {'schema', 'name', 'endpoint', 'kernel_id', 'session_id', 'notebook_hash', 'phase'}
IDENTIFIER = re.compile(r'[A-Za-z0-9_.:-]{1,256}\Z')
SCHEMA = 'picoagent.ephemeral-colab.v1'


MAX_TRANSFER_BYTES = 32 * 1024 * 1024
_LOG_LOCK = threading.RLock()
_LOG_USERS = 0
_LOG_TARGETS = []
_LOG_NULL = None


class _PrivateCallFilter(logging.Filter):
    def filter(self, record):
        return False


_LOG_FILTER = _PrivateCallFilter()


@contextmanager
def private_vendor_logging():
    """Drop logs during sensitive vendor calls, including background WS logs.

    Shared reference counting prevents overlapping transfers from prematurely
    restoring handlers. The lock covers setup/teardown only, never network IO.
    A temporary NullHandler prevents module-level logging from auto-configuring
    an unfiltered handler. Existing handlers/levels are left otherwise intact.
    """
    global _LOG_USERS, _LOG_TARGETS, _LOG_NULL
    with _LOG_LOCK:
        if _LOG_USERS == 0:
            root = logging.getLogger()
            _LOG_NULL = logging.NullHandler()
            root.addHandler(_LOG_NULL)
            loggers = [root, *(value for value in logging.Logger.manager.loggerDict.values()
                               if isinstance(value, logging.Logger))]
            handlers = {handler for logger in loggers for handler in logger.handlers}
            if logging.lastResort is not None:
                handlers.add(logging.lastResort)
            _LOG_TARGETS = list(handlers)
            for handler in _LOG_TARGETS:
                handler.addFilter(_LOG_FILTER)
        _LOG_USERS += 1
    try:
        yield
    finally:
        with _LOG_LOCK:
            _LOG_USERS -= 1
            if _LOG_USERS == 0:
                for handler in _LOG_TARGETS:
                    handler.removeFilter(_LOG_FILTER)
                logging.getLogger().removeHandler(_LOG_NULL)
                _LOG_TARGETS, _LOG_NULL = [], None


def private_call(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with private_vendor_logging():
            return function(*args, **kwargs)
    return wrapped


def remote_content_path(value):
    if not isinstance(value, str) or value.startswith('//'):
        raise ValueError('Expected a canonical content path')
    value = value.removeprefix('/')
    parts = value.split('/')
    if (len(parts) < 2 or parts[0] != 'content'
            or any(part in {'', '.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', part)
                   for part in parts)):
        raise ValueError('Remote path must be canonical and inside content/')
    return value


def local_transfer_path(path):
    path = safe_path(path)
    if path.exists() and not path.is_file():
        raise ValueError('Transfer path must be a regular file')
    return path


def safe_path(path):
    if '..' in Path(path).parts:
        raise ValueError('Parent traversal is prohibited')
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Local path cannot contain symlinks')
    return path


def validate_descriptor(value):
    if not isinstance(value, dict) or set(value) - FIELDS or value.get('schema') != SCHEMA:
        raise ValueError('Invalid nonsecret reconnect descriptor')
    for field in FIELDS - {'schema', 'phase'}:
        if value.get(field) is not None and not IDENTIFIER.fullmatch(str(value[field])):
            raise ValueError('Invalid reconnect identifier')
    if not value.get('name') or value.get('phase') not in {'ready', 'allocation_requested'}:
        raise ValueError('Incomplete reconnect descriptor')
    if value['phase'] == 'ready' and not value.get('endpoint'):
        raise ValueError('An exact endpoint is required')
    return value


def save_descriptor(path, value, *, exclusive=False):
    validate_descriptor(value)
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.reconnect-', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_descriptor(path):
    path = safe_path(path)
    if path.stat().st_size > 4096:
        raise ValueError('Oversized reconnect descriptor')
    return validate_descriptor(json.loads(path.read_text()))


@private_call
def vendor_client():
    # This property alone invokes the vendor's unchanged existing credential loader.
    from colab_cli.common import state
    try:
        return state.client
    except BaseException:
        raise RuntimeError('Vendor credential loading failed; private details suppressed') from None


class EphemeralController:
    def __init__(self, descriptor, *, client=None, runtime_factory=None,
                 contents_factory=None, session_factory=None, timeout=600, download_workers=1):
        self.path = safe_path(descriptor)
        self.record = load_descriptor(self.path)
        if self.record['phase'] != 'ready':
            raise RuntimeError('Allocation outcome is uncertain; inspect assignments, never auto-reallocate')
        if type(download_workers) is not int or not 1 <= download_workers <= 4:
            raise ValueError('Expected 1 to 4 download workers')
        from colab_cli.runtime import ColabRuntime
        from colab_cli.contents import ContentsClient
        from colab_cli.state import SessionState
        self.client = client if client is not None else vendor_client()
        self.runtime_factory = runtime_factory or ColabRuntime
        self.contents_factory = contents_factory or ContentsClient
        self.session_factory = session_factory or SessionState
        self.timeout, self.download_workers = timeout, download_workers
        self.runtime = self.session = None
        self._runtime_log_guard = None
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._refresh(force=True)
        except BaseException as error:
            self._disconnect()
            self.session = None
            raise sanitized_controller_error(error, 'reconnect',
                'Exact endpoint reconnect failed; no runtime was allocated') from None

    def _save_id(self, field, value):
        with self._lock:
            updated = {**self.record, field: value}
            save_descriptor(self.path, updated)
            self.record = updated
            setattr(self.session, field, value)

    @private_call
    def _disconnect(self):
        try:
            if self.runtime is not None:
                self.runtime.stop(shutdown_kernel=False)
        except Exception:
            pass  # Never expose vendor request URLs or tokens through cleanup errors.
        finally:
            self.runtime = None
            if self._runtime_log_guard is not None:
                self._runtime_log_guard.__exit__(None, None, None)
                self._runtime_log_guard = None

    @private_call
    def _refresh(self, *, force=False):
        if self._closed:
            raise RuntimeError('Controller is closed')
        if (not force and self.session is not None and self.session.token_expires_at is not None
                and self.session.token_expires_at > datetime.now(timezone.utc) + timedelta(minutes=5)):
            return self.session
        matches = [assignment for assignment in self.client.list_assignments()
                   if assignment.endpoint == self.record['endpoint']]
        if len(matches) != 1:
            self._disconnect()
            self.session = None
            raise RuntimeError('Exact saved endpoint is absent or ambiguous; no runtime was allocated')
        proxy = matches[0].runtime_proxy_info
        if self.session is not None and (self.session.token != proxy.token or self.session.url != proxy.url):
            self._disconnect()
        self.session = self.session_factory(
            name=self.record['name'], endpoint=self.record['endpoint'], token=proxy.token,
            url=proxy.url, token_expires_at=proxy.expires_at(),
            kernel_id=self.record.get('kernel_id'), session_id=self.record.get('session_id'))
        return self.session

    def refresh(self):
        with self._lock:
            try:
                self._refresh(force=True)
            except BaseException as error:
                self._disconnect()
                raise sanitized_controller_error(error, 'refresh', 'Session refresh failed') from None
        return dict(self.record)

    @private_call
    def execute(self, code):
        from colab_persistent_stage import parse_result
        with self._lock:
            try:
                session = self._refresh()
                if self.runtime is None:
                    # Websocket background threads can log after execute returns.
                    # Keep the same filter active for the connection's lifetime.
                    self._runtime_log_guard = private_vendor_logging()
                    self._runtime_log_guard.__enter__()
                    self.runtime = self.runtime_factory(
                        session.url, session.token, kernel_id=session.kernel_id,
                        session_id=session.session_id,
                        on_kernel_started=lambda value: self._save_id('kernel_id', value),
                        on_session_started=lambda value: self._save_id('session_id', value))
                return parse_result(self.runtime.execute_code(code, timeout=self.timeout))
            except BaseException as error:
                self._disconnect()
                raise sanitized_controller_error(error, 'execute', 'Remote execution failed') from None

    def transfer(self):
        return self

    @private_call
    def _contents(self):
        with self._lock:
            return self.contents_factory(self._refresh())

    @private_call
    def download(self, remote, local):
        temporary = None
        try:
            remote, local = remote_content_path(remote), local_transfer_path(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            contents = self._contents()
            # Official ContentsClient buffers its full response. Metadata and
            # post-read bounds protect ordinary transfers, not hostile servers
            # lying about size. No claim of a streaming memory cap is made.
            parent, name = remote.rsplit('/', 1)
            listing = contents.list_dir(parent)
            if listing.get('type') != 'directory' or not isinstance(listing.get('content'), list):
                raise ValueError('Expected a directory listing for transfer metadata')
            candidates = [entry for entry in listing['content'] if isinstance(entry, dict)
                          and (entry.get('name') == name or entry.get('path') == remote)]
            if (len(candidates) != 1 or candidates[0].get('name') != name
                    or candidates[0].get('path') != remote):
                raise ValueError('Remote transfer listing identity is absent or ambiguous')
            metadata = candidates[0]
            size = metadata.get('size')
            if metadata.get('type') != 'file' or type(size) is not int or not 0 <= size <= MAX_TRANSFER_BYTES:
                raise ValueError('Remote transfer metadata exceeds bounds')
            fd, name = tempfile.mkstemp(prefix='.colab-download-', dir=local.parent)
            os.close(fd)
            temporary = Path(name)
            contents.download(remote, str(temporary))
            local_transfer_path(temporary)
            if temporary.stat().st_size != size or temporary.stat().st_size > MAX_TRANSFER_BYTES:
                raise ValueError('Downloaded size differs from bounded metadata')
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            local_transfer_path(local)
            os.replace(temporary, local)
        except BaseException as error:
            with self._lock:
                self._disconnect()
            raise sanitized_controller_error(error, 'download', 'Download failed') from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @private_call
    def upload(self, local, remote):
        temporary = None
        try:
            remote, local = remote_content_path(remote), local_transfer_path(local)
            source_fd = os.open(local, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(source_fd, 'rb') as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= MAX_TRANSFER_BYTES:
                    raise ValueError('Upload source exceeds bounds')
                fd, name = tempfile.mkstemp(prefix='.colab-upload-', dir=local.parent)
                temporary = Path(name)
                with os.fdopen(fd, 'wb') as snapshot:
                    remaining = MAX_TRANSFER_BYTES
                    while block := source.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(block)
                        if remaining < 0:
                            raise ValueError('Upload source grew beyond bounds')
                        snapshot.write(block)
                after = os.fstat(source.fileno())
                if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise ValueError('Upload source changed while snapshotting')
            self._contents().upload(str(temporary), remote)
        except BaseException:
            with self._lock:
                self._disconnect()
            raise RuntimeError('Upload failed; private error details suppressed') from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def command(self, action, local, remote, *, capture=True):
        if action != 'upload':
            raise ValueError('Only explicit upload is supported by this adapter')
        return self.upload(local, remote)

    def close(self):
        with self._lock:
            try:
                self._disconnect()
            finally:
                self.session = None
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


@private_call
def allocate(descriptor, *, name, approved=False, variant=None, accelerator=None, shape=None,
             client=None):
    """A separate explicit allocation, never used as a reconnect fallback.

    Reserve the nonsecret notebook ID before assign. Uncertain errors retain the
    reservation for manual read-only reconciliation; no retry or unassign here.
    """
    if approved is not True or None in (variant, accelerator, shape):
        raise ValueError('Explicit allocation approval and exact hardware choices are required')
    from colab_cli.client import Variant, Accelerator, Shape
    variant, accelerator, shape = Variant(variant), Accelerator(accelerator), Shape(shape)
    record = {'schema': SCHEMA, 'name': name, 'phase': 'allocation_requested',
              'notebook_hash': str(uuid.uuid4())}
    save_descriptor(descriptor, record, exclusive=True)
    client = client if client is not None else vendor_client()
    try:
        assignment = client.assign(uuid.UUID(record['notebook_hash']),
                                   variant=variant, accelerator=accelerator, shape=shape)
        record = {**record, 'phase': 'ready', 'endpoint': assignment.endpoint}
        save_descriptor(descriptor, record)
    except BaseException:
        raise RuntimeError('Allocation outcome uncertain; inspect assignments before another action') from None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', required=True, type=Path)
    parser.add_argument('--allocate', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--variant', choices=['DEFAULT', 'GPU', 'TPU'])
    parser.add_argument('--accelerator', choices=['NONE', 'T4', 'L4', 'A100', 'H100', 'V5E1', 'V6E1'])
    parser.add_argument('--shape', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.allocate:
        result = allocate(args.descriptor, name=args.name, approved=True,
                          variant=args.variant, accelerator=args.accelerator, shape=args.shape)
    else:
        with EphemeralController(args.descriptor) as controller:
            result = dict(controller.record)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code == 0:
            raise
        print(json.dumps({'error': type(error).__name__, 'details': 'suppressed'}))
        raise SystemExit(1) from None
