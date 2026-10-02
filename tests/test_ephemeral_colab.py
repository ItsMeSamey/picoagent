from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytest.importorskip('colab_cli', reason='Optional Colab adapter requires the installed vendor SDK')

PROJECT = Path(__file__).parents[1]
SCRIPTS = PROJECT / 'scripts' if (PROJECT / 'scripts').is_dir() else PROJECT / 'picoagent/scripts'
sys.path[:0] = [str(Path(__file__).parent), str(SCRIPTS)]
from ephemeral_colab import EphemeralController, SCHEMA, allocate, load_descriptor, save_descriptor  # noqa: E402

SECRET = 'FAKE-SECRET-MUST-NOT-PERSIST'


def assignment(endpoint, token=SECRET):
    return SimpleNamespace(endpoint=endpoint, runtime_proxy_info=SimpleNamespace(
        token=token, url='https://runtime.invalid',
        expires_at=lambda: datetime.now(timezone.utc) + timedelta(hours=1)))


class Client:
    def __init__(self):
        self.assignments = [assignment('other'), assignment('exact')]
        self.allocations = 0
        self.reads = 0

    def list_assignments(self):
        self.reads += 1
        return self.assignments

    def assign(self, *args, **kwargs):
        self.allocations += 1
        return assignment('new-exact')


class Runtime:
    fail = False
    stopped = []
    calls = []

    def __init__(self, url, token, **kwargs):
        assert 'history' not in kwargs and 'session_name' not in kwargs
        self.token, self.options = token, kwargs
        self.calls.append(kwargs)

    def execute_code(self, code, timeout):
        self.options['on_kernel_started']('kernel-1')
        self.options['on_session_started']('session-1')
        if self.fail:
            raise RuntimeError(SECRET)
        return [{'output_type': 'stream', 'text': 'PICOAGENT_RESULT={"ok": true}\n'}]

    def stop(self, shutdown_kernel):
        assert shutdown_kernel is False
        self.stopped.append(self)


class Contents:
    def __init__(self, session):
        assert session.endpoint == 'exact'
        assert session.token == SECRET

    def list_dir(self, parent):
        assert parent == 'content'
        return {'type': 'directory', 'content': [
            {'name': name, 'path': parent + '/' + name, 'type': 'file', 'size': 7}
            for name in ('fixture', 'file')]}

    def download(self, remote, local):
        Path(local).write_text('payload')

    def upload(self, local, remote):
        assert Path(local).read_text() == 'payload'


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'reconnect.json'
    save_descriptor(path, {'schema': SCHEMA, 'name': 'fixture', 'phase': 'ready',
                           'endpoint': 'exact'}, exclusive=True)
    Runtime.fail, Runtime.stopped, Runtime.calls = False, [], []
    client = Client()
    return path, client


def test_reconnect_exact_endpoint_no_allocation_or_credential_serialization(setup):
    path, client = setup
    with EphemeralController(path, client=client, runtime_factory=Runtime,
                             contents_factory=Contents) as controller:
        assert controller.session.endpoint == 'exact'
        assert controller.runtime is None  # Read-only reconnect never starts a kernel.
        assert controller.execute('fixture') == {'ok': True}
        controller.download('content/fixture', path.parent / 'download')
        controller.upload(path.parent / 'download', '/content/fixture')
        assert controller.transfer() is controller
    record = load_descriptor(path)
    assert record['kernel_id'] == 'kernel-1' and record['session_id'] == 'session-1'
    assert client.allocations == 0
    assert len(Runtime.stopped) == 1
    assert SECRET not in path.read_text()
    assert 'url' not in record and 'token' not in record
    assert {p.name for p in path.parent.iterdir()} == {'reconnect.json', 'download'}


@pytest.mark.parametrize('endpoints', [[], ['other'], ['exact', 'exact']])
def test_missing_or_ambiguous_endpoint_never_allocates(setup, endpoints):
    path, client = setup
    client.assignments = [assignment(endpoint) for endpoint in endpoints]
    with pytest.raises(RuntimeError, match='no runtime was allocated'):
        EphemeralController(path, client=client)
    assert client.allocations == 0


def test_execution_error_closes_channels_without_shutdown_and_sanitizes(setup):
    path, client = setup
    Runtime.fail = True
    with EphemeralController(path, client=client, runtime_factory=Runtime) as controller:
        with pytest.raises(RuntimeError) as result:
            controller.execute('failure')
        assert SECRET not in str(result.value)
        assert controller.runtime is None
    assert len(Runtime.stopped) == 1
    assert SECRET not in path.read_text()


def test_refresh_updates_credentials_only_in_memory_and_reuses_ids(setup):
    path, client = setup
    with EphemeralController(path, client=client, runtime_factory=Runtime) as controller:
        controller.execute('first')
        client.assignments = [assignment('exact', token='NEW-SECRET')]
        controller.refresh()
        assert controller.session.token == 'NEW-SECRET'
        assert controller.runtime is None
        controller.execute('second')
        assert Runtime.calls[-1]['kernel_id'] == 'kernel-1'
        assert Runtime.calls[-1]['session_id'] == 'session-1'
    assert 'NEW-SECRET' not in path.read_text()
    assert client.allocations == 0


def test_no_store_history_or_logging_setup_access(setup, monkeypatch):
    from colab_cli import common
    path, client = setup

    class State:
        @property
        def client(self):
            return client

        def __getattr__(self, field):
            pytest.fail('Forbidden state access: ' + field)

    monkeypatch.setattr(common, 'state', State())
    monkeypatch.setattr(common, 'setup_logging', lambda *args: pytest.fail('Logging setup called'))
    with EphemeralController(path, runtime_factory=Runtime) as controller:
        controller.execute('fixture')
    assert client.allocations == 0


def test_allowlist_rejects_credentials_even_if_caller_supplies_them(tmp_path):
    path = tmp_path / 'descriptor.json'
    with pytest.raises(ValueError):
        save_descriptor(path, {'schema': SCHEMA, 'name': 'fixture', 'phase': 'ready',
                               'endpoint': 'exact', 'token': SECRET})
    assert not path.exists()


def test_allocation_requires_explicit_approval_and_is_never_implicit(tmp_path):
    client, path = Client(), tmp_path / 'descriptor.json'
    with pytest.raises(ValueError):
        allocate(path, name='fixture', client=client, variant='GPU', accelerator='T4', shape=0)
    assert client.allocations == 0 and not path.exists()
    result = allocate(path, name='fixture', client=client, approved=True,
                      variant='GPU', accelerator='T4', shape=0)
    assert result['endpoint'] == 'new-exact' and client.allocations == 1
    assert SECRET not in path.read_text()
    with pytest.raises(FileExistsError):
        allocate(path, name='fixture', client=client, approved=True,
                 variant='GPU', accelerator='T4', shape=0)
    assert client.allocations == 1


def test_uncertain_allocation_preserves_nonsecret_reservation_without_retry(tmp_path):
    client, path = Client(), tmp_path / 'descriptor.json'

    def uncertain(*args, **kwargs):
        client.allocations += 1
        raise RuntimeError(SECRET)

    client.assign = uncertain
    with pytest.raises(RuntimeError) as result:
        allocate(path, name='fixture', client=client, approved=True,
                 variant='GPU', accelerator='T4', shape=0)
    assert SECRET not in str(result.value) and SECRET not in path.read_text()
    assert json.loads(path.read_text())['phase'] == 'allocation_requested'
    with pytest.raises(RuntimeError, match='uncertain'):
        EphemeralController(path, client=client)
    assert client.allocations == 1


def test_transfer_error_cleans_existing_connection_and_sanitizes(setup):
    path, client = setup

    class BrokenContents(Contents):
        def download(self, *args):
            raise RuntimeError(SECRET)

    with EphemeralController(path, client=client, runtime_factory=Runtime,
                             contents_factory=BrokenContents) as controller:
        controller.execute('fixture')
        with pytest.raises(RuntimeError) as error:
            controller.download('content/fixture', path.parent / 'download')
        assert SECRET not in str(error.value)
        assert controller.runtime is None
    assert len(Runtime.stopped) == 1
    assert client.allocations == 0


@pytest.mark.parametrize('remote', ['../content/file', 'content/../secret', '/etc/passwd',
                                   '//content/file', 'content//file', 'content/./file',
                                   'content/a\\b', 'content/%2e%2e/file', 'content/file?token=x'])
def test_remote_traversal_and_non_content_paths_rejected_before_requests(setup, remote):
    path, client = setup

    def forbidden(*args):
        pytest.fail('Unsafe transfer reached ContentsClient')

    with EphemeralController(path, client=client, contents_factory=forbidden) as controller:
        with pytest.raises(RuntimeError):
            controller.download(remote, path.parent / 'download')
        with pytest.raises(RuntimeError):
            controller.upload(path, remote)


@pytest.mark.parametrize('kind', ['file_symlink', 'parent_symlink', 'parent_traversal', 'directory'])
def test_local_unsafe_transfer_paths_are_rejected(setup, kind):
    path, client = setup
    target = path.parent / 'unsafe'
    if kind == 'file_symlink':
        target.symlink_to(path)
    elif kind == 'parent_symlink':
        target.symlink_to(path.parent, target_is_directory=True)
        target = target / 'file'
    elif kind == 'parent_traversal':
        target = path.parent / '..' / 'file'
    else:
        target.mkdir()
    before = path.read_bytes()
    with EphemeralController(path, client=client, contents_factory=lambda *args: pytest.fail('Unsafe path')) as controller:
        with pytest.raises(RuntimeError):
            controller.upload(target, 'content/file')
        with pytest.raises(RuntimeError):
            controller.download('content/file', target)
    assert path.read_bytes() == before


def test_upload_oversize_rejected_before_contents_call(setup):
    from ephemeral_colab import MAX_TRANSFER_BYTES
    path, client = setup
    source = path.parent / 'large'
    with source.open('wb') as handle:
        handle.truncate(MAX_TRANSFER_BYTES + 1)
    with EphemeralController(path, client=client, contents_factory=lambda *args: pytest.fail('Oversize upload')) as controller:
        with pytest.raises(RuntimeError):
            controller.upload(source, 'content/file')
    assert not list(path.parent.glob('.colab-upload-*'))


@pytest.mark.parametrize('size', [None, -1, 32 * 1024 * 1024 + 1, True])
def test_download_oversize_or_invalid_metadata_rejected_before_payload(setup, size):
    path, client = setup

    class Oversize(Contents):
        def list_dir(self, parent):
            return {'type': 'directory', 'content': [
                {'name': 'file', 'path': 'content/file', 'type': 'file', 'size': size}]}

        def download(self, *args):
            pytest.fail('Invalid metadata reached payload request')

    with EphemeralController(path, client=client, contents_factory=Oversize) as controller:
        with pytest.raises(RuntimeError):
            controller.download('content/file', path.parent / 'download')
    assert not (path.parent / 'download').exists()
    assert not list(path.parent.glob('.colab-download-*'))


def test_download_size_changed_does_not_replace_existing_destination(setup):
    path, client = setup
    destination = path.parent / 'download'
    destination.write_text('previous')

    class Changed(Contents):
        def download(self, remote, local):
            Path(local).write_text('changed size')

    with EphemeralController(path, client=client, contents_factory=Changed) as controller:
        with pytest.raises(RuntimeError):
            controller.download('content/file', destination)
    assert destination.read_text() == 'previous'
    assert not list(path.parent.glob('.colab-download-*'))


def test_vendor_exception_logging_is_suppressed_including_cleanup(setup):
    import io
    import logging
    path, client = setup
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logging.getLogger().addHandler(handler)

    class NoisyRuntime(Runtime):
        def execute_code(self, *args, **kwargs):
            logging.error('private execution %s', SECRET)
            raise RuntimeError(SECRET)

        def stop(self, shutdown_kernel):
            assert shutdown_kernel is False
            try:
                raise RuntimeError(SECRET)
            except RuntimeError:
                logging.exception('vendor stop failed')

    try:
        with EphemeralController(path, client=client, runtime_factory=NoisyRuntime) as controller:
            with pytest.raises(RuntimeError):
                controller.execute('failure')
        logging.error('ordinary logging restored')
    finally:
        logging.getLogger().removeHandler(handler)
    assert SECRET not in stream.getvalue()
    assert 'vendor stop failed' not in stream.getvalue()
    assert 'ordinary logging restored' in stream.getvalue()


def test_parallel_log_filter_does_not_restore_before_last_call_exits():
    import io
    import logging
    import threading
    from ephemeral_colab import private_vendor_logging
    stream, entered, release = io.StringIO(), threading.Event(), threading.Event()
    handler = logging.StreamHandler(stream)
    logging.getLogger().addHandler(handler)

    def overlapping():
        with private_vendor_logging():
            entered.set()
            assert release.wait(5)
            logging.error(SECRET)

    thread = threading.Thread(target=overlapping)
    try:
        with private_vendor_logging():
            thread.start()
            assert entered.wait(5)
        logging.error(SECRET)
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        logging.error('restored after overlap')
    finally:
        release.set()
        thread.join(5)
        logging.getLogger().removeHandler(handler)
    assert SECRET not in stream.getvalue()
    assert 'restored after overlap' in stream.getvalue()


@pytest.mark.parametrize('entries', [[],
    [{'name': 'file', 'path': 'elsewhere/file', 'type': 'file', 'size': 7}],
    [{'name': 'other', 'path': 'content/file', 'type': 'file', 'size': 7}],
    [{'name': 'file', 'path': 'content/file', 'type': 'file', 'size': 7}] * 2,
])
def test_directory_listing_requires_unique_exact_child(setup, entries):
    path, client = setup

    class Listed(Contents):
        def list_dir(self, parent):
            assert parent == 'content'
            return {'type': 'directory', 'content': entries}

        def download(self, *args):
            pytest.fail('Ambiguous listing reached payload')

    with EphemeralController(path, client=client, contents_factory=Listed) as controller:
        with pytest.raises(RuntimeError):
            controller.download('/content/file', path.parent / 'download')
    assert not (path.parent / 'download').exists()


@pytest.mark.parametrize('phase', ['reconnect', 'refresh', 'execute', 'download'])
@pytest.mark.parametrize('kind,retry,status', [
    ('timeout', True, None), ('connection', True, None), ('503', True, 503),
    ('403', False, 403), ('429', False, 429), ('integrity', False, None),
    ('certificate', False, None),
])
def test_safe_typed_errors_preserve_transience_and_numeric_diagnostics(setup, capsys, phase, kind, retry, status):
    import requests
    from colab_safe_cli import report_error, retryable_transport_error
    path, client = setup
    if kind.isdecimal():
        error = requests.HTTPError(SECRET)
        error.response = SimpleNamespace(status_code=int(kind), url=SECRET)
    else:
        error = {'timeout': requests.Timeout, 'connection': requests.ConnectionError,
                 'integrity': ValueError, 'certificate': requests.exceptions.SSLError}[kind](SECRET)

    def fail(*args, **kwargs):
        raise error

    class BrokenRuntime(Runtime):
        execute_code = fail

    class BrokenContents(Contents):
        download = fail

    expected = ConnectionError if retry else RuntimeError
    if phase == 'reconnect':
        client.list_assignments = fail
        with pytest.raises(expected) as result:
            EphemeralController(path, client=client)
    else:
        with EphemeralController(path, client=client, runtime_factory=BrokenRuntime,
                                 contents_factory=BrokenContents) as controller:
            with pytest.raises(expected) as result:
                if phase == 'execute':
                    controller.execute('fixture')
                elif phase == 'download':
                    controller.download('content/file', path.parent / 'download')
                else:
                    client.list_assignments = fail
                    controller.refresh()
            assert controller.runtime is None
    safe = result.value
    assert retryable_transport_error(safe) is retry
    assert safe.phase == phase and safe.http_status == status
    assert not hasattr(safe, 'response')
    assert SECRET not in str(safe)
    assert safe.__suppress_context__
    report_error('fixture', safe)
    diagnostic = capsys.readouterr().err
    assert SECRET not in diagnostic and f'phase={phase}' in diagnostic
    if status:
        assert f'HTTP {status}' in diagnostic
    assert SECRET not in path.read_text()
    assert client.allocations == 0
    assert not list(path.parent.glob('.colab-download-*'))


def test_real_ephemeral_adapter_retries_execute_through_watcher(setup):
    import requests
    from colab_sdk_watch import watch
    path, client = setup
    attempts, sleeps = [], []

    class FlakyRuntime(Runtime):
        def execute_code(self, code, timeout):
            attempts.append(code)
            if len(attempts) == 1:
                raise requests.ConnectionError(SECRET)
            return super().execute_code(code, timeout)

    with EphemeralController(path, client=client, runtime_factory=FlakyRuntime) as controller:
        assert watch(controller, project='/p', run_dir='/r', export_root='/e',
                     destination=path.parent / 'archive',
                     collect_fn=lambda adapter, *args: adapter.execute('fixture'),
                     status_fn=lambda *args: {'run_status': {'status': 'completed'}},
                     sleep_fn=sleeps.append, emit=lambda _: None) == 0
    assert sleeps == [10] and len(attempts) == 3
    assert client.allocations == 0
