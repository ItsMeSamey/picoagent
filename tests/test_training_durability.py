"""No accelerator/network: fail-closed durability rendezvous and process restart."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from picoagent.training.data import sha256_file
from picoagent.training.durability import acknowledgement_from_receipt, wait_for_durable_ack
from picoagent.training.provenance import checkpoint_evidence, write_json


@pytest.fixture
def sealed(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run_manifest.json').write_text('{}')
    checkpoint = run / 'checkpoint-7'
    checkpoint.mkdir()
    for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        (checkpoint / name).write_bytes(b'fixture')
    (checkpoint / 'trainer_state.json').write_text('{"global_step":7}')
    checkpoint_evidence(checkpoint, sha256_file(run / 'run_manifest.json'))
    receipt = {'schema': 'picoagent.github-release-receipt.v1', 'published': True,
               'independent_readback_verified': True, 'release_id': 1, 'tag': 'fixture',
               'plan_sha256': 'a' * 64,
               'identity': {'visibility': 'public', 'repository': 'owner/repo',
                            'checkpoint': checkpoint.name,
                            'run_manifest_sha256': sha256_file(run / 'run_manifest.json'),
                            'checkpoint_manifest_sha256': sha256_file(checkpoint / 'checkpoint_manifest.json'),
                            'transfer_manifest_sha256': 'b' * 64, 'source_tree_sha256': 'c' * 64},
               'assets': {'transfer_manifest.json': {'id': 1, 'sha256': 'b' * 64, 'bytes': 10}}}
    return run, checkpoint, receipt


def publish(run, receipt):
    directory = run / 'durability'
    directory.mkdir(exist_ok=True)
    write_json(directory / 'checkpoint-7.json', acknowledgement_from_receipt(receipt), exclusive=True)


def test_controller_death_times_out_without_altering_checkpoint(sealed):
    run, checkpoint, _ = sealed
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    with pytest.raises(TimeoutError, match='training stopped'):
        wait_for_durable_ack(run, checkpoint, .01)
    assert before == {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    assert json.loads((run / 'durability/status.json').read_text())['status'] == 'timed_out'


@pytest.mark.parametrize('field', ['published', 'independent_readback_verified'])
def test_unverified_receipt_rejected(sealed, field):
    receipt = sealed[2]
    receipt[field] = False
    with pytest.raises(ValueError, match='verified published'):
        acknowledgement_from_receipt(receipt)


@pytest.mark.parametrize('field,value', [('checkpoint', 'checkpoint-8'),
                                        ('run_manifest_sha256', '0' * 64),
                                        ('checkpoint_manifest_sha256', '0' * 64)])
def test_stale_wrong_ack_fails_closed(sealed, field, value):
    run, checkpoint, receipt = sealed
    receipt['identity'][field] = value
    publish(run, receipt)
    with pytest.raises(ValueError, match='another run or checkpoint'):
        wait_for_durable_ack(run, checkpoint, .01)


def test_partial_or_tampered_ack_fails_closed(sealed):
    run, checkpoint, receipt = sealed
    publish(run, receipt)
    path = run / 'durability/checkpoint-7.json'
    ack = json.loads(path.read_text())
    ack['receipt_sha256'] = 'f' * 64
    path.write_text(json.dumps(ack))
    with pytest.raises(ValueError, match='integrity mismatch'):
        wait_for_durable_ack(run, checkpoint, .01)
    path.write_text('{')
    with pytest.raises(ValueError):
        wait_for_durable_ack(run, checkpoint, .01)


def test_fresh_process_after_timeout_requires_then_accepts_persisted_ack(sealed):
    run, checkpoint, receipt = sealed
    code = ('from pathlib import Path; from picoagent.training.durability import wait_for_durable_ack; '
            f'wait_for_durable_ack(Path({str(run)!r}), Path({str(checkpoint)!r}), .02)')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    failed = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True)
    assert failed.returncode != 0 and b'TimeoutError' in failed.stderr
    publish(run, receipt)
    success = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True)
    assert success.returncode == 0, success.stderr.decode()
    assert json.loads((run / 'durability/status.json').read_text())['status'] == 'verified'


def test_symlink_ack_rejected(sealed, tmp_path):
    run, checkpoint, receipt = sealed
    target = tmp_path / 'ack.json'
    target.write_text(json.dumps(acknowledgement_from_receipt(deepcopy(receipt))))
    (run / 'durability').mkdir()
    (run / 'durability/checkpoint-7.json').symlink_to(target)
    with pytest.raises(ValueError, match='Unsafe'):
        wait_for_durable_ack(run, checkpoint, .01)
