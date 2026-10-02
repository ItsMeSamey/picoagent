"""Bounded external upload scheduling; optional real CPU checkpoint/resume proof."""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from picoagent.training.checkpointing import AsyncCheckpointSchedule
from picoagent.training.config import TrainingConfig
from picoagent.training.train import _output_budget_reservation


def test_timer_starts_at_training_and_coalesces_backlog_without_deleting(tmp_path):
    with patch('picoagent.training.checkpointing.time.monotonic', return_value=0):
        schedule = AsyncCheckpointSchedule(tmp_path, 1800, 2)
    with patch('picoagent.training.checkpointing.time.monotonic', return_value=1000):
        schedule.begin()
    with patch('picoagent.training.checkpointing.time.monotonic', return_value=2799):
        assert not schedule.should_save()
    with patch('picoagent.training.checkpointing.time.monotonic', return_value=2800):
        assert schedule.should_save()
        for step in [1, 2]:
            (tmp_path / f'checkpoint-{step}').mkdir()
        assert not schedule.should_save()
        assert schedule.status()['coalesced_checkpoint_requests'] == 1
        assert schedule.should_save(boundary=True)
        assert len(schedule.local_checkpoints()) == 2


def test_slots_reopen_across_four_verified_retention_cycles(tmp_path):
    schedule = AsyncCheckpointSchedule(tmp_path, 1800, 2)
    with patch.object(schedule.timer, 'due', return_value=True):
        for step in range(1, 6):
            assert schedule.should_save()
            (tmp_path / f'checkpoint-{step}').mkdir()
            schedule.saved(f'checkpoint-{step}')
            if step > 1:
                # Simulate controller retaining only its newly verified latest;
                # never discard anything from the actual run under test.
                (tmp_path / f'checkpoint-{step-1}').rename(tmp_path / f'retained-externally-{step-1}')
        assert schedule.status()['local_checkpoint_count'] == 1
        assert schedule.status()['last_sealed_checkpoint'] == 'checkpoint-5'


def test_reserves_intermediate_plus_final_plus_export_and_final_model(tmp_path):
    with patch('picoagent.training.train.shutil.disk_usage', return_value=SimpleNamespace(free=10**12)):
        result = _output_budget_reservation(tmp_path, parameters=1000, budget_bytes=10**12,
                                            checkpoint_copies=2, export_copies=1)
    assert result['projected_output_bytes'] == (3 * result['checkpoint_reserve_bytes']
        + result['final_model_reserve_bytes'] + result['safety_margin_bytes'])


@pytest.mark.parametrize('updates', [dict(async_checkpoint_max_local=True), dict(async_checkpoint_max_local=1),
    dict(async_checkpoint_upload=True), dict(async_checkpoint_upload=True, checkpoint_before_eval=True,
                                            checkpoint_interval_seconds=None)])
def test_async_config_rejects_unsafe_settings(updates):
    with pytest.raises(ValueError):
        TrainingConfig(model_id='org/model', model_revision='a'*40, dataset_manifest='x', output_dir='x', **updates)


def wall_clock_every_two_steps():
    original = AsyncCheckpointSchedule.should_save
    def scheduled(self, *, boundary=False):
        self._test_steps = getattr(self, '_test_steps', 0) + 1
        with patch.object(self.timer, 'due', return_value=self._test_steps % 2 == 0):
            return original(self, boundary=boundary)
    return patch.object(AsyncCheckpointSchedule, 'should_save', scheduled)


ml = pytest.mark.skipif(os.environ.get('PICOAGENT_RUN_ML_TESTS') != '1', reason='optional local CPU ML integration')


@ml
def test_real_training_continues_with_dead_uploader_and_forces_final(tmp_path):
    from picoagent.training.smoke import run_smoke
    from picoagent.training.provenance import verify_checkpoint
    from picoagent.training.data import sha256_file
    with wall_clock_every_two_steps(), patch('picoagent.training.durability.wait_for_durable_ack',
                                               side_effect=AssertionError('must never wait')):
        result = run_smoke(tmp_path / 'blocked', max_steps=8, train_records=8,
            gradient_accumulation_steps=1, save_steps=1, eval_steps=50, dropout=.2,
            checkpoint_before_eval=True, async_checkpoint_upload=True, checkpoint_interval_seconds=1800)
    run = tmp_path / 'blocked/run'
    assert result['status'] == 'completed' and result['global_step'] == 8
    assert sorted(p.name for p in run.glob('checkpoint-*')) == ['checkpoint-2', 'checkpoint-4', 'checkpoint-8']
    status = json.loads((run / 'checkpoint_upload_status.json').read_text())
    assert status['coalesced_checkpoint_requests'] == 1
    assert status['last_sealed_checkpoint'] == 'checkpoint-8'
    for step in [2, 4, 8]:
        verify_checkpoint(run / f'checkpoint-{step}', sha256_file(run / 'run_manifest.json'))


@ml
def test_real_sealed_snapshot_dropout_resume_state_matches_uninterrupted(tmp_path):
    from picoagent.training.smoke import run_smoke
    from picoagent.training.train import run_training
    from test_training_efficiency import assert_same_training
    options = dict(max_steps=4, train_records=5, gradient_accumulation_steps=1, save_steps=1,
                   eval_steps=50, dropout=.2, checkpoint_before_eval=True, async_checkpoint_upload=True,
                   checkpoint_interval_seconds=1800, async_checkpoint_max_local=2)
    with wall_clock_every_two_steps():
        reference = run_smoke(tmp_path / 'reference', **options)
        first = run_smoke(tmp_path / 'resumed', segment_steps=2, continue_through_checkpoints=True, **options)
        assert first['global_step'] == 2 and first['status'] == 'paused'
        config = TrainingConfig.load(tmp_path / 'resumed/smoke-config.json')
        resumed = run_training(config, resume_from_checkpoint=first['checkpoint'])
    assert resumed['status'] == 'completed'
    assert_same_training(reference, resumed, tmp_path / 'reference/run', tmp_path / 'resumed/run', 4)


@ml
def test_real_periodic_space_failure_coalesces_but_final_failure_is_explicit(tmp_path):
    from picoagent.training import smoke, train
    original_run = train.run_training
    original_reserve = train._output_budget_reservation
    calls = []
    def reserve(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise OSError('temporary export space pressure')
        with patch.object(train.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)):
            return original_reserve(*args, **kwargs)
    def bounded_run(config, **kwargs):
        return original_run(config, output_budget_bytes=10**12, **kwargs)
    with wall_clock_every_two_steps(), patch.object(smoke, 'run_training', bounded_run), \
         patch.object(train, '_output_budget_reservation', reserve):
        result = smoke.run_smoke(tmp_path / 'space', max_steps=4, train_records=5,
            gradient_accumulation_steps=1, save_steps=1, eval_steps=2,
            checkpoint_before_eval=True, async_checkpoint_upload=True, checkpoint_interval_seconds=1800)
    assert result['status'] == 'completed'
    run = tmp_path / 'space/run'
    assert [p.name for p in run.glob('checkpoint-*')] == ['checkpoint-4']
    status = json.loads((run / 'checkpoint_upload_status.json').read_text())
    assert status['coalesced_checkpoint_requests'] == 1
    assert status['last_space_error'] == 'temporary export space pressure'
    assert calls[0]['checkpoint_copies'] == 3 and calls[0]['export_copies'] == 3
    assert calls[1]['checkpoint_copies'] == 2 and calls[-1]['checkpoint_copies'] == 1
    calls.clear()
    with wall_clock_every_two_steps(), patch.object(smoke, 'run_training', bounded_run), \
         patch.object(train, '_output_budget_reservation', reserve), \
         pytest.raises(OSError, match='temporary export space pressure'):
        smoke.run_smoke(tmp_path / 'final-space', max_steps=2, train_records=3,
            gradient_accumulation_steps=1, save_steps=1, eval_steps=50,
            checkpoint_before_eval=True, async_checkpoint_upload=True, checkpoint_interval_seconds=1800)
    failed = json.loads((tmp_path / 'final-space/run/run_status.json').read_text())
    assert failed['status'] == 'failed' and failed['error_type'] == 'OSError'
