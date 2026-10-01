import unittest

from picoagent.training.checkpointing import CheckpointTimer


class CheckpointTimerTests(unittest.TestCase):
    def test_deadline_and_successful_save_reset(self):
        timer = CheckpointTimer(300, now=10)
        self.assertFalse(timer.due(now=309))
        self.assertTrue(timer.due(now=310))
        timer.mark_saved(now=325)
        self.assertFalse(timer.due(now=624))
        self.assertTrue(timer.due(now=625))

    def test_disabled_timer(self):
        self.assertFalse(CheckpointTimer(None, now=0).due(now=10**9))

    def test_long_step_is_due_after_it_finishes(self):
        self.assertTrue(CheckpointTimer(300, now=0).due(now=1800))

    def test_clock_regression_does_not_force_save(self):
        self.assertFalse(CheckpointTimer(300, now=100).due(now=50))

    def test_bad_intervals(self):
        for interval in (0, -1, float("inf"), float("nan"), True, "300"):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                CheckpointTimer(interval)
