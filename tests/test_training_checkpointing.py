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


class EvaluationRNGIsolationTests(unittest.TestCase):
    def test_rng_restored_even_on_evaluation_failure(self):
        import pickle
        import random
        from pathlib import Path
        from picoagent.training.train import _evaluate_preserving_rng

        class Trainer:
            def _save_rng_state(self, directory):
                (Path(directory) / "rng").write_bytes(pickle.dumps(random.getstate()))

            def _load_rng_state(self, directory):
                random.setstate(pickle.loads((Path(directory) / "rng").read_bytes()))

        random.seed(772)
        original = random.getstate()

        def evaluate():
            random.random()
            return {"eval_loss": 1.0}

        self.assertEqual(_evaluate_preserving_rng(Trainer(), evaluate), {"eval_loss": 1.0})
        self.assertEqual(original, random.getstate())

        def failed_evaluate():
            random.random()
            raise RuntimeError("evaluation interrupted")

        with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
            _evaluate_preserving_rng(Trainer(), failed_evaluate)
        self.assertEqual(original, random.getstate())
