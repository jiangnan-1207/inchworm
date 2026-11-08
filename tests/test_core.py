import unittest
from dataclasses import dataclass
from inchworm_core import Config, entropy_shifts, generate, quantile, trigger


@dataclass
class Pred:
    prefix: tuple
    entropy: float


class FakeBackend:
    """Full logical prefixes act as immutable caches for controller tests."""
    def __init__(self):
        self.calls = []
        self.lookahead_prefixes = []

    def predict(self, state):
        if isinstance(state, Pred):
            return state
        self.calls.append(state)
        t = len(state)
        entropy = 1.0 if t < 4 else (0.8 if t < 6 else 4.0)
        if any(x == 1 for x in state):
            entropy = 0.3
        elif any(x == 2 for x in state):
            entropy = 0.4
        return Pred(state, entropy)

    def sample(self, pred):
        return 0

    def alternatives(self, pred, original, k):
        return [x for x in (1, 2, 3) if x != original][:k]

    def append(self, pred, token):
        return pred.prefix + (token,)

    def ready(self, pred):
        return pred


class CoreTests(unittest.TestCase):
    def test_quantile_includes_zero_shifts(self):
        self.assertEqual(quantile([0, 0, 2, 4], 0.5), 1)

    def test_shift_uses_recent_history(self):
        self.assertEqual(entropy_shifts([1, 1, 4, 2], 2), [0, 0, 3, -0.5])

    def test_and_before_max_or_at_max(self):
        cfg = Config(max_window=3)
        # Window mean is high, but no local positive shift exceeds the historic spike.
        entropy = [0, 10, 5, 6]
        yes, diag = trigger(entropy, 3, cfg)
        self.assertTrue(diag["mean_high"])
        self.assertFalse(diag["shift_high"])
        self.assertFalse(yes)
        yes, _ = trigger(entropy, 3, Config(max_window=1))
        self.assertTrue(yes)

    def test_no_backtracking_matches_forward_only_calls(self):
        backend = FakeBackend()
        result = generate(backend, (), 12, Config(warmup=4, max_backtracks=0))
        self.assertEqual(result["tokens"], [0] * 12)
        self.assertEqual(len(backend.calls), 12)
        self.assertEqual(result["backtracks"], 0)

    def test_branches_suffix_recomputed_and_winner_continued(self):
        backend = FakeBackend()
        cfg = Config(warmup=4, max_window=3, max_backtracks=1)
        result = generate(backend, (), 12, cfg)
        self.assertEqual(result["backtracks"], 1)
        event = next(e for e in result["events"] if e["kind"] == "backtrack")
        self.assertEqual((event["s"], event["f"]), (4, 7))
        # Only three positions exist in this window, yielding 3*2 candidates.
        self.assertEqual(len(event["candidates"]), 6)
        self.assertEqual(len(event["survivors"]), 2)
        self.assertEqual(result["tokens"][:4], [0] * 4)
        self.assertEqual(sum(x != 0 for x in event["selected_tokens"]), 1)
        # Every subsequent generation step must use the accepted revised prefix.
        last_prediction_prefix = backend.calls[-1]
        self.assertEqual(list(last_prediction_prefix), result["tokens"][:-1])
        # Accepted branch entropy replaces original history, not only token IDs.
        self.assertTrue(any(x < 0.8 for x in result["entropies"][4:7]))

    def test_final_window_has_no_out_of_image_lookahead(self):
        backend = FakeBackend()
        result = generate(backend, (), 7, Config(warmup=4, max_window=3))
        event = next(e for e in result["events"] if e["kind"] == "backtrack")
        self.assertTrue(event["terminal_mean_only"])
        self.assertTrue(all(c["lookahead_entropy"] is None for c in event["survivors"]))
        self.assertLess(max(map(len, backend.calls)), 7)

    def test_warmup_can_cover_short_sequence(self):
        result = generate(FakeBackend(), (), 3, Config())
        self.assertEqual(len(result["tokens"]), 3)
        self.assertEqual(result["backtracks"], 0)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            Config(mean_weight=2).validate()
        with self.assertRaises(ValueError):
            Config(warmup=0).validate()


if __name__ == "__main__":
    unittest.main()
