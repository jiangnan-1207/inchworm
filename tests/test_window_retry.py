import unittest
from dataclasses import dataclass

from inchworm_core import Config
from window_retry import generate, summarize


@dataclass(frozen=True)
class Prediction:
    prefix: tuple
    entropy: float


class Backend:
    def predict(self, prefix):
        length = len(prefix)
        if not length:
            entropy = 1.0
        elif length % 2 == 1:
            entropy = 0.5
        else:
            adjustment = (-1 if length % 4 == 0 else 1) if prefix[-1] == 1 else 0
            entropy = 4.0 + length + adjustment
        return Prediction(prefix, entropy)

    def append(self, prediction, token):
        return (*prediction.prefix, token)


class WindowRetryTests(unittest.TestCase):
    def test_all_twenty_windows_are_replayed_and_selected_cache_continues(self):
        for guard in (False, True):
            with self.subTest(guard=guard):
                initial = ()
                backend = Backend()
                original_prefixes, retry_prefixes = [], []
                events = []

                def first(prediction):
                    original_prefixes.append(prediction.prefix)
                    return 0

                def retry(prediction):
                    retry_prefixes.append(prediction.prefix)
                    return 1

                tokens, entropies = generate(
                    backend, initial, 41, first, retry, keep_lower_entropy=guard,
                    on_check=events.append, config=Config(warmup=1, max_window=2),
                )
                self.assertEqual(len(events), 20)
                self.assertEqual(len(original_prefixes), 41)
                self.assertEqual(len(retry_prefixes), 40)
                expected = [0]
                for index, check in enumerate(events):
                    slow, fast = 1 + 2 * index, 3 + 2 * index
                    self.assertEqual((check["slow"], check["fast"]), (slow, fast))
                    self.assertEqual(retry_prefixes[2 * index], tuple(expected))
                    self.assertEqual(retry_prefixes[2 * index + 1], (*expected, 1))
                    self.assertEqual(original_prefixes[slow], tuple(expected))
                    self.assertEqual(check["retry_tokens"], [1, 1])
                    self.assertEqual(check["retry_entropies"][0], check["original_entropies"][0])
                    take_retry = not guard or index % 2 == 1
                    expected.extend([int(take_retry)] * 2)
                    expected_entropy = check["retry_entropies"] if take_retry else check["original_entropies"]
                    self.assertEqual(entropies[slow:fast], expected_entropy)
                self.assertEqual(tokens, expected)
                summary = summarize(events)
                self.assertEqual(summary["resampled_tokens"], 40)
                self.assertEqual(summary["retry_selected"], 10 if guard else 20)
                self.assertEqual(summary["retry_lower"], 10)
                self.assertEqual(summary["retry_higher"], 10)

    def test_single_token_window_ties_preserve_original_but_still_draw_retry(self):
        class RisingEntropy(Backend):
            def predict(self, prefix):
                return Prediction(prefix, len(prefix) + 1.0)

        events = []
        tokens, _ = generate(
            RisingEntropy(), (), 3, lambda _: 0, lambda _: 1,
            keep_lower_entropy=True, on_check=events.append,
            config=Config(warmup=1, max_window=1),
        )
        self.assertEqual(tokens, [0, 0, 0])
        self.assertEqual([event["retry_tokens"] for event in events], [[1], [1]])
        self.assertEqual(summarize(events)["retry_equal"], 2)
        self.assertEqual(summarize(events)["retry_selected"], 0)
