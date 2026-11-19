"""Regenerate one whole window per Inchworm trigger, with or without a guard."""

from math import isfinite
from statistics import mean

from inchworm_core import Config, trigger


def generate(backend, initial_state, length, sample_first, sample_retry, *,
             keep_lower_entropy, on_check, config=Config()):
    config.validate()
    if length < 1 or config.max_backtracks is not None:
        raise ValueError("Expected a positive length and no backtrack cap")

    def predict(state):
        prediction = backend.predict(state)
        if not isfinite(prediction.entropy) or prediction.entropy < 0:
            raise ValueError("Invalid predictive entropy")
        return prediction

    tokens, entropies = [], []
    state = initial_state
    slow = 0
    check_number = 0
    first_prediction = None
    while len(tokens) < length:
        prediction = predict(state)
        if len(tokens) == slow:
            first_prediction = prediction
        token = sample_first(prediction)
        tokens.append(token)
        entropies.append(prediction.entropy)
        state = backend.append(prediction, token)
        fast = len(tokens)
        if fast <= config.warmup:
            slow = fast
            continue

        should_retry, diagnostics = trigger(entropies, slow, config)
        if should_retry:
            original_tokens = tokens[slow:fast]
            original_entropies = entropies[slow:fast]
            retry_tokens, retry_entropies = [], []
            # This prediction was made from exactly the prefix at the slow
            # pointer. Reuse it, but draw a fresh token even at the first site.
            retry_prediction = first_prediction
            for index in range(slow, fast):
                retry_token = sample_retry(retry_prediction)
                retry_tokens.append(retry_token)
                retry_entropies.append(retry_prediction.entropy)
                retry_state = backend.append(retry_prediction, retry_token)
                if index + 1 < fast:
                    retry_prediction = predict(retry_state)

            original_mean = mean(original_entropies)
            retry_mean = mean(retry_entropies)
            take_retry = not keep_lower_entropy or retry_mean < original_mean
            retry_would_trigger, _ = trigger(entropies[:slow] + retry_entropies, slow, config)
            if take_retry:
                tokens[slow:fast] = retry_tokens
                entropies[slow:fast] = retry_entropies
                state = retry_state
            check_number += 1
            on_check({
                "check": check_number,
                "slow": slow,
                "fast": fast,
                "window_length": fast - slow,
                "trigger": diagnostics,
                "original_tokens": original_tokens,
                "retry_tokens": retry_tokens,
                "original_entropies": original_entropies,
                "retry_entropies": retry_entropies,
                "original_mean_entropy": original_mean,
                "retry_mean_entropy": retry_mean,
                "delta_original_minus_retry": original_mean - retry_mean,
                "retry_would_trigger_at_same_boundary": retry_would_trigger,
                "changed_tokens": sum(a != b for a, b in zip(original_tokens, retry_tokens)),
                "selected": "retry" if take_retry else "original",
                "selected_mean_entropy": retry_mean if take_retry else original_mean,
            })
            slow = fast
            # Keep only the chosen state for the following window.
            del retry_state, retry_prediction
        elif fast - slow >= config.max_window or fast == length:
            slow = fast

    return tokens, entropies


def summarize(events):
    def counts(rows):
        lower = sum(r["delta_original_minus_retry"] > 0 for r in rows)
        equal = sum(r["delta_original_minus_retry"] == 0 for r in rows)
        selected = sum(r["selected"] == "retry" for r in rows)
        return {
            "checks": len(rows),
            "retry_lower": lower,
            "retry_equal": equal,
            "retry_higher": len(rows) - lower - equal,
            "retry_selected": selected,
            "original_selected": len(rows) - selected,
            "retry_selected_with_token_changes": sum(
                r["selected"] == "retry" and r["changed_tokens"] > 0 for r in rows
            ),
            "retry_still_triggers_at_same_boundary": sum(
                r["retry_would_trigger_at_same_boundary"] for r in rows
            ),
            "mean_delta": mean(r["delta_original_minus_retry"] for r in rows) if rows else None,
        }

    return {
        **counts(events),
        "resampled_tokens": sum(r["window_length"] for r in events),
        "single_token_windows": sum(r["window_length"] == 1 for r in events),
        "multi_token_windows": counts([r for r in events if r["window_length"] > 1]),
    }
