"""Backend-independent Inchworm controller, following manuscript equations 4-21.

This is a validation implementation, not a claim of exact paper reproduction.
Indices in Python are zero-based; s is the number of confirmed tokens.
"""
from dataclasses import asdict, dataclass
from math import floor, isfinite


@dataclass(frozen=True)
class Config:
    max_window: int = 12
    reference_length: int = 16
    shift_quantile: float = 0.9
    positions: int = 4
    alternatives: int = 2
    survivors: int = 2
    mean_weight: float = 0.5
    warmup: int = 16
    max_backtracks: int | None = None

    def validate(self):
        for name in ("max_window", "reference_length", "positions", "alternatives", "survivors", "warmup"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.shift_quantile <= 1 or not 0 <= self.mean_weight <= 1:
            raise ValueError("quantile and mean_weight must be in [0, 1]")
        if self.survivors >= self.positions * self.alternatives:
            raise ValueError("survivors must be less than positions * alternatives")
        if self.max_backtracks is not None and self.max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative or None")


def quantile(values, q):
    """Linear interpolation, including zero-valued clipped entropy shifts."""
    if not values:
        raise ValueError("empty quantile")
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = floor(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (at - lo) * (values[hi] - values[lo])


def entropy_shifts(entropies, r):
    # No estimate at token zero; partial recent history before r tokens is an
    # explicit engineering convention. Recompute after accepting a revision.
    shifts = [0.0] * len(entropies)
    for t in range(1, len(entropies)):
        history = entropies[max(0, t - r):t]
        shifts[t] = entropies[t] - sum(history) / len(history)
    return shifts


def trigger(entropies, s, cfg):
    f = len(entropies)
    if not 0 < s < f:
        raise ValueError("trigger requires a nonempty confirmed prefix and window")
    shifts = entropy_shifts(entropies, cfg.reference_length)
    threshold = quantile([max(0.0, x) for x in shifts[:s]], cfg.shift_quantile)
    mean_history = sum(entropies[:s]) / s
    mean_window = sum(entropies[s:f]) / (f - s)
    mean_high = mean_window > mean_history
    shift_high = max(shifts[s:f]) > threshold
    at_max = f - s >= cfg.max_window
    return (mean_high or shift_high) if at_max else (mean_high and shift_high), {
        "mean_history": mean_history, "mean_window": mean_window,
        "shift_threshold": threshold, "max_shift": max(shifts[s:f]),
        "mean_high": mean_high, "shift_high": shift_high, "at_max": at_max,
    }


def generate(backend, initial_state, length, cfg, on_event=None):
    """Backend exposes predict, sample, alternatives, append and entropy.

    Prediction contains a read-only cache of the prefix BEFORE the predicted
    token. append(prediction, token) creates a pending-token state with no model
    call. This avoids lookahead work for candidates discarded in stage one.
    """
    cfg.validate()
    if length < 1:
        raise ValueError("length must be positive")
    tokens, entropies, records, events = [], [], [], []
    state = initial_state
    s = 0
    backtracks = 0

    def emit(event):
        events.append(event)
        if on_event:
            on_event(event)

    def checked_predict(current):
        prediction = backend.predict(current)
        if not isfinite(prediction.entropy) or prediction.entropy < 0:
            raise RuntimeError("Invalid predictive entropy")
        return prediction

    while len(tokens) < length:
        pred = checked_predict(state)
        token = backend.sample(pred)
        tokens.append(token)
        entropies.append(pred.entropy)
        state = backend.append(pred, token)
        f = len(tokens)
        if f % 96 == 0:
            emit({"kind": "progress", "f": f})

        if f <= cfg.warmup:
            s = f
            if f == min(cfg.warmup, length):
                emit({"kind": "warmup", "confirmed": f})
            continue
        if cfg.max_backtracks is not None and backtracks >= cfg.max_backtracks:
            records.clear()
            s = f
            continue
        records.append(pred)
        should_backtrack, diagnostics = trigger(entropies, s, cfg)
        if not should_backtrack:
            if f - s >= cfg.max_window or f == length:
                emit({"kind": "confirm", "s": s, "f": f, **diagnostics})
                s = f
                records.clear()
            continue

        original_tokens = tokens[s:f].copy()
        original_entropies = entropies[s:f].copy()
        # Each position is branched independently, never a Cartesian product.
        indices = sorted(range(s, f), key=lambda i: (-entropies[i], i))[:cfg.positions]
        survivors = []
        summaries = []
        candidate_id = 0
        for i in indices:
            original_pred = records[i - s]
            for alternative in backend.alternatives(original_pred, tokens[i], cfg.alternatives):
                candidate_tokens = tokens[s:i] + [alternative]
                # Forced alternatives share H_i; token surprisal is NOT H_i.
                candidate_entropies = entropies[s:i + 1].copy()
                candidate_state = backend.append(original_pred, alternative)
                for _ in range(i + 1, f):
                    p = checked_predict(candidate_state)
                    z = backend.sample(p)
                    candidate_tokens.append(z)
                    candidate_entropies.append(p.entropy)
                    candidate_state = backend.append(p, z)
                mean = sum(candidate_entropies) / (f - s)
                candidate = {
                    "id": candidate_id, "position": i + 1, "alternative": alternative,
                    "mean_entropy": mean, "tokens": candidate_tokens,
                    "entropies": candidate_entropies, "state": candidate_state,
                }
                summaries.append({k: candidate[k] for k in ("id", "position", "alternative", "mean_entropy")})
                survivors.append(candidate)
                survivors.sort(key=lambda c: (c["mean_entropy"], c["id"]))
                del survivors[cfg.survivors:]
                candidate_id += 1
        if not survivors:
            raise RuntimeError("No replacement candidates")

        final_window = f == length
        for candidate in survivors:
            if final_window:
                # No f+1 image token exists. End-of-image selection uses mean
                # entropy only, an explicitly documented boundary convention.
                candidate["lookahead_entropy"] = None
                candidate["score"] = candidate["mean_entropy"]
            else:
                lookahead = checked_predict(candidate["state"])
                candidate["lookahead_entropy"] = lookahead.entropy
                candidate["score"] = (cfg.mean_weight * candidate["mean_entropy"]
                                        + (1 - cfg.mean_weight) * lookahead.entropy)
                candidate["state"] = backend.ready(lookahead)
        winner = min(survivors, key=lambda c: (c["score"], c["id"]))
        tokens[s:f] = winner["tokens"]
        entropies[s:f] = winner["entropies"]
        state = winner["state"]
        backtracks += 1
        emit({
            "kind": "backtrack", "s": s, "f": f, **diagnostics,
            "original_tokens": original_tokens, "original_entropies": original_entropies,
            "selected_tokens": winner["tokens"].copy(),
            "candidates": summaries,
            "survivors": [{k: c[k] for k in ("id", "mean_entropy", "lookahead_entropy", "score")} for c in survivors],
            "selected_id": winner["id"], "terminal_mean_only": final_window,
        })
        s = f
        records.clear()
        survivors.clear()
        # Drop references to losing GPU caches before the next window.
        del candidate, candidate_state, winner, original_pred

    return {"tokens": tokens, "entropies": entropies, "events": events,
            "backtracks": backtracks, "config": asdict(cfg),
            "search_cap_reached": cfg.max_backtracks is not None and backtracks >= cfg.max_backtracks}
