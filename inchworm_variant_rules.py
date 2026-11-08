"""Confirmed building blocks for a separate Inchworm variant.

No replacement sampling or acceptance policy is selected by this module.
The original inchworm_core.py and model adapters remain independent of it.
"""
from dataclasses import dataclass
from math import exp, isfinite, log


@dataclass(frozen=True)
class TemperatureConfig:
    # Initial engineering settings, not validated optimum or paper parameters.
    minimum: float = 0.5
    maximum: float = 1.0
    entropy_scale: float = 2.0  # nats

    def validate(self):
        values=(self.minimum,self.maximum,self.entropy_scale)
        if not all(isfinite(x) for x in values):
            raise ValueError('Temperature parameters must be finite')
        if not 0 < self.minimum <= self.maximum <= 1:
            raise ValueError('Require 0 < Tmin <= Tmax <= 1 for sharpening')
        if self.entropy_scale <= 0:
            raise ValueError('Entropy scale must be positive')


def adaptive_temperature(entropy, config=TemperatureConfig()):
    config.validate()
    if not isfinite(entropy) or entropy < 0:
        raise ValueError('Entropy must be finite and nonnegative, in nats')
    return config.minimum+(config.maximum-config.minimum)*exp(-entropy/config.entropy_scale)


def temperature_probabilities(probabilities, temperature):
    """Standard-library reference q(z) ∝ p(z)^(1/T), before any exclusions.

    The input is the unmodified base sampling distribution. A zero-probability
    token remains outside its support. No candidate exclusion or draw occurs.
    """
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    probabilities=list(probabilities)
    if not probabilities or any(not isfinite(p) or p < 0 for p in probabilities):
        raise ValueError('Invalid probability weights')
    positive=[log(p)/temperature for p in probabilities if p > 0]
    if not positive:
        raise ValueError('Distribution has empty support')
    largest=max(positive)
    weights=[exp(log(p)/temperature-largest) if p > 0 else 0.0 for p in probabilities]
    total=sum(weights)
    return [w/total for w in weights]


def probability_gap(max_probability, original_probability):
    """G = log(p_max) - log(p_original), in nats, before adaptive temperature."""
    if not (isfinite(max_probability) and isfinite(original_probability)
            and 0 < original_probability <= max_probability):
        raise ValueError('Require finite 0 < p_original <= p_max')
    return log(max_probability)-log(original_probability)


def mixed_positions(entropies, gaps):
    """Exactly the confirmed M=4 rule, with stable chronological tie breaks.

    Return local window offsets: first two by H, then two remaining by G.
    Short windows select every available position; no duplicate positions.
    """
    entropies=list(entropies); gaps=list(gaps)
    if len(entropies)!=len(gaps):
        raise ValueError('Entropy and gap lengths differ')
    if any(not isfinite(x) or x < 0 for x in entropies+gaps):
        raise ValueError('Entropy and gap must be finite and nonnegative')
    high=sorted(range(len(entropies)),key=lambda i:(-entropies[i],i))[:2]
    selected=set(high)
    suspicious=sorted((i for i in range(len(gaps)) if i not in selected),key=lambda i:(-gaps[i],i))[:2]
    return high+suspicious
