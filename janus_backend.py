"""Janus adapter pinned to Transformers 4.38.2 legacy KV tuples."""
from dataclasses import dataclass
import torch


@dataclass
class Prediction:
    probabilities: object
    entropy: float
    cache: object


@dataclass
class State:
    cache: object = None
    pending: int | None = None
    ready_prediction: Prediction | None = None
    prompt_inputs: object = None


def legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if not isinstance(cache, tuple) or not cache:
        raise TypeError(f"Unsupported KV cache: {type(cache)}")
    if any(not isinstance(layer, tuple) or len(layer) != 2 for layer in cache):
        raise TypeError("Expected tuple of (key, value) pairs")
    return cache


def crop_cache(cache, prefix_length):
    """Read-only prefix views; never mutate saved KV tensors in place."""
    cache = legacy_cache(cache)
    if prefix_length < 1 or prefix_length > cache[0][0].shape[-2]:
        raise ValueError("Invalid cache prefix length")
    return tuple((key[..., :prefix_length, :], value[..., :prefix_length, :])
                 for key, value in cache)


class JanusBackend:
    def __init__(self, model, cfg_weight=5.0, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.device = next(model.parameters()).device
        self.forward_calls = 0
        self.prefill_positions_per_branch = 0
        self.image_forward_calls = 0
        self.sample_calls = 0

    @torch.inference_mode()
    def predict(self, state):
        if state.ready_prediction is not None:
            return state.ready_prediction
        if state.prompt_inputs is not None:
            inputs = state.prompt_inputs
            self.prefill_positions_per_branch += inputs.shape[1]
        else:
            paired = torch.tensor([state.pending, state.pending], dtype=torch.long, device=self.device)
            inputs = self.model.prepare_gen_img_embeds(paired).unsqueeze(1)
            self.image_forward_calls += 1
        output = self.model.language_model.model(
            inputs_embeds=inputs, past_key_values=state.cache, use_cache=True,
        )
        self.forward_calls += 1
        cache = legacy_cache(output.past_key_values)
        logits = self.model.gen_head(output.last_hidden_state[:, -1, :])
        if logits.shape[0] != 2:
            raise ValueError("This validation adapter supports exactly one CFG pair")
        # Preserve original BF16 sampling path. Use FP32 only to measure the
        # normalized distribution actually passed to multinomial.
        guided = logits[1:2] + self.cfg_weight * (logits[0:1] - logits[1:2])
        probabilities = torch.softmax(guided / self.temperature, dim=-1)
        p = probabilities.float()
        p = p / p.sum(dim=-1, keepdim=True)
        entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum().item()
        return Prediction(probabilities, entropy, cache)

    def sample(self, prediction):
        self.sample_calls += 1
        return int(torch.multinomial(prediction.probabilities, num_samples=1).item())

    def alternatives(self, prediction, original, k):
        scores = prediction.probabilities[0].float().clone()
        scores[original] = -torch.inf
        count = min(k, scores.numel() - 1)
        return torch.topk(scores, count).indices.tolist()

    def append(self, prediction, token):
        return State(cache=prediction.cache, pending=int(token))

    def ready(self, prediction):
        return State(ready_prediction=prediction)

    def counters(self):
        return {k: getattr(self, k) for k in (
            "forward_calls", "image_forward_calls", "sample_calls", "prefill_positions_per_branch")}
