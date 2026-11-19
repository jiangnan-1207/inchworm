"""Run both single-image whole-window retry experiments on Janus-Pro-1B."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from inchworm_core import Config
from protocol import atomic_json, sha
from window_retry import generate, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt.strip() or not 0 <= args.seed < 2**63:
        parser.error("Supply a nonempty prompt and a seed in [0, 2**63)")

    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModelForCausalLM
    from janus.models import VLChatProcessor
    from janus_backend import JanusBackend, State

    if not torch.cuda.is_available():
        raise RuntimeError("Use a Slurm job with an allocated GPU")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    directory = args.data_root.resolve() / "models" / "Janus-Pro-1B"
    processor = VLChatProcessor.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(directory), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).cuda().eval()

    config = Config()
    retry_seed = args.seed + 2**63
    length = 576
    source = Path(__file__).resolve().parent
    specification = {
        "model": "Janus-Pro-1B",
        "model_directory": str(directory),
        "prompt": args.prompt,
        "first_pass_seed": args.seed,
        "retry_seed": retry_seed,
        "image_tokens": length,
        "trigger": {name: asdict(config)[name] for name in (
            "warmup", "max_window", "reference_length", "shift_quantile"
        )},
        "check_cap": None,
        "selection_score": "window_mean_entropy_nats",
        "cfg_scale": 5.0,
        "base_temperature": 1.0,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {name: sha(source / name) for name in (
            "inchworm_core.py", "janus_backend.py", "window_retry.py", "run_window_retry.py"
        )},
    }
    atomic_json(out / "run.json", specification)

    conversation = [
        {"role": "<|User|>", "content": args.prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prompt = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=processor.sft_format, system_prompt="",
    ) + processor.image_start_tag
    ids = processor.tokenizer.encode(prompt)
    input_ids = torch.tensor([ids, ids], device="cuda")
    input_ids[1, 1:-1] = processor.pad_id

    results = {}
    with torch.inference_mode():
        initial = State(prompt_inputs=model.language_model.get_input_embeddings()(input_ids))
        for name, guard in (("retry_always", False), ("retry_keep_lower", True)):
            backend = JanusBackend(model)
            first_rng = torch.Generator(device=backend.device).manual_seed(args.seed)
            retry_rng = torch.Generator(device=backend.device).manual_seed(retry_seed)

            def draw(prediction, rng):
                return int(torch.multinomial(
                    prediction.probabilities, num_samples=1, generator=rng
                ).item())

            events = []
            started = perf_counter()
            with (out / f"{name}.checks.jsonl").open("x", encoding="utf-8") as log:
                def record(event):
                    events.append(event)
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                    log.flush()
                    print(
                        f"{name} check={event['check']} window=({event['slow']},{event['fast']}] "
                        f"original={event['original_mean_entropy']:.6f} "
                        f"retry={event['retry_mean_entropy']:.6f} selected={event['selected']}",
                        flush=True,
                    )

                tokens, entropies = generate(
                    backend, initial, length, lambda p: draw(p, first_rng),
                    lambda p: draw(p, retry_rng), keep_lower_entropy=guard,
                    on_check=record, config=config,
                )
            generation_seconds = perf_counter() - started
            codes = torch.tensor([tokens], device=backend.device, dtype=torch.int)
            decoded = model.gen_vision_model.decode_code(codes, shape=[1, 8, 24, 24])
            pixels = decoded.float().cpu().numpy().transpose(0, 2, 3, 1)[0]
            image = Image.fromarray(np.clip((pixels + 1) * 127.5, 0, 255).astype(np.uint8))
            image_path = out / f"{name}.png"
            image.save(image_path)
            atomic_json(out / f"{name}.tokens.json", {"tokens": tokens, "entropies": entropies})
            stats = summarize(events)
            # Every retry draw after its first one needs an additional forward.
            expected_forwards = length + stats["resampled_tokens"] - stats["checks"]
            if backend.forward_calls != expected_forwards:
                raise RuntimeError("Unexpected number of whole-window forwards")
            results[name] = {
                **stats,
                "final_image_mean_entropy": mean(entropies),
                "generation_seconds": generation_seconds,
                "model_forward_calls": backend.forward_calls,
                "total_token_draws": length + stats["resampled_tokens"],
                "image_sha256": sha(image_path),
            }
    atomic_json(out / "summary.json", {"status": "complete", **results})
    print("WINDOW_RETRY_DONE", out, json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
