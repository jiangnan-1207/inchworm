# Third-party notices

This repository adapts the official Janus image-token sampling loop and
LlamaGen sampling/model-loading conventions. It calls the upstream GenEval
evaluator. The notices below are retained for those components; no project-wide
license has yet been selected for newly written project code.

| Upstream | Pinned source | License text |
| --- | --- | --- |
| DeepSeek Janus, Copyright (c) 2023 DeepSeek | [source](https://github.com/deepseek-ai/Janus/tree/1daa72fa409002d40931bd7b36a9280362469ead) | [MIT](licenses/Janus.txt) |
| LlamaGen, Copyright (c) 2024 FoundationVision | [source](https://github.com/FoundationVision/LlamaGen/tree/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec) | [MIT](licenses/LlamaGen.txt) |
| GenEval, Copyright (c) 2023 Dhruba Ghosh | [source](https://github.com/djghosh13/geneval/tree/af4902f24d3ca90ebbb446dd9891a59e0f82725f) | [MIT](licenses/GenEval.txt) |
| ModelScope mPLUG, Copyright (c) Alibaba, Inc. and its affiliates | [preprocessor](https://github.com/modelscope/modelscope/blob/v1.9.5/modelscope/preprocessors/multi_modal.py) / [wrapper](https://github.com/modelscope/modelscope/blob/v1.9.5/modelscope/models/multi_modal/mplug_for_all_tasks.py) | [Apache-2.0](licenses/ModelScope.txt) |
| MMDetection | [source and Apache-2.0 license](https://github.com/open-mmlab/mmdetection/tree/e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1) | Remains in fetched source |

The bootstrap script fetches separate upstream repositories including their
license files; it does not vendor their full trees here. Model weights and
other dependencies are downloaded separately and retain their own terms.
Publication of this code does not publish the manuscript or model weights.

The optional Lumina adapter calls [Alpha-VLLM/Lumina-mGPT-2.0](https://github.com/Alpha-VLLM/Lumina-mGPT-2.0/tree/0a54b9e0f1bf4ea3577e8a3b1b27b652fbc1e6cc), which is Apache-2.0 licensed. Its model/processor/decoder code and license remain in the separately fetched upstream checkout; the adapter does not vendor them. [MoVQGAN](https://huggingface.co/ai-forever/MoVQGAN) weights are downloaded separately with their upstream terms. The new persistent cache uses the public Hugging Face Cache interface; it is an independent implementation, not a modification of model weights.
