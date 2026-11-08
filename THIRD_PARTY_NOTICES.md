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
| MMDetection | [source and Apache-2.0 license](https://github.com/open-mmlab/mmdetection/tree/e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1) | Remains in fetched source |

The bootstrap script fetches separate upstream repositories including their
license files; it does not vendor their full trees here. Model weights and
other dependencies are downloaded separately and retain their own terms.
Publication of this code does not publish the manuscript or model weights.
