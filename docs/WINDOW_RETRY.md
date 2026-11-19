# Janus-Pro-1B 单图整窗重生成

同一个提示词和种子运行两条生成轨迹，各产出一张完整图。每条轨迹都沿用 Inchworm 的触发规则。触发一次，就从慢指针后第一个 token 开始，按 Janus 基础采样重抽到快指针。每次都完整重生成窗口一次，之后确认该窗口并继续生成，直到完成 576 个图像 token。

- `retry_always` 每次继续使用重生成的窗口，同时记录首次生成和重生成的熵。
- `retry_keep_lower` 每次比较两条路径的窗口平均预测熵，重生成的平均熵严格更低才采用，否则保留原路径。

两条轨迹开头使用相同的随机种子，首次生成与重生成使用各自的随机数流。只要决策不同，后续的上下文和回溯窗口就可能不同。两条轨迹上的同编号检查仅在窗口和前缀相同的情况下才对应同一个生成状态。每次检查内部，两条窗口路径都有共同的已确认前缀。评分采用基础采样分布的窗口平均熵，单位为 nats。这里的比较只需要一次重生成，不加候选选点、温度调整、候选筛选或前瞻熵。

在已经配好 Janus-Pro-1B 的 Slurm 服务器上，从仓库根目录执行：

```bash
export CONFIG_FILE="$PWD/configs/local.sh"
export WINDOW_RETRY_ROOT="$PWD/../janus_window_retry_results"
bash scripts/submit_window_retry.sh --submit 'a photo of three red apples on a wooden table' 42
```

launcher 会打印作业 ID 和结果目录。配置文件使用现有的模型目录、Python 环境和 Slurm 账户。提交申请一张 GPU，也可将 `--submit` 换成 `--check-only` 预检。已有 GPU 分配时可以直接运行 `run_window_retry.py --data-root ... --out ... --prompt ... --seed 42`，其中 `--out` 为新目录。

- `summary.json` 含全部检查数、重生成后熵降低／相等／升高的次数、最终用了哪条路径，以及只统计多 token 窗口时的同类结果。
- `*.checks.jsonl` 每次检查一行，含窗口范围 `(slow, fast]`、两条 token 路径、每位置的熵、窗口平均熵和最终选择。正的 `delta_original_minus_retry` 表示重生成后降熵。
- `retry_always.png` 和 `retry_keep_lower.png` 是完整图片。`*.tokens.json` 是最终生成序列及其预测熵。

第一组的 `retry_lower` 和 `retry_higher` 可以回答触发后重抽一次是否经常降熵。`retry_still_triggers_at_same_boundary` 是在同一前缀和边界下给重生成结果计算的触发诊断，不引发第二次重抽。第二组的选中熵按规则不会高于原路径，这不能单独验证熵能判断图片质量。对于单 token 窗口，两次采样使用相同的预测分布，因此即使 token 不同，窗口熵也必然相等。解释结果时一并查看 `multi_token_windows`，并对照两张最终图片与提示词。
