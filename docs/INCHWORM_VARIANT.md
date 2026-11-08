# 独立新方法：inchworm_variant

状态：已实现用户确认的规则，单独入口和结果目录；模型 GPU 数值及图像效果尚待实际运行。不替换原 Inchworm，也不预填效果结论。

## 已确认的算法

1. **混合位置**：M=4，先取窗口内预测熵 H 最大的2个位置，再从剩余位置取 G 最大的2个位置。`G = log(p_max) - log(p_original)`。不足4个位置时全部选取，并列按位置先后。H 和 G 均来自调温前原采样分布。
2. **温度提议**：每个位置独立抽 k=2 次，**允许抽到原 token，也允许重复**。每次抽样都独立重生成到原窗口末尾；后缀仍用原始采样策略，不施加自适应温度，也不擅自去重。
3. **保留两阶段筛选和原路径保底**：最多8条候选先按窗口平均预测熵取 N=2 条，再算加权完整评分。原窗口连同可恢复的 KV 状态额外保留，不占这2条名额。只有最佳候选严格降分才接受，即 `S_candidate < S_original - delta`，初始 `delta=0`。同分或更差均保留原路径。

触发检查、warmup、窗口阈值等与原工程设置一致。最后一个图像窗口沿用原来的均值熵评分，不做超出图像末尾的 lookahead。

## 温度公式与初值

熵使用自然对数，单位 nats。令 p 为当前实现原本用于采样的分布：Janus 为 CFG 与原温度处理后；LlamaGen 为 CFG、原温度及既定 top-k 处理后。保持该分布的零概率支持集，不重新引入被过滤 token。

```text
T_i = 0.5 + 0.5 * exp(-H_i / 2.0)
q_i(z) = exp(log(p_i(z)) / T_i) / sum_v exp(log(p_i(v)) / T_i)
```

| H（nats） | T |
| --- | --- |
| 0 | 1.000 |
| 1 | 0.803 |
| 2 | 0.684 |
| 4 | 0.568 |
| 8 | 0.509 |

高熵得到较低温度；对非均匀分布，当 `0<T<1` 时，高概率 token 相对低概率 token 的优势扩大：`q(a)/q(b) = [p(a)/p(b)]^(1/T)`。不会改变概率排序；完全均匀的分布仍均匀。

这是标准温度变换，可参考 [Hugging Face TemperatureLogitsWarper](https://huggingface.co/docs/transformers/v4.39.1/internal/generation_utils#transformers.TemperatureLogitsWarper)。有放回抽样使用 [PyTorch multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) 的 `replacement=True`。

**Tmin=0.5、Tmax=1、h0=2 是可调整的工程初值，不是已验证最优参数或论文参数。** 降低温度不保证提升图像质量，不能把它当作实验结论。

提议温度只影响分支抽样；窗口 H、G、候选后缀熵和 lookahead 熵全部取原分布。这样不会因为人为把提议分布压尖，就自动把该候选评为更好。

## 评分和计数

完整评分为 `S = 0.5 * 窗口平均熵 + 0.5 * 一步lookahead熵`。原路径与替代路径采用同一个公式。保底仅保证被接受候选在这次实际比较的评分上严格改善，不保证最终图片质量，也不保证经过均值筛选的两条候选覆盖全局最优解。

每张图分别记录：

- `checks`：触发检查的窗口数。
- `accepted_edits`：接受替代路径的窗口数。
- `kept_original`：检查后保留原路径的窗口数。
- `backtracks`：兼容字段，等于 `accepted_edits`，不可与 checks 混用。

恒有 `checks = accepted_edits + kept_original`。统计均值包括零值；逐次事件记录温度、抽样 token、位置、候选分数与原路径分数。检查被拒绝后确认该原窗口并继续；分支计算已消耗随机数，因此保留当前原窗口不等于保证后续整张图片与未检查版本逐像素相同。

## 文件与运行

原 `inchworm_core.py`、`janus_backend.py`、`llamagen_backend.py`、`run_requested_experiments.py` 及旧提交脚本未修改。

- `inchworm_variant.py`：独立控制器。
- `inchworm_variant_rules.py` / `inchworm_variant_backend.py`：温度、G、混合位置和抽样。
- `inchworm_variant_scoring.py`：对称评分与严格保底。
- `variant_model_runner.py`：独立模型入口，同时支持 backbone、原 Inchworm、新方法。
- `run_variant_experiments.py`：三方法对比与独立统计。
- `scripts/submit_variant.sh`：独立 Slurm 提交，不调用旧任务入口。

在一个新的代码 checkout 中，复用已经准备完成的资产。先从旧任务目录复制本机 `configs/local.sh`，再运行 CPU 资产准备脚本来建立本 checkout 的 benchmarks 引用（见 [追加实验说明](REQUESTED_EXPERIMENTS.md)）。不要更新已经运行或排队任务所使用的旧 checkout。

默认先做9个提示词、三方法各一张，总共27张图（每个模型3个不同提示词，均与旧定性实验一致）；每模型仅一GPU作业，三方法复用已加载模型。各方法一次完整 warmup，不纳入结果。

```bash
# 在新目录、CPU 资产准备成功后
bash scripts/submit_variant.sh --check-only variant_pairs01 pairs
bash scripts/submit_variant.sh --submit variant_pairs01 pairs
```

想统计同一批100个提示词的三方法性能和检查/接受/保留次数，显式选择 profile；不自动额外提交这些较长的任务：

```bash
bash scripts/submit_variant.sh --check-only variant_profile01 profile janus1b llamagen
bash scripts/submit_variant.sh --submit variant_profile01 profile janus1b llamagen
```

普通已获分配的 GPU 服务器也可运行（Python 选择对应 Janus 或 LlamaGen 环境）：

```bash
python run_variant_experiments.py --root /path/to/checkout --data-root /path/to/data \
  --out /path/to/new-output --model janus1b --scope pairs
# 只运行新方法，复用已有基线结果进行定性检查时：
python run_variant_experiments.py --root /path/to/checkout --data-root /path/to/data \
  --out /path/to/another-new-output --model janus1b --scope pairs --methods inchworm_variant
```

可用 `--tmin --tmax --h0 --delta` 修改初值；修改任何配置后使用新的输出目录。Slurm 默认使用上表初值。

输出位于 `$DATA_ROOT/outputs/<RUN_TAG>/<MODEL>/<METHOD>/pairs/` 或 `profile/`。`pairs_summary.json` 是3提示词定性子集统计，`profile_summary.json` 是100张性能子集统计；两者都不是完整 GenEval 分数。保存原图、逐图JSON、各方法汇总和 completed.json。

模型作业先运行新方法规则与缓存测试、Janus/LlamaGen 既有适配器检查，再做 warmup 和生成。纯 Python 测试通过不等于真实 GPU、性能或质量验证完成。
