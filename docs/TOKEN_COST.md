# 原版 Inchworm：100张图 token 开销补测

入口 `scripts/submit_token_profile.sh --submit janus1b RUN_TAG`，或模型参数 `llamagen`。独立普通服务器可用所需生成环境运行 `python run_token_profile.py --model janus1b --assets-root /existing/source --data-root /existing/data --out /new/output`。

这是补充计数实验，不是新增评分集。复用 `experiment_plan.select_100` 的同一批100条分层GenEval提示词及 `42+4*prompt_id` 种子，原版 `inchworm_core.Config()`，不加新方法的保底、温度或高G规则。每张图单独计数，预热一张不计入；最终均值除以100，包括零回溯图。原版core及采样逻辑不变，计数包装器仅记录调用；测试比较完整控制器输出、候选事件、后端调用和RNG序列。

旧的 `extra_20261109/janus1b/profile_summary.json` 已有平均回溯70.21（7021/100）。旧性能记录没有保留候选事件或模型调用计数，不能据此准确反推token开销。新实验保存在新目录，不改旧latency/peak-memory。新计数运行不输出性能结论，避免将额外记录开销混入旧计时。

## 定义

设最终图像长度L（Janus=576，LlamaGen=1024）；每次检查窗口结束位置f、候选替换位置i均按一基编号。每条候选从i到f重新构造，共 `1+(f-i)` 个token；i之前的缓存前缀复用，不重复计数。

| JSON字段 | 含义 |
|---|---|
| `backtracks` | 原版实际窗口替换次数，候选条数不算回溯次数 |
| `regeneration_tokens` | 所有候选的 `1+(f-i)` 之和，包括被淘汰候选；不是只数最终图中改变的token |
| `forced_alternative_tokens` | 替换位置的top-k指定token数；指定token不调用随机采样 |
| `resampled_suffix_tokens` | 所有候选的后缀随机重采样数，即上述 `f-i` 之和 |
| `selected_candidate_regeneration_tokens` | 每次胜出候选的重新构造长度之和，作为辅助统计 |
| `logical_generated_image_tokens` | L＋regeneration_tokens：原路径抽样加所有候选构造量，不含仅评分的lookahead |
| `image_probability_evaluations_per_branch` | 实际产生图像token概率分布的前向次数：含所有候选后缀及lookahead，缓存分布复用不重复计 |
| `model_input_image_token_positions_per_branch` | 实际送进语言模型的图像token位置总数，包含被丢弃候选和lookahead的输入；作为**图像token consumption**主列 |
| `prompt_prefill_token_positions_per_branch` | 文本提示词/模板预填充位置，另列，不计T5文本编码器内部运算 |
| `model_input_image_token_positions_with_cfg` | 上述图像输入位置乘2，对应条件/无条件CFG分支 |
| `model_input_token_positions_with_cfg` | （预填充位置＋图像输入位置）乘2，作为**含文本及CFG的总token consumption** |

`lookahead_prediction_requests` 单列前瞻请求数；`cached_prediction_reuses` 是胜出lookahead分布被下一主循环复用的次数。前瞻预测分布用于计算熵，不抽取额外图像token，所以它增加模型处理量，不增加生成token数。

可交叉核验：设候选后缀随机token数A，lookahead请求Q，缓存分布复用C，则实际前向分布数 `F=L+A+Q-C`；随机采样调用 `L+A`；一次文本预填充产生第一个图像分布，其余 `F-1` 次输入图像token。每步同时处理CFG双分支，逻辑图像token仍只算一份。

例如原生Janus输出576个token，只需575个图像token输入加一次提示词prefill：prefill预测第1个token，第576个输出token直接供VQ解码，不再输入语言模型。故输入计数575不是漏算。原生基线 `backbone_reference` 按此循环解析计算，没有重新跑基线GPU推理；必须标注**解析参考值**，不能称新测量结果。

这些是token位置处理计数，不是attention总读写量、FLOPs、文本编码器/VQ计算量或GPU时间；KV缓存被attention读到的历史位置不另按整段重复计数。不同定义的token指标不能混用。

## Phoenix使用

在新checkout复制现有 `configs/local.sh`，保留已有环境、权重。设置 `ASSETS_ROOT=/storage/project/r-rg1002-0/twang730/CVPR/inchworm` 复用该目录下的geneval/LlamaGen源码及setup_records。配置里需设置正确JANUS_PYTHON/LLAMA_PYTHON、DATA_ROOT、账户、分区和GPU资源。

```bash
export ASSETS_ROOT=/storage/project/r-rg1002-0/twang730/CVPR/inchworm
bash scripts/submit_token_profile.sh --check-only janus1b tokens100_20261110
bash scripts/submit_token_profile.sh --submit janus1b tokens100_20261110
```

每个模型一个GPU作业，默认4小时，可用 `TOKEN_WALLTIME=08:00:00` 调整。输出 `$DATA_ROOT/outputs/tokens100_20261110/janus1b/token_summary.json`，逐图PNG、完整事件和计数在该目录的 `inchworm/` 下；只有100张全部完成才写summary和完成标记。已有PNG及receipt的哈希/实验规格会检查后复用。保留提交日志，避免重复提交；需要恢复时先检查原作业，再以新提交标签启动driver并指定原输出路径。

**LlamaGen当前先做原生诊断及修复cache概率一致性失败，不先提交大批补测。** 新入口保留相同硬检查，不提高阈值、不绕过失败；修复通过后用同一脚本的 `llamagen` 参数，使用包含修复的新checkout和新输出标签。此处提供入口不表示LlamaGen已经修好。已有旧质量分数与原图保留。
