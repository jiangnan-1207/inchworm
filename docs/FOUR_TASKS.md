# 四项任务：交给其他服务器运行

全部代码在同一个公开仓库中，无需老师服务器的账号、VPN、固定路径或已有安装。运行平台为 Linux x86_64、Python 3.10、支持 BF16 的 NVIDIA GPU，优先用 A100/H100/H200/L40S，建议单卡48GB或更大显存。固定 PyTorch 2.5.1/CUDA12.4 的生成环境和 PyTorch2.0/CUDA11.8 的 GenEval 环境分别安装；Blackwell 等新架构尚未验证。

## 任务定义

| 任务 | 模型 | 输出规模 | 目的 |
| --- | --- | --- | --- |
| task1 | Janus-Pro-1B、LlamaGen GPT-XL | 每模型 backbone100张 + 原Inchworm100张，共400张 | batch=1 latency和peak memory对比 |
| task2 | LlamaGen + 原Inchworm | 读取task1的100张结果，不重复生成 | 平均回溯次数，零回溯图片计入分母 |
| task3 | Janus-Pro-1B、LlamaGen、Janus-Pro-7B | 每模型30个不同提示词，共90对、180张 | 定性候选图，供用户挑选展示 |
| task4 | Janus-Pro-1B、LlamaGen各3个消融版本 | 每版本每模型553×4=2212，6组共13272张 | 完整GenEval分数 |

“原Inchworm”与消融版本分开，任务1、2仍为原方法。任务4改为：`guard_only`（仅保底，4H、原top-k）、`guard_temperature`（保底+温度，4H）、`guard_gap`（保底+2H/2G，原top-k、不调温）。三个版本均保留8选2、原路径额外入选、delta=0严格降分接受。原三改动全开的 `combined` 仍保留，默认不纳入此次六组评分。参数见 [新方法说明](INCHWORM_VARIANT.md)。100提示词性能子集不是完整GenEval分数。

## 每位同学先完成一次安装

```bash
git clone https://github.com/jiangnan-1207/inchworm.git
cd inchworm
cp configs/example.sh configs/local.sh
```

编辑 `configs/local.sh`：将 `DATA_ROOT` 改成代码目录以外、实际有空间的数据盘绝对路径，例如 `/data/yourname/inchworm-data`。三个 Python 路径默认随 DATA_ROOT 变化。请预留足够空间，三套环境和模型可能占用上百GB；模型/缓存/输出不写入 home。

在普通服务器运行以下命令；Slurm集群应把安装命令提交到获批的CPU计算节点，不在登录节点安装或下载模型。

```bash
# 把路径替换成 configs/local.sh 中相同的实际值
python3.10 tools/bootstrap.py --data-root /data/yourname/inchworm-data --janus-models 1b 7b
```

脚本下载固定源码、权重并建立独立环境。1B与7B共用Janus环境。模型修订号初值保存在 `configs/model_revisions.json`，实际记录位于 `benchmarks/setup_records/model_revisions_v1.json`。

如果按模型分工，可减少下载：

```bash
# 只负责1B：task1、task3的1B部分，以及task4的1B完整分数
python3.10 tools/bootstrap.py --data-root /data/yourname/inchworm-data --components janus geneval --janus-models 1b

# 只负责LlamaGen
python3.10 tools/bootstrap.py --data-root /data/yourname/inchworm-data --components llamagen geneval

# 只负责7B的task3，不需要GenEval评分
python3.10 tools/bootstrap.py --data-root /data/yourname/inchworm-data --components janus --janus-models 7b
```

等待 `BOOTSTRAP_COMPLETE`。CPU安装完成不等于GPU兼容性或图片效果已验证；GPU入口会执行适配器检查，新方法还会运行额外测试，失败应保留日志并修复，不跳过检查。

## 普通GPU服务器：逐项运行

以下命令在已经获得GPU使用权的终端或tmux会话内运行；可用 `CUDA_VISIBLE_DEVICES=0` 等为进程指定已分配的卡。每个命令默认一张GPU。任务之间使用不同RUN_TAG。

### task1：原方法latency、peak memory

```bash
bash scripts/four_tasks.sh task1 janus1b perf01
bash scripts/four_tasks.sh task1 llamagen perf01
```

每模型运行同一批固定100提示词，baseline与Inchworm各100张。模型只加载一次，每方法先完整热身一次，热身不计入统计，方法先后顺序交替。

```bash
source configs/local.sh
cat "$DATA_ROOT/outputs/perf01/janus1b/profile_summary.json"
cat "$DATA_ROOT/outputs/perf01/llamagen/profile_summary.json"
```

读取 `results.backbone`、`results.inchworm`：

- `mean_latency_seconds`：文本准备、生成token、VQ解码和CPU像素回传的平均总时间，不含加载模型、图片写盘。
- `mean_token_seconds`：图像token生成平均时间。
- `max_peak_allocated_bytes`、`max_peak_reserved_bytes`：100张中的最大PyTorch CUDA显存峰值；除以 `1024**3` 得GiB。模型、VQ和LlamaGen的T5常驻显存计入峰值。

每图都保留JSON。对比两个方法时使用相同GPU型号、batch=1，不能把旧四图批次的分摊时间混入。本实现使用功能性前缀cache，其实际开销计入测量；结果不能直接称为论文原实现的效率。

### task2：LlamaGen 100张平均回溯

task1的LlamaGen完成后直接读取，**不申请GPU、不再生成100张**：

```bash
bash scripts/four_tasks.sh task2 perf01
```

输出 `mean_backtracks = total_backtracks / 100`。原方法每次触发后的窗口替换计一次，不把每个候选分支算一次。若不足100张，程序拒绝输出完整汇总。

### task3：每模型30对图片

```bash
bash scripts/four_tasks.sh task3 janus1b pairs30_01
bash scripts/four_tasks.sh task3 llamagen pairs30_01
bash scripts/four_tasks.sh task3 janus7b pairs30_01
```

90个不同提示词固定在 `configs/qualitative_prompts.json`；每模型30个，三个模型之间也不重复，每词两方法同prompt、同seed。路径：

```text
<DATA_ROOT>/outputs/pairs30_01/<MODEL>/backbone/pairs/
<DATA_ROOT>/outputs/pairs30_01/<MODEL>/inchworm/pairs/
```

每模型60张，共180张，逐图JSON包含提示词和种子。每模型 `completed.json` 应显示 `qualitative_pairs=30`、`qualitative_images=60`。可以从中挑选定性展示，但保留原始完整结果，展示时注明选例，不能用挑选后的图片替代task4完整集合算分。

### task4：三个消融版本的完整GenEval

只有一张GPU时，以下命令会对每个模型依次完成三个版本的完整生成、覆盖检查和评分。每模型3×2212=6636张，两模型共13272张，不会只跑100张：

```bash
bash scripts/four_tasks.sh task4 janus1b ablations01
bash scripts/four_tasks.sh task4 llamagen ablations01
```

可按整个模型分给两位同学独立完成。三个版本位置/提议策略已分开；仅第二版使用 Tmin=.5、Tmax=1、h0=2 nats 的有放回温度采样，允许原token和重复。第一、三版沿用原top-k，排除原token、不重复。所有后缀不额外调温，检查无次数上限，delta=0。

只跑某一版时，在末尾明确添加版本名：

```bash
bash scripts/four_tasks.sh task4 janus1b guard_only01 guard_only
bash scripts/four_tasks.sh task4 llamagen guard_gap01 guard_gap
# 旧的三改动全开方案仍可显式运行
bash scripts/four_tasks.sh task4 janus1b combined01 combined
```

结果文件：

```text
<DATA_ROOT>/outputs/ablations01/merged/evaluation/janus1b_guard_only.summary.json
<DATA_ROOT>/outputs/ablations01/merged/evaluation/janus1b_guard_temperature.summary.json
<DATA_ROOT>/outputs/ablations01/merged/evaluation/janus1b_guard_gap.summary.json
<DATA_ROOT>/outputs/ablations01/merged/evaluation/llamagen_guard_only.summary.json
<DATA_ROOT>/outputs/ablations01/merged/evaluation/llamagen_guard_temperature.summary.json
<DATA_ROOT>/outputs/ablations01/merged/evaluation/llamagen_guard_gap.summary.json
```

必须满足 `prompts=553`、`images=2212`、`formal_complete_benchmark=true`，再读取 `overall` 与 `task_scores`。Overall为六任务准确率的等权平均，0–1；不是2212张直接混算的成功率，也不是best-of-4。生成完图片尚未代表取得分数，必须完成GenEval评分。

## 追加：原Inchworm在Janus-Pro-1B上的完整分数

与新方法三版分开，可显式选择 `original`，调用原Inchworm控制器，三个新改动都关闭，仍为553×4=2212张：

```bash
bash scripts/four_tasks.sh task4 janus1b original1b01 original
# Slurm，先用 --check-only 检查，再提交
ABLATIONS=original bash scripts/submit_four_tasks.sh --submit task4 original1b01 janus1b
```

结果文件为 `outputs/original1b01/merged/evaluation/janus1b_original_inchworm.summary.json`，与新方法六份结果独立。任务1依然是1B和LlamaGen的性能，任务3依然是三模型各30对；这项原方法分数不替代新方法三版评分。

## task4 多卡分片（普通服务器）

例如一台机器上已有4张获准使用的GPU，用同一份代码、模型记录和数据目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/four_tasks.sh task4-generate janus1b newfull4 0 4 guard_only > /tmp/newfull4-0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash scripts/four_tasks.sh task4-generate janus1b newfull4 1 4 guard_only > /tmp/newfull4-1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash scripts/four_tasks.sh task4-generate janus1b newfull4 2 4 guard_only > /tmp/newfull4-2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash scripts/four_tasks.sh task4-generate janus1b newfull4 3 4 guard_only > /tmp/newfull4-3.log 2>&1 &
wait
# 检查四份日志成功，再使用一张卡评分
CUDA_VISIBLE_DEVICES=0 bash scripts/four_tasks.sh task4-score janus1b newfull4 4 guard_only
```

LlamaGen将模型参数改为 `llamagen`；其他版本替换最后一个参数。分片数量在开始后不能改变；缺片或配置不同将拒绝合并。建议按整个模型分给不同服务器，避免跨机器搬运单组分片和混入不同版本。

## Slurm服务器：提交式运行

配置本机 `ACCOUNT/PARTITION/QOS/GPU_REQUEST/CPUS_PER_TASK/MEMORY/WALLTIME`。任务1可先预留8小时时限，这不是预计耗时或排队时长。task4默认 `SHARDS=32`，`CONCURRENT` 为每模型并行生成任务数，每份一张GPU。同一模型三个版本按“本版生成→本版评分→下一版生成”依赖串行；两个模型独立。因此默认不会把并发申请扩大三倍。若每模型CONCURRENT=4，两个模型生成最多同时8张卡；按授权资源调整。

```bash
# task1：两个模型的性能，同时产出task2所需统计
bash scripts/submit_four_tasks.sh --check-only task1 perf01
bash scripts/submit_four_tasks.sh --submit task1 perf01

# task3：三个模型，每模型30对、共180张
bash scripts/submit_four_tasks.sh --check-only task3 pairs30_01
bash scripts/submit_four_tasks.sh --submit task3 pairs30_01

# task4：三版本×两模型，共6个生成数组和6个评分作业
bash scripts/submit_four_tasks.sh --check-only task4 ablations01
bash scripts/submit_four_tasks.sh --submit task4 ablations01
```

如只接手一个模型，在末尾添加模型名，例如：

```bash
bash scripts/submit_four_tasks.sh --submit task4 ablations01 janus1b
# 或仅选择一个版本；分次提交请使用不同RUN_TAG
ABLATIONS=guard_temperature bash scripts/submit_four_tasks.sh --submit task4 temperature01 janus1b
```

task2仍使用 `bash scripts/four_tasks.sh task2 perf01` 在读取结果时执行。提交记录在数据目录logs；出错先查记录和日志，不盲目重跑提交脚本。依赖任务不会在生成失败时冒充完成。

## 协作交回什么

1. task1/2：两个 `profile_summary.json`；需要复核时附逐图JSON。
2. task3：180张原图及对应JSON，保留MODEL/METHOD目录，避免同名覆盖。
3. task4：六组 `.summary.json`、`.jsonl` 逐图评分和 `.coverage.json`；保存生成图片以便复核。
4. 每台服务器的代码commit、实际 `model_revisions_v1.json`、环境freeze记录和GPU型号。

task4 的 merged 图片目录使用符号链接引用 shards；交回图片时保留 shards，或打包时显式跟随链接，不能只复制链接而丢掉实际图片。

原代码、原方法和新方法分别保留。所有图片检查通过才生成相应完整汇总；不会填造分数。源码/模型/配置发生变化时使用新目录或RUN_TAG，运行中不要git pull改变文件。
