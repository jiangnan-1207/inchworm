# 四项任务：交给其他服务器运行

全部代码在同一个公开仓库中，无需老师服务器的账号、VPN、固定路径或已有安装。运行平台为 Linux x86_64、Python 3.10、支持 BF16 的 NVIDIA GPU，优先用 A100/H100/H200/L40S，建议单卡48GB或更大显存。固定 PyTorch 2.5.1/CUDA12.4 的生成环境和 PyTorch2.0/CUDA11.8 的 GenEval 环境分别安装；Blackwell 等新架构尚未验证。

## 任务定义

| 任务 | 模型 | 输出规模 | 目的 |
| --- | --- | --- | --- |
| task1 | Janus-Pro-1B、LlamaGen GPT-XL | 每模型 backbone100张 + 原Inchworm100张，共400张 | batch=1 latency和peak memory对比 |
| task2 | LlamaGen + 原Inchworm | 读取task1的100张结果，不重复生成 | 平均回溯次数，零回溯图片计入分母 |
| task3 | Janus-Pro-1B、LlamaGen、Janus-Pro-7B | 共9个不同提示词，每词backbone/原Inchworm各1张，共18张 | 定性对比，不选最佳图 |
| task4 | Janus-Pro-1B + 新方法、LlamaGen + 新方法 | 每模型553提示词×4张=2212，共4424张 | 完整GenEval分数 |

“原Inchworm”与独立新方法 `inchworm_variant` 分开。新方法为原路径保底、温度有放回抽样、2H+2G位置，并保留8选2的两阶段筛选。参数和限制见 [新方法说明](INCHWORM_VARIANT.md)。性能/回溯用的100提示词子集不是完整GenEval分数。

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

### task3：九对图片

```bash
bash scripts/four_tasks.sh task3 janus1b pairs01
bash scripts/four_tasks.sh task3 llamagen pairs01
bash scripts/four_tasks.sh task3 janus7b pairs01
```

9个不同提示词固定在 `experiment_plan.py`；每模型3个，每词两方法同prompt、同seed。路径：

```text
<DATA_ROOT>/outputs/pairs01/<MODEL>/backbone/pairs/
<DATA_ROOT>/outputs/pairs01/<MODEL>/inchworm/pairs/
```

共18张，逐图JSON包含提示词和种子。每个模型的 `completed.json` 应显示 `qualitative_images=6`。

### task4：新方法完整GenEval

只有一张GPU时，以下命令依次完成完整生成、覆盖检查和评分。每模型2212张，需要较长运行时间，不会只跑100张：

```bash
bash scripts/four_tasks.sh task4 janus1b newfull01
bash scripts/four_tasks.sh task4 llamagen newfull01
```

其中任意一组可交给一位同学在自己的服务器独立完成，无需等待另一组。新方法配置固定为 Tmin=.5、Tmax=1、h0=2 nats、delta=0，允许原token和重复抽样；后缀不额外调温，回溯检查无次数上限。

结果文件：

```text
<DATA_ROOT>/outputs/newfull01/merged/evaluation/janus1b_inchworm_variant.summary.json
<DATA_ROOT>/outputs/newfull01/merged/evaluation/llamagen_inchworm_variant.summary.json
```

必须满足 `prompts=553`、`images=2212`、`formal_complete_benchmark=true`，再读取 `overall` 与 `task_scores`。Overall为六任务准确率的等权平均，0–1；不是2212张直接混算的成功率，也不是best-of-4。生成完图片尚未代表取得分数，必须完成GenEval评分。

## task4 多卡分片（普通服务器）

例如一台机器上已有4张获准使用的GPU，用同一份代码、模型记录和数据目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/four_tasks.sh task4-generate janus1b newfull4 0 4 > /tmp/newfull4-0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash scripts/four_tasks.sh task4-generate janus1b newfull4 1 4 > /tmp/newfull4-1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash scripts/four_tasks.sh task4-generate janus1b newfull4 2 4 > /tmp/newfull4-2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash scripts/four_tasks.sh task4-generate janus1b newfull4 3 4 > /tmp/newfull4-3.log 2>&1 &
wait
# 检查四份日志成功，再使用一张卡评分
CUDA_VISIBLE_DEVICES=0 bash scripts/four_tasks.sh task4-score janus1b newfull4 4
```

LlamaGen将模型参数改为 `llamagen`。分片数量在开始后不能改变；缺片或配置不同将拒绝合并。建议按整个模型分给不同服务器，避免跨机器搬运单组分片、重建符号链接和混入不同版本。

## Slurm服务器：提交式运行

配置本机 `ACCOUNT/PARTITION/QOS/GPU_REQUEST/CPUS_PER_TASK/MEMORY/WALLTIME`，不要照搬其他学校的账户。任务1可先预留8小时时限；这不是预计耗时或排队时长。task4默认 `SHARDS=32`，`CONCURRENT` 为**每模型**并行生成任务数，每份一张GPU，两模型可同时排队。若每模型CONCURRENT=4，两模型生成最多同时8张卡；按授权资源调整。

```bash
# task1：两个模型的性能，同时产出task2所需统计
bash scripts/submit_four_tasks.sh --check-only task1 perf01
bash scripts/submit_four_tasks.sh --submit task1 perf01

# task3：三个模型，九对图片
bash scripts/submit_four_tasks.sh --check-only task3 pairs01
bash scripts/submit_four_tasks.sh --submit task3 pairs01

# task4：两个模型的完整分片生成，评分依赖各自所有分片成功
bash scripts/submit_four_tasks.sh --check-only task4 newfull01
bash scripts/submit_four_tasks.sh --submit task4 newfull01
```

如只接手一个模型，在末尾添加模型名，例如：

```bash
bash scripts/submit_four_tasks.sh --submit task4 newfull01 janus1b
```

task2仍使用 `bash scripts/four_tasks.sh task2 perf01` 在读取结果时执行。提交记录在数据目录logs；出错先查记录和日志，不盲目重跑提交脚本。依赖任务不会在生成失败时冒充完成。

## 协作交回什么

1. task1/2：两个 `profile_summary.json`；需要复核时附逐图JSON。
2. task3：18张原图及对应JSON，保留MODEL/METHOD目录，避免同名覆盖。
3. task4：两组 `.summary.json`、`.jsonl` 逐图评分和 `.coverage.json`；保存生成图片以便复核。
4. 每台服务器的代码commit、实际 `model_revisions_v1.json`、环境freeze记录和GPU型号。

task4 的 merged 图片目录使用符号链接引用 shards；交回图片时保留 shards，或打包时显式跟随链接，不能只复制链接而丢掉实际图片。

原代码、原方法和新方法分别保留。所有图片检查通过才生成相应完整汇总；不会填造分数。源码/模型/配置发生变化时使用新目录或RUN_TAG，运行中不要git pull改变文件。
