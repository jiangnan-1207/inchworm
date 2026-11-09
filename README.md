# Inchworm + Janus-Pro-7B / LlamaGen / GenEval

**交给同学运行的四项任务：[从新服务器安装到逐项运行、查看结果](docs/FOUR_TASKS.md)。** 包含原方法1B/LlamaGen性能、100张平均回溯、每模型30对定性图（共180张），以及三个消融版本×两个模型的完整GenEval（六组共13272张）。统一入口为 `scripts/four_tasks.sh`，可选Slurm入口为 `scripts/submit_four_tasks.sh`。

把 Inchworm 解码接到 Janus-Pro-7B，并对三种方法运行**完整 GenEval**。支持普通 Linux GPU 服务器，也支持 Slurm；不依赖某所学校的 VPN、账号或固定目录。

追加任务入口：[Janus-Pro-1B / LlamaGen 延迟与显存、LlamaGen 100 张平均回溯、三模型定性图](docs/REQUESTED_EXPERIMENTS.md)。LlamaGen Inchworm 的真实 GPU 兼容性需运行验证，作业会先检查再继续生成。

独立新方法：[inchworm_variant：原路径保底＋自适应温度提议＋混合位置](docs/INCHWORM_VARIANT.md)。保留原方法及旧任务入口；新方法的 GPU 效果仍待实际验证。

| 方法名 | 内容 | 完整输出 |
| --- | --- | --- |
| `janus_baseline` | Janus-Pro-7B 原生采样，CFG 5，384px | 553 prompts × 4 = 2212 张 |
| `janus_inchworm` | 同一 Janus 模型 + Inchworm，无回溯次数上限 | 2212 张 |
| `llamagen_baseline` | LlamaGen GPT-XL，stage2 512px，CFG 7.5 | 2212 张 |

三组共 6636 张。正式分数必须覆盖所有 553 个提示词及每词 4 个样本。Overall 是六个任务准确率的等权平均，数值范围 0–1。程序会校验图片数量、元数据和校验和，缺样本就报错。

**当前状态：**这是根据论文方法实现的工程版本，不是作者发布的官方实现。原文缺少部分参数，默认设置及边界约定见 [docs/METHOD.md](docs/METHOD.md)。已有核心单元测试和单图回溯运行验证；这个可迁移安装流程仍需在目标服务器运行，完整 benchmark 分数尚未取得。

## 1. 在新服务器安装

要求 Linux x86_64、Python **3.10**（含 venv）、Git、联网下载权限、支持 BF16 的 NVIDIA GPU。优先使用 A100 / H100 / H200 / L40S，建议单卡至少 48GB 显存。24GB 显存以及 Blackwell 尚未验证；固定的旧版 GenEval CUDA 环境不适合直接承诺支持所有新显卡。需要能运行 CUDA 12.4 PyTorch 的驱动。模型、三套隔离环境和输出需要较多空间，请选择有充足容量的数据盘。

在普通 GPU 服务器终端中执行；如果是集群，安装和模型下载放到获批的计算节点，遵守本地登录节点规则。

```bash
git clone https://github.com/jiangnan-1207/inchworm.git
cd inchworm
cp configs/example.sh configs/local.sh
```

编辑 `configs/local.sh`，把 `DATA_ROOT` 改成**代码目录以外**的数据盘绝对路径，例如 `/data/yourname/inchworm-data`。其余 Python 路径会随它变化。然后安装：

```bash
python3.10 tools/bootstrap.py --data-root /data/yourname/inchworm-data
```

请把示例路径替换成刚才填写的实际路径。安装脚本会拉取固定版本的官方源码，分别建立 `janus`、`llamagen`、`geneval` 环境并下载权重；无需手动把它们装到同一个环境。下载得到的模型修订号会记录在 `benchmarks/setup_records/model_revisions_v1.json`。

如果同学只接手 Janus + Inchworm 两组，可以只安装：

```bash
python3.10 tools/bootstrap.py --data-root /data/yourname/inchworm-data --components janus geneval
```

`--plan` 只展示部署计划；`--skip-downloads` 只装环境，不能在没有权重的情况下开始生成。重新执行安装会使用已有目录，不会主动删除现有环境；安装中断需保留完整报错再处理。

## 2. 普通 GPU 服务器直接跑完整评测

以下每个 `generate` 都覆盖完整集合，默认单 GPU 顺序生成 2212 张。先确认 `nvidia-smi` 能看到分配给你的 GPU。可以在自己的 `tmux` 会话内运行，避免 SSH 断开导致进程退出。

```bash
# 一次 CUDA 兼容性检查；不计入 benchmark，随后直接进行完整生成。
bash scripts/run.sh preflight janus_baseline run01

# 第一组：Janus 原生
bash scripts/run.sh generate janus_baseline run01
bash scripts/run.sh evaluate janus_baseline run01

# 第二组：Janus + Inchworm
bash scripts/run.sh generate janus_inchworm run01
bash scripts/run.sh evaluate janus_inchworm run01

# 第三组：LlamaGen（可由另一位同学单独负责）
bash scripts/run.sh generate llamagen_baseline run01
bash scripts/run.sh evaluate llamagen_baseline run01
```

只接手某一组就执行对应两行。可以在不同机器分别评测各组，再汇总 `.summary.json`；不要把不同源码、不同模型修订号生成的同组分片混合。两台机器的模型修订记录、代码 commit 和配置应保留并核对。

完成后查看：

```bash
source configs/local.sh
cat "$DATA_ROOT/outputs/run01/merged/evaluation/"*.summary.json
```

每个方法有独立的 `*.summary.json`，其中 `images` 应为 2212、`prompts` 为 553、`formal_complete_benchmark` 为 true。只看生成日志不能代替 GenEval 分数；`overall` 是要汇报的总体分数，`task_scores` 是分项。

## 3. 多卡分片（普通服务器）

例如分成 4 份，在 4 张已获授权的 GPU 上并行运行。每份固定覆盖原始提示词编号 `i % 4 == shard_index`，合并后仍为 553 × 4。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh generate janus_inchworm run02 0 4 > /tmp/inchworm-0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh generate janus_inchworm run02 1 4 > /tmp/inchworm-1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh generate janus_inchworm run02 2 4 > /tmp/inchworm-2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash scripts/run.sh generate janus_inchworm run02 3 4 > /tmp/inchworm-3.log 2>&1 &
wait
# 检查四份日志均成功后再评分；评分需要一张 GPU。
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh evaluate janus_inchworm run02 4
```

## 4. Slurm 提交（可选）

在 `configs/local.sh` 填写**目标集群自己的** `ACCOUNT`、`PARTITION`、`QOS`、`GPU_REQUEST`，并设置每任务 CPU、内存、时限。`SHARDS=32`、`CONCURRENT=4` 表示每组 32 个分片、最多同时运行 4 份，每份一张 GPU；并非同时申请 32 张。默认墙钟时间 2 小时，需要按目标硬件吞吐调整。

```bash
# 只检查调度请求，不实际提交
bash scripts/submit.sh --check-only run03
# 提交完整三组及其评分
bash scripts/submit.sh --submit run03
# 或只提交同学负责的 Inchworm 一组，用新的 RUN_TAG
bash scripts/submit.sh --submit only_inchworm janus_inchworm
```

生成组依次衔接，前组评分可与后组生成重叠。`CONCURRENT=4`、三组时最多 4 个生成任务加 2 个前组评分任务；实际启动由调度器决定。作业号和提交记录保存在数据目录 `logs/<RUN_TAG>.submissions.txt`。失败时检查对应日志及依赖状态，**不要重新提交整个启动脚本来碰运气**。

## 5. 续跑与实验记录

普通服务器重跑相同 `generate` 命令，会核对每张图片的收据并跳过已完成图片。Janus baseline 部分完成时会重算该提示词的四图批次，但只补存缺失图片，以保持固定批次形状。不要在同一运行目录中改源码、分片数、模型或配置；改动后使用新的 RUN_TAG。

主要文件：

- `inchworm_core.py`：熵触发、候选回溯、两阶段筛选。
- `janus_backend.py`：Janus CFG、熵和只读 KV cache 分支。
- `generate_full_shard.py`：完整提示集生成与同模型原生循环对齐检查。
- `merge_full.py` / `protocol.py`：完整覆盖、校验和、断点续跑。
- `evaluate_benchmark.py`：调用固定版本的 GenEval 检测与颜色分类评分。
- `tools/bootstrap.py` / `tools/download_assets.py`：安装、下载及版本记录。

本地可执行不需要 GPU 的检查：

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

原论文、模型权重、生成图片、环境文件和个人配置不放在此仓库。第三方来源及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。仓库公开供协作访问；本项目新增代码的项目级开源许可证尚未指定。
