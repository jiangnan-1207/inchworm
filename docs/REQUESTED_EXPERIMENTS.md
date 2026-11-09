# 三项追加实验

## 输出与口径

1. Janus-Pro-1B、LlamaGen GPT-XL 各选同一批 100 个提示词，各生成 backbone 100 张 + Inchworm 100 张。batch=1，包含一次完整 warmup / 方法，warmup 不纳入统计。100 个提示词使用固定种子按 GenEval 六类别近似比例抽样，不是完整 benchmark 分数。
2. LlamaGen 的上述 100 张 Inchworm 图片同时用于平均回溯次数：回溯事件数总和 / 100，包含零回溯图片。这里的“次数”是一整个窗口的分支选择事件，不是候选分支数或 token 数。
3. Janus-Pro-1B、LlamaGen、Janus-Pro-7B 各用30个不同提示词，三个模型合计90个不同提示词，每词生成 baseline / Inchworm 一对，共180张。两方法同提示词、同种子，全部保存。用户可从中选定性展示图，不能用选例代替完整benchmark评分。

测量记录 `latency_seconds`（文本准备 + 图像 token + VQ 解码 + 像素回传 CPU）、`token_seconds`、PyTorch CUDA allocated / reserved 峰值，模型加载与 PNG 写盘不计入 latency。模型、T5、VQ 均常驻 GPU，显存峰值包含它们。样本间释放无用张量并清理缓存，CUDA 同步后计时；每方法执行一次完整热身，运行时交替两方法顺序。不要把这份 batch=1 latency 与旧四图批次的分摊耗时混比。

两模型最好使用同型号 GPU；各自 baseline/Inchworm 在同一 GPU 作业内运行。汇报该工程实现的测量值，并保留 GPU、Torch、CUDA、模型 revision、源码哈希和全部逐图数据。1B、7B 为 384px；LlamaGen 为 512px，不宣称模型间是同分辨率效率对比。

LlamaGen 新适配器使用不可变前缀 cache 拼接和裁剪后注意力 mask，拼接及分支缓存开销计入实际测量。它不是原生可变 cache 的零开销封装。GPU 作业先执行 cache 分支单元测试、32 个固定前缀上的原生概率分布与恢复检查（total variation ≤0.01），通过后继续 warmup 和正式任务。该数值检查不是论文质量的证明。初次运行若检查失败，应修复，不能跳过检查给出分数。

## 复用已有老师服务器资产

为避免改变已排队的完整 benchmark，使用一个**新代码目录**，指向已有模型和环境。配置仅存在服务器本地，不上传账号路径到仓库。假设你已有 `janus-pro` 和 `llamagen-v1` 环境以及之前下载的 LlamaGen 权重。

先在项目盘新目录 clone 当前仓库。在 `configs/local.sh` 填写：

```bash
DATA_ROOT="你的原有 scratch/inchworm 绝对路径"
JANUS_PYTHON="$DATA_ROOT/envs/janus-pro/bin/python"
LLAMA_PYTHON="$DATA_ROOT/envs/llamagen-v1/bin/python"
GENEVAL_PYTHON="$DATA_ROOT/envs/geneval-v1/bin/python"
ACCOUNT="目标账户"
PARTITION="gpu-l40s"
QOS="inferno"
GPU_REQUEST="gpu:l40s:1"
CPUS_PER_TASK=4
MEMORY="64G"
WALLTIME="08:00:00"
```

先给 CPU 下载作业导出：`REPO_ROOT`（本次新代码路径）、`EXISTING_ROOT`（之前已有 benchmarks 的 inchworm 根目录）、`CONFIG_FILE="$REPO_ROOT/configs/local.sh"`，用 `sbatch --export=ALL scripts/requested_setup_worker.sh` 提交 CPU 作业，并显式带账户、QoS、cpu-small 分区、4CPU、16G 内存、2小时时限以及 scratch 日志绝对路径。

该 CPU 作业只新增 Janus-Pro-1B 权重、校验旧源码版本、在新目录引用旧源码，保留已有 7B 和 LlamaGen 权重及环境。新目录原先不能含其他 `benchmarks/geneval` 或 `benchmarks/LlamaGen` 实体目录，脚本不会主动覆盖它们。

CPU 日志出现 `REQUESTED_ASSETS_OK` 且 sacct 为 COMPLETED 后执行：

```bash
bash scripts/submit_requested.sh --check-only extra01
bash scripts/submit_requested.sh --submit extra01
```

三个独立作业，每个一张 GPU，分别完成 1B 的两个任务、LlamaGen 的三个任务、7B 的定性图任务；不要求同时空出三张卡。8小时是运行时限，不是预计耗时，也不保证排队时长。提交一次后保留三个作业号，避免重复提交。

结果在 `$DATA_ROOT/outputs/extra01/`：

- `janus1b/profile_summary.json`：1B 两方法平均/中位总延迟、平均 token 时间、最大峰值显存、平均回溯。
- `llamagen/profile_summary.json`：LlamaGen 的同类指标，其中 `results.inchworm.mean_backtracks` 对应任务 2。
- 三个模型各自 `backbone/pairs/` 和 `inchworm/pairs/`：每模型30对，共90对图片及逐图JSON；提示词固定在 `configs/qualitative_prompts.json`。
- 三个模型各自 `completed.json`：完整完成标记。

峰值显存报告 `max_peak_allocated_bytes / 1024**3`（GiB）；reserved 峰值另列，不能用 `nvidia-smi` 某一次截图替代。表格尚无完整 100 张时不出汇总；可在相同配置与 GPU 型号上手工重新运行该失败 worker，已有逐图结果校验后跳过。不要改源码后继续写进同一 RUN_TAG。
