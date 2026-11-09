# Lumina-mGPT 2.0：30对原生/原Inchworm图片

本任务使用 `Alpha-VLLM/Lumina-mGPT-2.0` 文生图模型和MoVQGAN，768×768，每提示词原生约束采样与原Inchworm各一张，共30对60张。不是Lumina-Image 2.0扩散模型，也不是mGPT 1.0或Omni模型。

官方原生768生成的显存参考约80GB，非本适配器测量值。老师服务器优先申请一张H200/worker；本次不默认使用48GB L40S，不加量化或Speculative Jacobi，以免混入其他方法。端到端GPU部署和图片效果仍须在实际作业验证。

官方来源：[代码与原生资源参考](https://github.com/Alpha-VLLM/Lumina-mGPT-2.0)、[模型](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0)。固定源码/权重版本保存在 `configs/lumina_sources.json`。

## 独立环境与适配边界

- 使用单独的Linux/Python3.10 venv，Torch2.3.1+cu121、torchvision0.18.1、Transformers4.46.2，不更改Janus/LlamaGen环境。
- 本入口使用官方模型的SDPA实现，不安装可选的外部FlashAttention扩展、torchao量化、训练/Gradio组件。推理依赖固定在 `requirements/lumina.txt`，这是本工程的最小依赖配方，仍待目标GPU环境验证。
- 调用官方模型、CFG、图像码约束、top-k和MoVQ解码；两组均固定CFG=4、top-k=4096、温度1.0。
- 两组输入都显式添加同样的图像开始/768尺寸头，省去自由文本或尺寸预测。原生组仍由官方 `model.generate` 和logits processors采样。这是固定尺寸头条件下的对比，不宣称覆盖官方自由文本/尺寸预测的完整流程。
- 768图像对应96×96=9216个实际图像码。每行的固定换行token会进入模型缓存，但不作为Inchworm的搜索位置；图像结束token也固定。原生生成长度为9216+96+1=9313（不含头）。
- Inchworm调用未修改的 `inchworm_core.generate` 和原Config：4个高熵位置、每位置2个排除原token的top-k替代、先均值筛2条再lookahead选优，不加入原路径保底、自适应温度或H/G混合位置，检查次数不设上限。
- 为保存条件/无条件两条分支状态，新增共享不可变128-token块缓存。分支只能复制末尾局部块，避免每个候选保留一整份长前缀；注意力计算时临时还原连续K/V。它是工程缓存实现，实际开销计入该实现，尚无GPU效率结论。
- 每个GPU worker先比对官方采样在相同前缀上的概率分布，包含跨行换行和缓存恢复。总变差阈值0.005，超过就中止，不跳过检查出图。数值容差检查不等于图片质量证明。
- 原生采样还会对固定结构token执行采样操作，随机数消耗与只在图像码上运行的控制器不同；同种子不保证关闭回溯后逐像素相同。每对用相同固定种子，逐图保留全部配置与回溯事件。

## 1. 新目录准备

在一个新的代码checkout中复制或编辑本机 `configs/local.sh`。DATA_ROOT继续指向数据盘，模型/环境/输出不放home。老师现有Janus环境的Python3.10可以作为安装解释器；新环境会单独建在 `$DATA_ROOT/envs/lumina-mgpt2`。

普通服务器可以在获准使用的CPU/计算节点直接运行：

```bash
python3.10 tools/setup_lumina.py --data-root /actual/path/to/inchworm-data
```

Slurm集群使用CPU作业：

```bash
export REPO_ROOT="$PWD"
export CONFIG_FILE="$REPO_ROOT/configs/local.sh"
# 默认使用configs/local.sh的JANUS_PYTHON作为Python3.10安装解释器。
# 如果没有Janus环境，设置可用的Python3.10路径：
# export LUMINA_SETUP_PYTHON=/actual/path/to/python3.10
sbatch --partition=YOUR_CPU_PARTITION --account=YOUR_ACCOUNT --qos=YOUR_QOS \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 --export=ALL \
  --output=/actual/data/logs/lumina_setup_%j.out --error=/actual/data/logs/lumina_setup_%j.err \
  scripts/lumina_setup_worker.sh
```

提前建立日志目录。确认作业COMPLETED/0:0且日志出现 `LUMINA_SETUP_OK`，再进入下一步。CPU安装通过仅表示权重/导入准备完成。

## 2. 提交30对生成

先按目标集群配置账户/QoS；Lumina脚本单独默认 `gpu-h200` / `gpu:h200:1`、4CPU、96G内存、8小时，不覆盖原配置的L40S字段。

```bash
bash scripts/submit_lumina.sh --check-only lumina30_01
bash scripts/submit_lumina.sh --submit lumina30_01
```

默认6个分片，每片5对10张，最多同时运行2个H200 worker。模型每片只加载一次；检查通过后直接继续正式生成。8小时是运行时限，实际耗时/显存未取得，不保证这个上限足够；若超时可按相同源码配置恢复已完成图片。

如只允许一张卡并行：

```bash
LUMINA_CONCURRENT=1 bash scripts/submit_lumina.sh --submit lumina30_serial
```

分片数可通过 `LUMINA_SHARDS` 在首次提交前配置，开始后不能更改；首次不要盲目重复提交启动脚本。脚本不保证某个GPU分区可用或排队多久。使用其他集群可明确设置 `LUMINA_PARTITION`、`LUMINA_GRES`、`LUMINA_WALLTIME`，确认显存满足后再提交。

普通已分配GPU服务器使用单进程：

```bash
source configs/local.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$DATA_ROOT/envs/lumina-mgpt2/bin/python" -u run_lumina_pairs.py \
  --root "$PWD" --data-root "$DATA_ROOT" --out "$DATA_ROOT/outputs/lumina30_01"
```

## 3. 结果核对

30条提示词在 `configs/lumina_prompts.json`，彼此不同，也不与之前三个模型的90条重复。

图片和逐图JSON在：

```text
<DATA_ROOT>/outputs/lumina30_01/backbone/
<DATA_ROOT>/outputs/lumina30_01/inchworm/
```

全部worker完成后可在登录节点做轻量完整性检查（不加载模型）：

```bash
source configs/local.sh
python3 tools/check_lumina_outputs.py --run "$DATA_ROOT/outputs/lumina30_01"
```

应显示pairs=30、images=60、status=complete。保留全部图片、逐图JSON、workers中的版本和parity记录；可以选定性图展示，但不能当作完整GenEval分数。
