# 原版 Inchworm：DPG-Bench / TIIF-Bench testmini

本入口对 Janus-Pro-1B 和 LlamaGen 分别生成 backbone 与 **原版** Inchworm 两组，四组都用同一完整提示词集和固定种子规则。原控制器 Config() 不变，不加入 guard、adaptive temperature、H/G；回溯不限次数。沿用工程参数，不宣称精确复现论文作者未公开参数。

DPG使用官方ELLA的全部1065提示词，每词4张，单组4260张、四组17040张。官方文件名同时有数字和字符串，不重新编号。逐图评分等价于官方2×2图的四个裁剪，保留问题依赖关系，先四图均值再跨提示词均值。修正上游CSV解析器跳过首条有效问题的错误；因此报告应注明 all-CSV-questions 修正，不能声称逐字执行未修正脚本。记录所有回答，异常直接失败，不跳过坏图汇总。

TIIF使用固定官方仓库 `data/testmini_prompts` 及对应 `testmini_eval_prompts`，全部39维、全部短/长提示词，各一图。数量由源码实际枚举后写manifest，避免把不同版本的554/完整集混用。结果必须写 **TIIF testmini / Qwen2.5-VL-7B + PaddleOCR，local evaluation**，不是完整TIIF、不是GPT-4o、不是新TIIF-Evaluator榜单。Qwen固定greedy/SDPA/BF16、max_new_tokens=2048，固定提示词、不提供目标caption；图像处理min_pixels=256×28²，max_pixels=1280×28²。非文字维按官方yes/no问题评分；文字维用PaddleOCR英文模型和官方GNED函数，转换1-GNED。每维先汇总，再在官方9组间宏平均，分别报告short/long。格式不匹配即失败，不伪造no答案。7B仅作为成本可控的本地评分器，未经该基准可靠性校准；公平比较应同配置重测backbone。

来源：[ELLA](https://github.com/TencentQQGYLab/ELLA)、[TIIF](https://github.com/A113N-W3I/TIIF-Bench)、[Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)。源码/权重固定版本在configs/alignment_sources.json；Paddle模型文件记录SHA，ModelScope mPLUG使用v1.0.0。各自许可证由原仓库/模型页提供。

## 老师服务器：新checkout，复用生成环境

```bash
cd /storage/project/r-rg1002-0/twang730/CVPR
git clone https://github.com/jiangnan-1207/inchworm.git inchworm_alignment_20261109
cd inchworm_alignment_20261109
cp ../inchworm_v2_20261109/configs/local.sh configs/local.sh
export REPO_ROOT="$PWD"
export CONFIG_FILE="$REPO_ROOT/configs/local.sh"
bash scripts/submit_alignment.sh --check-only prepare align_20261109
bash scripts/submit_alignment.sh --submit prepare align_20261109
```

CPU作业4核/32G/2小时，固定下载官方数据/源码，建两个独立评分环境，下载mPLUG、Qwen、PaddleOCR；不会升级Janus、LlamaGen、GenEval、Lumina环境。需要已有JANUS_PYTHON/LLAMA_PYTHON指向正常生成环境，DATA_ROOT指向scratch。

等待COMPLETED/0:0，日志同时出现ALIGNMENT_DATA_READY和ALIGNMENT_JUDGES_READY。它只是数据/导入准备完成，首次GPU评分仍须实际验证；依赖安装或下载失败请先看err，不要重复启动生成。

```bash
bash scripts/submit_alignment.sh --check-only generate align_20261109
bash scripts/submit_alignment.sh --submit generate align_20261109
```

默认2个独立生成数组（每模型16片、每片申请一张L40S、最多2片并行），加4个afterok评分作业（每模型DPG、TIIF各一）。每模型一片复用同一个模型生成两方法、两基准。单个模型所有分片成功后，才开始该模型评分。默认生成/评分最多4张GPU同时使用；每任务8小时、4CPU/64G，不保证排队时间或实际运行时间足够。首次提交前可设置 ALIGNMENT_SHARDS、ALIGNMENT_CONCURRENT、ALIGNMENT_PARTITION、ALIGNMENT_GRES、ALIGNMENT_TIME；并行上限随配置变化。check-only不提交，真实启动器禁止同标签重复提交。

作业失败时依赖评分等待，先定位修复，不删除已有结果，不改变源码混跑。相同源码配置和原分片编号可跳过已验证图片；启动器journal需人工审阅后定向补交失败分片，不要盲目重提全套。

## 输出与完整性

`$DATA_ROOT/outputs/align_20261109/images/<janus1b|llamagen>/<backbone|inchworm>/<dpg|tiif>/...` 保存PNG及配置/种子/回溯/延迟/显存凭证。

`$DATA_ROOT/outputs/align_20261109/scores/<model>/<benchmark>/summary.json` 保存该模型两方法总分。只有完整性检查、逐图评分全部通过才写status=complete；记录目录不是正式总分。生成源码哈希/manifest/评分配置绑定，防止恢复时混用。

快速检查状态：

```bash
squeue -u "$USER" -o "%.26i %.36j %.2t %.10M %R"
find "$DATA_ROOT/outputs/align_20261109/scores" -name summary.json -print
```

单独补交/普通已分配GPU下的评分入口（先source configs/local.sh）：

```bash
"$DATA_ROOT/envs/alignment-dpg-v1/bin/python" score_alignment.py --root "$PWD" --data-root "$DATA_ROOT" \
  --run "$DATA_ROOT/outputs/align_20261109" --model janus1b --benchmark dpg
"$DATA_ROOT/envs/alignment-tiif-v1/bin/python" score_alignment.py --root "$PWD" --data-root "$DATA_ROOT" \
  --run "$DATA_ROOT/outputs/align_20261109" --model janus1b --benchmark tiif
```

LlamaGen替换--model llamagen。禁止在登录节点执行模型生成/评分。CPU安装/数据测试与实际GPU运行结果分开记录；尚未取得端到端GPU成功证据。
