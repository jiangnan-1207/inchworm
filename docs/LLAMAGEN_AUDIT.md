# LlamaGen stage2 原生生成诊断

触发条件：GenEval 553×4完成但Overall=0.0911198，`a photo of a bench`出现建筑/乱码图；另有Inchworm cache概率检查t=0失败。单张坏图和低分不能独立证明权重或文本编码损坏，也不能说明Inchworm造成基线低分。

官方[README](https://github.com/FoundationVision/LlamaGen/tree/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec)区分stage1_256和stage2_512。当前任务为stage2_512；未经核查不能与其他checkpoint的文献分数等同比较，更不能换权重后混入旧目录。

`tools/check_llamagen_native.py`执行固定官方sample_t2i.py，只替换示例的硬编码提示词和保存位置，不改其模型加载、FLAN-T5、左padding、native_generate、VQ解码。三个提示词为bench、red apple、dog，CFG7.5/top-k1000/BF16/seed42、batch3。batch与初始化流程和早先batch1不同，因此不要求图片逐像素相同。此项诊断不进入任何benchmark计数。

使用新checkout，复用已有configs/local.sh、生成环境和权重，不改正在运行的旧任务。导出REPO_ROOT、CONFIG_FILE、LLAMA_SOURCE_ROOT（现有固定源码目录），通过sbatch执行scripts/llama_native_audit_worker.sh，4CPU/64G/1GPU即可。语法和参数入口已检查，真实GPU结果尚未获得。

输出：`$DATA_ROOT/outputs/llamagen_native_audit_<JOBID>/00.png`、01.png、02.png，以及official_grid.png和diagnostic.json。日志结束标记LLAMAGEN_OFFICIAL_NATIVE_AUDIT_OK只说明脚本完成，不是质量合格证明。

若原生结果也持续偏离简单提示词，再核对权重哈希、环境、stage1/stage2差异；若原生结果合理，再进一步对齐项目生成链。无论哪种情况，保留旧0.0911和所有原图记录；在适配器一致性检查通过前，不重启LlamaGen的大规模Inchworm评测。

## 三条简单提示词均偏离时：先核对实际文件

2026-11-10的15420761作业使用官方原生stage2脚本，用户提供的三图仍未对应长椅、苹果和狗。这证明故障不只存在于Inchworm分支；还不能只凭图片断定哪个权重或软件组件错误。

新增 `tools/audit_llamagen_assets.py` 和 `scripts/llama_asset_audit_worker.sh`：在CPU计算节点逐个流式读取GPT、VQ、T5权重及tokenizer，与 `configs/llama_asset_fingerprints.json` 固定官方版本元数据比对；LFS文件使用内容SHA256，普通文件使用带header的Git blob SHA1。清单来自HF的 `model API ?blobs=true`，revision与现有下载配置一致。检查实际存在的两种权重格式，防止额外旧格式被Transformers优先加载；索引引用的分片必须齐全。记录环境版本、模块路径及官方源码HEAD/工作区状态，不加载模型、不申请GPU、不联网、不安装或改写权重。

使用新checkout和已有local.sh，导出REPO_ROOT、LLAMA_SOURCE_ROOT，然后以cpu-small/2CPU/8G/30分钟提交该worker。日志末尾应是 `LLAMAGEN_ASSET_HASHES_AND_SOURCE_OK`；结果在 `$DATA_ROOT/outputs/llamagen_asset_audit_<JOBID>.json`。通过仅代表文件与固定来源一致，不代表质量问题修好了。

后续依据报告处理：有hash/size/missing失败时，在独立目录重新获取对应固定文件并复验，不覆盖正在使用的共享权重；全部通过时，进一步查文本特征、加载完整性和数值后端，使用同一组简单提示词做受控对照，不能靠挑选种子或直接调整Inchworm掩盖基础链路问题。原版适配器的概率一致性失败仍需单独修复，禁止单纯放宽阈值。
