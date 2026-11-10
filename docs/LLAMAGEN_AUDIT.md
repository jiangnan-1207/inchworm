# LlamaGen stage2 原生生成诊断

触发条件：GenEval 553×4完成但Overall=0.0911198，`a photo of a bench`出现建筑/乱码图；另有Inchworm cache概率检查t=0失败。单张坏图和低分不能独立证明权重或文本编码损坏，也不能说明Inchworm造成基线低分。

官方[README](https://github.com/FoundationVision/LlamaGen/tree/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec)区分stage1_256和stage2_512。当前任务为stage2_512；未经核查不能与其他checkpoint的文献分数等同比较，更不能换权重后混入旧目录。

`tools/check_llamagen_native.py`执行固定官方sample_t2i.py，只替换示例的硬编码提示词和保存位置，不改其模型加载、FLAN-T5、左padding、native_generate、VQ解码。三个提示词为bench、red apple、dog，CFG7.5/top-k1000/BF16/seed42、batch3。batch与初始化流程和早先batch1不同，因此不要求图片逐像素相同。此项诊断不进入任何benchmark计数。

使用新checkout，复用已有configs/local.sh、生成环境和权重，不改正在运行的旧任务。导出REPO_ROOT、CONFIG_FILE、LLAMA_SOURCE_ROOT（现有固定源码目录），通过sbatch执行scripts/llama_native_audit_worker.sh，4CPU/64G/1GPU即可。语法和参数入口已检查，真实GPU结果尚未获得。

输出：`$DATA_ROOT/outputs/llamagen_native_audit_<JOBID>/00.png`、01.png、02.png，以及official_grid.png和diagnostic.json。日志结束标记LLAMAGEN_OFFICIAL_NATIVE_AUDIT_OK只说明脚本完成，不是质量合格证明。

若原生结果也持续偏离简单提示词，再核对权重哈希、环境、stage1/stage2差异；若原生结果合理，再进一步对齐项目生成链。无论哪种情况，保留旧0.0911和所有原图记录；在适配器一致性检查通过前，不重启LlamaGen的大规模Inchworm评测。
