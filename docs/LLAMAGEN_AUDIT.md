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

## 文件校验通过后的单GPU运行对照

用户的15428523已通过全部文件及源码校验，不应重复下载同一权重。`scripts/llama_runtime_probe_worker.sh` 复用现有环境/权重，要求 `LLAMA_ASSET_REPORT` 指向该报告。执行 `tools/diagnose_llamagen_runtime.py`，在一个GPU作业内完成三组对照：默认T5 BF16＋GPT BF16；T5 FP32特征转BF16＋GPT BF16；T5/GPT全FP32。禁用TF32，不改CFG7.5/top-k1000/温度1、512分辨率或提示词；每组重设seed42。FP32均从原始磁盘权重重新加载，避免仅上转换BF16后冒充完整精度。

同时保存T5加载缺失/不匹配键、有效文本特征统计、提示词特征相似度，以及默认/slow tokenizer的实际token IDs；仅在这三条提示词的IDs/masks不同的情况下，额外生成slow-tokenizer BF16对照。该比较不能证明所有文本的tokenizer行为相同。默认三组共9张，必要时12张；VQ逐张解码以控制峰值显存。种子重设时点、TF32设定和VQ解码批次与旧demo有差异，不要求逐像素复现旧图；用于当前对照内部定位问题，不作性能或质量benchmark。

输出 `$DATA_ROOT/outputs/llamagen_runtime_<JOBID>/runtime_report.json` 及各case的 `grid.png`、00.png（bench）、01.png（apple）、02.png（dog）和图像tokens。结束标记 `LLAMAGEN_RUNTIME_PROBE_OK` 只表示运行结束，图片仍需核对。若高精度组恢复提示词对应关系，则再缩小差异、落实统一配置并重新验证适配器；若全部异常，继续核对实际加载与文本条件响应，不能以脚本成功作为修复成功。

## 三种精度都失败后的代码审计

用户报告15428919的所有precision case仍偏离提示词。已逐段对照官方sample、generate、GPT、训练数据集/训练脚本及T5提取器：当前GPT-XL/t2i、caption_dim2048、120条件token、512/16=32网格、1024图像token、CFG顺序/公式、左padding和VQ码本配置与官方路径一致。输入和权重文件一致不等于语义质量已经验证。当前stage2_512尚未建立与其他报告的checkpoint对应关系，不同checkpoint不能直接对比分数，也不能承诺换成stage1就修好。

找到一个**独立的适配器检查错误**：`check_native_prefixes` 曾将prefill也强制到math SDPA，实际原生generate的prefill使用默认后端，仅decode强制math。现已对齐这个调用策略，保持TV阈值0.01；回归测试用会因kernel策略产生不同分布的后端证明旧实现会误拒绝。此修改不改Inchworm算法，也不能解释不经过该检查的原生生成图片为何异常。

另一个**上游模型行为**：固定GPT源码的 `precompute_freqs_cis_2d` 文本前缀填入全零的cos/sin对，`apply_rotary_emb` 因而把文本Q/K置零；V及残差路径仍有文本内容，不能说模型完全没有文本条件。该实现与训练路径共用，上游[issue54](https://github.com/FoundationVision/LlamaGen/issues/54)有讨论，但不是已证实的本次图片错误根因。不能直接用单位旋转或普通RoPE替换后仍声称保持原预训练backbone。[issue41](https://github.com/FoundationVision/LlamaGen/issues/41)也报告512原生推理复现困难，目前该讨论没有给出已验证解决方案。

新增 `scripts/llama_structure_worker.sh`：复用通过hash校验的资产、同一环境与512配置，读取已有Janus红苹果图片作为参考，不再随机生成三组图。先用LlamaGen VQ做encode/decode和decode_code的重建对照，保存 `vq_roundtrip.png`；再在同一图像token前缀下测不同文本的概率响应与teacher-forced NLL，并用FP32比较原生KV缓存和完整前缀计算的概率。最后分别记录已修正adapter gate的FP32/BF16结果。保持原RoPE及预训练参数，未修改后端阈值。

设置 `LLAMA_REFERENCE_IMAGE` 可指定参考图片，默认 `$DATA_ROOT/outputs/janus_test/apple_seed42.png`。结果在 `$DATA_ROOT/outputs/llamagen_structure_<JOBID>/structure_report.json`。`LLAMAGEN_STRUCTURE_DIAGNOSTIC_COMPLETED` 仅表示各诊断已执行；其中adapter_gate可明确为failed，不得把脚本完成误认为适配器通过。单张参考图的NLL排序也不能作为正式质量通过标准。该报告用于把VQ重建、原生cache、文本条件响应、adapter四条链路区分定位。
