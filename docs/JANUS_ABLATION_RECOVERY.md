# 恢复被依赖链阻塞的两个 Janus-Pro-1B 消融

适用于 guard_temperature、guard_gap 两个生成数组尚未开始，而前一版本评分已失败的情况。入口为 scripts/submit_janus_ablation_recovery.sh。必须设置 ASSETS_ROOT（原有 benchmark 仓库）和 ABLATION_RUN_ROOT（新共享输出目录）；配置继续读取 configs/local.sh，DATA_ROOT 只用于已有环境与权重。先 --check-only，再 --submit。旧提交器保持原样，不要重新运行旧提交器。

这两个版本独立提交，评分仅 afterok 依赖本版本的生成数组。默认每版本32分片、最多2个生成工作进程，两个版本合计最多4张GPU（一个评分器可能替换已结束的另一版本生成工作进程）。单生成工作进程时限2小时，评分30分钟；可用 RECOVERY_SHARDS、RECOVERY_CONCURRENT、RECOVERY_WALLTIME 调整。不保证排队或总完成时间。

每版本仍为553提示词×4图，共2212张；版本定义、种子、模型、分辨率与原消融一致：guard_temperature=保底+4个高H位置+自适应温度；guard_gap=保底+2高H/2高G+原替代token规则，不包含自适应温度。沿用原两阶段筛选N=2和严格降分接受规则。新运行记录当前源码哈希，不能与旧运行收据混合。

Slurm日志、临时文件、可写缓存、图片、收据、合并链接和评分全部放 ABLATION_RUN_ROOT；不复制权重、不删除或改动 scratch 中已完成的保底版图片。project 是这两组恢复任务的临时输出位置，需有数GB空间，后续应归档；不能据此搬入所有数据。原任务的日志目录及 DATA_ROOT/tmp 必须已经存在（common.sh 沿用现有目录）。

先确认新提交四个编号均已返回并保存在 logs/submissions.txt，再取消四个旧的阻塞任务。脚本不会自动取消旧任务或其他用户作业。若提交中途失败，检查日志记录的已提交任务，不要盲目重跑或删除提交记录。

最终汇总：merged/evaluation/janus1b_guard_temperature.summary.json 和 janus1b_guard_gap.summary.json。完整覆盖验证必须通过，汇总需有formal_complete_benchmark=true、prompts=553、images=2212；overall 为六类等权平均分。
