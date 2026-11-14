# Janus-Pro-1B 原生完整 GenEval 基线

30对展示图片、100张性能测试均不是完整GenEval分数。`backbone` 是独立版本，方法名 `janus1b_backbone`，调用原生采样分支，不调用任何Inchworm控制器。保留现有三消融的默认列表，原版Inchworm仍单独标识。

范围：官方553个提示词，每词4张，共2212张；384px、576图像tokens、CFG5、temperature1、BF16、batch1，种子42+4*prompt_id+sample_id，与现有1B消融的提示词和种子规则一致。GenEval按六类等权平均，不筛选图片、不把30张或100张子集作为完整分数。baseline本身仍执行既有Janus原生循环对齐检查，检查使用的图片不计入2212张。

入口 `scripts/submit_janus1b_backbone.sh --check-only` / `--submit`。复制已有local.sh，设置ASSETS_ROOT为已有benchmark源码目录，BASELINE_RUN_ROOT为新建的共享可写目录。本次为单独补齐的1B基线，允许临时使用project存放这一组图片、收据与评分（约数百MB到1GB级别，需核对project实际配额），不能据此搬入所有模型/数据。DATA_ROOT仍指向原环境和权重，无需重新下载。Slurm日志、临时缓存、图片、合并链接和评分均在BASELINE_RUN_ROOT。模型/评测器分别使用JANUS_PYTHON和GENEVAL_PYTHON。

一个GPU作业依次执行完整生成与完整评分，生成进程退出释放显存后才启动评分，不需再次申请评分作业；任一失败即停止并保留已完成图片和收据。默认时限8小时：此前L40S batch1测得平均约9秒/张，2212张生成粗估约5.5小时，另加I/O、加载和评分；实际吞吐与排队时间均不保证。可用BACKBONE_WALLTIME调整。

输出：BASELINE_RUN_ROOT/merged/evaluation/janus1b_backbone.summary.json。只有通过完整覆盖验证、输出553/2212和formal_complete_benchmark=true，才作为完整分数；日志最终标记JANUS1B_NATIVE_FULL_GENEVAL_OK。提交记录在logs/submission.txt，防重复提交。需要恢复时先检查原作业；原生成命令会验证并复用已完成收据，不能在同一目录修改源码或配置。新基线与旧性能/消融运行分目录存放。
