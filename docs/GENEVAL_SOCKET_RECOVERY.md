# 修复评分进程通信路径过长

日志明确显示：两组图片已经 FULL_COVERAGE_OK 553/2212，随后在 preflight 的颜色分类过程中，Torch 通过 multiprocessing.resource_sharer 共享 tensor，Listener 创建 Unix socket 时抛出 AF_UNIX path too long。后台队列线程失败后，主进程等待数据直到 Slurm 超时。不能通过增加时限单独解决此错误。

evaluate_benchmark.main 在预检与正式评分各自的独立进程中、导入 Torch 和建立工作进程前，创建 /tmp/iw-随机短目录，同时设置 TMPDIR/TMP/TEMP 和 Python tempfile.tempdir。只把小型临时文件及进程通信端点放在节点本地 /tmp；图片、收据、评分、权重和下载缓存仍保留原共享目录。正常退出自动清理本次临时目录；不改上游评测阈值、不关闭或替换评分模型。

必须克隆到独立评分 checkout，不要更新正在运行或尚有待启动生成分片的源码目录：生成收据绑定全部顶层 Python 文件哈希，原目录更新会导致不同分片的源码记录不一致。已完成图片不需要重新生成。

入口 scripts/submit_janus_scores_only.sh --check-only / --submit，必需 ASSETS_ROOT、BASELINE_RUN_ROOT、ABLATION_RUN_ROOT、SCORE_LOG_ROOT、GAP_GENERATION_JOB。三个评分作业，默认每个1 GPU、2小时，前两组无依赖，高H高G评分仅 afterok 依赖既有生成数组。先取消旧高H高G评分任务，保留其生成数组，再提交新评分；不得同时给同一输出目录运行两个评分进程。任何作业提交失败都要先核对日志记录，不要盲目重复运行。

生成目录及 run_spec 均只读验证；最终评分仍放对应运行目录 merged/evaluation，文件名不变。preflight 和正式评分日志应出现 GENEVAL_SHORT_TMP /tmp/iw-...。这只验证了针对当前异常的修复，实际服务器上的完整评分结果仍须等待作业执行。
