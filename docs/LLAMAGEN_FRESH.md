# 独立重新部署 LlamaGen

目的：新环境、新官方代码，直接运行官方原生 demo，再运行正式完整基线；不把旧环境的依赖覆盖进新环境，不继续以多轮结构诊断替代实际部署。没有预先保证任何分数或图文一致性。

官方源码固定 ce98ec41803a74a90ce68c40ababa9eaeffeb4ec。官方仅要求torch>=2.1，没有发布完整锁版本；这里选择PyTorch2.1.2/cu121、Transformers4.36.2及requirements/llamagen-fresh.txt形成独立配方，不称为作者原始环境。使用Python3.10 venv、不继承旧site-packages。CPU任务安装环境并克隆官方源码；复核已固定来源的大文件哈希，以只读软链接复用T5/VQ/stage2，额外下载固定HF版本stage1_256约3.1GB。新环境、下载、源码、日志、图片全在FRESH_LLAMA_ROOT，约需额外10—15GB可用空间；需检查project当前配额，不能放回文件数超限的scratch。临时文件用节点短/tmp路径，任务退出清理；其他数据持久保留。

scripts/submit_llamagen_fresh.sh --submit deploy 提交CPU安装+依赖它的GPU官方出图两个任务。worker不使用Inchworm适配器，也不改官方示例脚本的四个提示词、采样或加载逻辑；运行256和512两套官方命令，在各自native_images/llamagen_fresh_stage{1,2}_{256,512}_backbone/sample_t2i.png保存四图网格。demo完成不等于质量验证通过，必须查看原生图；两个stage是不同检查点/分辨率，不得混为同一复现或只挑好图掩盖另一版问题。

确认对应原生图正常后，可用同一入口 --submit full256 或 full512 启动正式benchmark。每版553词×4图，默认32分片并发2、生成每片2小时、评分2小时；生成、评分各自独立进程，评分依赖本版本数组成功。run_llamagen_fresh.py调用官方GPT、T5、VQ、generate，使用官方CFG7.5/top-k1000/top-p1/temperature1/BF16和默认左padding；保留官方TF32/high设置，种子42+4*prompt_id+sample_id，batch1。与旧实现/旧TF32设置不同，需以新方法名、环境和源码记录独立汇报；不是新Inchworm实验。基线跑通后再单独接入已通过BF16一致性检查的Inchworm。

权重严格加载；所有提示词/图像覆盖、源码与环境指纹、收据/图片哈希均验证。阶段名称不同、目录独立。分数在runs/llamagen_fresh_stage{1,2}_{256,512}_backbone/merged/evaluation/对应方法名.summary.json。沿用已通过验证的独立GenEval环境和固定评测器；新生成环境无需再次安装旧MMCV。

CPU时限2小时、官方demo90分钟；不保证队列等待或实际完成时间。提交器有每阶段独立防重复记录，部分提交失败或安装失败时先检查记录和日志，不盲目删除新环境或重复提交。在单独新checkout中运行，不修改仍有待启动分片的其他源码目录。
