# 当前云端基线入口（2026-10-13）

项目 `/root/autodl-tmp/open_clip-main`；服务器 `autodl-westd`。
用户最新要求优先完成所有基线，暂不启动 Full、RAF 消融或共享教师学生。

执行入口：`src/open_clip_train/fer_baselines.py`。
单卡队列：`artifacts/training_prepare_20261013/run_baseline_queue.py`。
旧四阶段 runner 不能直接用于本轮官方 Vim 基线。

学生采用官方 hustvl/Vim Small bimambav2、384维、24层、patch16、middle CLS，
源码固定到 `dd0358ad1e42701f22afbefa0717cc8825cf9f45`。
没有提供官方学生预训练权重，因此准确标为官方架构、seed42随机初始化，
不声明已加载官方学生预训练参数。保留官方前/反向 dt 初始化策略。
Hybrid 使用同一骨干＋随机 ResNet18 至 layer3＋现有融合分类结构。

九个学生基线：三个数据集各 Mamba CE、Hybrid CE、logits-only KD；每个30epoch。
KD 依赖各自本域教师 T1(5epoch)→T2(10epoch)，共六个准备阶段。
CE没有教师、文本、feature、W或AU损失。KD只使用CE＋0.2×KL(teacher||student)，T=2。
T1使用七类AU原型CE，文本塔冻结，有效 logit_scale 上限100。
T2只微调CLIP视觉塔及FER头。教师按现有归档中实际QuickGELU激活加载。

输入256、RGB、CLIP normalize、bicubic；师生收到同一次增强的tensor。
ViT-L14网格18×18，右/下各4像素未被patch卷积覆盖。
训练直接读固定manifest并核验hash，不重新随机划分。
SFEW清理后763/88/402；RAF11040/1231/3068；AF255507/28393/3500。
只在AffectNet使用训练集逆频率、均值归一的CE类权重，所有对照一致。
模型选择使用selection UAR，准确率用于同分判定；T1按selection loss选择。
最终评估只在正式训练结束加载best后进行；不依据它调参。

学生 batch32/drop_last、AdamW lr1e-4、betas(.9,.98)、eps1e-6、wd.01。
教师视觉lr1e-5，新头/尺度lr1e-4。BF16 AMP，梯度checkpoint，clip-norm1。
学生LR统一10＋20epoch分段cosine，每段warmup5%；baseline不重置Adam状态，
后续Full如果阶段切换重置优化器，必须披露该剩余差异。
固定epoch/sample增强种子，shuffle使用独立seed42+epoch；保存CPU/CUDA/Python/NumPy RNG。

产物位于 `/root/autodl-fs/fer_suite_simplified_20261013`。
每个训练只保留best.pt、last.pt：best纯权重，last同步权重/optimizer/RNG/进度/LR状态。
每1000update及epoch末更新last，原子替换，不保留逐epoch副本。
教师只保存visual/head/scale；冻结文本从相同原始CLIP文件重建并核验SHA256。
不保存图片/整套教师特征/attention缓存；仅小型曲线、有效配置、指标与最终预测CSV。
工作进程4、CPU线程8，单GPU单训练进程，剩余存储不足12GiB时停止下一作业。

队列先对五种路径做实际batch32×2梯度和检查点roundtrip验收，全部通过才能正式运行。
重启同一队列会跳过已完成阶段、从未完成阶段last恢复，身份不匹配拒绝续训。

```bash
cd /root/autodl-tmp/open_clip-main
PYTHONPATH=src /root/miniconda3/envs/clip/bin/python artifacts/training_prepare_20261013/verify_official_vim.py
/root/miniconda3/envs/clip/bin/python -u artifacts/training_prepare_20261013/run_baseline_queue.py --accept-only
# 上面验收通过后，正式队列：
nohup /root/miniconda3/envs/clip/bin/python -u artifacts/training_prepare_20261013/run_baseline_queue.py > /root/autodl-fs/fer_suite_simplified_20261013/queue.log 2>&1 < /dev/null &
```

运行状态以 `queue_status.json` 和各作业 `status.json`/日志为准。
本文件是入口说明，不代表训练或GPU验收已经成功。
