# 一份代码运行 CNN–VMamba–SASF 实验

日期：2026-10-17；分支：dev。基于用户本轮提供的最终分析实现。本文的实验名 T1/T2/T3 是学生对照编号，教师阶段另用小写 `t1`/`t2`，两者不要混淆。

## 实现范围

所有学生实验使用同一组装入口 `make_student`、同一训练循环、同一损失入口 `objectives`、同一断点恢复和评估代码；通过 schema_version=3 配置声明差异。现有 schema_version=2 的历史计划保持兼容。

骨干：官方 Vim Small（仅作为 E1/E2 对照）、官方 VMamba-T s1l8、CNN-only。不会把本地自写 SSM 作为官方骨干替代。Vim 与 VMamba 的容量和预训练来源不同，因此 E1/E3 是骨干方案对照，不能单独解释为 SS2D 算子的净贡献。

学生结构支持：

- SASF 开关及 e/m/l 位置。VMamba 在 256px 下对应 192×32×32、384×16×16、768×8×8；`S_middle` 使用 16×16 的中间阶段。
- 单尺度 CNN 保留原 ResNet18 through layer3 的 256×16×16 输出。多尺度融合 layer2 的 128×32×32 和 layer3 的 256×16×16：先对中层做 1×1 通道投影，将深层上采样后等权融合；再投影到 768 维并形成 1024 个 K/V tokens，不提前全部池化。
- 单 Query 或三 Query。三 Query 分别来自 e/m/l 的 mean+max 投影，SASF 残差加到对应 Query；注意力输出取均值，同时保留原全局聚合表示的残差。
- 768 维 fused_raw 接口保持一致。分类头继续读取三阶段表征与 fused_raw；CNN-only 使用自己的线性头，不实例化任何 SSM；VMamba-only 不实例化 CNN。
- 可选训练时 K/V token dropout=.3。它只删除注意力输入的 CNN tokens，不破坏 VMamba 二维网格，也不删除 SASF 的输入。它与旧设计中“骨干输入 patch dropout=.3”不是同一种干预，不能混称。
- `SASF` 在均值池化之前操作真实空间特征，其输出通过残差 Query 进入融合和分类。门控 alpha=0 初始化，第一步只学习门控，后续卷积开始获得非零梯度。

## 实验目录

每个数据集 24 个唯一训练配置；三个数据集全部选择时共 72 个。默认生成 RAF-DB 的配置，不自动训练。并不建议必须一次跑完所有配置，可先选择 E3/E4。

| 实验 | 改动 | 对照 |
| --- | --- | --- |
| E1 | Vim + CNN，无 SASF，CE | 起点 |
| E2 | E1 + late SASF | E1 |
| E3 | VMamba + CNN，无 SASF，CE | E1；骨干与权重来源同时变化 |
| E4 | E3 + late SASF | E3 |
| B_cnn | CNN-only，CE | E3 |
| B_vmamba | VMamba-only，CE | E3 |
| S_early | SASF 移到 early | E4 |
| S_middle | SASF 移到 middle | E4 |
| C_multiscale | CNN layer2/layer3 空间融合 | E4 |
| Q_multi | 三个 Query | E4 |
| F_recommended | middle SASF + 多尺度 CNN，单 Query，CE | S_middle；另可对照 C_multiscale |
| T1 | 推荐结构 + CE + logits KD(.2) | F_recommended |
| T2_w0p05 / T2_w0p1 / T2_w0p2 | T1 + 简单文本原型 CE(.05/.1/.2) | T1 |
| T3_w0p05 / T3_w0p1 / T3_w0p2 | T1 + 区别性描述原型 CE(.05/.1/.2) | T1；相同权重时还应对照 T2 |
| D_feature | 推荐结构 + CE + feature KD(.2) | F_recommended |
| D_two_stage | S1 feature KD；S2 CE + logits KD(.2) | T1；阶段/教师/监督预算共同变化 |
| D_affinity | T1 + batch pairwise cosine affinity(.1)，不加文本 | T1 |
| A_patchdrop | 推荐结构 + K/V dropout(.3)，CE | F_recommended |
| A_balanced_sampler | 推荐结构 + 训练集逆频率采样，CE | F_recommended |
| A_weighted_ce | 推荐结构 + 训练集逆频率加权 CE | F_recommended |

别名 `D_ce` 指 F_recommended、`B_hybrid` 指 E3、`T2` 指 T2_w0p1、`T3` 指 T3_w0p1；同时选择别名和原名只会生成一次。别名不创建额外目录。text-only、affinity-only、KD-only 因素不默认同时叠加。

F_recommended 是组合方案，不把它与 E4 的差值作为单因素证据。D_two_stage 的总 epoch 与单阶段相同，S1 占用总预算，不额外增加 epoch：RAF 10+50、SFEW 5+45、AffectNet 5+25。

## 损失与公平性

- S1：仅 feature KD = cosine distance + **原始 768 维投影空间** MSE，两项系数均为 1。MSE 与 cosine 的量纲不同；日志分项记录加权值和梯度范数，不能单看原始 loss 值调参。
- S2：按实验配方开启普通 CE、teacher→student logits KL（T=2，乘 T²）、feature KD、text prototype CE 或 affinity。关闭项不读取相应教师／文本输入。
- Affinity 是 batch 内归一化视觉特征的两两 cosine 矩阵、去对角线后 MSE；这是明确实现的候选损失，未声称复刻未提供的历史 Affinity 公式。
- 文本损失为 `CE(10 * normalize(fused_raw) @ prototypes.T, labels)`。只与七类原型比较，不把同批同类图像互当负样本；无需恢复局部 AU loss。
- 原型由冻结原始 CLIP ViT-L/14 文本塔生成：按类取各 prompt 的**原始投影 embedding 均值，再归一化**。simple 和 distinctive 两份候选描述都带来源标记。distinctive 为常见可能表情描述，不是经过标注的逐图 AU 真值，不保证每幅图包含全部特征。
- 文本原型文件校验文件 SHA、类别顺序、原始 CLIP SHA、prompt 文件 SHA。推理仅加载学生权重，不读取原型、教师或预训练文件。
- 同一结构的 CE/KD/text 实验共享同一初始化快照；新增可选模块也不会改变公共头部参数的初始化随机序列。共享 head 的随机种子按模块名字固定，backbone/CNN 预训练张量不被头部初始化覆盖。
- 保持现有 frozen manifest、train/selection_val/final_eval，seed42，batch32，输入几何及 ImageNet/CLIP 两套归一化。数据增强先共享几何，学生两视觉分支使用同一张已归一化输入。
- 只对训练样本计数做类别平衡；selection_val/final_eval 不平衡采样。选择标准仍为 UAR、ACC、较早 epoch；所有实验汇总逐类 Recall/F1、混淆矩阵。

## 配置与命令

从仓库根目录运行，先安装正常项目依赖和 requirements-vmamba.txt。根据云电脑实际文件修改 `configs/vision_mamba_clip/experiment_paths.example.json`，保存为自己的 paths.experiments.json。示例中的路径、CLIP/manifest/teacher SHA 与 train=0 是明确占位，不能直接正式运行；SFEW/AffectNet 需按同一结构增加各自数据条目。

保留既有冻结数据划分与同域教师。T2 配置额外保留 T1 来源权重校验信息，但不会重复训练教师。新实验必须使用全新的 output_root，不覆盖历史 Vim/VMamba 结果。

查看实验并先生成小规模对照：

```bash
python -m vim_v2 list-experiments
python -m vim_v2 prepare-experiments --paths paths.experiments.json --output configs/fer_suite --datasets rafdb --experiments E3 E4
```

生成完整目录（只生成配置，不开跑）：

```bash
python -m vim_v2 prepare-experiments --paths paths.experiments.json --output configs/fer_suite --datasets rafdb sfew affectnet
```

文本实验先准备冻结原型，输出的 text_prototypes.json 包含两个真实 SHA，应将其字典填回 paths.experiments.json，再生成新计划：

```bash
python -m vim_v2 prepare-text --paths paths.experiments.json --output /absolute/cloud/path/text_assets --device cpu
python -m vim_v2 prepare-experiments --paths paths.experiments.json --output configs/fer_text --datasets rafdb --experiments D_ce T1 T2 T3
```

上述原型准备读取原始 CLIP 权重，不微调教师。普通 CE 实验不需要教师或原型；文本实验要求原型资产；KD 要求对应教师及完成标记。

资产和 GPU 验收沿用同一入口：

```bash
python -m vim_v2 validate-plan --plan configs/fer_suite/plan.json --dry-run
python -m vim_v2 verify --plan configs/fer_suite/plan.json --gpu
python -m vim_v2 budget --plan configs/fer_suite/plan.json
python -m vim_v2 lock-plan --plan configs/fer_suite/plan.json
```

`verify --gpu` 会做实际数据检查、GPU smoke、过拟合诊断、初始化与恢复验收；不是纯检查耗时的轻量命令。文本配置还会运行 teacher/text 语义诊断。`accepted=true` 在该语义报告中仅表示资产验证且诊断成功执行，不代表“语义一定兼容”或自动筛选出较优 prompt。

只有在用户另行授权云端训练后，再运行：

```bash
python -m vim_v2 run-plan --plan configs/fer_suite/plan.json --resume-incomplete
python -m vim_v2 run-plan --plan configs/fer_suite/plan.json --group architecture --resume-incomplete
python -m vim_v2 resume --config configs/fer_suite/rafdb__E4.json --plan configs/fer_suite/plan.json
```

可选 group：architecture、branches、combined、supervision、text、augmentation、imbalance；默认 core 表示所选计划全部配置。单 GPU 串行，已完成或训练完等待最终评估的任务会校验完成身份后跳过；不自动调整失败实验参数。

评估与汇总：

```bash
python -m vim_v2 evaluate --config configs/fer_suite/rafdb__E4.json --plan configs/fer_suite/plan.json --split selection_val
python -m vim_v2 evaluate --config configs/fer_suite/rafdb__E4.json --plan configs/fer_suite/plan.json --split final_eval
python -m vim_v2 aggregate --plan configs/fer_suite/plan.json
```

输出包括每任务 metrics/predictions、reports/results.csv、results.json、per_class.csv。SASF 位置、文本权重和结构选择应使用 selection_val，final_eval 仅作最终报告，不据最终集反复调参。

## 诊断

训练日志定期记录 Mamba/CNN/SASF/注意力输出 RMS、SASF 门控、新参数梯度和实际更新。每 1000 update 及第 0 个 update，记录**同一个 fusion_projection.weight 上**各加权损失的梯度范数及两两 cosine；这是共享融合层局部诊断，不等价于整个网络的梯度夹角。

只读的 selection_val 诊断命令：

```bash
python -m vim_v2 diagnose-branches --config configs/fer_suite/rafdb__E4.json --device cuda
python -m vim_v2 audit-semantics --config configs/fer_text/rafdb__T3_w0p1.json --teacher-stage t2 --device cuda
```

分支诊断在同一模型上测 full、no_cnn、no_mamba、no_sasf、shuffle_cnn、shuffle_cnn_values。no_mamba 同时清除分类头的三层直通输入，no_cnn 同时清除 CNN global descriptor 和注意力更新；不是只遮掉一条仍有旁路的张量。

**联合打乱 K/V 是顺序不变性对照**：当前注意力没有位置编码，联合置换不改变数学结果，不能由此判定 CNN 无用。shuffle_cnn_values 只打乱 V 相对 K 的内容对应，用来测试注意力依赖；它也不是五官位置的直接定位验证。遮断会改变特征分布，分数变化不能直接当作因果贡献的精确估计。

语义诊断同时给出教师分类头与文本原型分类的 ACC/UAR/F1、原型相似度矩阵、真实类与最近错误类 cosine margin 和逐类 margin。诊断不修改权重，不从 final_eval 自动选择 prompt。

分析文字中 3×3、dilation=3 的有效跨度应为 `(3-1)*3+1=7`，不是 9；选择中间层的理由是保留细节，并不能靠一个核跨度公式证明 late SASF 无效。

## 本地验证边界

真实 VMamba SS2D、所有 VMamba/CNN 目录变体、768 维接口、多尺度梯度、三 Query、文本/KD 数学、同类原型监督、干预路径、采样和恢复均有 CPU 测试。

本轮完整测试命令（85 项通过、0 项失败）：

```bash
python -m pytest tests/test_experiment_suite.py tests/test_vmamba.py tests/test_sasf.py tests/test_vision_mamba_clip.py tests_vim_v2/test_contracts.py -q
```

本机缺少官方 Vim CUDA 扩展，所以 E1/E2 本轮只验证了**明确标为 stub 的 token 接口和统一组装路径**；正式实验必须在云端通过官方 Vim 预训练全量加载和 GPU 检查。没有把它报告为真实 Vim 核心验收。

未启动云端训练、未完成 CUDA/BF16 batch32 验收、没有重新测量准确率。24 配置/数据集是代码支持范围，不代表授权重新启动所有历史暂停任务。
