# RAF-DB 报告后的 E3 / CNN 融合修改

本轮根据用户粘贴的 RAF-DB 验证分析修改代码。没有读取原始 validation_report.md 或云端检查点；其中分数是用户提供的诊断背景，不是本轮重新测量的结果。代码修改与推送不启动或恢复任何云端训练。

## 优先路径

从 E3（VMamba-T + 单尺度 CNN + 单 Query、无 SASF）出发。保留已有 Cross-Attention、融合投影、主分支残差以及原四路分类头，768 维接口和已有 CE/KD 逻辑不变。CNN 原本已有注意力加权 global descriptor 拼接；本轮新增的是**不经过空间权重的独立 CNN 直通残差**，没有把原 E3 描述为完全没有 CNN 旁路。

新增 `CNNResidualAux`：

```text
CNN layer3 feature → cnn_projection → raw spatial tokens [B,N,768]
                                     ├→ existing spatial attention → existing E3 fusion
                                     └→ mean pool → LayerNorm → independent CNN global
                                                            ├→ training auxiliary Linear(768,7)
                                                            └→ sigmoid(gate_logit) × global
                                                                 ↓
                   old E3 fused_raw + CNN residual → LayerNorm → new fused_raw
```

该独立通路在空间注意力、跨分支注意力和 K/V dropout 之前取数，辅助 CE 单独反传时只到 CNN、CNN 投影、独立 LayerNorm 和辅助头，不到 VMamba 或注意力权重。门控 logit 初始化为 log(.2/.8)，初始 gate=.2。辅助头存在时只在训练或显式辅助评估中执行；普通学生推理不执行辅助头，也不读取文本或教师资产。检查点仍保留辅助头参数，确保完整恢复训练。

**仅加辅助头时不加最终 LayerNorm，也不改主 logits 的数据流。** 这样 R1 不会混入残差或额外融合归一化因素。R2 则按建议公式 `LN(old_fused + sigmoid(gate)*cnn_global)` 增加残差与其归一化；R3 同时启用两者。

原 E3/E4/F 及历史配方仍可运行，旧 checkpoint 的模型 state_dict 结构不因关闭的新模块而增加参数。代码变更后旧训练身份/plan.lock 不能直接冒充新运行身份；不要把旧检查点作为新结构的严格 resume。原 F_recommended 名称保留为历史实验，不再表示本轮优先选择。

## R0–R6

| 名称 | 结构/损失 | 主要对照 |
| --- | --- | --- |
| R0 | E3 的别名，普通 CE，无新模块 | 原始起点；目录实际名 E3 |
| R1 | E3 + CNN 辅助头，main CE + .2 aux CE | R0 |
| R2 | E3 + CNN 门控残差，main CE | R0 |
| R3 | E3 + 辅助头 + 门控残差，main CE + .2 aux CE | R1，另可与 R2 对照 |
| R4 | 所选 R0/R1/R2/R3 + Class-Balanced main CE | 所选基准 |
| R5 | 所选基准 + .05 简单七类文本原型 CE | 所选基准 |
| R6 | 所选基准 + middle SASF | 所选基准 |

R1–R3 不启用 SASF、多尺度 CNN、多 Query、KD 或文本。R4/R5/R6 各自独立从 `--refinement-base` 指定的结构派生，不逐步叠加 R4→R5→R6。默认 base=R3 仅是明确记录的候选配置，不表示已经证明 R3 最好；应根据完整选择集结果指定基准后生成新计划。

R4 的 beta=.999。权重使用**训练集**七类计数：`(1-beta)/(1-beta**count)`，再除以七类权重均值。稳定实现用 float64 的 expm1；不使用选择集计数，不同时启用平衡采样。仅主 CE 使用该权重，若所选基准有辅助头，aux CE 仍是未加权 CE，系数仍为 .2。实际权重写入阶段 effective_config.json；对应代码和配置均纳入身份校验。

R5 继续计算全部七类原型 logits 后做 CE，不按真实标签提前选唯一文本作为输入；辅助文本仍为独立因素。R6 才重新打开中间层 SASF。原型文件沿用已有 `prepare-text` 和 SHA/类别顺序校验。

原 24 项目录保留，新增 R1–R6 后每数据集每 seed 为 30 个唯一实验；R0 和 E3 不重复训练。不要因为目录支持这些配置就默认全部启动。

## 生成计划与三个随机种子

继续用同一代码入口、相同 frozen manifest、输入预处理、batch32、正式 epoch 预算（RAF60/SFEW50/AffectNet30）、选择集 UAR 优先策略。此处没有把报告中的三轮诊断改成正式训练预算，也没有拿测试集 85.85% 与三轮选择集分数比较。

先仅生成 R0–R3：

```bash
python -m vim_v2 prepare-experiments --paths paths.experiments.json --output configs/raf_cnn_r --datasets rafdb --experiments R0 R1 R2 R3 --seeds 42 43 44
```

每个 seed 对应独立 run_id、输出目录和初始化身份。seed42 保持已有路径命名（如 rafdb/R1），seed43/44 使用 rafdb/R1_seed43 / rafdb/R1_seed44。随机头部种子为 seed+1000；几何增强、训练采样、诊断初始化和断点恢复均使用对应 seed。**数据划分本身保持原 frozen manifest，不因训练 seed 改变。** 同一个 seed 下的不同结构，共同头部参数按模块名固定初始化，预训练骨干保持同源全量加载。

根据结果，例如选择 R1 后再生成后续独立对照；换一个新的 output_root 和配置目录，保持实验身份可追溯：

```bash
python -m vim_v2 prepare-experiments --paths paths.followup.json --output configs/raf_cnn_followup --datasets rafdb --experiments R1 R4 R5 R6 --refinement-base R1 --seeds 42 43 44
```

验证、预算、锁定、运行、恢复与评估继续使用现有命令：

```bash
python -m vim_v2 validate-plan --plan configs/raf_cnn_r/plan.json --dry-run
python -m vim_v2 verify --plan configs/raf_cnn_r/plan.json --gpu
python -m vim_v2 budget --plan configs/raf_cnn_r/plan.json
python -m vim_v2 lock-plan --plan configs/raf_cnn_r/plan.json
# 以下实际训练命令仅在用户另行授权云端训练后执行
python -m vim_v2 run-plan --plan configs/raf_cnn_r/plan.json --resume-incomplete
```

新 group 为 refinement（R1/R2/R3）和 refinement_followup（R4/R5/R6）；R0/E3 仍属于 architecture。若计划只含 R0–R3，默认 core 可运行全部所选项；只用 group=refinement 会跳过 R0。

## 日志与诊断

- 每次更新已有 main/aux 加权 loss；每轮增加 cnn_aux_train_metrics 与选择集 cnn_aux_metrics（ACC/UAR/Macro-F1、七类 Recall/F1、混淆矩阵）。主 checkpoint 仍按主头指标选择，不按辅助头挑选。
- 既有首/末 batch 和 smoke 监测点在 backward、AMP unscale 后、梯度裁剪前记录 CNN/VMamba/SASF/辅助头整体 L2 梯度范数；同时记录参数总量、可训练参数量、trainable tensor 数和 grad=None 数。不能仅凭不同分支绝对梯度大小比较贡献。
- 分支输出记录 cnn_gate、cnn_independent_rms、cross_attn_norm、fused_feat_norm，以及 SASF gate、增强前后空间 RMS 差、SASF 全局残差相对于原 Mamba global 的范数比。
- no_cnn 同时切断旧 CNN descriptor/注意力更新和**新增残差路径**；no_mamba 保持清除三层分类旁路；no_sasf 禁用相应增强。模型训练状态与权重在诊断中不被修改。
- 干预报告增加逐样本 sample_id、真实标签、full/干预预测、logits/probability 最大与平均变化；汇总 changed_predictions 和全局差异。可区分置信度变化与决策变化。
- 联合打乱 K/V 仍是顺序不变性对照；仅打乱 V 是内容对应干预，不能直接当作五官定位证据。置零干预会改变分布，敏感性不等于独立模型精度或因果贡献的精确估计。

示例：

```bash
python -m vim_v2 diagnose-branches --config configs/raf_cnn_r/rafdb__R3.json --device cuda
python -m vim_v2 evaluate --config configs/raf_cnn_r/rafdb__R3.json --plan configs/raf_cnn_r/plan.json --split selection_val
python -m vim_v2 aggregate --plan configs/raf_cnn_r/plan.json
```

aggregate 配对同一 seed 的模型，额外报告计划/完成 seed 列表及跨 seed 均值、样本标准差；未完成的 seed 不填作零分，也不声称三 seed 足以建立统计显著性。

## 配对 Bootstrap

对保存的选择集逐样本预测做 class-stratified paired percentile bootstrap：每类内部有放回抽样，同一组图像索引同时用于两个模型。输出 candidate−baseline 的 ACC/UAR/Macro-F1 差值（百分点）及 95% percentile 区间。

```bash
python -m vim_v2 paired-bootstrap --baseline /absolute/path/rafdb/E3/selection_predictions.csv --candidate /absolute/path/rafdb/R3/selection_predictions.csv --output /absolute/path/paired_R3_E3.json --resamples 2000 --seed 42
```

读取时校验概率、argmax、一一对应的 sample_id 和真实标签，不依赖两文件行顺序。该区间以已训练模型及观察到的类别计数为条件，不包含训练 seed 不确定性，也没有多重比较修正。只有结构和超参数敲定后再评估独立 final_eval；final_predictions.csv 需要显式 --split final_eval，不冒充选择集结果。

## 验证边界

本轮代码测试包括旧配置回归、R0–R6/多 seed 配置生成、辅助头独立梯度、R1 主路径与 E3 精确一致、残差公式/gate=.2、Class-Balanced CE 数值、干预新增旁路、逐样本差异、辅助指标、Bootstrap 输入校验，以及真实官方 VMamba R3 的反向传播和无辅助推理检查点重建。

本地最终回归：106 项通过（119.80 秒）。68 条警告来自第三方 CPU fallback、旧 AMP 接口与 checkpoint 提示；没有测试失败。

没有执行云端训练、CUDA/BF16 batch32 验收或新的准确率测量。收益必须通过后续完整训练验证，不能把新增模块的正确计算等同于 UAR 已提升。
