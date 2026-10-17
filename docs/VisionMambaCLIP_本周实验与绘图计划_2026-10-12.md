# VisionMambaCLIP 本周实验、绘图与写作执行计划

编制时间：2026-10-12（周一），北京时间。资源：云服务器单张 vGPU，32GB 显存，全天可用。硬截止：**10 月 17 日（周六）22:00 完成实验、图表和论文可交付版本**；10 月 18 日（周日）仅留复查/故障缓冲，以满足“周天之前做完”。

本计划基于当前源码、两篇用户 PDF 和实际本地数据审计。计划不等于实验已经运行，所有结果数值待真实实验填入。本次只编制计划和审计记录，未修改原始 PDF、删除数据或启动云训练。

## 1. 本周交付的研究问题

论文主线应回答四件事：

1. 在相同 FER 数据、初始化和验证协议下，CLIP 教师知识能否提高双向 Vim 学生的分类与语义对齐能力？
2. CNN 局部分支、S1、logit 蒸馏、W-Align、AU 空间项分别是否有贡献？
3. AU 描述相对类别名称，是否带来可测量收益？
4. 学生在相同设备、精度和分辨率下，实际部署成本如何？

本周必交：**干净 RAF-DB 协议 + 完整四阶段主模型 + 两个 CE 基线 + 六项必要消融 + 成本测量 + 7 张主图 + 4 张必需主表（跨域完成后增加第5张）+ 实验与结论正文**。完整模型争取 3 个学生随机种子；预算不够时先完成所有必要对照，再补重复。不预设必须达到 92.3%，也不把结果未到目标当成失败。

扩展优先级：SFEW/可用目标域的跨数据集评估，随后 SFEW 域内四阶段，最后 AffectNet 全量域内实验。若坚持三数据集完整训练，必须先按服务器实测计算预算；它不能仅凭 32GB 显存承诺在本周完成。缺少某数据集的正式结果时，正文缩小相应结论。

## 2. 阅读两篇论文后的结构安排

当前稿为 25 页 PDF：方法部分已有完整蒸馏公式，但实验尚含占位符和未核实数字。旧稿为 12 页 PDF：实验顺序是数据集、实现细节、对比、消融、跨域、定性分析、局限性。

建议沿用旧稿的章节组织，替换成当前方法的证据：

| 当前稿建议章节 | 需要的内容 | 本周产物 |
|---|---|---|
| 4.1 Datasets and Protocols | 真实类别映射、训练/选择/最终评估 split、去重、版本 | 表 1，数据审计清单 |
| 4.2 Implementation Details | T1/T2/S1/S2、各阶段冻结、真实初始化、温度/系数/步数/硬件 | 配置表、图 1(b) |
| 4.3 Comparison | 本地公平基线与完整学生；外部文献结果另列来源 | 表 2 |
| 4.4 Ablation Studies | 结构、阶段、损失、文本、必要重复 | 表 3，图 2/图 3 |
| 4.5 Efficiency Analysis | 学生推理和含教师训练分别测量 | 表 4，图 7 |
| 4.6 Cross-Dataset Generalization | 仅真实已跑的方向，不强行凑 3×3 | 表 5，预算不足时移补充 |
| 4.7 Qualitative Analysis | 同样本特征图、混淆矩阵、空间注意力、失败案例 | 图 4/5/6 |
| 4.8 Limitations | 数据规模、伪目标、可训练 phi、成本、泛化边界 | 实际结果支持的文字 |

旧稿中的 GCN 邻接矩阵、六 ROI 重要度和双向跨分支融合不在当前学生中，不能照搬成新稿实验。当前 cross-attention 聚合的是三个 Vim 状态，并非六个面部 ROI；局部注意力是 CNN 的空间图。

## 3. G0：正式训练前必须通过的检查（周一）

### 3.1 已核实的数据泄漏，优先于所有训练

2026-10-12 对本地 `RAF_DB` 所有 JPG 做了只读 SHA-256 检查：

| 项目 | 实际值 |
|---|---:|
| Train 文件数 | 13,810 |
| Train 中 train_ 前缀 | 12,195 |
| Train 中 test_ 前缀 | 1,615 |
| Test 文件数 | 3,056 |
| Train/Test 同名文件 | 1,615 |
| Train/Test 字节完全相同的跨 split 图片 | **1,615** |
| 旧 manifest 中 train 与 Test 重叠 | 1,451 |
| 旧 manifest 中 val 与 Test 重叠 | 164 |

因此旧 `artifacts/raf_db_protocol.json` 的 12,429/1,381/3,056 协议不可用于正式实验。先保存隔离/排除清单，保留原始文件，再从可信原始标签/划分文件重建。目录中的 `test_` 样本不能进入训练或模型选择。检查内容哈希和原始样本身份，不仅看文件名；补做感知相似度/近重复排查、异常标签和损坏图像检查。

若只能使用当前资产，单纯按 `train_` 筛选得到候选训练池 12,195，分层留出约 10% 后为 10,976 训练 + 1,219 选择验证，最终评估 3,056。**这只是预算用候选数，并非已经完成清理的正式协议**。标准协议缺失样本需找到原因；不能把本地子集写成完整官方 RAF-DB。优先恢复官方原始划分，否则正式报告“清理后的本地 RAF 子集”，外部论文结果仅作背景比较。

旧 `src/open_clip_train/data_preprocess.py::CLASS_DESC` 存在本地目录到 Angry/Disgust/Fear/Happy/Neutral/Sad/Surprise 的映射线索；旧 CSV 还合并遍历 Train/Test。该映射应与原始标签核对，不能作为权威标签证据；CSV 不能直接拿来训练。完整 AU 表仍缺失。

交付：`dataset_audit.json`、`excluded_samples.json`、`label_mapping.json`、新的 train/selection_val/final_eval manifest 与哈希。G0 标准：跨 split 身份/字节重叠为 0，类别顺序、来源、排除原因明确。

### 3.2 真实骨干与代码验收

当前代码已有模型/损失骨架和 10 项单元测试，尚不是完成了真实 Vim 和全部正式实验功能的训练系统。以下工作要在主实验前落实：

| 当前文件/行为 | 正式实验前工作 | 完成标准 |
|---|---|---|
| `teacher.checkpoint`、`student.factory/checkpoint`、AU 路径均为空 | 云端准备 ViT-L/14、一个固定真实双向 Vim 型号、带来源 AU JSON | B=2、256 分辨率真实前向/反向成功，真实加载身份与哈希保存 |
| `VimAdapter.forward` 用 `hidden[:, num_special_tokens:, :]` | 支持真实 CLS 位置；官方 Vim 的 middle CLS 不能靠删首 token 处理 | 根据实际 token 索引删特殊 token，patch 数与网格一致 |
| Vim hook 取 tuple 第一项 | 明确选取 mixer 输出还是含 residual 的状态，核对模型语义 | 三个状态确为最后三层，定义与报告一致 |
| 官方 Vim patch embedding 检查固定尺寸 | 正确设置 256 网格并插值位置编码，保留 CLS | checkpoint 严格加载；特殊位置与边界行为有记录 |
| `ablations.json` 只是描述集合 | 将覆盖转成可运行配置；变体 profile 不受 full 非零限制阻塞 | 六项消融真正改变相应执行路径 |
| S2 装配强制 student_s1 | 仅显式 `no_s1` 变体允许从同一 student_init 开始 | 普通 S2 仍强制来自 S1 |
| `class_name_ablation_schema` 将 ID 直接当文本 | 使用已核实情绪名称，如 Angry，而不是数字 1 | prompt 键保持标签 ID；内容为正确类别名 |
| runner 没有启动时统一 seed | 固定 Python/NumPy/Torch/CUDA、sampler/worker，存 RNG 状态 | 同 seed 复现实验；不同学生 seed 身份独立 |
| runner 只终局保存，缺 CLI 中断恢复 | 每 epoch 保存 last/best 和 optimizer/scheduler/RNG；模型选择与恢复身份分开 | 故障后恢复步数、LR、输出一致，已有 checkpoint 不覆盖 |
| 当前将 best model 与末尾 optimizer 混合保存 | 同步保存被选 epoch 的状态，或明确 best 只做推理、last 做恢复 | 恢复不会使用不对应的 optimizer |
| 只打印 step loss、最终梯度 | 保存 JSONL/TensorBoard：loss、LR、val 指标、attention、梯度、时间、显存 | 绘图所需原始数据齐全 |
| 导出只有指标 | 导出 sample ID/y/pred/logits、fused/隐藏层特征、attention/patch 可选 | 单次最终评估可生成所有主图 |
| 文本接口只检查token张量形状 | 额外检查原始token长度、截断与空文本；长AU描述按原CLIP context审查 | 无静默超长截断，保留原文与tokenIDs |
| config 只记录 CNN 初始化字符串，默认随机 | 主配置明确 ResNet18 随机或真实加载来源；对照一致 | 不把文字字段当成已加载预训练 |
| FP16 runner 未接 GradScaler | 主线实测 BF16；若回退 FP16 则接 scaler，概率/loss 用 FP32 | AMP 梯度有限，回退原因记录 |
| CNN 关闭后 AU/TV/稀疏仍可被计算 | 显式结构消融同时去掉相应不可定义项并记录 | 表中归因是“局部分支及配套目标”的组合移除 |

官方 Vim 源码/权重依据：[官方仓库](https://github.com/hustvl/Vim)、[models_mamba.py](https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py)。当前默认工厂的 bidirectional 参数是调用方声明，应核对实际 `bimamba_type`/模型配置与加载权重，不仅检查布尔值。

建议先固定 **Vim-tiny patch16、stride16、真实预训练** 作为一周主型号；若云端已有可用且确认无误的 Vim-small，则以它为主。型号今天冻结，不同时扫 tiny/small/base。CNN 默认用 ResNet18-layer3 并记录随机初始化；若本周切为 ImageNet 初始化，则所有公平基线重新统一。

### 3.3 训练时长与 warmup

候选干净训练集 10,976、batch32、drop_last 时，343 step/epoch；20 epoch 为 6,860 optimizer step。原配置 warmup10,000 会被严格拒绝。工程周计划采用记录清楚的 `warmup_ratio=0.05`（此例 343 steps），并同步修改方法/实现参数表；不能写用了有效的 10,000-step warmup。最终数字从修复后的数据规模重新计算。

保留图像256、AdamW lr1e-4/betas(0.9,0.98)/eps1e-6、完整损失系数初值。每阶段20 epoch是工程安排。32GB先测 batch32；OOM时采用有记录的较小 batch，所有相关对照统一，并报告 BN/对比 batch 改变。当前 runner 不支持 accumulation>1，不能只改数字假装生效。

云端前 20 step 预热、后 100 step 计时；计时包含真实前后向与优化器，CUDA同步，另测整 epoch（含 loader/验证）。记录真正 GPU 型号、驱动、CUDA、PyTorch、BF16能力、mamba内核版本；vGPU 显存大小不能确定吞吐。

## 4. 需要运行的实验矩阵

下面的“epoch20”先作为默认固定预算；只用验证集调整一次计划，并对各对照应用同一政策。完整模型选 epoch 不依据测试准确率。每一行输出 Acc、macro-F1、BAcc（逐类 recall 均值）、混淆矩阵、run ID、初始化、步数和 best-selection-val 指标。

### 4.1 主实验与两个公平基线

| ID | 模型/训练 | 回答的问题 | 复用关系 |
|---|---|---|---|
| T-AU-1 | 原始 ViT-L/14 → T1，AU 文本配对，20 epoch | 教师语义空间来源 | 一次训练，后续固定 |
| T-AU-2 | T1 → T2，仅 CE，20 epoch | 强教师及学生 logit 监督 | 一次训练，供对应 S2 |
| B-VIM-CE | 同一预训练 Vim + 同型投影/MLP，仅 CE，20 epoch，去 CNN | 预训练 Vim 的分类基线 | 从真实 student/Vim 初始化开始 |
| B-HYB-CE | 同一 Hybrid 学生初始化，仅 CE，20 epoch | 收益是否来自蒸馏，而非 CNN/更大结构 | 与 full 同尺寸/初始化；无梯度的 phi 冻结 |
| F-42 | 初始化 → S1完整五项20epoch → S2完整目标20epoch | 完整方法主结果 | S1用T1，S2用T2 |
| F-43、F-44 | full 学生 seed43/44，分别 S1→S2 | 稳定性与误差条 | 复用同一冻结教师对；明确是“条件于固定教师的学生3seed” |

报告原始 CLIP 的 AU 文本预测、T1 的 AU 文本预测、T2 分类头预测，分别命名。S1 语义指标使用训练到的 `state_l_norm`（也可分别记录 e/m/l）及 `cnn_distill`，不要把 S1 未训练的 fusion/query/classifier 输出当作已学好的主预测。

原始 CLIP 在目标数据上未适配时可单列零样本文本基线。经过 T1/S1 目标任务标签监督后的文本锚点分类应称“语义锚点分类”，不继续叫零样本。

### 4.2 六项必要消融

| ID | 相对 full 的唯一研究因素 | 要跑的阶段 | 复用与注意事项 |
|---|---|---|---|
| A-NOCNN | 移除 CNN 局部分支及其配套目标 | S1+S2，各20epoch | 无 AU/TV/CNN-pre/局部稀疏；记录这是结构组合删除 |
| A-NOS1 | 不做学生 S1 | 直接S2，20epoch | 同 student_init/教师T2；总优化步数较少，要报告 |
| A-NOLOGIT | `beta_logit=0` | S2，20epoch | 从 F-42 的同一个 S1 checkpoint 开始 |
| A-NOWALIGN | `lambda_anchor=0` | S2，20epoch | 同 S1；phi 无其它用途时冻结，避免无梯度参数进优化器 |
| A-NOAU | `lambda_au=0` | S2，20epoch | 同 S1；该因素包含空间 KL 和配套稀疏项 |
| A-NAME | AU描述改为核实的类别名 | T1/T2/S1/S2，各20epoch | 单独教师链+学生链；新 prompt hash，禁止硬复用 AU 教师 |

不要把 logit/W/AU 三项一起关掉然后分别解释每一项贡献。每个消融 seed42，与 F-42 做成对比较。3seed主结果和单seed消融分开标注；只有多次运行才能给 seed 标准差。

额外预算顺序：A-NOS1 的 S2=40epoch（优化步数匹配）→ hybridCE40epoch → 关键消融第二seed → 冻结phi → 单项温度实验。若论文声称“阶段分离缓解梯度冲突”，需要另做联合训练/梯度夹角测量；仅 no-S1 不能证明这件事，本周不做该实验就收缩这句话。

### 4.3 跨数据集的分级方案

**P0 必须：RAF完整实验。** 先完成它，再允许大规模扩展抢 GPU。

**P1 低训练成本：RAF训练 → SFEW明确可用评估split、AffectNet七类评估split。** 不做目标域微调，先统一语义类别再重排 logits；数据版本/对齐方式需记录。RAF训练所见标签不影响目标域评估标签隔离。跨域矩阵本周可只有 RAF 来源这一行，不填不存在的反方向结果。

**P1 视数据条件：SFEW域内四阶段完整模型 + HybridCE。** 公共验证集若承担文献中的最终评估，则从官方Train内部再留 selection_val；官方Val只在方案冻结后评估。教师阶段若先在RAF适配再迁移，标成“RAF→SFEW迁移”，不得称SFEW独立训练。若官方Test无公开标签，不写Test准确率。

**P2 只有实测预算可容纳：AffectNet七类全量域内四阶段。** 它的训练规模远超RAF。分层30k/50k子集可作pilot，必须保存抽样seed和manifest，并在表中明确子集，不能与全量官方结果混列。本周若没有全量实验，删掉“在AffectNet域内取得某准确率”的结论，保留真实跨域结果。

数据集规模描述以作者原始资料、当前下载版和有效manifest为准。AffectNet原始论文区分大规模收集与人工/自动标注，不能将总体规模直接写成实际监督训练数，也不把AU图文伪目标写成数据集原生AU标签。[AffectNet原始论文](https://arxiv.org/abs/1708.03985)

## 5. 单卡预算：测完再排，不凭显存猜速度

以候选343 step/epoch估算：包含full学生3seed、两条CE基线、六项消融，复用AU教师与共用S1后，约 **80个教师epoch + 320个学生epoch**。这是假定CE各20epoch、消融单seed、类别名重跑教师链的核心矩阵；不包含附加数据集和可选实验。

令平均教师训练步耗时为 tT 秒、学生含冻结教师前向的步耗时为 tS 秒：

`核心训练小时 ≈ (27440*tT + 109760*tS)/3600`。

另加验证、保存、导出及排队开销；初期可留25%裕量，最后以整epoch真实计时替换。以下仅算术情景，不是服务器测速：

| tT / tS（秒/步） | 纯训练小时 | 加25%裕量 |
|---|---:|---:|
| 0.5 / 0.2 | 9.9 | 12.4 |
| 1.5 / 0.6 | 29.7 | 37.2 |
| 3.0 / 1.2 | 59.5 | 74.3 |
| 6.0 / 2.4 | 118.9 | 148.6 |

周一晚到周六中午约可给GPU110小时，但计划承诺最多100小时任务，剩余留故障。若核心预计>80小时：先取消P2与可选扫描，优先单seed全矩阵；主模型补seed只有剩余预算足够才做。如果必须缩短所有学生阶段epoch，需在开跑前统一更改并公开实际步数；不能单独给full更多训练再归因结构收益。

推荐队列：T-AU-1→T-AU-2→F-42(S1,S2)→B-HYB-CE→B-VIM-CE→A-NOAU→A-NOWALIGN→A-NOLOGIT→A-NOS1→A-NOCNN→A-NAME全链→F-43→F-44→可选扩展。单卡不同时跑两个训练进程。GPU运行期间，CPU并行整理引用、写方法、画结构图、核对数据。

## 6. 必须画的7张图（先定数据接口，再跑训练）

统一类别颜色、字体、图宽；最终输出矢量PDF + 高分辨率PNG，保留绘图脚本和原始数据。所有test图是方案冻结后的描述分析，不用于再挑checkpoint或调参。

| 图 | 设计与面板 | 必要原始数据 | 最早开工 |
|---|---|---|---|
| **图1 总体结构与四阶段训练** | (a)冻结文本/教师与Vim-CNN学生；(b)T1/T2→S1/S2、冻结/可训练状态；CLS去除、三态pool、concat fusion标明 | 当前模型和loss契约，实际D/C/网格 | 周一；周二真实骨干确认后定稿 |
| **图2 训练稳定性与语义迁移** | S1每项loss单独小图；S2 raw与weighted各项、valAcc曲线；语义状态与教师cosine/分布KL随epoch | JSONL、学习率、验证指标，e/m/l与教师对齐指标 | 周二起自动生成；每个阶段独立epoch轴 |
| **图3 消融贡献** | 主图Full与六项消融Acc/macroF1，表3提供完整数值；full3seed误差条，单seed消融不伪造误差条 | 每个run最终预测、seed、初始化、协议 | 周四/五 |
| **图4 表征可视化** | 2×2：HybridCE/Full的fused_norm，以及S1 state_l/S2 state_l；说明S1与fused空间不同 | 同一组样本ID、特征、真实标签，仅评估用 | 周四模板，周六最终导出 |
| **图5 混淆矩阵** | HybridCE vs Full，行归一化，旁表逐类recall；跨域可附第三面板 | 统一class_order、y、pred | 周六 |
| **图6 AU空间注意力与失败案例** | 七类各固定样本：原图、A、空间softmax(A)、CNN-AU伪目标；Full与noAU同样本比较，附成功/失败 | attention、patch_norm、AU锚点、样本ID、预测置信度 | 周五模板，周六最终 |
| **图7 精度与部署成本** | Acc-学生推理延迟/参数量散点；完整CNN带来的耗时；可附batch1延迟、batch32吞吐条图 | 同硬件同精度真实计时、峰值显存、参数、支持范围明确的FLOPs | 周五/六 |

图4：预先固定取样seed，最多每类200张，样本ID各模型完全一致；可PCA到50维再t-SNE，固定perplexity/seed/迭代次数。不同模型单独t-SNE坐标不直接比簇距离。加原始空间的类内cosine离散度/类间中心cosine距离作为数值补充，不把t-SNE外观当显著性证据。

图6：A是非负权重，softmax(A)才是空间概率；不能混称。展示范围/颜色标尺固定，按样本分别min-max的热图只能说明位置而非强度。伪目标由学生CNN与真实类别文本产生，是训练解释性分析；真实y不进入预测。它不是独立AU真值，也不能证明识别出了每个AU。不要将其冠名“ViT教师attention”。预先固定验证集样本与最终test展示规则，保留弱激活、常数图和失败样本，不只挑好看图。

图7：记录50次预热、至少200次同步计时，batch1/32，学生eval/no_grad；输出中位数和P95、images/s、峰值分配显存。图像预处理计时与模型forward分别报告；文字编码和教师不计入学生推理，含教师训练另表。Mamba自定义算子FLOPs工具可能遗漏：报告支持的算子范围/缺失量，不把部分FLOPs当总FLOPs。成本对照本地实测；外部方法若拿不到同设备代码，只比较其论文报告参数并明确来源。

## 7. 5张主表与结果记录

| 表 | 必填内容 |
|---|---|
| 表1 数据协议/实现参数 | split有效数、class_order、去重、256输入、公共维度、真实Vim型号、初始化、四阶段epoch/步数/warmup、AMP |
| 表2 主结果 | originalCLIP锚点、T1锚点、T2头、VimCE、HybridCE、Full；Acc/macroF1/BAcc，mean±sampleSD仅多seed行 |
| 表3 六项消融 | Full42与6行消融；总训练步数、相同S1复用注释、文本链身份、分支组合删除脚注 |
| 表4 部署成本 | 视觉学生与教师：参数、可用FLOPs、batch1 latency/P95、batch32 throughput、峰值显存；训练成本另列 |
| 表5 跨域/额外数据集 | 训练来源→评估目标、标签映射、使用split、是否微调、是否全量；没跑的格子留“未运行”或不建表 |

外部文献比较表可做附表：预先固定纳入方法集合，保留更强结果，不按是否低于自己结果筛文献。旧稿比较表标题曾明确排除高于AG-VCLIP的结果，本篇不采用这种筛法。每行核对原论文年份、七/八类、输入、split、额外预训练和指标，不能沿用旧稿不一致数值。

推荐结果文件夹：

```text
experiments/2026-10-12/
  protocol/              # 数据、AU表、哈希、排除清单
  runs/<run_id>/         # 展开配置、环境、JSONL、last/best、RNG
  exports/<run_id>/      # sample_id/y/pred/logits/features/attention
  metrics/               # 每run指标与聚合，不重复手填
  figures/source/        # 绘图脚本与图数据
  figures/final/         # PDF/PNG
  manuscript/            # 正文、表格、caption、待补事实清单
```

每个run带：dataset_protocol_hash、prompt_hash、model/code_version、init_hash、teacher_T1/T2_hash、student_init/S1_hash、training_seed、selection_seed、实际步数、precision、实际warmup、关闭项。冻结split_seed42；training_seed42/43/44与它分开。

## 8. 从周一到周六的执行安排

以下为北京时间目标窗口，周一实际开始时间晚于窗口时向后平移，不能跳过G0。云上主实验运行不依赖本地MX450。没有云端连接信息时，本计划不表示我已经远程启动任务。

| 日期 | GPU任务 | 同步人工/CPU任务 | 当日验收与停止线 |
|---|---|---|---|
| **10/12 周一** | 环境/资产→真实B2 smoke→100step计时→启动T1 | 清理泄漏、核对标签/AU表、补运行控制、图1草稿、实验章节骨架 | 22:00前G0全通过；无法通过时停止正式结果训练，先排除原因 |
| **10/13 周二** | 完成T1/T2→Full42 S1/S2 | 图1定稿、实时图2、主表模板、查文献、数据/实现细节写完 | Full完整一遍；loss/attention/梯度有限，保存最佳验证与可恢复last |
| **10/14 周三** | CE基线→noAU/noW/noLogit；若队列快则noS1 | 梳理主结果、导出验证特征，图4/5/6模板，开始消融正文 | 核心损失消融完成；当晚重算剩余GPU预算 |
| **10/15 周四** | noS1/noCNN→类别名教师与学生链 | 空间机制诊断、注意力公平样例、图3初稿、表3 | 六项消融进入完成状态；18:00决定是否开启额外数据集 |
| **10/16 周五** | Full43/44、成本测量；预算内跨域/小型SFEW | 方法与代码口径统一，结果/局限性初稿、所有图caption | 20:00冻结结构/超参/样本规则；未开始的P2取消 |
| **10/17 周六** | 12:00前结束新训练；模型冻结后最终评估与导出 | 图2–7最终渲染、统一表值、摘要/结论/参考文献、全文检查 | 18:00图表全齐；22:00可提交论文+代码/配置/日志归档 |
| **10/18 周日** | 不安排新的研究训练 | 只复查可重现性、拼写、格式、交付 | 保留应急缓冲，不扩大实验 |

周四决策：若必需矩阵和3seed预计还能占用>剩余时间80%，不启动AffectNet训练。周五决策：没完成的数据集直接如实缩小摘要和实验范围。已经完成并方案冻结的行才能做最终评估；不根据test重新排实验优先级。

## 9. 当前论文中需要同步修改的地方

| 当前稿位置（PDF页码） | 问题 | 本周处理 |
|---|---|---|
| p1–2 | 图形摘要空白，Highlights仍是占位句 | 由图1裁出摘要图，按真实结论写2–3条 |
| p3–6 | 48%、67.12%、78%、xx%，监督适配后“zero-shot”，预设SOTA/计算更低 | 找原始日志并验证协议；没有干净证据先替换占位，最终用本周真实结果 |
| p5–6 | 长文本编码器、部分解冻、联合对比/分类描述和当前实现不一致 | 原始CLIP tokenizer/context、共享冻结文本、四阶段与完整目标口径一致 |
| p11–14 | 教师T1段落/公式大幅重复 | 合并一遍，正确标注tau与exp(logit_scale)，省篇幅给实验 |
| p14–18 | e/m/l称early/middle/late，“teacher guidance map”易误读 | 明确最后三层别名；空间target为学生CNN+冻结文本的停止梯度伪目标 |
| p18–20 | 数据集版本、规模、AU标注和SFEW来源/数量描述未核实 | 按原始数据说明和下载版本修正，不沿用旧文数字 |
| p20–21 | Vim/VMamba混称、教师B32/L14冲突、全部冻结只解一组、warmup10000 | 用实际Vim/ViT-L14、四阶段冻结、实际步数和warmup表重写 |
| p21–22 | 92.3%与“超过POSTER++ 1%”/更低成本未有本周证据 | 引用原论文准确值与真实自己结果；成本用图7实测 |
| p22–23 | xxxx/Fig.xx、提前断言簇更好/混淆更少 | 看真实图后写，可以报告无收益或下降 |
| 参考文献 | AdaCLIP条目为异常检测论文，却在正文作FER适配依据；多处?引用 | 逐一核实适用性、作者/年份/DOI，不将旧文参考列表机械复制 |

关于W-Align：phi可训练时，需分别测原始文本与W的cosine、phi(T)与W的cosine，必要时报告pairwise文本几何变化。投影后对齐提高不能直接证明冻结文本原始几何被保留。关于AU图：缺少独立AU区域标注时，只作定性伪目标一致性，不称AU定位准确率。

## 10. 完成标准

- [ ] G0：去重/原始划分/标签/AU来源确认，真实骨干跑通，环境/计时记录。
- [ ] 主线T1→T2和S1→S2身份正确，完整目标启用，teacher/text无更新。
- [ ] 两个CE基线与六项必要消融完成，每个run配置和初始化可追溯。
- [ ] Full学生3seed，或明确写单seed及本周无法完成重复的范围。
- [ ] 仅用selection_val选模型；冻结后最终评估，保存逐样本预测与特征。
- [ ] 图1–7、表1–4齐全；表5/附加数据集按实际运行范围填写。
- [ ] 部署成本为实测，有真实设备与精度；不包含教师或文本编码成本混入学生forward。
- [ ] 论文中的数值、表、图、摘要、结论、命名、初始化、训练阶段一一一致。
- [ ] 10/17 22:00交付论文PDF、可编辑源文件、图表源、日志/checkpoint清单、运行说明。

若周一G0延误或服务器速度落在慢情景，本周仍能完成可信的限定范围稿件，但不能保证未经测量的三数据集全量实验全部收敛。以完整证据链和可复现交付为截止标准，而不是追逐稿件已写入的准确率。

## 阅读依据与核对入口

- 当前论文：`C:/Users/xiaochai/Desktop/Vmamba_Chai.pdf`（方法p10–18、实验p18–23）。
- 旧论文：`D:/TEMP/Second_Chai.pdf`（实验p6–10，结构作为参考，不复用结果）。
- 当前代码：`src/open_clip/vision_mamba_clip/`、`src/open_clip_train/vision_mamba_clip.py`、`vision_mamba_clip_assembly.py`、`configs/vision_mamba_clip/`。
- 真实Vim接口：[官方源代码](https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py)，2026-10-12核对。
- AffectNet定义：[作者原始论文](https://arxiv.org/abs/1708.03985)。
