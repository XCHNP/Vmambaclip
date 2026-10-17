# Vim 学生预训练接入与第二轮训练方案

日期：2026-10-16。状态：交给执行 agent 的实施规范，尚未修改训练代码或执行新实验。

本轮目标是验证 Vim 预训练、训练预算与视觉特征蒸馏的作用，并改善 SFEW 泛化。保留官方 bidirectional Vim Small 和 CLIP ViT-L/14 师生关系，避免把骨干、初始化、轮次、输入和全部损失同时改动后归因于单一因素。以下超参数是预先规定的工程起点，不保证准确率。

## 1 事实与执行边界

1. 本地代码审阅目录是 `D:\桌面文件\学习\open_clip-main\open_clip-main`。本地不代表云端正式源码；当前本地目录的 `git status` 返回非 Git 仓库，不能虚构 commit、分支或 PR。无 Git 时用文件清单、SHA256、修改前备份和 patch 记录变化。
2. 事实依据是桌面 `RAFDB_SFEW训练结果与权重保存说明_20261016.docx`，以及云端本轮 effective_config、源码快照、曲线和 manifest。旧审阅问题必须逐条判定已修复、未修复或待核验，不能照搬为本轮故障。
3. 报告记载云端已使用官方 hustvl Vim Small bimambav2、384 维、24 层、patch16、middle CLS；Vim 和 ResNet18 均随机初始化；教师已使用预训练 CLIP、QuickGELU、T1 七类原型 CE。禁止退回本地自写 SSM，禁止换成 VMamba/SS2D。
4. 报告记载 RAF train/selection/final 为 11040/1231/3068，SFEW 为 763/88/402，实际以核验后的 manifest 为准。SFEW 原 val 是 final_eval，不用于调参。统一类别为 Angry、Disgusted、Fearful、Happy、Neutral、Sad、Surprised。
5. 原结果根目录报告为 `/root/autodl-fs/fer_suite_simplified_20261013`，服务器标识为 `autodl-westb-50061`；这是定位线索，不是可用 SSH 地址或当前在线状态。原结果只读保留。
6. AffectNet 属于必需数据集，报告说明其在另一个任务继续。先只读检查现状，不结束其进程、不替换其文件、不擅自取消旧队列。新旧任务用独立输出目录、源码和资源锁；不能把未完成的旧实验写成已完成。
7. 延续旧简化方案默认每配置仅 seed=42。多 seed 与交叉验证作为可选增强，默认不执行，不报告多 seed 方差。不得默认调用其他聊天或自动创建新任务。
8. 本文定义新版本 V2；旧版 Full、消融、共享教师、跨域计划的状态另列清单，不自动全部重跑，也不宣布取消旧任务。本文必做矩阵与扩展矩阵明确分开。

## 2 先收集证据再改代码

执行 agent 应先生成 `v2_inventory.md/json`，核验以下内容：

- 云端实际项目根目录、当前入口、Python/CUDA/PyTorch/timm/mamba/causal-conv1d 版本、官方 Vim 来源 commit 或源码哈希。
- 旧实验实际配置、学生结构、层索引、取特征语义、KL 方向、优化器分组、增强、模型选择规则。
- 旧 `curve.jsonl` 的 train/selection 曲线、best epoch、最后 20% 更新趋势、学习率；若没有 train Accuracy/UAR，标为未记录，不从 loss 猜造。
- 原始 CLIP、教师 T1/T2、Vim 预训练、CNN 预训练、已有学生权重的真实路径和 SHA。原始 CLIP 不等于任务教师。
- 已运行进程、GPU 占用、可用磁盘、旧输出清单。若已有合格的云端修复，应迁移或在其基础上改，不用本地旧源码覆盖。

报告结果摘要：RAF Mamba CE/Hybrid CE/logits KD/Full 的 Accuracy 分别 58.60/73.14/74.64/76.24%，Full UAR 67.36%；no_feature_align UAR 63.56%，no_logits UAR 69.55%。SFEW Full Accuracy/UAR 为 15.92/14.31%，T2 为 65.42/57.76%。这些只用于提出假设，不作为 V2 checkpoint 或超参数选择标准。

若无法访问云端：继续完成本地代码、测试、配置生成器、队列与说明；使用明显占位路径并在运行前拒绝占位配置，准确交付缺失信息，不能假称已训练。若已有被用户授权且可用的连接，则在授权范围内继续，不重复索要已有信息。

## 3 代码修改清单

下列路径是本地对应关系；agent 必须定位云端等价实现，并交付实际修改位置和差异。

| 模块 | 修改要求 | 验收证据 |
|---|---|---|
| `src/open_clip/vision_mamba_clip/vim_backbone.py` 或云端官方工厂 | 官方 Vim 构建、随机/预训练初始化、严格加载、位置编码适配 | 官方参数匹配报告、伪 checkpoint 拒绝测试 |
| `adapters.py` | 原生最终特征接口；多层 hook 的 residual/CLS 语义；冻结教师 | 原生输出一致性、CLS 保留/排除、梯度检查 |
| `model.py` | 新增简单 Vim CE 诊断模式；保留原 Hybrid，支持 CNN 预训练 | 各结构 forward、真实参数量与训练参数列表 |
| `data.py` 和数据准备入口 | 固定 manifest；同几何增强、各自归一化；三数据集与裁剪版本 | 分割/映射/重复核查与同视图测试 |
| `losses.py` | 独立 feature/logits/AU/W 系数；正向 KL；关闭项不执行 | 数值参照、开关隔离与梯度测试 |
| `stages.py`、`training.py` | 按目标管理可训练参数；评估后恢复 train；统一优化器更新 | 参数变化、冻结参数不变、BN/dropout 状态测试 |
| `config.py` | schema_version=2；新旧配置区分；阶段/域/初始化/输入显式配置 | 合法配置通过，错误组合拒绝 |
| `checkpoint.py` | 区分 init/resume/transfer；原子 best/last 保存；恢复契约 | 中断恢复与加载身份验证 |
| `src/open_clip_train/vision_mamba_clip.py`、`vision_mamba_clip_assembly.py` 或云端 runner | CE 不依赖教师；数据与阶段 DAG；train/selection 指标、评估与推理 | dry-run、真实短跑、严格重建与预测复算 |
| 新增 `configs/vision_mamba_clip/v2/`、计划生成器和队列工具 | 展开本文矩阵、依赖、资源估计、断点续跑 | 配置校验、无重复运行、无 final_eval 选择依赖 |

### 3.1 官方预训练加载

- 模型匹配 `vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2`，记录 patch stride=16，不能把 stride8 的 finer-granularity 变体混入。
- 官方入口可能未提供可用的 `pretrained=True` 自动下载 URL。显式解析官方权重文件，使用记录过来源的本地云端路径；缺权重时绝不静默退回随机。
- 去除明确的模型包装前缀，兼容经过识别的 `model`/`state_dict` 容器；拒绝含糊匹配。仅 ImageNet 分类头允许丢弃；FER 新模块由本项目初始化。
- 骨干关键张量的键与 shape 必须完整匹配。除明确的分类头删除和位置编码转换，其他 missing/unexpected 均应报错；覆盖率按参数元素数统计，不能用覆盖率阈值代替关键层检查。
- 256 输入时，patch 位置网格从 14×14 转为 16×16；按 checkpoint 的真实 middle CLS 索引拆开特殊位置，再插值 patch，插回目标实际 CLS 索引。记录转换前后 shape、顺序和方法。不要假定 CLS 在序列开头。
- 先在 224 上验证包装后的原生最终特征与官方 forward_features 一致，再单独测 256 插值和 token 顺序。FP32 eval 下先用 atol=1e-5、rtol=1e-4；若 fused kernel 带来差异，报告实际误差和合理容差依据。
- 记录 checkpoint_load_report.json：来源、SHA、官方版本、输入分辨率、键数、元素数、丢弃键、转换项、匹配率、是否成功。测试无关 checkpoint 必须失败。
- 不对加载后的模型调用全局 apply 初始化，不覆盖 SSM 的 dt/A/D 等已加载参数。随机初始化对照也用同一官方工厂。

### 3.2 模型与特征接口

提供两种显式架构，不偷换旧结果名称：

1. `vim_native_ce`：原生最终归一化/池化特征（确认实际返回 384 维）接新的 7 类线性头。用于预训练×轮次诊断，不加载 CNN、教师或 AU 文本。
2. `hybrid_v2`：保留云端现有三状态投影、聚合、CNN、融合和分类路径；修复经验证的接口错误，Vim 与 ResNet18 layer3 均接预训练。所有 Hybrid 基线与 V2 Full 共享同一架构、所有初始可学习参数与输入协议。

若三层实际是最后三层，记录具体索引，不能描述成浅/中/深三阶段。hook 输出的 mixer 分量不能未经核验就称完整 residual hidden state；明确特征定义、位置和归一化。不要简单把 tuple 两项相加就声称与官方最终输出一致，应按官方前向的残差和 norm 顺序实现。

S1 V2 主要对齐与最终分类实际使用的 `fused_raw`，让融合、投影和参与该输出的骨干均得到梯度，分类头冻结。避免 S1 只训练旁路而 S2 的主要融合路径仍随机。旧多层同目标、CNN 独立 cosine、语义 KL、TV/sparse 目标保留为 `legacy_full` 或可选消融，不默认混入最小目标。

只要真实计算图显示目标模块未被损失触及，必须修正 stage policy 或准确标记未训练；不能用 requires_grad=True 代替参数实际得到更新的证据。

### 3.3 数据与预处理

- 复用已核验 manifest 和分割身份，不重新随机划分。AffectNet 固定七类并显式统计 contempt、无效、无法匹配和重复样本的过滤。
- RAF 保留已有 aligned 图；SFEW/AffectNet 先抽查 face bbox、面部占比和方向。已为脸图不再次盲裁。若确需裁剪，产生新 manifest/preprocess_version，对应所有比较方法共同使用；旧模型与新裁剪结果不能混成单变量比较。
- 第一轮固定 256 输入，与旧云端教师协议一致；224 实验放扩展，不同时改变分辨率来解释初始化收益。
- 一次读取 RGB、一套随机几何增强（crop/flip 等），生成共享几何图像；分别按教师 CLIP、学生 Vim/CNN 的预训练规范 Normalize。不能分别抽样裁剪后逐样本做特征匹配。
- 推荐 batch 字段：`image_teacher, image_student, label, sample_id`。若学生两分支预训练规范不同，再提供明确的分支 tensor。CE-only 不要求构建教师。
- 学生对照统一采用相同的 ImageNet 输入规范，包括随机初始化诊断组；教师仍用其 checkpoint 对应规范。记录实际 resize/crop、ratio、mean/std，而非仅写预处理名称。
- train 保留 batch32、shuffle、drop_last=True 以便与旧更新计数对应；selection/final 必须 shuffle=False、drop_last=False。若云端有效 batch/梯度累积改变，重新计算更新数，不能抄旧数字。
- 原域内不平衡方案先保持 unweighted CE；平衡采样/类别权重作为扩展。所有权重只能从 train 统计，不能使用 selection/final 频次。
- 域内及迁移前检查 train 与 selection/final 的路径、内容重复及可用的身份/视频分组信息；无相关标注不能声称完成身份级无泄漏证明。

### 3.4 损失的精确定义

教师输出冻结，FP32 计算 softmax/log_softmax、cosine 和 loss 聚合，避免低精度数值异常。

```python
# T 是温度，teacher_logits 必须 detach。结果为 KL(teacher || student)。
loss_kd = F.kl_div(
    F.log_softmax(student_logits.float() / T, dim=-1),
    F.softmax(teacher_logits.detach().float() / T, dim=-1),
    reduction="batchmean",
) * (T * T)

loss_feature = (1 - F.cosine_similarity(
    student_feature.float(), teacher_feature.detach().float(), dim=-1, eps=1e-6
)).mean()
loss = lambda_ce * loss_ce + lambda_feature * loss_feature + lambda_logits * loss_kd
```

独立系数，禁止再用一个共享 `lambda_distill` 隐式耦合 feature 与 logits。主方案：

| 模式 | 目标 | 教师 |
|---|---|---|
| CE | CE | 无 |
| logits KD | CE + 0.2×KD，T=2 | 本域 T2 |
| feature KD 单阶段 | CE + 0.2×feature | 本域 T2 |
| V2 Full S1 | 1.0×fused feature cosine | 本域 T1 |
| V2 Full S2 | CE + 0.2×fused feature cosine | 本域 T2 |
| feature+logits 扩展 | CE + 0.2×feature + 0.2×KD | 本域 T2 |

V2 Full 默认关闭 logits、W Align、AU spatial、文本语义 KL 和相关正则；保留它们的配置和旧模型重建能力，但不得把 V2 Full 宣称为旧 Full 完整损失的复现。零系数的项不构造无关目标、不前向无关模块、不读取其资产。每项记录 raw loss、weighted loss 以及诊断期梯度范数。

保留经过核验的 T1 七类原型 CE，不退回 batch 对角线伪负样本。若扩展启用 S1 文本 KL，显式核查 cosine logits 的 scale/temperature、分布熵和梯度强度，不能把温度数字相同当作教师分布已匹配。AU 空间目标仍是伪目标，不是图像级 AU 真值。

主要特征使用 CLIP 投影后的实际维度（ViT-L/14 通常 768，需运行核验），不得称为已实现 1024 维 patch/token 或真正多层 ViT 蒸馏。T1/T2 是微调知识；若要声称保留原始 CLIP 知识，另设冻结 T0 特征对照，不能通过名称替代实验。

### 3.5 优化器与阶段切换

- AdamW，betas=(0.9,0.98)，eps=1e-6，基础 weight_decay=0.01；尊重官方 SSM 参数的 no_weight_decay 声明，bias/norm/特殊 token/位置参数策略显式记录，各组对照一致。
- 参数组初始 LR：RAF/AffectNet 的 Vim/CNN 为 3e-5，新头/投影/聚合/融合为 3e-4；SFEW 分别为 1e-5、1e-4。
- RAF 的随机与预训练诊断四组使用完全相同 LR=3e-5/3e-4 和其余超参数。这只能回答“相同微调配方下初始化和预算的作用”，不能宣称随机模型已得到最佳专属配方。
- BF16 支持时启用；FP16 要正确 unscale 后 clip；全局 grad norm=1。记录 AMP fallback，不能静默变更精度。
- 每个完整 run 使用一条 cosine，warmup 为总 optimizer updates 的 5%，最低 LR 为各组峰值的 1%；不让用户误以为追加 epoch 会自动恢复学习率。
- `V2 Full` S1→S2 默认使用 S1 **last** 参数，保留已有优化器 state，保持全程 scheduler 连续，分类头在 S2 开始产生梯度。参数组从启动时就固定包含该 run 会训练的所有参数，冻结模块 grad=None，不因切阶段重复入组。S1 best 仍保存供分析，不用于本主协议跳回早期状态。
- 上一条有意区别于旧版“S1 best→重建 Adam”，目的是减少优化器/学习率重置混杂；旧版行为保留为独立兼容模式。S1 teacher T1→S2 teacher T2 的变化与监督预算差异仍存在，不能把两阶段结果解释为纯粹多训练了若干轮 CE。
- 核心矩阵固定预算、不用早停改变对应方法更新数；每轮按 selection UAR 最大、同分 Accuracy 最大保存 best（再同分保留更早者）。S1 best 按其 selection loss 最小；S1 不报告未训练分类头准确率。
- 若损失非有限、样本/标签错误、骨干关键梯度长期为零，标记 failed 并停止受影响任务，修复后用新 attempt 或配置身份重跑。不要为了“跑满轮次”继续错误训练。
- 每次 validation 之后恢复学生正确 train 状态，并重新施加冻结子模块 eval 策略，教师始终 eval。不要让 val 后剩余 epoch 意外停留 eval。

### 3.6 checkpoint 与可复现性

- `init_pretrained` 仅初始化；`resume` 恢复同一次运行；`transfer` 为跨域微调，三种模式的允许加载键和身份检查不同。
- 本域 resume 强校验模型结构、优化器参数组、数据/预处理身份、类别、run 配置与调度 horizon。不同总轮次不能伪装精确 resume。
- 保存 best/last；last 含 model、optimizer、scheduler、AMP scaler（如有）、epoch、global_update、sampler 进度、Python/NumPy/torch CPU/CUDA RNG、best 指标及所有父权重 SHA。best 至少能严格重建学生推理。
- 同文件系统临时文件写入、flush/fsync、完整加载检查后 atomic replace；失败时保留先前可用 last。别在跨盘时谎称原子。
- 多 worker 随机增强的 mid-epoch 恢复不能只存主进程 RNG 就称精确：选择可恢复 sampler+按样本/epoch确定的增强种子，或明确只保证 epoch 边界恢复。分别记录确定性测试和真实 GPU 非确定性限制。
- 中断丢弃的更新/重算时间与有效优化器步数分别统计。原结果不覆盖，新 run 目录不可复用为不同配置。
- transfer 记录 source/target 域和源选择协议，移除源 optimizer/scheduler，以新的目标域 run 开始；默认重置目标分类头保留特征提取路径，对照一致。不能为了跨域关闭所有身份校验。

## 4 必做实验矩阵

所有正式 run 默认 seed=42。新目录建议 `fer_suite_v2_pretrained_20261016`，实际根路径由云端确认。没有通过工程验收不启动完整矩阵。

### 阶段 A 预训练与预算诊断

RAF，`vim_native_ce`，同一数据/输入/头/优化器配方：

| ID | Vim 初始化 | 总 epoch | 原 manifest 下更新数 |
|---|---|---:|---:|
| A1 | 官方随机 | 30 | 10350 |
| A2 | 官方 ImageNet 预训练 | 30 | 10350 |
| A3 | 官方随机 | 60 | 20700 |
| A4 | 官方 ImageNet 预训练 | 60 | 20700 |

30 与 60 轮必须各自从对应相同初始参数重新开始，并各有完整 horizon 的 cosine；不能把 60 轮的第 30 轮叫作独立的 30 轮 cosine 实验。头部初始化用独立固定 RNG，防止 checkpoint 构造路径改变头部随机值。相同初始化内保存并复用 init snapshot。

A 组首先在 selection 上分析差值、曲线和尾段趋势；不根据 final_eval 改 B 组。A4 可精确复用为 B 组 RAF 的 native CE，配置哈希完全相同才复用。

### 阶段 B 三数据集主比较

每域必须完成 5 条学生训练路径：

| 模型 ID | 架构 | 初始化 | 训练 |
|---|---|---|---|
| native_ce | Vim 原生特征+线性头 | Vim 预训练 | 全程 CE |
| hybrid_ce | 现有 Hybrid 的 V2 合格实现 | Vim+CNN 预训练 | 全程 CE |
| hybrid_logits_kd | 与 hybrid_ce 完全相同 | 同一个初始快照 | 全程 CE+正向 logits KD |
| hybrid_feature_kd | 与 hybrid_ce 完全相同 | 同一个初始快照 | 全程 CE+视觉特征 |
| full_v2 | 与 hybrid_ce 完全相同 | 同一个初始快照 | S1 fused feature→S2 CE+fused feature |

| 数据集 | 单阶段总 epoch | Full S1/S2 | 参考更新数 |
|---|---:|---:|---|
| RAF | 60 | 10/50 | 20700，Full 为 3450+17250 |
| SFEW | 50 | 5/45 | 1150，Full 为 115+1035 |
| AffectNet-7 | 30 | 5/25 | 按核验后的训练集和 loader 实算 |

这些是总轮次，不再额外加 3–5 轮 head warmup。核心比较不使用独立 head warmup，以减少冻结日程混杂；小样本拟合和线性诊断属于验收，不混入正式初始化。

如果 A 组或真实短跑证明预设配方无法正常优化，在只看 train/selection 的前提下记录修订原因，统一修改受影响的同组配置、重算预算后再开始；不能只救援 Full 而给基线保留坏配置，也不能无限试参直到 final 达标。

计数：A 4 条 + B 15 条 - A4/B RAF native_ce 1 条复用 = **18 条不同学生训练路径**。B 中三域 Full 各有两个阶段，因此对应 **21 个学生阶段**；阶段数不等于独立优化器 run 数。教师准备、工程短跑、旧实验与扩展不计入这些数字。

### 阶段 C 教师复用与缺失补齐

- RAF/SFEW 的旧 T1/T2 优先复用，条件是能够严格重建、来源/类别/数据/预处理兼容，并复算 selection 指标；不能仅凭文件名确定。
- AffectNet 在既有任务合格产物就绪后复用；没有合格教师时，本任务配置 T1=5、T2=10，原始 CLIP→T1→T2，视觉 LR=1e-5、头/scale=1e-4，文本冻结，T1 scale 上限100。具体执行与正在运行任务协调资源，避免重复训练。
- T1 按 selection prototype CE 最小选择；T2 按 selection UAR、同分 Accuracy。不得因为已见 final 教师分数而改 checkpoint。
- 教师使用自身的实际 256 CLIP 输入；学生改用 ImageNet Normalize 本身不要求重训教师。同域重复/裁剪协议若发生实质变化，则重新验证教师数据资格，必要时单独产生新教师 run。

## 5 有条件扩展及明确预算

默认先完成 A/B 与工程验收；不自动展开大量组合搜索。下列扩展生成禁用配置与执行命令，只有用户明确选定，或本轮执行委托明确包含对应扩展时再运行。已经由其他任务获得授权的旧计划不受本开关撤销。

| 扩展 | 有限配置 | 目的 |
|---|---|---|
| 同结构 feature+logits | RAF Hybrid，60轮；复用相同 init | 在 feature 基础上测正向 KD 是否增益 |
| 简化 backbone-only feature | RAF native 架构增加视觉投影，CE基线需匹配头与投影结构 | 判断预训练后 CNN 是否仍必要，不能用旧 no_cnn 直接归因 |
| 恢复 AU/W/多层项 | 在固定选定基础上每次只加一个；最多3项 | 逐项验证贡献，不做全组合 |
| 保留原始 CLIP | 增加 T0 视觉特征对照，记录 teacher source | 区分原始预训练与任务微调知识 |
| SFEW 学生迁移 | RAF→SFEW、AffectNet→SFEW，每个来源做 CE 与 feature KD 一对，共4条，各30轮 | 区分额外 FER 数据与目标域 KD 贡献 |
| AffectNet 共享教师 | AF T1/T2 指导 RAF/SFEW 的 Full，学生仍从 ImageNet初始化，共2条 | 与域内教师方案比较，披露额外监督数据 |
| 跨域评估 | 三域 hybrid_ce、hybrid_logits_kd、full_v2 各六向，共18次，无训练 | 使用源域选择的 checkpoint，不做目标域适配或BN更新 |
| 稳定性 | 仅最终少数候选加 seed=43、44 | 有预算后估计运行方差 |

迁移源 checkpoint 必须只由源 selection 选定，与目标 final 无关；跨域重复核验不通过时不能启动迁移。对照采用同一种源架构/初始化快照：例如同一 source hybrid_ce best 初始化目标 CE 与 feature KD，避免把来源差异误当作目标蒸馏效果。源训练成本另报。

## 6 正式训练前验收

必须产出自动化报告；mock/unit 测试不能代替 GPU 实际官方骨干验收。

1. 三域 manifest、映射、分割、损坏/重复统计和样例图；selection/final 样本全覆盖。
2. 权重严格加载：正确 checkpoint 成功，无关 checkpoint、错误尺寸/变体明确失败；224 原生特征一致性；256 middle CLS/插值顺序测试。
3. 真实 forward/backward：所有模式 loss 有限；优化器包含所有且仅含预期 run 参数，无重复；目标骨干参数实际变化，教师和冻结参数不变。
4. KL 与显式 `sum(p_t*(log(p_t)-log(p_s)))` 参照一致；feature 近似相同时 loss 接近0；关闭项对总 loss 无贡献，也不依赖被关闭的资产。
5. 同几何增强测试；CE 路径不构造教师；validation 后 train 状态恢复；head/CNN BN 状态符合阶段策略。
6. 32–64 张平衡的 train 图片，关闭随机增强、蒸馏和强正则，用 CE 做一次有上限的拟合诊断（建议最多1000 updates）。目标接近完全拟合，例如 train accuracy≥95%；达不到先诊断标签、优化与接口，不能据此保证泛化。诊断权重不得进入正式 run。
7. 官方 GPU 短跑覆盖 BF16 和 checkpointing；检查 backward 和中间特征 hook 是否一致。没有 GPU/官方依赖则标待验收，不用 CPU toy 证明生产可运行。
8. 同一小实验连续运行与保存恢复运行对比；分别说明确定性测试、epoch边界和mid-epoch保证级别。
9. 学生 checkpoint 严格重建，在真实样本上比较保存前后 logits；最终推理无教师/文本运行依赖，导出预测与复算指标一致。
10. 计时使用正式架构/分辨率/精度，预热后记录训练和 validation 开销、峰值显存、磁盘预计需求，按实际更新数估算每条任务时长。

## 7 命令与任务调度契约

优先扩展云端已有 runner；下列是需要实现或映射的接口，不声称现在可执行。最终交付必须替换为实际可运行的命令，禁止只交伪代码。

```text
python <actual_entry> inventory --output <v2_root>/inventory
python <actual_entry> prepare-v2 --paths <paths.json> --output <v2_configs>
python <actual_entry> validate-plan --plan <plan.json> --dry-run
python <actual_entry> audit-data --plan <plan.json>
python <actual_entry> audit-init --config <config.json>
python <actual_entry> smoke --config <config.json> --max-updates <N>
python <actual_entry> run-plan --plan <plan.json> --group core --resume-incomplete
python <actual_entry> evaluate --run <run_dir> --split selection_val
python <actual_entry> evaluate --run <run_dir> --split final_eval --require-locked-plan
python <actual_entry> aggregate --plan <plan.json> --output <report_dir>
```

- plan.json 显式列出 run_id、architecture、dataset、init_id、teacher依赖、stage epochs、total updates、seed、config hash、输出目录与状态；DAG 不含 final_eval→训练/选型边。
- `paths.json` 只存当前机器路径，不提交密钥。生成器校验所有必需文件，不能把 Windows 路径直接传给 Linux。
- 同一 GPU 一次一个训练任务，用进程/锁机制避免重复启动；遇到别的任务占用则排队，不杀进程。
- 任务状态至少区分 planned、implemented、validated、running、completed、failed、blocked；resume-incomplete 跳过已核验完成项，损坏 last 要报错并保留证据。
- long-run 日志、进程号、完成标记、异常堆栈都写到运行目录。只有 checkpoint、预测、指标与身份验收全通过才能标 completed。
- 固定主矩阵后写 plan.lock；final_eval 不参与参数选择。旧 final 已被看过，本轮应标为迭代开发后同 benchmark 复评，不宣称全新盲测；若要独立确认，需要另行准备未使用测试集。

## 8 配置必须显式表达的字段

实现最终 schema 时至少涵盖：

```text
schema_version, run_id, seed, dataset, class_order, manifest_hashes
student.architecture, official_model_name, source_revision, initialization
student.pretrained_path, pretrained_sha256, input_size, cls_position_policy
student.feature_readout, layer_indices, cnn_initialization, init_snapshot_sha
teacher.source_domain, stage, checkpoint_sha, clip_base_sha, output_dim
preprocess.geometry_version, student_normalize, teacher_normalize
loss.lambda_ce, lambda_feature, lambda_logits, temperature, kl_direction
loss.lambda_au, lambda_weight_align, lambda_semantic, feature_target
optimizer.parameter_groups, betas, eps, weight_decay_policy, gradient_clip
schedule.total_updates, warmup_updates, min_lr_ratio, stage_boundaries
schedule.transition_source, optimizer_reset, scheduler_reset
checkpoint.resume_contract, selection_metric, tie_breaker
transfer.source_run, target_domain, classifier_reset
output.root, code_hash, environment_hash
```

仅相关模式要求 teacher/transfer 字段。旧 schema 走显式兼容路径并记录转换，不能默默把旧反向 KL 改成新方向后仍沿用旧 run 身份。

## 9 交付与结果判断

交付修改源码与 patch、配置、实际启动/恢复命令、环境与数据审计、权重加载报告、真实短跑与恢复测试、训练队列、曲线、best/last 清单、结果汇总和未完成项。

每个训练阶段保存 effective_config、curve.jsonl、parent hashes；每个最终学生保存 metrics、per-class 指标、混淆矩阵、sample_id/真值/预测/probabilities。Accuracy/UAR/macro F1 都报告；主选择指标 UAR，单位标明百分比或0–1。UAR 始终按固定七类计算，selection某类缺失则先暴露协议问题，不静默缩小分母。

最终报告分开回答：

- 预训练收益：A2−A1、A4−A3。
- 预算收益：A3−A1、A4−A2；同时展示 train/selection 曲线，不能只看 best final。
- 同结构蒸馏收益：hybrid_logits_kd、hybrid_feature_kd 相对 hybrid_ce。
- 两阶段方案效果：full_v2 相对 hybrid_feature_kd，披露教师切换、S1无CE、loss日程差异，不称单因素因果证明。
- SFEW 是否摆脱接近随机、是否出现少数类预测坍缩；若仍失败，交付可复现诊断，不以训练更多无关实验掩盖。
- 模型复杂度：参数、实测延迟/吞吐、显存和训练GPU时；不要套用官方论文效率数字。
- 单seed不提供运行方差或“显著提升”结论。可基于固定样本做明确标注的配对bootstrap，但不把它当多seed不确定性。

本方案实施成功的标准是代码正确、比较可解释、记录完整、执行结果真实，不设“必须达到某个准确率”的伪验收门槛。若复杂模型无收益，应如实推荐简单基线。

## 10 依据

- 用户 2026-10-16 训练报告及项目根目录 AGENTS.md。
- 旧简化执行计划 `docs/AGENT_EXPERIMENT_PLAN_SIMPLIFIED_2026-10-13.md`，仅继承事实、单seed偏好和未完成任务身份，不自动重跑其矩阵。
- [官方 Vim 仓库及模型权重入口](https://github.com/hustvl/Vim)。
- [官方 Vim 模型实现](https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py)。
- [官方 Vim 预处理实现](https://github.com/hustvl/Vim/blob/main/vim/datasets.py)。
- [官方 Vim Small middle CLS 权重](https://huggingface.co/hustvl/Vim-small-midclstok)。
- [知识蒸馏论文](https://arxiv.org/abs/1503.02531)。

官方资料用于确认模型/权重/输入接口；本文实验预算、分组学习率、简化损失和执行门槛是针对本项目的建议，不是这些来源已经验证的 FER 最优方案。
