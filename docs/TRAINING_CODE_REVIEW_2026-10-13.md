# 云端训练前代码审阅（2026-10-13）

## 结论与审阅边界

**教师应基于原始 CLIP ViT-L/14，并保留其 ViT 图像塔；改好的 VisionMambaCLIP 是学生。** 两阶段教师微调产生 T1/T2 权重，再冻结教师给学生提供监督。当前教师装配和阶段冻结基本符合这个方向，但数据接入、T1 目标和学生骨干身份尚有关键问题，不能仅凭现有单元测试通过就认定正式训练准备完成。

用户确认的事实已写入根目录 `AGENTS.md`：正式训练和数据均在云电脑；ViT-L/14 原始预训练权重已在云电脑；教师微调权重要自行训练得到；RAF-DB、SFEW、AffectNet 都属于实际训练范围。本次读取的是当前代码，未连接云电脑、未读取云端数据/权重、未启动正式训练。本地缺少文件不是云端缺少文件的证据。旧文档中的本地数据数量、泄漏记录、缺权重判断不能直接用于云端数据。

本次新增事实记录、审阅报告和可重复的 CPU 检查脚本/结果，没有修改模型或训练实现。下文“建议”均是后续修改方案，不是已实现功能。

## 优先解决的发现

| 优先级 | 发现、代码位置 | 影响与建议 |
|---|---|---|
| P1 | `data.py:54` 的标签映射只有 RAF-DB；`build_protocol:311` 只接受 RAF 风格平铺标签文件或 Train/Test 类别目录 | `sfew`、`affectnet` 都直接报 Unknown dataset。不能只补映射：还要接入真实标注格式、官方划分、人脸裁剪、过滤类别及统计。为三个数据集分别准备云端配置与 manifest。 |
| P1 | `training.py:39` 用 `anchors[labels]`；`losses.py:27` 用 batch 对角线作为唯一正样本 | 同类图片的文本完全相同，却互为负样本；类别频次改变损失的不可消除部分。B=32 全同类且图像特征相同时，实测 loss=3.465736≈ln(32)，图像梯度为 0。混合类别仍可学习，但目标与类级文本监督不匹配。改成全类别文本原型 CE，或有同类正样本掩码的多正样本对比损失。 |
| P1 | `vim_backbone.py:131,201` 自写 SSM；`create_vim:394` 宽松加载后无条件报告预训练成功 | 当前骨干不是与官方 Vim 权重对应的实现。完全无关的 checkpoint（匹配键数为 0）实测仍报告 `pretrained_loaded=True`。应使用已确认骨干，核验关键层、shape、参数覆盖率，禁止关键权重缺失；自写结构若保留，应准确命名并声明随机初始化。 |
| P1／目标缺口 | `adapters.py:61` 只返回 `encode_image`；`losses.py:136` 将学生三个状态都对齐同一教师全局向量 | 当前实现有 **768 维全局视觉特征蒸馏**，不能说只有 logits 蒸馏。但没有取教师 1024 维隐藏状态或 patch tokens，不能称为 ViT 多层/局部知识蒸馏。如果“高维”仅指相对 7 类 logits 的特征级迁移，现有方向合理；如果要求学习 ViT 中间表示，则须增加教师中间层接口及相应蒸馏目标。 |
| P1／条件性 | `vision_mamba_clip_assembly.py:100` 用本地文件路径初始化 `ViT-L-14`；`config.py:33` 只允许该名字 | 若下载的是 OpenAI 原始权重，应匹配 QuickGELU；当前标准 `ViT-L-14.json` 未启用，而路径加载不会自动获取对应 pretrained tag 的激活配置。需记录实际权重来源，按来源设置 `force_quick_gelu` 或允许匹配的模型配置。OpenCLIP 来源可能采用 GELU，不能一概切换。 |
| P2 | `vision_mamba_clip_assembly.py:103` 强制图像尺寸 256；`config.py:29` 拒绝其他尺寸 | ViT patch14 与 Vim patch16 的输入约束被绑在一起。256 对 ViT-L/14 得到 18×18 网格，底部/右侧各 4 像素未被 patch 卷积覆盖，位置编码也须从原网格插值。不是完全无法前向，但并非原始 224 协议。若无硬性 256 要求，224 同时适配 patch14/16；若保留 256，应显式记录并设计师生视图及网格对齐。 |
| P2 | `adapters.py:241` 固定删除最前面的特殊 token | 与自写 prepend CLS 一致；接官方 middle-CLS Vim 后会删掉真实 patch、留下 CLS。必须按真实位置排除 CLS，位置插值同样不能默认第一项就是 CLS。hook 输出还需明确是否包含 residual。 |
| P2 | `predata_rafdb.py:272,321` 用未排序的目录扫描和全局随机数；`data.py` 用排序扫描和局部随机数重建划分 | 两条预处理路径不能保证“相同 seed 就相同样本划分”。若比较 BaseCLIP 与学生，会影响公平性；混用不同验证集合还可能把一方选择样本放入另一方训练。统一读取保存好的 manifest，不独立重新划分。 |
| P2 | `data_preprocess.py:10,29` 采用另一套数字类别描述，并把 train/test 写入同一个 CSV | 该映射不能直接用于 RAF 官方数字标签；若生成的 CSV 被用作训练集，会引入测试样本。正式流程应停用这条旧入口，统一映射与分离的 split manifest。云端是否实际使用该入口仍未知。 |

官方 Vim 的模型结构、middle CLS 和 mixer 可核对 [官方模型源码](https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py)、[官方 Mamba mixer](https://github.com/hustvl/Vim/blob/main/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py)。原始 CLIP 的激活见 [OpenAI CLIP model.py](https://github.com/openai/CLIP/blob/main/clip/model.py)。源码差异支持上述兼容性判断；本次没有声称加载或测试了云端官方权重。

另一个明确的初始化错误：`SelectiveSSM.__init__` 将 dt bias 设为 softplus 后落在 0.001–0.1 的值，但随后 `BidirectionalVim._init_weights()` 把所有 Linear bias 清零。实测 dt bias 全为 0、softplus 全为 0.693147。自写骨干保留时，也要保护这类 SSM 专用初始化。当前 Python 逐 token 扫描并展开 `[B,L,D,N]` 张量，不能据此套用官方 Vim 的吞吐/显存数据。

## 数据预处理：什么已正确，什么还需要补齐

主入口 `FERDataset.__getitem__` 将图片转为 RGB，图像 transform 由 OpenCLIP 创建，训练/验证分别使用随机和确定性变换。师生收到同一次变换后的相同 tensor，目前不会发生两次独立随机裁剪导致的视图错位。分类器标签和文本原型顺序也有严格检查，这些逻辑应保留。

### CLIP 输入

对于 OpenAI 原始 CLIP，输入流程应是：面部区域 → RGB → 双三次 resize/crop → float tensor（归一化前为 0–1）→ CLIP Normalize。标准均值为 `(0.48145466, 0.4578275, 0.40821073)`，标准差为 `(0.26862954, 0.26130258, 0.27577711)`；验证/最终评估采用确定性变换。依据是 [OpenAI CLIP 的预处理实现](https://github.com/openai/CLIP/blob/main/clip/clip.py)。

注意当前 `get_model_preprocess_cfg` 返回的是构建模型的有效配置。给 `pretrained` 传单独文件路径，并不等于自动读取旁边的 `preprocessor_config.json`。如果云端权重来自不同 OpenCLIP 训练来源，须显式匹配其 resize、mean/std、激活、分辨率和 tokenizer，不能只相信配置中的“read from actual CLIP checkpoint”描述字符串。

RAF 已对齐脸图可以直接使用；SFEW 或 AffectNet 若存的是原图，应先按标注框或统一的人脸检测策略取脸，保留眉、眼、鼻、口和下巴。若已提供裁剪脸图，不应再次盲目裁剪。建议轻量、保留表情区域的增强；当前 scale=0.9–1.0 较温和，但非 timm 路径没有传入 `ratio`，其配置不能被当成实际生效的裁剪纵横比。正式记录应保存实际 transform。

224 是同时整除 14 和 16 的简单选择。若研究设计要求学生输入 256，可从同一面部区域、同一次随机几何增强产生教师 224 与学生 256 视图，并在训练中传递两个 tensor；当前接口尚不支持。做空间 token 对齐时还要对齐归一化坐标和覆盖范围，不能仅 reshape 后直接比较不同网格。

### 三个数据集的划分与标签

当前统一输出顺序为：`Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised`（0–6）。RAF 官方原始 1–7 在当前映射中对应 `6,2,1,3,5,0,4`。如果云端文件夹已经重新编号，必须用实际标注映射，不能再次套用官方原始编号。

SFEW 应读取所用版本的实际 train/validation/test 划分；有官方验证划分时，不应像 RAF 一样无条件从 Train 再派生 10%。如果公开 validation 是论文最终报告集合，最好从 training 内划出 checkpoint 选择集，冻结超参数后评估公开 validation，并准确报告 split 名称。是否具备可用 test 标签要以云端数据为准。[SFEW 2.0 原始实验论文](https://talhassner.github.io/home/projects/cnn_emotions/LeviHassner_ICMI15.pdf)明确分别报告 validation 与 test。

AffectNet 必须先明确 **7 类还是 8 类**。当前协议固定 C=7，仅能表达 7 类；若使用 7 类，应显式过滤 contempt 和无效/不确定/非脸标注并统计排除量，不能把它们重映射为某个已有类别。若使用 8 类，则必须扩展类别顺序、AU 表、分类头及配置校验。其原始标注包含这些额外类别，且具有独立划分、显著类别不平衡；见 [AffectNet 原始论文](https://www.mohammadmahoor.com/wp-content/uploads/2017/08/AffectNet_oneColumn-2.pdf)。具体数字编号仍须以云端标注说明核验。

当前 folders 模式固定要求 Train/Test，且 `build_protocol` 没有透传自定义目录名；label_file 模式固定 train_/test_ 前缀，忽略传入的 `validation_ratio`，硬编码 90/10。建议统一 manifest 字段：`relative_path, raw_label, canonical_label, split, dataset, bbox(optional)`，每个数据集记录版本、类别映射、过滤统计、路径/内容重复与损坏图片检查。云端迁移时只变更 root，保持相对路径、split 身份及 manifest 哈希。

AffectNet 的采样/损失权重应依据训练集频次设定并对所有对照一致；至少记录 accuracy、macro recall/UAR、macro F1 和混淆矩阵。当前 loader 只是 `shuffle=True`，没有类别平衡；runner 的 T2/S2 checkpoint 仅按 accuracy 选择，需要明确是否改用适合不平衡数据的指标。

## AU 文本审阅

现有文本大体覆盖了典型眉、眼、鼻和口部运动，可用作 **类别级 AU 启发的语义先验**。它们由手写 `AU_FEATURES` 生成，没有图像级 AU 标注依据，因此不能写成“每张图片真实发生了这些 AU”。当前空间 AU 目标还是学生 CNN patch 特征与类别文本的相似度构造的伪目标，不是教师 ViT 的 AU 热图，也不是真实 AU 监督。

实测七条原型为 **41–48 tokens**（含 SOT/EOT）；穷举离线 caption 的 **264 个组合**为 35–52 tokens，无一超过 CLIP 的 77-token context。因此当前文本不存在实际超长截断问题；不过 tokenizer 本身会静默截断，后续增加模板仍应做原始 token 长度检查。

建议修正 `a Angry` 等语法、弱化“deeply/significantly/firmly/as high as possible”等强度断言，以核心肌肉运动为主；避免把 watery eyes、calm gaze、皱纹深浅等外观现象等同 AU。可使用下面的短文本作为待验证候选，先与现有表做文本消融，不预设哪套一定更好：

| 类别 | 候选描述 |
|---|---|
| Angry | A face showing anger, with brows lowered and drawn together, tightened eyelids, and pressed lips. |
| Disgusted | A face showing disgust, with a wrinkled nose and a raised upper lip. |
| Fearful | A face showing fear, with raised and drawn-together brows, widened eyes, and horizontally stretched lips. |
| Happy | A smiling face, with raised cheeks and upward-pulled mouth corners. |
| Neutral | A neutral face, with relaxed brows, eyelids, and mouth, without a pronounced expression. |
| Sad | A face showing sadness, with raised inner brows and downward-pulled mouth corners. |
| Surprised | A face showing surprise, with raised brows, widened eyes, and a relaxed open jaw. |

同一类别顺序下，三个数据集可共享相同原型表，避免 dataset-specific 文本改变实验因素。若采用多模板，每类文本先分别编码、归一化、平均再归一化，并记录模板及哈希；目前 schema 只支持每类一个字符串。类别级模板不能消除同类图片多正样本问题。

预处理入口还有两个不一致：`generate_training_prompt` 随机选择区域描述，而 runner 只使用七条固定原型，随机 captions 没有进入四阶段 runner；`write_prompt_schema()` 已定义但 `__main__` 只调用 `generate_split_csv()`，直接执行脚本不会自动更新 AU JSON。应明确使用固定原型还是动态模板，并统一生成入口，不能声称随机 CSV 文本与原型完全相同。

## 教师两阶段应如何训练

**BaseCLIP 指保持原始 CLIP 双塔的模型，不是重新随机训练 CLIP。** 云端已有 ViT-L/14 权重作为初始化，外接 FER 线性头并不改变教师的 ViT 身份。若 vmambaclip 已将 ViT 替换成 Mamba，用它训练教师将变成 Mamba→Mamba 蒸馏，无法直接提供原始 ViT 表示，这与本项目目标不符。

| 阶段 | 初始化与可训练部分 | 目标及产物 |
|---|---|---|
| T1 | 云端原始 CLIP；冻结文本编码器；训练 ViT 图像塔与可选 logit_scale | 类级图像–AU 文本对齐。建议 `CE(exp(logit_scale) * norm(image) @ anchors.T, labels)`，或正确的多正样本对比；保存 `teacher_t1.pt`。 |
| T2 | 从同一数据协议下 T1 初始化；文本塔冻结；训练线性 FER 头和 ViT 的选定层 | 分类 CE；保存 `teacher_t2.pt`。当前代码会解冻整个 visual，这应是明确的实验选择。 |
| S1 | 学生初始化；冻结 T1 教师与文本；训练 Mamba、视觉投影及局部分支 | 视觉/语义知识迁移；保存 `student_s1.pt`。 |
| S2 | 从 S1 初始化；冻结 T2 教师与文本；训练学生 | FER CE + 视觉特征蒸馏 + 辅助 logits/文本/空间目标；保存 `student_s2.pt`。 |

当前 `stages.py` 正确冻结教师文本、T1 不训练分类头、T2 训练分类头，学生阶段教师完全冻结并 eval；assembly 正确实现 T1→T2、T1→S1、T2→S2、S1→S2。修改损失时应保留这些边界。

T2 若只做 CE，视觉空间可能偏离 T1 文本空间。为了保留预训练知识，可先训练头，再用较小视觉学习率解冻后几层，按验证选择是否扩展到全塔；必要时加冻结原始 CLIP 或 T1 的特征保持项。**这是工程建议，需要实验验证，不能保证比全量微调更好。** 当前所有阶段/所有参数都用 lr=1e-4，建议支持阶段和参数组学习率；例如视觉塔从 1e-6–1e-5、新头从 1e-4–1e-3 试起，作为验证搜索起点，不作为已验证最优配置。SFEW 小样本场景尤其要检查过拟合及教师泛化。

T1 修改为七类原型目标后，验证可用原型分类 accuracy/UAR 或相应类级 loss，避免用 batch-dependent 错误对比 loss 选择权重。保留原始权重 T0，便于检验“原始预训练知识”和“FER 微调知识”各自的贡献。

推荐默认按三个数据集分别跑完整链路，以保证域内结果易解释：

```text
RAF-DB:   T0 → T1_rafdb → T2_rafdb；S1_rafdb → S2_rafdb
SFEW:     T0 → T1_sfew  → T2_sfew； S1_sfew  → S2_sfew
AffectNet:T0 → T1_affectnet → T2_affectnet；S1_affectnet → S2_affectnet
```

这是建议的实验协议，并非用户已指定必须独立训练。联合训练或 AffectNet→SFEW 迁移也可行，但应单独命名协议、只使用授权的训练 split、记录源/目标数据，且不能与“仅目标数据集训练”的基线直接混为一项。当前 checkpoint 强制相同 manifest，尚不支持直接跨数据集复用 T1/T2；迁移需要显式的身份校验规则，不能简单删除所有校验。

当前 runner 的阶段调用形式如下。在云端配置真实路径、修复上述关键问题后，分别运行；下列命令只是入口说明，本次未执行：

```bash
python -m open_clip_train.vision_mamba_clip inspect-data --config configs/rafdb_config.json
python -m open_clip_train.vision_mamba_clip run-stage --config configs/rafdb_config.json --stage teacher_t1 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/rafdb/teacher_t1.pt
# 将 checkpoints.teacher_t1 更新为实际 T1 输出，再运行 T2 和 S1。
python -m open_clip_train.vision_mamba_clip run-stage --config configs/rafdb_config.json --stage teacher_t2 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/rafdb/teacher_t2.pt
python -m open_clip_train.vision_mamba_clip run-stage --config configs/rafdb_config.json --stage student_s1 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/rafdb/student_s1.pt
# 将 checkpoints.teacher_t2 / student_s1 更新为实际输出，再运行 S2。
python -m open_clip_train.vision_mamba_clip run-stage --config configs/rafdb_config.json --stage student_s2 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/rafdb/student_s2.pt
```

每阶段使用独立路径，当前代码拒绝覆盖已存在的输出。SFEW、AffectNet 还没有对应可运行配置和数据适配，不能只把命令里的 dataset 名字替换后就认为已经支持。通用 `open_clip_train/main.py` 的 CSV 数据集只返回 image/text、未读取 labels，也不会产生此 runner 要求的教师阶段元数据；不要把它的普通 CLIP checkpoint 当成 `teacher_t1.pt/teacher_t2.pt` 直接传入。

## 高维 ViT 知识迁移：建议把目标分成两种

当前 ViT-L/14 配置的隐藏宽度为 **1024**、视觉文本共同空间为 **768**、FER logits 为 **7**。`normalize=False` 表示没有 L2 归一化，**不表示没有经过 CLIP 的视觉投影**。当前 `teacher.encode_image` 返回的是投影后的 768 维；S1 把三个学生层各自池化/投影后对齐该向量，S2 对齐融合向量。这是有效的全局特征蒸馏基线，应准确描述。

若要求更丰富的 ViT 表示，建议分开设置：

1. **768 维语义目标**：保留 CLIP 图像 embedding、冻结文本原型和 image/text 概率分布蒸馏，保持 CLIP 的视觉语言空间。
2. **1024 维视觉目标**：教师额外输出投影前归一化 CLS 或明确选定的多层 hidden/patch tokens；学生使用训练期专用投影头匹配教师宽度，或将师生映射到明确的共同蒸馏空间。语义路径仍保持 768 维，不能直接把所有 common_dim 改成 1024。

先增加投影前全局 CLS 蒸馏，可以避开 token 网格对齐的复杂性；再按验证收益考虑跨层或 patch/关系蒸馏。具体 ViT 层与学生层要记录实际索引，不能把学生最后三层都对齐同一个全局向量称为“教师三层对齐”。如果是官方 VMamba 而非 Vim，学生返回的是多尺度特征图，适配逻辑也要相应改变，不能沿用 CLS/token 假设。

对空间目标，224 时 ViT patch14 为 16×16，Vim patch16 为 14×14；不同 token 数不能直接逐元素比较。可对齐归一化空间网格、使用插值后的 patch 表示或尺寸无关的关系目标；必须保持同一几何视图。只用于训练的投影/蒸馏头，部署时应明确移除并单独报告其训练成本。

若特别强调保留“原始预训练”知识，可将冻结 T0 的视觉目标与 T2 的 FER logits 分别使用，或约束 T1/T2 对 T0 的偏移。当前只使用任务微调后的 T1/T2，因此迁移的是“预训练初始化后经 FER 适配的视觉表示”，并不保证原始知识完全保留。

## 其他训练可靠性问题

- S1 的类文本分布使用单位余弦除以 tau_it=2，却没有 CLIP logit_scale。余弦范围为 [-1,1]，分布倾向较平；应观察真实熵、KL 大小和梯度，再调整尺度/温度，避免把弱语义项当主要知识源。
- S2 采用 `KL(student || teacher)`，与常见的 `KL(teacher || student)` 不同。前者并非数学错误，但对教师概率很低的类别惩罚更强；方向应由明确的方法目标决定，记录并做必要对照，不能以单元测试匹配现有公式为正确研究设计的证据。
- teacher `logit_scale.exp()` 没有限幅；T1 修复后仍应监控尺度，必要时做显式 clamp 及有限梯度检查。
- runner 只记录 validation seed，没有统一设置 Python/NumPy/Torch/CUDA 和 worker RNG；模型初始化、shuffle 和增强不是完整可复现的实验种子。
- 当前 `strict_warmup=True, paper_warmup_steps=10000` 的通用 full config 会拒绝总步数不足的阶段；`rafdb_config.json` 已使用非严格比例回退。每个数据集和阶段应记录实际 warmup，不应机械套用相同步数。
- runner 只在最终保存模型，没有阶段内断点恢复；加载 best model 后却保存末尾 optimizer/scheduler。作为下阶段初始化尚可，但作为恢复 checkpoint 状态不一致，应分开保存 best/last 及同步的 optimizer/RNG。
- AMP 请求 float16 时 runner 没有创建/传递 GradScaler；BF16 路径可用，但实际云端硬件与精度应实测，不能从本地 CUDA 可用推断。
- 独立 `inference.py` 硬编码自写 Vim 工厂、默认 heads/use_cnn/use_coords/特殊 token 等配置，不能忠实重建外部工厂或所有消融；还可能把 metadata 中的二维 size 直接用于 Resize，改变训练时 shortest resize 语义。应统一从完整架构元数据和同一 transform 工厂重建。
- `cnn_initialization` 只是说明字符串；当前构建实际调用 `ResNet18(weights=None)`。主实验/对照须统一并准确报告随机或真实预训练初始化。
- `data.formal_training_ready`、`student.selected_layers` 等说明字段不都是执行控制。代码实际生效路径与文档应同步，清理旧的“AU 表/ViT-L 权重不存在”等断言，并按本地/云端范围准确表述。

## 本次验证

- 当前 `tests/test_vision_mamba_clip.py`：**10 passed**。使用 Python 3.11、PyTorch 2.5.1、本地已有 ftfy 包路径及 `src` 路径；无依赖安装、无权重下载。最初默认环境缺 ftfy，另一环境测试遇到 Windows 依赖/临时目录问题，最终通过上述现有运行时完成验证。
- `artifacts/training_review_probe_2026_10_13.py` 的小规模 CPU 检查结果保存在同名 JSON：三数据集支持状态、T1 同类损失、256/patch14 覆盖、无关 Vim checkpoint 的错误成功标记、dt 初始化、七条原型与所有离线 caption 的 token 长度。
- 检查使用合成 tensor/临时无关 checkpoint，没有云端训练、真实教师精度、官方 Vim 吞吐、三数据集收敛或准确率结果。10 项测试使用接口 stub，未覆盖上述真实骨干与数据协议缺口。

建议修改顺序：**统一云端三数据集 manifest/标签与配置 → 确认实际 Mamba 骨干并修正严格权重加载 → 匹配 CLIP 来源/输入协议 → 修复 T1 类级文本目标 → 明确并实现所需的高维蒸馏层级 → 补齐可复现与 checkpoint 恢复 → 云端真实小批量验收 → 完整训练。**
