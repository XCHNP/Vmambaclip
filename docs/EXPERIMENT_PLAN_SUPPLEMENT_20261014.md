# FER 训练计划补充（2026-10-14）

依据用户 `C:\Users\xiaochai\Desktop\train.md`，在已有基线计划上新增 Full、RAF 消融及教师/零样本参考。服务器改为 SSH 别名 `autodl-westb`（connect.westb.seetacloud.com:32848）。项目 `/root/autodl-tmp/open_clip-main`，结果根目录 `/root/autodl-fs/fer_suite_simplified_20261013`。

本次用户“继续训练”授权恢复先前暂停的后续实验。保留暂停原始记录，不重复完成的 Mamba CE，不改变其代码身份、数据、初始化、预算或已报告分数。任何验收失败均停止，不盲目重试或自动改超参。

## 1. 实际范围与数量

| 类别 | 域 | 数量 | 预算/依赖 |
|---|---|---:|---|
| Mamba CE、Hybrid CE、logits-only KD | RAF / SFEW / AffectNet | 9 学生模型 | 各30epoch，KD依赖本域T2 |
| Full MambaClip | 三域 | 3 学生模型 | T1→S1(10epoch)，T1→T2(10epoch)→S2(20epoch) |
| −S1 / −CNN / −Feature / −Logits / −W-Align / −AU | 仅RAF | 6 学生模型 | 见消融表 |
| 本域教师T1、T2 | 三域 | 6 教师阶段 | T1=5epoch；T2=10epoch |
| T2参考、原始CLIP zero-shot | 三域 | 6 评估参考行 | T2复用已保存最终评估；zero-shot另外3次纯评估 |

`train.md` 中“22个训练实验”是计数笔误：按其条目为 **18个学生模型＋6个教师阶段**。考虑S1/S2和S1复用后，实际 **29个独立训练阶段作业**，另加3个zero-shot作业。不是22个学生模型。旧计划中的共享教师学生、18次跨域评估及其他域消融不在此次队列中。

执行顺序：跳过已完成三域Mamba CE → 剩余Hybrid CE → 本域T1/T2和logits KD → 完整基线报告验收 → 三域Full → RAF六项消融 → zero-shot → 完整补充报告。单GPU串行，每条训练路径先实际batch32×2、反向传播和严格checkpoint roundtrip验收。

## 2. 不变的比较协议

- 官方hustvl/Vim Small bimambav2，commit `dd0358ad1e42701f22afbefa0717cc8825cf9f45`；384维、24层、patch16、固定middle CLS。**seed42随机初始化，没有Vim预训练权重**，不能称为ImageNet预训练Vim结果。
- CLIP ViT-L/14使用云端原始归档及QuickGELU；冻结教师文本塔，按原始文件SHA重建。Hybrid的ResNet18至layer3同样随机初始化。
- 冻结manifest、类别顺序、train/selection/final分离不变；RAF 11040/1231/3068，SFEW 763/88/402，AffectNet 255507/28393/3500。SFEW/AffectNet原val仅作final_eval。train按类别留出10%，seed42。
- RGB256、CLIP normalize、bicubic；师生共享同一个增强tensor，训练按需解码，不新增全数据解码检查或缓存。原图片不删除。
- batch32、drop_last；AdamW lr1e-4、betas(.9,.98)、eps1e-6、wd.01；BF16、gradient checkpoint、clip-norm1；4 workers、8 CPU线程。仅AffectNet CE使用训练集逆频率权重。
- 每个学生总预算30epoch。RAF每轮345更新，SFEW23，AffectNet7984。对应30轮10350/690/239520更新；S1/S2相加匹配该预算。
- T1按selection原型CE选best；S1按selection S1目标最小值选best，**不评估未训练分类头的准确率**；T2、CE、KD和S2按selection UAR、同分Accuracy选best。最终集仅在训练结束加载best报告，不据其调参或换初始化。

## 3. 损失定义与修复边界

T1已采用七类AU原型CE并限制scale≤100，避免`anchors[labels]`成对CLIP损失造成同类伪负样本。本次入口不调用旧的成对T1路径。T2训练视觉塔和FER分类头。原基线runtime SHA保持不变，因此已完成模型不会因补充代码而身份失配。

**logits-only KD**：`CE + 0.2×KL(teacher || student)×T²`，T=2。本域T2冻结；无feature cosine、W-Align、AU或S1，不使用包含其他项的Stage2Loss作为它的目标。

**Full S1**：三层semantic KL（teacher||student，T=2）＋三层feature cosine（各权重1/3）＋CNN cosine，系数分别1/1/1；TV=1e-4，attention sparsity=1e-3。训练Vim、状态投影、CNN与空间注意力；分类头、融合/聚合和文本投影冻结。使用本域T1 best。

**Full S2**：CE + 0.2×fused feature cosine + 0.2×logit KL×T² + 0.05×W-Align + 0.1×AU空间目标。T=2，tau_low=.1、gamma_sparse=.001、空间目标detach。使用本域T2 best及本模型S1 best。AffectNet CE类权重与基线一致。文本投影可训练；各有效项独立系数，消融不会误关另一项。

沿用旧Full损失规定的logit KL方向`student||teacher`；它与已冻结logits-only KD的`teacher||student`不同，报告须披露，不能把Full相对KD变化全部归因于新增组件。Full阶段切换从S1 best开始并重建Adam；基线不重建Adam。此优化器状态差异也须披露，不宣称严格只有loss不同的对照。

## 4. RAF单变量消融

| 变体 | S1来源/预算 | S2预算 | 精确改动 |
|---|---|---:|---|
| no_s1 | 无，随机同源Hybrid | 30epoch | 不做S1；仍使用10+20 LR分段，保留总更新预算 |
| no_cnn | 独立S1，10epoch | 20epoch | 删除CNN/空间注意力；关闭CNN cosine、TV、sparse及AU；其余保留 |
| no_feature_align | 独立S1，10epoch | 20epoch | S1三层和CNN cosine=0，S2 fused cosine=0；semantic KL/其余项保留 |
| no_logits | 引用Full-RAF S1 best，不复制 | 20epoch | 仅S2 logit系数=0 |
| no_w_align | 引用Full-RAF S1 best，不复制 | 20epoch | 仅S2 W-Align系数=0 |
| no_au | 引用Full-RAF S1 best，不复制 | 20epoch | 仅S2 AU总项系数=0（含该项sparse） |

no_cnn先构造完整seed42初始化再去掉CNN，因此保留模块的初值不因随机数消耗顺序改变。三项S2-only消融与Full严格引用同一个S1 SHA。消融报告的Δ=各消融最终UAR−Full最终UAR；不以最终结果选checkpoint。

## 5. 纯评估、验收与产物

zero-shot使用原始、未经T1/T2修改的CLIP，固定七条class-name prompt：`a photo of a {angry/disgusted/fearful/happy/neutral/sad/surprised} face.`。按归一化视觉/文本相似度分类，固定后不依final_eval选择模板。没有人为保证的参数量，实际加载后统计。

完成验收包括：29阶段预算与状态，冻结manifest和源代码SHA，teacher/S1父checkpoint SHA，严格模型加载、32张真实样本有限前向与预测核对，预测行/标签/混淆矩阵及Acc/UAR/macroF1复算。技术能运行不等于可部署；不虚构生产阈值、统计显著性或多seed结论。

- 计划：`expanded_plan.json`；状态：`expanded_workflow_status.json`。
- 原基线/教师：`<dataset>/<variant>/`，保持原路径不动。
- 新Full/消融：`<dataset>/<variant>/s1/`或`s2/`；只保留每阶段`best.pt`、`last.pt`、配置、曲线、指标与预测，不留逐epoch权重。
- zero-shot：`<dataset>/clip_zero_shot/`，仅配置、指标和预测，无权重副本。
- 完整报告：`analysis/EXPANDED_ANALYSIS_REPORT.md`、`analysis/expanded_report_data.json`、`analysis/expanded_report_status.json`。原完整基线报告仍保留。

显存/内存/存储优先：不下载模型、不缓存全数据或teacher特征、不复制复用S1；每1000update和epoch末原子保存last。磁盘空闲不足12GiB不启动下一作业。任何代码/检查点身份不兼容或验收失败须停止并报告，不能盲目换配置恢复。
