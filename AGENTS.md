# 项目事实（用户于 2026-10-13 确认）

- 正式训练在云电脑运行，真实数据也在云电脑。数据根目录、标注文件、预训练权重和阶段权重的路径，均须以云电脑实际目录为准。此工作区是代码审阅和修改环境，不能用本地资产情况推断云端情况。
- 用户已下载 CLIP ViT-L/14 预训练权重并放在云电脑。该权重是原始 CLIP 初始化；教师的任务微调权重需要用户运行教师训练后获得，不能把两者混为一谈。
- 当前数据处理范围（用户于 2026-10-13 更新）：处理 RAF-DB、SFEW、AffectNet。RAF-DB 标注位于项目根目录 rafdb_labels.txt，原始 1–7 映射到统一索引 6/2/1/3/5/0/4。SFEW 原始 0–6=Angry/Disgusted/Fearful/Happy/Neutral/Sad/Surprised（用户更正后确认）；AffectNet 原始 0–6=Neutral/Happy/Sad/Surprised/Fearful/Disgusted/Angry。SFEW、AffectNet 现有 val 保留为最终评估，从 train 每类留出 10% 作模型选择，seed=42。本次仅做数据处理，不做图片逐张解码或结构检查。
- 蒸馏目标是让学生的 Mamba 视觉骨干学习 CLIP ViT 图像塔的预训练视觉知识。审查时须区分高维视觉特征蒸馏与类别 logits 蒸馏。
- 用户称学生为 vmambaclip；当前源码实现声明为 bidirectional Vim。Vim 与官方 VMamba（SS2D/Cross-Scan）不是同一骨干，后续改变骨干前应明确实际目标。

审阅记录：`docs/TRAINING_CODE_REVIEW_2026-10-13.md`。其中建议是审阅结论，不代表训练代码已经完成修复，也不代表云端实验已经执行。

## 学生骨干替换（用户于 2026-10-17 确认）

- 最新 RAF-DB 分析后的优先方向为 E3 + CNN 辅助头／门控残差（R0–R6），详见 `docs/CNN_REFINEMENT_2026-10-17.md`。R1–R3 不启用 SASF、多尺度、多 Query 或文本；R4/5/6 独立从显式 refinement_base 派生。
- 支持训练 seeds 42/43/44，保持原 frozen manifest，独立输出与初始化身份；Class-Balanced CE 使用训练计数，不叠加平衡采样。本轮授权本地修改、测试及推送 dev，不自动执行云端实验。

- 用户后续要求按最终分析用一份代码覆盖所有实验：当前统一入口为 `python -m vim_v2 prepare-experiments`（schema 3），说明见 `docs/UNIFIED_EXPERIMENTS_2026-10-17.md`。Vim 仅作为 E1/E2 明确对照，主体仍为 VMamba-T s1l8。
- 多尺度 CNN、SASF e/m/l、多 Query、文本原型监督、KD/affinity 和类别采样均由实验配置控制；不能把不同配方的结果混作单因素结论。此轮没有额外授权云端训练或恢复暂停任务。

- 当前开发目标改为官方 VMamba-T（Tiny，s1l8，SS2D/Cross-Scan），替换此前开发模型中的 Vim；覆盖历史文档中禁止替换为 VMamba 的架构限制。
- 学生加入 SASF，并使用 VMamba 全局 Query、CNN 空间 tokens 作为 Key/Value 的跨分支注意力，保留全局特征补充融合。
- 新入口为 `python -m vim_v2 prepare-vmamba`；旧 Vim 配置/初始化/训练记录仅作为历史，不能当作 VMamba 结果或直接续训。
- 本轮授权为本地代码审查、修改和测试，没有授权自动启动、恢复或更改云端训练。云端暂停和数据资产限制仍按用户后续指令执行。

## 当前训练授权更新（2026-10-13）

- 用户后续明确要求依据简化实验计划训练；之前“本次仅做数据处理”属于历史阶段范围，不再禁止本轮训练。
- 使用SSH别名autodl-westd，项目仍为/root/autodl-tmp/open_clip-main；先完成三个数据集全部九个基线，必要的本域T1/T2教师准备允许执行，暂不排Full/消融/共享学生。
- 用户指定官方Vim，禁止用原项目自写双向SSM冒充。使用固定官方源码的Vim Small bimambav2，未提供官方学生预训练权重，准确标注seed42随机初始化。
- 单GPU串行，注意内存和存储，只保留每作业best/last及小型指标日志；禁止图片/全数据教师特征缓存和逐epoch检查点堆积。
- 当前基线入口open_clip_train.fer_baselines，官方middle-CLS/residual适配见official_vim.py，旧runner不能直接用于本轮官方基线。执行说明docs/BASELINE_RUNTIME_2026-10-13.md；实际验收和训练状态以文件存储中的日志/JSON为准。


## 补充训练授权更新（2026-10-14）

- 用户明确要求依据本地train.md补充当前计划并在autodl-westb继续训练，覆盖之前完成AffectNet Mamba CE后暂停和基线先行暂不Full的历史限制。
- 当前授权范围为18学生模型（9基线、3Full、RAF6消融）、6教师阶段、3新增zero-shot评估；T2最终评估复用。29个唯一训练阶段，不含共享教师、跨域或其他域消融。
- 不重复已完成模型，不更改其不可变baseline runtime/manifest/随机初始化与预算。新入口artifacts/training_prepare_20261013/run_expanded_workflow.py；补充协议docs/EXPERIMENT_PLAN_SUPPLEMENT_20261014.md。
- 必须单GPU串行、实际batch32验收、best/last和依赖SHA核验；失败先诊断，不自动调参。暂停原记录已归档，后续暂停请求优先。


## 教师优先并暂停更新（2026-10-15）

- 用户最新指令：先暂停当前AffectNet训练，训练全部教师模型后暂停计划。当前AffectNet Hybrid CE已在新检查点安全暂停，不能自动恢复。
- 只允许三个数据集各teacher_t1=5epoch、teacher_t2=10epoch及其必要验收，共六阶段。禁止KD、Full、消融、zero-shot或其他学生任务。
- 专用入口artifacts/training_prepare_20261013/teacher_only_workflow.py；结束时设置全局暂停并禁止续跑，需用户明确恢复。


## RAF/SFEW范围授权更新（2026-10-15）

- 用户最新指令：当前AffectNet T1完整结束并评估后，不启动AffectNet T2，转而执行现有补充计划全部RAF/SFEW相关实验。覆盖先前仅教师阶段的限制。
- 两域已完成Mamba CE/Hybrid CE/T1/T2不重复；剩余logits KD、Full，以及RAF六消融，维持既定初始化/manifest/loss/更新预算。
- 禁止其余AffectNet训练（含T2、Hybrid CE恢复、KD/Full）、共享教师、跨域、新seed或看最终集调参；单GPU串行，作用域结束暂停。
