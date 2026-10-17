# VisionMambaCLIP V2 实现与运行说明

本版本在 `autodl-westb-50061` 的 `/root/autodl-tmp/open_clip-main` 内增加 `vim_v2/` 独立入口，复用云端官方 Vim 工厂、原有 Hybrid 结构和固定 manifest。没有将本地自写 SSM 上传覆盖云端。旧基线与 Full 的入口、结果、队列及 AffectNet 任务保持原有身份。本轮按用户最后指令不启动正式训练。

## 实现边界

已实现核心 18 条学生路径和 21 个阶段的配置生成、严格初始化、训练、恢复、选择集评估、锁定后最终评估、逐样本概率输出、指标复算、计算预算估算与学生推理计时。所有启动脚本使用真实服务器路径，默认单 GPU 串行。

GPU 尚不可见时，不能声称官方核验 forward/backward、BF16、checkpointing、学习能力或泛化已通过。代码会拒绝越过这些验收。SFEW 固定 manifest 发现跨 train/selection 的内容重复与标签冲突，因此保留原分割并阻止验收；没有擅自删除样本。

扩展配置处于禁用状态，列明独立预算；本版本不提供绕过核心身份校验的 transfer。`legacy_full.disabled.json` 指向保留的旧 Full 入口并注明其 KL 和优化器差异。禁用扩展的检查命令会明确退出，不会假称启动了扩展实验。

## 模块对应

| 新模块 | 作用与兼容性 |
|---|---|
| `vim_v2/models.py` | 调用云端 `official_vim.py` 的官方工厂；固定标准 Small stride16 权重 SHA；仅丢弃 ImageNet head；middle CLS 插值；独立头 RNG；ResNet18 V1 严格加载；冻结、核验 T1/T2 |
| `vim_v2/data.py` | 读取三域已有 manifest，不重新划分；一次 RGB 几何增强，分别 ImageNet/CLIP Normalize；全量解码、字节重复及路径审计 |
| `vim_v2/losses.py` | 独立 CE/feature/logits 系数，正向 KL、T=2、T²；FP32 运算；零项不读取目标 |
| `vim_v2/optim.py` | 固定参数组包含 S2 头；S1 头冻结；SSM no-weight-decay；全程 optimizer-update cosine |
| `vim_v2/runtime.py` | 单卡锁、实际梯度/参数变化探针、学生和教师模式管理、原子 best/last、固定预算、无教师学生推理 |
| `vim_v2/checkpoint.py` | init/resume/inference 区分；optimizer/scheduler/scaler/RNG 保存；强身份校验；epoch 边界恢复 |
| `vim_v2/acceptance.py` | 官方 224/256 接口、checkpointing 梯度一致性、56 张训练图 CE 拟合、真实保存恢复与教师 selection 复算 |
| `vim_v2/config.py` | schema v2；18 路径/21 阶段；固定配方、分割、预算与阶段规则；拒绝旧 Full 行为混入 |
| `vim_v2/__main__.py` | inventory/prepare/validate/audit/smoke/fit/run/resume/evaluate/aggregate/budget/benchmark/lock |

Hybrid 的层为零基索引 21、22、23，即最后三层。保留的接口取 post-block mixer output + residual，排除 middle CLS 后做 mean/max pooling；它不是官方 final norm 的输出。Hybrid 的 `norm_f` 不参与分类路径，因此显式冻结。native 模式直接使用官方最终 norm 后 middle CLS 的 384 维特征。

主 feature 是 CLIP 视觉投影后的全局向量，构建教师时运行检查为 768 维。未实现或宣称 1024 维 patch/多层 ViT 蒸馏。Hybrid 的 Vim 和 CNN 都预训练；与 native 的差异不能单独归因为 Vim 预训练。

## 真实命令

在服务器执行以下命令。当前阶段仅执行检查；正式训练由用户在 GPU 验收通过后启动。

```bash
cd /root/autodl-tmp/open_clip-main
bash scripts/vim_v2.sh inventory --output artifacts/vim_v2_20261016/runtime_inventory.json
bash scripts/vim_v2.sh prepare-v2 --paths artifacts/vim_v2_20261016/paths.json --output configs/vision_mamba_clip/v2
bash scripts/vim_v2.sh validate-plan --plan configs/vision_mamba_clip/v2/plan.json --dry-run

# 一键 CPU 数据、配置与初始化验证。会完整解码三个固定 manifest。
bash scripts/verify_vim_v2.sh

# GPU 可见且数据问题处理后，执行真实 GPU 验收、拟合、恢复和教师复算。
# 该命令包含有上限的工程短跑，不执行正式矩阵。
bash scripts/verify_vim_v2.sh --gpu

bash scripts/vim_v2.sh budget --plan configs/vision_mamba_clip/v2/plan.json
bash scripts/vim_v2.sh lock-plan --plan configs/vision_mamba_clip/v2/plan.json

# RAF 四个初始化×预算诊断优先；A4 同时就是 RAF native_ce。
bash scripts/vim_v2.sh run-plan --plan configs/vision_mamba_clip/v2/plan.json --group diagnostic --resume-incomplete

# 核心矩阵串行训练。脚本会重新检查预算与验收，不自动 final_eval。
bash scripts/train_vim_v2.sh
bash scripts/resume_vim_v2.sh rafdb__native_ce
bash scripts/evaluate_vim_v2.sh rafdb__native_ce selection_val
bash scripts/evaluate_vim_v2.sh rafdb__native_ce final_eval
bash scripts/vim_v2.sh aggregate --plan configs/vision_mamba_clip/v2/plan.json
bash scripts/vim_v2.sh benchmark --config configs/vision_mamba_clip/v2/rafdb__native_ce.json --plan configs/vision_mamba_clip/v2/plan.json

# 查看禁用扩展及预算；明确退出，不启动训练。
bash scripts/extension_vim_v2.sh student_transfer
```

`lock-plan` 只允许通过全部对应验收的任务进入 qualified_runs，允许 CE 任务先于未就绪教师任务获得资格。没有合格任务时不会写 `plan.lock`。数据、配置、代码、实际 GPU 初始化、所有阶段 smoke、学习拟合、恢复和所需教师复算均是门槛。锁定后代码或配置变化会拒绝启动；需要新计划版本，不能修改已训练 run 的身份。

预算中的 GPU hours 基于实测暖机后训练 step 与 selection 外推，未计入数据等待、教师构建、checkpoint I/O、重算和重试，不能当作完整排程或计费时长保证。正式 curve 的 elapsed_seconds 会记录运行中的这些开销。显存字段使用 PyTorch `max_memory_allocated`，不代表驱动或其他进程的总占用。

若其他任务更新 AffectNet 教师，应在其完成后重新只读核验 SHA、完成状态与 selection，再生成新计划版本。不要将暂存 best 或原始 CLIP 当作合格任务教师。现有原始 CLIP、任务教师与其训练协议保持分别记录。

## 保存与恢复契约

正式输出根目录为 `/root/autodl-fs/fer_suite_v2_pretrained_20261016`。每个 run 保存 effective_config、initialization_report、updates.jsonl、curve.jsonl、best.pt、last.pt、status.json；Full 另保存 s1_best.pt、s1_last.pt、stage_transition.json 及阶段有效配置。S1→S2 逐项检查模型、Adam 参数组与 moments、scheduler 游标，保存并记录 S1 last 的 SHA 后继续；保持 optimizer 与全程 cosine，不回跳 best。S2 的 best/last 均携带此父 SHA。

last 只保证 epoch 边界恢复。多 worker 的几何增强由 seed/epoch/sample 确定；中途终止时重新执行未完成的整轮。丢弃更新会保留在 updates.jsonl 中，恢复前的 curve 归档，新的有效曲线从 last 内的记录恢复。不能称为 mid-epoch 精确续训，也不能把 model-only 加载称为 resume。跨硬件或非确定 CUDA kernel 不承诺逐位一致。

分类按 selection UAR、Accuracy、较早 epoch 选 best；S1 按 selection feature loss 最小选 best。训练不自动访问 final。最终评估输出七类混淆矩阵、逐类 precision/recall/F1、Accuracy/UAR/macro F1 与逐样本七类概率；汇总时校验样本顺序、标签、概率与指标复算。

## 结果解释

本轮尚未产生新的正式曲线、best/last、预测或性能结论。结果表使用空值和 not_started，不能把旧探索数值填入 V2。单 seed42 不报告多 seed 方差或显著性。

预训练作用分别比较 A2−A1、A4−A3；预算作用比较 A3−A1、A4−A2。30/60 各自从共享初值运行自己的完整 cosine。同结构 logits/feature KD 对 Hybrid CE；Full 对 feature KD 仍有教师切换、S1 无 CE 和监督日程差异。SFEW 需检查 train/selection 差距和预测类别分布，不能只看总体 Accuracy。

旧 final 已被看过，本轮只能表述为迭代开发后的同 benchmark 复评，不能称全新盲测。达到特定精度不属于验收条件。
