# VMamba-T + CNN + SASF 审查与实现记录

日期：2026-10-17。范围：本地 `dev` 分支代码审查、修改和验证；没有启动、恢复、暂停或修改任何云端实验。

## 已落实的用户选择

- 骨干替换为官方 VMamba-T（Tiny，s1l8），使用 SS2D/Cross-Scan；历史 Vim Small 不再作为本次开发目标。
- 加入 SASF；融合优先考虑充分利用两个分支的特征。
- VMamba 增强后的全局表征作 Query，CNN 空间 tokens 作 Key/Value；同时保留两个分支的全局信息。
- 沿用当前 V2 的分类与特征蒸馏入口，不把参考总结中的旧 AU 方案、3.22 方案与现有损失混合。

## 审查发现与处理

| 发现 | 影响 | 本次处理 |
| --- | --- | --- |
| 当前学生实际为官方 Vim Small，而非 VMamba | 不能沿用 VMamba 名称描述结果，权重也不兼容 | 新增固定版本的官方 VMamba-T 源码和独立工厂；新模型身份为 `vmamba_hybrid` |
| 当前 e/m/l 对应最后三层 21/22/23 | 不是不同空间尺度，也不是网络前/中/后部 | VMamba 改取 stage 1/2/3 的降采样前二维特征，对应 stride 8/16/32 |
| SASF 尚不存在 | 没有空间增强路径 | 新增 dilation=1/2/3 的深度可分离卷积分支、融合、残差、池化投影 |
| 原注意力只在 Mamba 三个池化状态之间聚合，CNN 先池化再拼接 | CNN 空间位置在跨分支交互之前丢失 | 保留原聚合得到全局 Query，新增真正的 VMamba→CNN 空间 Cross-Attention |
| 官方 Vim 通用旧组装入口使用前缀 CLS 假设、忽略 tuple 的 residual | 与中间 CLS 的官方 Vim 不匹配 | 旧入口遇到 OfficialBackbone 时改用已有 OfficialPatchAdapter；VMamba 新路径不使用 CLS/token hook |
| 官方 VMamba 命名工厂忽略传入 kwargs | 请求的 drop path/activation checkpointing 可能无效 | 使用同一官方 VSSM 类和完全相同的 s1l8 结构参数显式构建；drop path=.1、checkpointing=True 沿用当前配方 |
| 官方单文件在没有 Triton 时仍定义 `@triton.jit` 函数 | 本地 CPU 导入失败 | 仅为 Triton 内核声明增加 WITH_TRITON 条件，不修改 SS2D、扫描、合并或 reference 数学运算 |
| 官方发布权重早于 grouped Linear 参数命名 | 直接比较 state_dict 键名会误判不兼容 | 只按官方 SS2D 自带映射转换 x_proj_weight/dt_projs_weight；随后完整键、形状、有限值校验及 strict=True 加载 |
| 原独立推理入口只认识旧 stage 格式 | 无法重建 V2 学生检查点 | 支持 V2 identity.config 校验、完整模型重建及严格加载；推理不读取教师或预训练资产 |

## 最终数据流

```text
Image [B,3,256,256]
  ├─ Official VMamba-T s1l8 (SS2D, 4 stages)
  │    ├─ e: [B,192,32,32] → mean+max → Linear(384,768)
  │    ├─ m: [B,384,16,16] → mean+max → Linear(768,768)
  │    └─ l: [B,768, 8, 8] → mean+max → Linear(1536,768)
  │         └─ SASF(l): dilated DW+PW conv d=1/2/3
  │              → concat original and 3 branches → 1×1 + GN + GELU
  │              → spatial residual → mean pool → Linear(768,768)
  │    e/m/l projections → learned-query attention → m_global [B,768]
  │    m_query = m_global + tanh(alpha) * sasf_global
  └─ ResNet18 stem through layer3 → [B,256,16,16]
       ├─ spatial attention + projection → weighted CNN global descriptor
       └─ 1×1 projection → [B,768,16,16]
            × normalized spatial attention → CNN K/V [B,256,768]

CrossAttention(Q=m_query[B,1,768], K/V=CNN tokens[B,256,768])
  → LayerNorm(m_query + attention_output) [B,768]
  → concat CNN global descriptor → Linear(1536,768) → fused_raw
  → concat(e_projection, m_projection, l_projection, fused_raw)
  → classifier [B,7]
```

SASF 接收真实二维特征，VMamba 没有 CLS，不从全局向量猜网格。CNN 的 K/V 在注意力之前不作全局池化。SASF 输出进入最终融合，因此能接受分类和 V2 fused feature 蒸馏的监督。

alpha 初始化为 0：第一次反向传播只更新 SASF 门控，内部卷积的梯度为 0；门控更新后卷积开始学习。这是可控残差的预期行为。测试检查了连续两步，不能把首步零梯度当作 SASF 断路。CNN 和 VMamba 本体、跨分支注意力从开始就参与优化。

现有分类头仍同时读取三阶段表征和 fused_raw；`fused_raw` 的蒸馏目标维度保持 768。旧 AU/坐标通道/landmark 设计没有额外引入。参考中的 CNN Spatial Mixer、单独 Mamba channel gating 不是本轮必选组件，没有据此扩展损失或声称复现完整旧 b 版本。

## 源码和预训练权重

- 官方仓库：[MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)。固定 commit：`2ed52ead062a51a64521ed3871d52914bf532876`。
- 官方单文件及 MIT License 位于 `third_party/VMamba/`。原始 LF SHA256、补丁后 LF SHA256 和兼容补丁说明均记录于 `official_source.json`，每次构造前检查。
- 官方权重：[vssm1_tiny_0230s_ckpt_epoch_264.pth](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth)。已下载到本地工作目录并实际校验，SHA256 为 `d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a`。
- 实际加载：完整覆盖官方分类模型的所有张量，按官方规则转换 28 个投影参数后 strict=True 成功；随后才移除 ImageNet 分类头。Hybrid 使用无 CLS 的 stage maps，不保留未使用的最终分类归一化参数；native CE 对照保留官方 final norm 和 spatial mean。
- `initialize=False` 只用于完整学生检查点重建，不把随机骨干声明为预训练。正式新建 Hybrid 仍需显式提供 VMamba 和 CNN 权重与 SHA256。

## 配置与使用

新入口：

```bash
python -m vim_v2 prepare-vmamba --paths paths.vmamba.json --output configs/vision_mamba_clip/vmamba_sasf
```

`paths.vmamba.json` 基于现有云端 paths 文件，保留真实 datasets/manifest/counts/teachers/cnn/clip_base 条目，增加 `vmamba` 权重条目，并设置**新的** output_root。不要根据本机目录推断云端资产路径。新增字段示例见 `configs/vision_mamba_clip/vmamba_pretrained.example.json`，它是字段示例，不是可以直接训练的完整配置。

`prepare-vmamba` 默认启用 SASF 和跨分支注意力；`--no-sasf`、`--legacy-fusion` 仅用于显式变体。生成计划本身不启动训练。它复用现有 V2 计划的预算结构，并不意味着本轮重新授权全部训练任务；云端运行范围仍需服从用户最新指令。

新初始化 ID 示例：`vmamba_hybrid_sasf_crossattn_pretrained_seed42`。旧 Vim 配置、初始化、checkpoint 和历史结果保持原身份，不能作为 VMamba 续训。新建计划必须重新完成当前代码/资产/GPU 验收；不复用旧 plan.lock。

学生推理支持 V2 检查点：

```bash
python -m open_clip.vision_mamba_clip.inference --checkpoint /absolute/path/best.pt --image /absolute/path/face.jpg --device cuda
```

从仓库运行（V2 运行器位于仓库根目录）。正常依赖之外需要 `requirements-vmamba.txt`。本地 CPU 测试使用官方 PyTorch selective-scan reference；GPU 加速需要与云端 Python/PyTorch/CUDA 匹配的 Triton 和 selective_scan_cuda/mamba-ssm，不要假设 Vim 环境已经满足。旧 `prepare-v2` 和旧 stage runner 保留为历史 Vim 路径，当前 VMamba 使用 `prepare-vmamba`。

## 验证与边界

- 已通过真实官方 VMamba 的 256×256 三尺度输出检查；小尺寸 CPU 前向/反向传播；与官方 s1l8 工厂参数逐项一致；native 特征与官方 readout 数值一致。
- 已验证 SASF 的二维/矩形网格、CLS/尺寸异常拒绝、零门控初始等价性与两步后非零梯度。
- 已验证 Q 为 [B,1,768]、CNN K/V 保留所有空间位置；真实 VMamba+Cnn+SASF 在 V2 S1 feature 蒸馏和 S2 loss 下均有梯度，教师分支规则未更改。
- 已验证真实 SS2D activation checkpointing 的前向和反向一致性、优化器覆盖新模块、SSM A_logs 无权重衰减、检查点严格重建及无教师推理。
- 回归覆盖原 VisionMambaCLIP 和 V2 contract 测试，共 44 项通过、0 项失败。测试命令：`python -m pytest tests/test_vmamba.py tests/test_sasf.py tests/test_vision_mamba_clip.py tests_vim_v2/test_contracts.py -q`。
- 额外使用真实官方预训练 VMamba、随机 CNN/新融合头完成 256×256 整体前向验证，得到 fused=[1,768]、logits=[1,7] 且均为有限值；当前完整 Hybrid 共 48,147,785 个参数。它是接口 smoke，不是已训练的学生结果。
- 尚未执行云端 CUDA/BF16/实际 batch32 验收、正式训练、准确率比较或显存/速度测量。不能据此声称准确率提升；跨分支融合的收益仍需固定数据划分下的训练验证。
