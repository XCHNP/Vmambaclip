"""
包注解
1、OrderedDict：是一个字典子类，除了具有标准字典的功能外，它还记住了元素的插入顺序。
这意味着你在 OrderedDict 中插入元素的顺序会被保留下来
2、typing：Python 标准库中的一个模块，提供了多种用于类型注解的工具，
帮助开发者在代码中声明变量、函数参数、返回值等的类型。
3、torch：PyTorch 的核心模块，提供张量操作、深度学习模型构建等功能。
4、nn：用于构建神经网络的模块，包含层、损失函数等。
5、functional：包含常用的神经网络操作函数。
6、checkpoint：用于优化内存使用，减少训练过程中的显存占用。
7、to_2tuple 和 get_2d_sincos_pos_embed：自定义函数，用于处理元组转换和生成位置编码等任务。
"""
from collections import OrderedDict
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):  #继承标准语法，"class 子类名(父类名)"
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back).
    这个类继承自 torch.nn.LayerNorm，即它是 PyTorch 中层归一化（Layer Normalization）的扩展。
    LayerNorm 通常用于深度学习中对每个样本的特征进行归一化。
    这个类的目的，是为了处理半精度浮点数（fp16）的输入数据。
    由于 LayerNorm 在 fp16 类型下可能存在精度问题，类会将输入数据转换为 float32（单精度浮点数），
    执行层归一化操作，然后再将结果转换回原来的数据类型（即 x 的原始类型）。
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        #层归一化
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        #转回原来的数据类型
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype).
    在执行层归一化操作后，将结果从 float32 或其他类型转换回输入张量的原始数据类型。
    这通常是为了处理具有不同精度（如 float16）的张量，避免精度损失。
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    """
    一个自定义的激活函数类，继承自 nn.Module。该类实现了 快速 GELU 激活函数（Quick GELU）
    该实现是通过一种简化形式来近似 GELU 激活函数，以提高计算效率和减少 GPU 内存的消耗。
    通常，nn.GELU 是通过精确的数值计算（如使用误差函数 erf）来实现的，
    而 QuickGELU 使用了 sigmoid 函数的近似形式，能更快地计算。
注意：在注释中提到，尽管 QuickGELU 更快，但它的计算速度慢于 PyTorch 内建的 nn.GELU 或 nn.SiLU，
并且它的 GPU 内存消耗更多。
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    """
    是一个自定义的 PyTorch 模块，通常用于 缩放操作，
    其主要功能是对输入张量进行逐元素的缩放（乘以一个可学习的缩放因子）。
    这个类通过使用 nn.Parameter 定义了一个可训练的缩放因子 gamma，
    并在 forward 方法中根据需要对输入 x 进行缩放。
    """

    #dim:缩放因子维度，即gamma向量的大小
    #init_values = 1e-5：这是初始化gamma参数的初始值
    #inplace 是一个常用于 PyTorch 中的参数，它用于控制是否对张量执行 in-place 操作。In-place 操作指的是直接修改原始张量，而不是创建一个新的张量进行返回。这样可以节省内存，因为不需要分配新的内存来存储结果。
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        #gamma 是一个可训练的参数，初始化为一个大小为 dim 的张量，所有元素都设置为 init_values。nn.Parameter 将 gamma 定义为模型的参数，这样它将在训练过程中被优化。
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        #x.mul直接修改x，x*self.gamma是返回新的值
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    自定义的 PyTorch 模块，实现了一个与传统 Dropout 类似的功能，但是它是应用于图像的 patch（小块）上的。
    该模块的目标是通过 随机丢弃一部分 patches，来提高模型的鲁棒性。
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        # prob：这是一个浮动参数，表示丢弃的概率（即每个 patch 被丢弃的概率）。该值应该在 [0, 1) 范围内，prob=0 表示没有任何 patch 被丢弃，而 prob 越接近 1，丢弃的概率越大。
        self.prob = prob
        #CLS token 用于表示整个图像的语义信息
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.: #判断是否处于训练模式，或者丢弃概率为0时，不进行任何丢弃操作（避免了推理时执行不必要的丢弃操作）
            return x

        if self.exclude_first_token:
        #如果 true，那么将输入张量 x 的第一个 token（通常是 CLS token）单独分离出来，并将其余的部分作为 patches 存储在 x 中。
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))  #实际保留的 patches 的数量。确保至少保留一个 patch。

        rand = torch.randn(batch, num_tokens)  #生成随机正态分布张量，用来为每个patch生成随机值
        #使用 topk 方法选择保留的 patches。rand.topk(num_patches_keep, dim=-1) 返回每个样本中最大的 num_patches_keep 个随机值的索引，表示选择的 patches。
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int, # 输入张量的最后一维大小，也就是每个输入样本的特征维度
            num_heads: int = 8,
            qkv_bias: bool = True, # 是否为qkv添加偏置项
            scaled_cosine: bool = False, # 是否使用缩放余弦相似度来计算注意力分数。
            scale_heads: bool = False, # 是否对每个注意力头的输出进行缩放
            logit_scale_max: float = math.log(1. / 0.01), # 对计算出的注意力分数进行缩放的最大值   logit_scale 是一个可学习的参数，它控制着注意力分数的尺度
            batch_first: bool = True, # 是否将批次维度放在第一个位置
            # 补充：PyTorch 中的张量默认形式是 (sequence_length, batch_size, feature_size)
            attn_drop: float = 0., # 注意力权重的 dropout 概率
            # 该参数控制了丢弃的概率（即随机丢弃某些注意力连接）。例如，attn_drop=0.1 表示 10% 的注意力权重将被丢弃。
            proj_drop: float = 0. # 输出投影层的 dropout 概率
    ):
        super().__init__() # 用于继承 nn.Module父类方法
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads' # 这是一个断言语句，确保 dim（每个输入样本的特征维度）能够被 num_heads（注意力头的数量）整除。如果不能整除，就会抛出一个错误，提醒用户检查参数。
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # 计算缩放因子 scale
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale) # 用于生成输入投影权重的张量
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype) # orch.zeros_like():创建一个与 attn_mask 形状相同的张量，并且所有元素初始化为 0
            new_attn_mask.masked_fill_(attn_mask, float("-inf")) # masked_fill_ 是一个原地操作函数，它会根据 attn_mask 中的布尔值来填充 new_attn_mask
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))   # torch.bmm 执行批量矩阵乘法，计算q和k的点积，得到注意力分数 attn
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp() # 对 self.logit_scale 进行裁剪，确保其值不超过 self.logit_scale_max。.exp() 对裁剪后的 self.logit_scale 取指数，得到缩放因子 logit_scale
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.use_fsdpa: # 直接调用 F.scaled_dot_product_attention 函数进行缩放点积注意力计算，该函数会自动处理查询、键、值的缩放、掩码和 Dropout 操作
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale # 对查询 q 进行缩放，self.scale 是预先计算好的缩放因子
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C) # 这里的 -1 表示该维度的大小由其他维度和张量中元素的总数自动推断得出

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):  # 基于注意力机制的池化器
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model)) # 定义可学习的查询参数 self.query
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True) # 初始化多头注意力层 self.attn，使用 nn.MultiheadAttention 模块
        # 初始化两个归一化层 self.ln_q 和 self.ln_k，分别用于对查询和输入特征进行归一化处理
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0] # 获取张量x的批次大小
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module): # 实现了一个带有残差连接的注意力块
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None, # 前馈神经网络（MLP）隐藏层相对于输入维度的扩展比例，默认值为 4.0
            act_layer: Callable = nn.GELU, # 激活函数，默认使用 nn.GELU
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False
        # 初始化一个由多个 ResidualAttentionBlock 组成的模块列表 self.resblocks
        self.resblocks = nn.ModuleList([  # nn.ModuleList 是 PyTorch 中的一个容器类，用于存储多个 nn.Module 子类的实例
            # 列表推导式的基本语法是 [expression for item in iterable];其实本质就是循环生成残差链接层
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype: # -> 是函数返回值类型注解的语法,允许为函数的参数和返回值指定预期的数据类型
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x

# 自定义的 Transformer 模块，继承自 nn.Module。
# 它允许用户使用不同类型的注意力块来构建 Transformer 架构，通过灵活配置 block_types 参数，可以在不同层使用不同的注意力块
class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run trasnformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> torch.dtype:
        weight = self.resblocks[0].get_reference_weight()
        if hasattr(weight, 'int8_original_dtype'):
            return weight.int8_original_dtype
        return weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int, # 图像分割成的小块的尺寸
            patch_size: int,
            width: int,
            layers: int, # Transformer 层的数量
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None, # LayerScale 层的初始化值
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,  # 是否禁用预归一化层
            pos_embed_type: str = 'learnable',  # 位置嵌入的类型，如 learnable 或 sin_cos_2d
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False, # 是否输出所有小块的特征表示
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) # 可学习的参数,用于表示整个图像的类别信息
        # 位置嵌入
        if pos_embed_type == 'learnable':  # 认为是可学习的位置嵌入
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d': # 使用固定的二维正弦余弦位置嵌入
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False) # self.grid_size 是一个二元组 (height_grid, width_grid)，其中 self.grid_size[0] 代表网格的行数，self.grid_size[1] 代表网格的列数
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity() # 小块丢弃

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width) # 预归一化层
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

#！！！！！！！！！！！！！！！！训练需要调整！！！！！！！！！！！！！！
    # 锁定（冻结）VisionTransformer 模型中的部分或全部参数，使其在训练过程中不参与梯度更新
    def lock(self, unlocked_groups=0, freeze_bn_stats=False): # freeze_bn_stats参数用于控制是否冻结批量归一化层（Batch Normalization，简称 BN 层）的统计信息，true冻结
        # unlocked_groups = 0 时，意味着不解锁任何参数组，所有参数都被锁定，在训练过程中不会更新，unlocked_groups = 1 时，最后一个参数组（这里是 self.proj）会被解锁，该组中的参数可以在训练时更新，而其他组的参数仍然被锁定
        for param in self.parameters(): # self.parameters() 会返回模型中所有可训练的参数
            param.requires_grad = False # false为不更新

        if unlocked_groups != 0:
            groups = [ # 学习参数分组
                [
                    self.conv1, # 卷积层
                    self.class_embedding, # 分类嵌入
                    self.positional_embedding, # 位置嵌入
                    self.ln_pre, # 预归一化层
                ],
                *self.transformer.resblocks[:-1],  # 将 self.transformer.resblocks 中除最后一个块之外的所有残差注意力块作为单独的组展开添加到 groups 列表中
                [
                    self.transformer.resblocks[-1], # self.transformer.resblocks 的最后一个块
                    self.ln_post, #后归一化层 self.ln_post
                ],
                self.proj, # 投影层
            ]

            def _unlock(x): # 递归函数，用于解锁参数组中的参数
                if isinstance(x, Sequence): # 如果输入 x 是一个序列（例如列表），则遍历该序列中的每个元素，并递归调用 _unlock 函数对每个元素进行处理。
                    for g in x:
                        _unlock(g)
                else: # 通过前面的递归得到每个序列中最底层的数据，判断类型进行解锁
                    if isinstance(x, torch.nn.Parameter): # 此类直接解锁
                        x.requires_grad = True
                    else:
                        for p in x.parameters(): # nn.Module 对象（如 nn.Conv2d、nn.Linear 等），则遍历该模块的所有参数
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self): # 对 vit 模型中的参数进行初始化操作
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer # CLIP模型没有为视觉 Transformer 部分定义初始化函数
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore # PyTorch 的特定装饰器，用于排除方法被 TorchScript 编译
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: Optional[int] = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_type: str = 'linear',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_type == 'none' or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if self.cls_emb is not None:
            no_wd.add('cls_emb')
        return no_wd

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
            batch_first: bool = True,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
        )
        self.context_length = context_length # 文本序列的最大长度，这个值会用于构建注意力掩码，以确保模型在处理文本时遵循因果关系，即每个位置的输出只能依赖于该位置及其之前的输入
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

        # 将注意力掩码注册为模型的缓冲区（buffer）。缓冲区是一种特殊的张量，它不是模型的可训练参数（即不会在反向传播中更新），但会随着模型一起保存和加载。
            # 'attn_mask'：缓冲区的名称，后续可以通过 self.attn_mask 访问该注意力掩码
            # 调用 self.build_attention_mask 方法生成注意力掩码。这个方法通常会创建一个上三角矩阵，用于在自注意力机制中屏蔽未来的信息
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self): # 初始化模型的参数
        # 计算不同层的权重初始化标准差 proj_std、attn_std 和 fc_std
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        # 将自注意力层和跨注意力层的权重初始化为正态分布
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # 对文本投影层的权重进行初始化
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal # 使用 triu_ 函数将下三角部分置为零，上三角为-inf(softmax中映射为0)，得到一个上三角矩阵作为注意力掩码
        return mask

    def forward(self, image_embs, text_embs):
        seq_len = text_embs.shape[1]
        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND
 # 使用 zip 函数同时遍历 self.resblocks（自注意力块列表）和 self.cross_attn（跨注意力块列表），对每一组自注意力块和跨注意力块依次进行处理。
        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            # self.grad_checkpointing：这是一个布尔类型的标志，用于指示是否启用梯度检查点技术：通过在反向传播时重新计算一些中间结果，避免在正向传播时保存所有中间激活值，从而降低内存开销
            # torch.jit.is_scripting() 函数用于检查当前代码是否正在被 TorchScript 脚本化
            if self.grad_checkpointing and not torch.jit.is_scripting(): # 启用梯度检查点且未使用 TorchScript 脚本化的情况
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
