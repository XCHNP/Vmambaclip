'''
作为一个工厂模块，负责创建和管理项目中的各种模型、变换函数、损失函数等。
它提供了一些创建模型的函数，如create_model、create_model_and_transforms等，
方便用户根据不同的需求快速创建所需的模型和相关组件
'''
import json
import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .convert import convert_state_dict
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from .coca_model import CoCa
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    """ Fetch model config from builtin (local library) configs.
    """
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def _get_hf_config(
        model_id: str,
        cache_dir: Optional[str] = None,
):
    """ Fetch model config from HuggingFace Hub.
    """
    config_path = download_pretrained_from_hf(
        model_id,
        filename='open_clip_config.json',
        cache_dir=cache_dir,
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
):
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX):]
        try:
            config = _get_hf_config(model_name, cache_dir=cache_dir)['model_cfg']
        except Exception:
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                cache_dir=cache_dir,
                **kwargs,
            )
            return tokenizer
    else:
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        tokenizer = HFTokenizer(
            text_config['hf_tokenizer_name'],
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def load_state_dict(
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
        model: Union[CLIP, CustomTextCLIP],
        checkpoint_path: str,
        strict: bool = True,
        weights_only: bool = True,
        device='cpu',
):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        # Separate path loading numpy big_vision (SigLIP) weights
        from ..open_clip.convert import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # Detect & convert 3rd party state_dicts -> open_clip
    state_dict = convert_state_dict(model, state_dict)

    # Detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # correct if logit_scale differs in being scaler vs 1d param
    if 'logit_scale' in state_dict and model.logit_scale.ndim != state_dict['logit_scale'].ndim:
        state_dict['logit_scale'] = state_dict['logit_scale'].reshape(model.logit_scale.shape)

    # correct if logit_bias differs in being scaler vs 1d param
    if 'logit_bias' in state_dict and model.logit_bias.ndim != state_dict['logit_bias'].ndim:
        state_dict['logit_bias'] = state_dict['logit_bias'].reshape(model.logit_bias.shape)

    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if 'logit_bias' not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str, # 要创建的模型架构名称
        pretrained: Optional[str] = None, # 预训练模型权重的标签或路径
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        load_weights_only: bool = True,
        **model_kwargs,
):
    """Creates and configures a contrastive vision-language model.

    Args:
        model_name: Name of the model architecture to create. Can be a local model name
            or a Hugging Face model ID prefixed with 'hf-hub:'.
        pretrained: Tag/path for pretrained model weights. Can be:
            - A pretrained tag name (e.g., 'openai')
            - A path to local weights
            - None to initialize with random weights
        precision: Model precision/AMP configuration. Options:
            - 'fp32': 32-bit floating point
            - 'fp16'/'bf16': Mixed precision with FP32 for certain layers
            - 'pure_fp16'/'pure_bf16': Pure 16-bit precision
        device: Device to load the model on ('cpu', 'cuda', or torch.device object)
        jit: If True, JIT compile the model
        force_quick_gelu: Force use of QuickGELU activation
        force_custom_text: Force use of custom text encoder
        force_patch_dropout: Override default patch dropout value
        force_image_size: Override default image size for vision encoder
        force_preprocess_cfg: Override default preprocessing configuration
        pretrained_image: Load pretrained weights for timm vision models
        pretrained_hf: Load pretrained weights for HF text models when not loading CLIP weights
        cache_dir: Override default cache directory for downloaded model files
        output_dict: If True and model supports it, return dictionary of features
        require_pretrained: Raise error if pretrained weights cannot be loaded
        load_weights_only: Only deserialize model weights and unpickling torch checkpoints (for safety)
        **model_kwargs: Additional keyword arguments passed to model constructor

    Returns:
        Created and configured model instance

    Raises:
        RuntimeError: If model config is not found or required pretrained weights
            cannot be loaded

    Examples:
        # Create basic CLIP model
        model = create_model('ViT-B/32')

        # Create CLIP model with mixed precision on GPU
        model = create_model('ViT-B/32', precision='fp16', device='cuda')

        # Load pretrained OpenAI weights
        model = create_model('ViT-B/32', pretrained='openai')

        # Load Hugging Face model
        model = create_model('hf-hub:organization/model-name')
    """

    # 判断模型来源并获取配置
    # 根据输入的model_name是否以HF_HUB_PREFIX开头，来判断要创建的模型是否是来自 Hugging Face Hub 的模型，然后进行不同的处理
    model_tag=0 # 用来测试走了哪个判断语句
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix: # 处理来自huggingface hub的模型
        model_id = model_name[len(HF_HUB_PREFIX):]  # 提取模型 ID
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir) # 下载预训练权重，cache_dir 是下载文件的缓存目录
        config = _get_hf_config(model_id, cache_dir=cache_dir)  #获取模型配置
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg']) # 合并预处理配置
        model_cfg = config['model_cfg'] # 从获取的配置信息 config 中提取出模型的配置字典 model_cfg
        pretrained_hf = False  # 禁用加载 Hugging Face 文本模型的预训练权重# override, no need to load original HF text weights
    else: # 处理非huggingace hub的模型
        model_tag=1
        model_name = model_name.replace('/', '-')  # 将 model_name 中的 / 替换为 -，这是为了兼容旧的命名方式，例如 ViT-B/32 会被转换为 ViT-B-32 # for callers using old naming with / in ViT names
        # 初始化检查点路径和模型配置 表示还没有获取到对应的检查点路径和模型配置，后续会根据具体情况进行处理
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):  # isinstance函数用于检查一个对象是否是指定类或其子类的实例
        device = torch.device(device)

    # print(f'facory.py 312 Loaded {model_tag}')

    # 获取配置
    model_cfg = model_cfg or get_model_config(model_name) # 检查 model_cfg 是否已经存在（之前可能从 Hugging Face Hub 配置中获取到），如果不存在，则调用 get_model_config(model_name) 函数，根据传入的 model_name 获取对应的模型配置
    if model_cfg is not None: # 如果成功获取到模型配置（model_cfg 不为 None）
        logging.info(f'Loaded {model_name} model config.') # 使用 logging.info 记录一条信息，表明已成功加载该模型的配置
    else: # 如果没有获取到模型配置（model_cfg 为 None），并列出所有可用的模型（通过 list_models() 函数获取）
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.') # 使用 logging.error 记录错误信息，显示找不到该模型的配置
        raise RuntimeError(f'Model config for {model_name} not found.') # 然后抛出 RuntimeError 异常，提示找不到该模型的配置

    if force_quick_gelu: # 如果 force_quick_gelu 为 True，则将模型配置中的 quick_gelu 字段设置为 True，表示强制使用 QuickGELU 激活函数
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if force_patch_dropout is not None: # 覆盖默认的 patch dropout 值
        # override the default patch dropout value
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout # 将模型配置中视觉编码器配置（vision_cfg）的 patch_dropout 字段设置为 force_patch_dropout 的值，从而覆盖默认的 patch dropout 值

    if force_image_size is not None: #覆盖模型配置的图像大小
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    # 检查是否为 timm 模型并处理预训练图像权重
    is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
    if pretrained_image:
        if is_timm_model:
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else: # 如果不是 timm 模型（is_timm_model 为 False），则使用 assert 语句抛出 AssertionError 异常，提示当前仅支持 timm 模型加载预训练图像权重
            assert False, 'pretrained image towers currently only supported for timm models'

    # 处理模型配置和创建模型
    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = get_cast_dtype(precision) # 获取数据类型
    # 判断是否为 Hugging Face 模型并设置预训练标志
    is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
    if is_hf_model:
        # load pretrained weights for HF text model IFF no CLIP weights being loaded
        model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained # pretrained_hf 表示是否要加载 Hugging Face 文本模型的预训练权重，pretrained 表示是否要加载 CLIP 模型的预训练权重
    custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model # 布尔标志，用于决定是否使用自定义文本编码器

    model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
    if custom_text: # 根据custom_text标志选择创建CoCa、CustomTextCLIP或CLIP模型
        if "multimodal_cfg" in model_cfg:
            model = CoCa(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
    else:
        model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)
            from .transformer import LayerNormFp32

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)
            model.apply(_convert_ln)
        else:
            model.to(device=device)
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    pretrained_loaded = False
    if pretrained: # 当 pretrained 参数有值时，程序会尝试获取预训练配置并下载预训练权重
        # 获取预训练配置和检查点路径
        checkpoint_path = ''
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg: # 如果能获取到配置pretrained_cfg
            checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir) # 下载预训练权重并得到检查点路径
            preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg) # 合并预处理配置
            pretrained_quick_gelu = pretrained_cfg.get('quick_gelu', False)
            model_quick_gelu = model_cfg.get('quick_gelu', False)
            if pretrained_quick_gelu and not model_quick_gelu:
                warnings.warn(
                    f'These pretrained weights were trained with QuickGELU activation but the model config does '
                    f'not have that enabled. Consider using a model config with a "-quickgelu" suffix or enable with a flag.')
            elif not pretrained_quick_gelu and model_quick_gelu:
                warnings.warn(
                    f'The pretrained weights were not trained with QuickGELU but this activation is enabled in the '
                    f'model config, consider using a model config without QuickGELU or disable override flags.')
        elif os.path.exists(pretrained): # 如果获取不到配置，但 pretrained 本身是一个存在的文件路径，就将其作为检查点路径
            checkpoint_path = pretrained

        # 根据与训练权重路径加载预训练权重并更新标记
        if checkpoint_path: # 不为空，说明找到了预训练权重的路径
            # 使用 logging.info 记录一条信息，表明正在加载指定模型的预训练权重，其中 model_name 是模型的名称，pretrained 是预训练权重的标识（如标签名或路径）
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            # 调用 load_checkpoint 函数将预训练权重加载到 model
            load_checkpoint(model, checkpoint_path, weights_only=load_weights_only) # weights_only=load_weights_only 参数表示是否仅加载模型的权重
        else:# 预训练权重未找到
            error_str = ( # 构建错误字符串error_str
                f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
            logging.warning(error_str) # 记录警告信息
            # 抛出异常，后续代码不会继续执行（相当于直接异常退出了，因此pretrained_loaded不会设为true)
            raise RuntimeError(error_str) # 抛出 RuntimeError 异常，终止程序并显示错误信息，让用户知道加载预训练权重失败
        pretrained_loaded = True # 正常加载预训练权重后才会设为true，异常退出不会执行到这行语句
    elif has_hf_hub_prefix: # 检查模型名称是否以 HF_HUB_PREFIX 开头
        # 记录一条信息，表明正在加载来自 Hugging Face Hub 的指定模型的预训练权重
        # 注意此时预训练权重是从本地读取
        logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
        # 调用 load_checkpoint 函数将预训练权重加载到 model 中
        load_checkpoint(model, checkpoint_path, weights_only=load_weights_only)
        pretrained_loaded = True # 表示已成功加载来自huggingfacehub的预训练权重

    # 检查预训练权重是否成功加载
    if require_pretrained and not pretrained_loaded: # require_pretrained(bool)表示调用者是否要求必须加载预训练权重
        # callers of create_model_from_pretrained always expect pretrained weights
        raise RuntimeError( # 抛出异常信息,指出是哪个模型(model_name)，要求加载哪个预训练权重(pretrained)没有加载成功
            f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')
    # 根据参数设置模型的输出格式
        # output_dict(bool)表示调用者是否希望模型的输出是一个字典格式
        # hasattr(model, "output_dict") 用于检查模型对象 model 是否有 output_dict 这个属性。这一步是为了确保模型本身支持以字典形式输出
    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True # 运行时以字典形式输出结果

    if jit: # jit(bool)表示是否要对模型进行 JIT（Just-In-Time）编译
        model = torch.jit.script(model) # torch.jit.script(model) 是 PyTorch 提供的一个函数，它可以将模型 model 转换为 JIT 脚本模块

    # 设置模型的图像预处理配置
    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None: # getattr(model.visual, 'image_size', None) 用于获取模型视觉部分（model.visual）的 image_size 属性值。存在返回值，不存在返回none用于判断
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size # 确保预处理配置中的图像大小与模型实际使用的图像大小一致
    # 合并预处理配置并设置到模型中
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model


def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, "Horovod not currently supported for SigLip"
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
            dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
        )

    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(  #
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        load_weights_only: bool = True,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs( # 将传入的关键字参数合并到一个字典中
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        load_weights_only=load_weights_only,
        **model_kwargs,
    )

    # 获取预处理配置
    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg) # 模型视觉部分的预处理配置字典，通过解包操作将其传入 PreprocessCfg 类的构造函数，创建一个预处理配置对象 pp_cfg

     # 生成训练和验证的预处理转换函数
    preprocess_train = image_transform_v2( # 训练阶段预处理转换函数
        pp_cfg,
        is_train=True,  # 训练阶段is_train 参数设置为 True
        aug_cfg=aug_cfg,  # 传入数据增强配置 aug_cfg
    )
    preprocess_val = image_transform_v2( # 验证阶段预处理转换函数
        pp_cfg,
        is_train=False, # 验证阶段is_train 参数设置为 False，不进行数据增强操作
    )

    return model, preprocess_train, preprocess_val # 将创建好的模型、训练预处理转换函数和验证预处理转换函数作为元组返回


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        load_weights_only: bool = True,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        load_weights_only=load_weights_only,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess
