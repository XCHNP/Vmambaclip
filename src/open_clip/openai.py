'''
该文件主要处理与 OpenAI 相关的模型操作。
例如，load_openai_model函数可能用于加载 OpenAI 的预训练模型，list_openai_models函数用于列出可用的 OpenAI 模型，方便用户使用 OpenAI 的模型资源。
'''

""" OpenAI pretrained model functions
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url

__all__ = ["list_openai_models", "load_openai_model"]

# 返回可用的 OpenAI CLIP 模型的名称列表
def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')

# 加载指定名称的 OpenAI 预训练 CLIP 模型
def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    # 设备和精度设置
    # 如果未指定设备和精度，会根据当前环境自动设置。如果有可用的 CUDA 设备，默认使用 CUDA 并采用 'fp16' 精度；否则使用 CPU 并采用 'fp32' 精度
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'
    # 模型路径获取
    if get_pretrained_url(name, 'openai'): # 尝试从预训练模型的 URL 下载模型文件，或者使用本地文件路径。如果找不到模型，会抛出 RuntimeError 异常
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")
    # 模型加载
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        state_dict = torch.load(model_path, map_location="cpu")

    # Build a non-jit model from the OpenAI jitted model state dict
    # 模型构建
    cast_dtype = get_cast_dtype(precision)

    try:
        model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

    # 模型属性设置
    # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
    model = model.to(device)
    # FIXME support pure fp16/bf16 precision modes
    if precision != 'fp16':
        model.float()
        if precision == 'bf16':
            # for bf16, convert back to low-precision
            convert_weights_to_lp(model, dtype=torch.bfloat16)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model
