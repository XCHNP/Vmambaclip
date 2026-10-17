'''
用于对 OpenCLIP 模型进行简单的推理测试。
它定义了一组简单的测试模型及其配置（包括模型名称、预训练名称、是否使用 JIT、是否强制使用自定义文本等），
并使用pytest对这些配置进行参数化测试。
对于每个配置，会加载模型和相应的预处理函数，对一张示例图片和一组示例文本进行编码，
计算文本概率，并通过断言来验证计算结果是否符合预期。
'''
import torch
from PIL import Image
from src.open_clip.factory import get_tokenizer
import pytest
import src.open_clip
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)


test_simple_models = [
    # model, pretrained, jit, force_custom_text
    ("ViT-B-32", "laion2b_s34b_b79k", False, False),
    ("ViT-B-32", "laion2b_s34b_b79k", True, False),
    ("ViT-B-32", "laion2b_s34b_b79k", True, True),
    ("roberta-ViT-B-32", "laion2b_s12b_b32k", False, False),
]


@pytest.mark.parametrize("model_type,pretrained,jit,force_custom_text", test_simple_models)
def test_inference_simple(
        model_type,
        pretrained,
        jit,
        force_custom_text,
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained,
        jit=jit,
        force_custom_text=force_custom_text,
    )
    tokenizer = get_tokenizer(model_type)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert torch.allclose(text_probs.cpu()[0], torch.tensor([1.0, 0.0, 0.0]))
