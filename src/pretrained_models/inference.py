
import torch
from PIL import Image
import open_clip



def main():
    local_model_path = "/src/pretrained_models/open_clip_pytorch_model.bin"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=local_model_path)
    # 将模型设置为评估模式
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

    # 获取文本分词器：将输入的文本转换为模型可以处理的 token 序列
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # 加载并预处理图像
    # Image.open("docs/CLIP.png") 使用 PIL 库打开指定路径下的图像文件;
    # preprocess是之前函数返回的图像预处理函数用于对图像进行一系列的预处理操作
    # .unsqueeze(0) 是 PyTorch 中的一个方法，用于在张量的第 0 个维度上添加一个维度，将图像从 (C, H, W) 形状转换为 (1, C, H, W) 形状，以模拟批量输入
    image = preprocess(Image.open("/docs/CLIP.png")).unsqueeze(0)
    # 对文本进行分词：将每个文本字符串转换为对应的 token 序列
    text = tokenizer(["a diagram", "a dog", "a cat"])

    # 进行推理并计算相似度
    with torch.no_grad(), torch.cuda.amp.autocast(): # torch.no_grad() 是一个上下文管理器，用于在推理过程中禁用梯度计算，以减少内存消耗并提高推理速度；torch.cuda.amp.autocast() 是 PyTorch 中的自动混合精度训练和推理工具，用于在 GPU 上使用混合精度（FP16 和 FP32）进行计算，以提高计算效率
        image_features = model.encode_image(image) #对图像编码，将其转换为特征向量
        text_features = model.encode_text(text) # 对文本编码，将其转换为特征向量
        image_features /= image_features.norm(dim=-1, keepdim=True) # 对图像特征进行归一化处理，使其具有单位长度
        text_features /= text_features.norm(dim=-1, keepdim=True) # 对文本特征进行归一化处理，使其具有单位长度

        # 计算图像特征和文本特征之间的相似度，并通过 softmax 函数将相似度转换为概率分布
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

if __name__ == "__main__":
    # main()
    # 检查 CUDA 是否可用
    print(torch.cuda.is_available())

    # 获取当前可用的 CUDA 设备数量
    print(torch.cuda.device_count())

    # 获取当前 CUDA 设备的名称
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

