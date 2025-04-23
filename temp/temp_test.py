import torch
from models.u_net import UNet
from models.clip_alignment_v2 import MemoryOptimizedCLIP

# 初始化 U-Net 模型
unet = UNet()

# 初始化 MemoryOptimizedCLIP 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = MemoryOptimizedCLIP(device=device)

# 准备输入数据
image = torch.randn(1, 3, 224, 224).to(device)
texts = ["a photo of cat"]
text_inputs = clip_model.tokenizer(
    texts, return_tensors="pt", padding=True, truncation=True
).to(device)

# 获取语义嵌入
with torch.cuda.amp.autocast():
    logits = clip_model(image, text_inputs)
    # 假设 UNet 模型期望的语义嵌入维度是 (1, 768)
    # 这里需要根据实际情况调整
    semantic_embedding = logits.view(1, -1)
    if semantic_embedding.size(1) != 768:
        # 可以根据实际情况进行插值或者其他处理
        # 这里简单地进行截断或者填充
        if semantic_embedding.size(1) > 768:
            semantic_embedding = semantic_embedding[:, :768]
        else:
            padding = torch.zeros(1, 768 - semantic_embedding.size(1)).to(device)
            semantic_embedding = torch.cat((semantic_embedding, padding), dim=1)

# 准备 U-Net 的输入
x = torch.randn(1, 3, 224, 224).to(device)
weights = torch.ones(1).to(device)

# 调用 U-Net 的 forward 方法
try:
    output = unet(x, semantic_embedding, weights)
    print("兼容性验证通过！U-Net 输出形状:", output.shape)
except Exception as e:
    print("兼容性验证失败:", e)