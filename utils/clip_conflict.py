import torch
import torch.nn.functional as F
from transformers import ViTConfig, BertModel, BertTokenizer, ViTModel
from collections import deque
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.clip_alignment_v2 import MemoryOptimizedCLIP, OptimizedCLIPLoss

# 定义请求体
class ClipConflictRequest(BaseModel):
    images: list  # 假设输入是图像数据列表
    texts: list   # 假设输入是文本数据列表

# 初始化FastAPI应用
app = FastAPI()

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MemoryOptimizedCLIP(device=device).to(device)
loss_fn = OptimizedCLIPLoss(model)


@app.post("/clip-conflict")
async def clip_conflict(request: ClipConflictRequest):
    try:
        # 检查输入数据的格式
        if len(request.images) == 0 or len(request.texts) == 0:
            return {"error": "Invalid input data: images or texts list is empty"}

        # 打印输入数据的形状，用于调试
        print(f"Input images shape: {len(request.images)}")
        print(f"Input texts shape: {len(request.texts)}")

        # 将输入转换为张量
        images = torch.tensor(request.images, dtype=torch.float32).to(device)
        # 检查 images 张量的维度，如果是 3D 张量，添加一个维度
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # 检查通道维度，如果是 1，复制为 3 个通道
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        text_inputs = model.tokenizer(
            request.texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # 打印转换后的张量形状，用于调试
        print(f"Images tensor shape: {images.shape}")
        print(f"Text inputs tensor shape: {text_inputs['input_ids'].shape}")

        # 前向传播
        logits = model(images, text_inputs, labels=torch.eye(images.size(0), device=device))
        loss = loss_fn(logits, logits.t())

        # 获取模型动态信息
        dynamics = model.get_dynamics()

        return {
            "loss": loss.item(),
            "scale": dynamics['scale'],
            "f1": dynamics['f1']
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈信息
        return {"error": str(e)}    