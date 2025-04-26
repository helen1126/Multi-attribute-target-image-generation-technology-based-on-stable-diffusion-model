import torch
import torch.nn.functional as F
from fastapi import FastAPI, status
from pydantic import BaseModel
from transformers import BertTokenizer
from models.clip_alignment_v3 import MemoryOptimizedCLIP, OptimizedCLIPLoss

# 定义请求模型
class ClipConflictRequest(BaseModel):
    images: list
    texts: list

# 初始化FastAPI应用
app = FastAPI()

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和损失函数
model = MemoryOptimizedCLIP(device=device).to(device)
loss_fn = OptimizedCLIPLoss(model)

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.post("/clip-conflict")
async def clip_conflict(request: ClipConflictRequest):
    try:
        # 验证输入数据类型
        if not isinstance(request.images, list) or not isinstance(request.texts, list):
            return {
                "status": "error",
                "message": "Invalid input data type: images and texts should be lists",
                "error_code": 400
            }, 400

        # 验证图像和文本列表是否为空
        if len(request.images) == 0 or len(request.texts) == 0:
            return {
                "status": "error",
                "message": "Invalid input data: images or texts list is empty",
                "error_code": 400
            }, 400

        # 尝试将图像数据转换为张量
        try:
            images = torch.tensor(request.images, dtype=torch.float32).to(device)
        except ValueError:
            return {
                "status": "error",
                "message": "Invalid image data: cannot convert to tensor",
                "error_code": 400
            }, 400

        # 检查图像维度
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # 检查图像尺寸
        if images.shape[2:] != (224, 224):
            return {
                "status": "error",
                "message": "Invalid image size: images should be 224x224",
                "error_code": 400
            }, 400

        # 处理文本数据
        text_inputs = tokenizer(
            request.texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # 前向传播
        logits = model(images, text_inputs, labels=torch.eye(images.size(0), device=device))
        loss = loss_fn(logits, logits.t())

        # 获取模型动态信息
        dynamics = model.get_dynamics()

        # 获取统计信息
        precision = dynamics.get('precision', 0)
        recall = dynamics.get('recall', 0)

        return {
            "status": "success",
            "loss": loss.item(),
            "scale": dynamics['scale'],
            "f1": dynamics['f1'],
            "precision": precision,
            "recall": recall,
            "error_code": 200
        }, status.HTTP_200_OK

    except IndexError as e:
        return {
            "status": "error",
            "message": f"Index error occurred: {str(e)}",
            "error_code": 400
        }, 400
    except RuntimeError as e:
        return {
            "status": "error",
            "message": f"Runtime error occurred: {str(e)}",
            "error_code": 500
        }, 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
            "error_code": 500
        }, 500