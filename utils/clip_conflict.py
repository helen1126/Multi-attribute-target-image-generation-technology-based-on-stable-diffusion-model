import torch
import torch.nn.functional as F
from transformers import ViTConfig, BertModel, BertTokenizer, ViTModel
from collections import deque
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.clip_alignment_v2 import MemoryOptimizedCLIP, OptimizedCLIPLoss

# ����������
class ClipConflictRequest(BaseModel):
    images: list  # ����������ͼ�������б�
    texts: list   # �����������ı������б�

# ��ʼ��FastAPIӦ��
app = FastAPI()

# ��ʼ��ģ��
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MemoryOptimizedCLIP(device=device).to(device)
loss_fn = OptimizedCLIPLoss(model)


@app.post("/clip-conflict")
async def clip_conflict(request: ClipConflictRequest):
    try:
        # ����������ݵĸ�ʽ
        if len(request.images) == 0 or len(request.texts) == 0:
            return {"error": "Invalid input data: images or texts list is empty"}

        # ��ӡ�������ݵ���״�����ڵ���
        print(f"Input images shape: {len(request.images)}")
        print(f"Input texts shape: {len(request.texts)}")

        # ������ת��Ϊ����
        images = torch.tensor(request.images, dtype=torch.float32).to(device)
        # ��� images ������ά�ȣ������ 3D ���������һ��ά��
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # ���ͨ��ά�ȣ������ 1������Ϊ 3 ��ͨ��
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        text_inputs = model.tokenizer(
            request.texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # ��ӡת�����������״�����ڵ���
        print(f"Images tensor shape: {images.shape}")
        print(f"Text inputs tensor shape: {text_inputs['input_ids'].shape}")

        # ǰ�򴫲�
        logits = model(images, text_inputs, labels=torch.eye(images.size(0), device=device))
        loss = loss_fn(logits, logits.t())

        # ��ȡģ�Ͷ�̬��Ϣ
        dynamics = model.get_dynamics()

        return {
            "loss": loss.item(),
            "scale": dynamics['scale'],
            "f1": dynamics['f1']
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # ��ӡ��ϸ�Ĵ����ջ��Ϣ
        return {"error": str(e)}    