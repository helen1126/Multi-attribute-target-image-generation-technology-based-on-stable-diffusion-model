import torch
import torch.nn.functional as F
from fastapi import FastAPI, status
from pydantic import BaseModel
from transformers import BertTokenizer
from models.clip_alignment_v3 import MemoryOptimizedCLIP, OptimizedCLIPLoss

# ��������ģ��
class ClipConflictRequest(BaseModel):
    images: list
    texts: list

# ��ʼ��FastAPIӦ��
app = FastAPI()

# ��ʼ���豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ��ʼ��ģ�ͺ���ʧ����
model = MemoryOptimizedCLIP(device=device).to(device)
loss_fn = OptimizedCLIPLoss(model)

# ��ʼ���ִ���
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.post("/clip-conflict")
async def clip_conflict(request: ClipConflictRequest):
    try:
        # ��֤������������
        if not isinstance(request.images, list) or not isinstance(request.texts, list):
            return {
                "status": "error",
                "message": "Invalid input data type: images and texts should be lists",
                "error_code": 400
            }, 400

        # ��֤ͼ����ı��б��Ƿ�Ϊ��
        if len(request.images) == 0 or len(request.texts) == 0:
            return {
                "status": "error",
                "message": "Invalid input data: images or texts list is empty",
                "error_code": 400
            }, 400

        # ���Խ�ͼ������ת��Ϊ����
        try:
            images = torch.tensor(request.images, dtype=torch.float32).to(device)
        except ValueError:
            return {
                "status": "error",
                "message": "Invalid image data: cannot convert to tensor",
                "error_code": 400
            }, 400

        # ���ͼ��ά��
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # ���ͼ��ߴ�
        if images.shape[2:] != (224, 224):
            return {
                "status": "error",
                "message": "Invalid image size: images should be 224x224",
                "error_code": 400
            }, 400

        # �����ı�����
        text_inputs = tokenizer(
            request.texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # ǰ�򴫲�
        logits = model(images, text_inputs, labels=torch.eye(images.size(0), device=device))
        loss = loss_fn(logits, logits.t())

        # ��ȡģ�Ͷ�̬��Ϣ
        dynamics = model.get_dynamics()

        # ��ȡͳ����Ϣ
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