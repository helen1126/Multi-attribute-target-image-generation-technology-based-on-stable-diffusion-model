import torch
from fastapi.testclient import TestClient
from utils.clip_conflict import app
from torchvision import transforms
import numpy as np

def test_clip_conflict_endpoint():
    client = TestClient(app)
    
    # ����ͼ��ת������
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # ģ��һ��ͼ������
    image = torch.randn(3, 2, 3)  # ���������һ�� 3 ͨ�����ߴ�Ϊ 2x3 ��ͼ��
    image = image.numpy()  # ת��Ϊ numpy.ndarray
    image = transform(image).unsqueeze(0).tolist()  # �����ߴ粢ת��Ϊ�б�
    
    data = {
        "images": image,
        "texts": ["a photo of something"]
    }
    response = client.post("/clip-conflict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "loss" in result
    assert "scale" in result
    assert "f1" in result