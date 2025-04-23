import torch
from fastapi.testclient import TestClient
from utils.clip_conflict import app
from torchvision import transforms
import numpy as np

def test_clip_conflict_endpoint():
    client = TestClient(app)
    
    # 定义图像转换函数
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 模拟一个图像数据
    image = torch.randn(3, 2, 3)  # 这里假设是一个 3 通道，尺寸为 2x3 的图像
    image = image.numpy()  # 转换为 numpy.ndarray
    image = transform(image).unsqueeze(0).tolist()  # 调整尺寸并转换为列表
    
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