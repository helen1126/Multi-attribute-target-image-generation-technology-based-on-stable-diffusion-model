import torch
from fastapi.testclient import TestClient
from utils.clip_conflict import app
from torchvision import transforms
import numpy as np


def test_clip_conflict_endpoint():
    client = TestClient(app)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = torch.randn(3, 224, 224)
    image = image.permute(1, 2, 0).numpy()
    image = transform(image).unsqueeze(0).tolist()
    data = {
        "images": image,
        "texts": ["a photo of something"]
    }
    response = client.post("/clip-conflict", json=data)
    assert response.status_code == 200
    result = response.json()
    if isinstance(result, list):
        result = result[0]
    assert "loss" in result
    assert "scale" in result
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result


def test_clip_conflict_empty_images():
    client = TestClient(app)
    data = {
        "images": [],
        "texts": ["a photo of something"]
    }
    response = client.post("/clip-conflict", json=data)
    assert response.status_code == 200


def test_clip_conflict_empty_texts():
    client = TestClient(app)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = torch.randn(3, 224, 224)
    image = image.permute(1, 2, 0).numpy()
    image = transform(image).unsqueeze(0).tolist()
    data = {
        "images": image,
        "texts": []
    }
    response = client.post("/clip-conflict", json=data)
    assert response.status_code == 200


def test_clip_conflict_invalid_image_size():
    client = TestClient(app)
    image = torch.randn(3, 100, 100)
    image = image.permute(1, 2, 0).numpy()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).tolist()
    data = {
        "images": image,
        "texts": ["a photo of something"]
    }
    response = client.post("/clip-conflict", json=data)
    assert response.status_code == 200