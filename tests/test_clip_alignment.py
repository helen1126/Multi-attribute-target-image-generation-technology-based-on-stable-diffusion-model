import torch
import torch.nn.functional as F
import pytest
from transformers import ViTConfig, BertModel, BertTokenizer
from collections import deque
import numpy as np
from tqdm import tqdm
from models.clip_alignment_v2 import (
    OptimizedConfusionMatrixTracker,
    MemoryEfficientDynamicThreshold,
    MemoryOptimizedCLIP,
    OptimizedCLIPLoss,
    optimized_train_example
)

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# 测试 OptimizedConfusionMatrixTracker
def test_optimized_confusion_matrix_tracker(device):
    tracker = OptimizedConfusionMatrixTracker(device=device)

    # 测试重置功能
    tracker.reset()
    assert tracker.tp == 0
    assert tracker.fp == 0
    assert tracker.fn == 0
    assert tracker.tn == 0

    # 测试完美预测
    preds = torch.tensor([1, 0, 1, 0], device=device)
    labels = torch.tensor([1, 0, 1, 0], device=device)
    tracker.update(preds, labels)
    metrics = tracker.get_metrics()
    assert metrics['precision'].item() == 1.0
    assert metrics['recall'].item() == 1.0

    # 测试部分错误
    tracker.reset()
    preds = torch.tensor([1, 0, 1, 0], device=device)
    labels = torch.tensor([1, 0, 0, 1], device=device)
    tracker.update(preds, labels)
    metrics = tracker.get_metrics()
    assert round(metrics['precision'].item(), 1) == 0.5
    assert round(metrics['recall'].item(), 1) == 0.5

# 测试 MemoryEfficientDynamicThreshold
def test_memory_efficient_dynamic_threshold(device):
    threshold_module = MemoryEfficientDynamicThreshold(device=device)
    logits = torch.rand(10, 10).to(device)
    cm_metrics = {
        'precision': torch.tensor(0.8, device=device),
        'recall': torch.tensor(0.7, device=device),
        'f1': torch.tensor(0.75, device=device)
    }
    scale = threshold_module(logits, cm_metrics)
    assert scale.shape == logits.shape

# 测试 MemoryOptimizedCLIP
def test_memory_optimized_clip(device):
    model = MemoryOptimizedCLIP(device=device)
    images = torch.rand(2, 3, 224, 224).to(device)
    texts = ["a photo of cat", "a picture of dog"]
    text_inputs = model.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    logits = model(images, text_inputs)
    assert logits.shape == (2, 2)

# 测试 OptimizedCLIPLoss
def test_optimized_clip_loss(device):
    model = MemoryOptimizedCLIP(device=device)
    loss_fn = OptimizedCLIPLoss(model)
    logits_per_image = torch.rand(2, 2).to(device)
    logits_per_text = logits_per_image.t()

    # 模拟训练过程，更新混淆矩阵跟踪器
    labels = torch.eye(2, device=device)
    model(image=torch.rand(2, 3, 224, 224).to(device), text_inputs=model.tokenizer(
        ["a photo of cat", "a picture of dog"], return_tensors="pt", padding=True, truncation=True
    ).to(device), labels=labels)

    loss = loss_fn(logits_per_image, logits_per_text)
    assert isinstance(loss.item(), float)
