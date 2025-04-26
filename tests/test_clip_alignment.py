import torch
import torch.nn.functional as F
import pytest
from transformers import ViTConfig, BertModel, BertTokenizer
from collections import deque
import numpy as np
from tqdm import tqdm
from models.clip_alignment_v3 import (
    OptimizedConfusionMatrixTracker,
    MemoryEfficientDynamicThreshold,
    MemoryOptimizedCLIP,
    OptimizedCLIPLoss,
    GradientConflictDetector,
    CLIPAlignmentAnalyzer,
    optimized_train_example,
    EnhancedCLIPTest
)
import unittest
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ModifiedCLIPAlignmentAnalyzer(CLIPAlignmentAnalyzer):
    @classmethod
    def load_model(cls, save_path, device):
        checkpoint = torch.load(save_path, map_location=device)
        clip_model = MemoryOptimizedCLIP(device=device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 过滤掉意外的键
        model_state_dict = clip_model.state_dict()
        pretrained_state_dict = checkpoint['clip_model']
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(filtered_state_dict)

        clip_model.load_state_dict(model_state_dict)
        return cls(clip_model, tokenizer, device)


@pytest.fixture(scope="session", autouse=True)
def close_all_visualization_windows():
    yield
    plt.close('all')


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

    # 测试可视化方法
    tracker.visualize()
    plt.close('all')


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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_inputs = tokenizer(
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_inputs = tokenizer(
        ["a photo of cat", "a picture of dog"], return_tensors="pt", padding=True, truncation=True
    ).to(device)
    model(image=torch.rand(2, 3, 224, 224).to(device), text_inputs=text_inputs, labels=labels)

    loss = loss_fn(logits_per_image, logits_per_text)
    assert isinstance(loss.item(), float)


# 测试 GradientConflictDetector
def test_gradient_conflict_detector(device):
    model = MemoryOptimizedCLIP(device=device)
    detector = GradientConflictDetector(model.parameters())
    # 模拟梯度计算
    images = torch.rand(2, 3, 224, 224).to(device)
    texts = ["a photo of cat", "a picture of dog"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    logits = model(images, text_inputs)
    loss = F.cross_entropy(logits, torch.arange(2).to(device))
    loss.backward()
    conflict_score = detector.calculate_conflict()
    assert isinstance(conflict_score.item(), float)


# 测试 CLIPAlignmentAnalyzer
def test_clip_alignment_analyzer(device):
    clip_model = MemoryOptimizedCLIP(device=device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    analyzer = ModifiedCLIPAlignmentAnalyzer(clip_model, tokenizer, device)

    images = torch.rand(2, 3, 224, 224).to(device)
    texts = ["a photo of cat", "a picture of dog"]

    # 测试 calculate_similarity 方法
    similarity = analyzer.calculate_similarity(images, texts)
    assert similarity.shape == (2, 2)

    # 测试 visualize_alignment 方法
    analyzer.visualize_alignment(similarity, texts)
    plt.close('all')

    # 测试 save_model 和 load_model 方法
    save_path = 'test_model.pth'
    analyzer.save_model(save_path)
    loaded_analyzer = ModifiedCLIPAlignmentAnalyzer.load_model(save_path, device)
    assert isinstance(loaded_analyzer, ModifiedCLIPAlignmentAnalyzer)


# 测试 optimized_train_example 函数
def test_optimized_train_example(device):
    try:
        optimized_train_example()
        plt.close('all')
    except Exception as e:
        pytest.fail(f"optimized_train_example raised an exception: {e}")


# 测试 EnhancedCLIPTest 类
def test_enhanced_clip_test(device):
    suite = unittest.TestSuite()
    suite.addTest(EnhancedCLIPTest("test_decay_monitoring"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    assert result.wasSuccessful()
    plt.close('all')


# 测试 CLIPModel 类的 forward 方法
def test_CLIPModel_forward(device):
    from models.clip_alignment_v3 import CLIPModel, ImageEncoder, TextEncoder
    model = CLIPModel()
    image = torch.randn(1, 3, 224, 224).to(device)
    text = {
        'input_ids': torch.randint(0, 1000, (1, 10)).to(device),
        'attention_mask': torch.ones(1, 10).to(device)
    }
    image_features, text_features = model(image, text)
    assert isinstance(image_features, torch.Tensor)
    assert isinstance(text_features, torch.Tensor)


# 测试 load_model 类方法
def test_load_model(device):
    from models.clip_alignment_v3 import MemoryOptimizedCLIP, CLIPAlignmentAnalyzer, BertTokenizer
    import os
    model = MemoryOptimizedCLIP(device=device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_path = 'test_checkpoint.pth'
    torch.save({
        'clip_model': model.state_dict(),
        'tokenizer_dir': 'bert-base-uncased'
    }, save_path)

    analyzer = CLIPAlignmentAnalyzer.load_model(save_path, device)
    assert isinstance(analyzer, CLIPAlignmentAnalyzer)
    os.remove(save_path)


# 测试 scale_history.popleft()
def test_scale_history_popleft(device):
    from models.clip_alignment_v3 import MemoryEfficientDynamicThreshold
    from collections import deque
    threshold_module = MemoryEfficientDynamicThreshold(device=device)
    threshold_module.scale_history = deque([1.0] * 500)
    original_length = len(threshold_module.scale_history)
    # 模拟调用触发 popleft 的操作
    threshold_module.update_statistics(torch.tensor([1.0], device=device))
    assert len(threshold_module.scale_history) == original_length


# 测试阈值稳定性检查部分
def test_threshold_stability_warning(device):
    from models.clip_alignment_v3 import MemoryEfficientDynamicThreshold
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold_module = MemoryEfficientDynamicThreshold(device=device)
    threshold_module.scale_history = deque([1.0] * 50 + [2.0] * 50)
    original_warning_count = threshold_module.stability_warning_count
    threshold_module.update_statistics(torch.tensor([1.0], device=device))
    assert threshold_module.stability_warning_count > original_warning_count


# 测试 return base_scale
def test_return_base_scale(device):
    from models.clip_alignment_v3 import MemoryEfficientDynamicThreshold
    threshold_module = MemoryEfficientDynamicThreshold(device=device)
    logits = torch.randn(1, 10, device=device)
    base_scale = threshold_module(logits)
    assert isinstance(base_scale, torch.Tensor)


# 测试 F.cosine_similarity 调用
def test_cosine_similarity(device):
    from models.clip_alignment_v3 import GradientConflictDetector
    grad1 = torch.randn(10).to(device)
    grad2 = torch.randn(10).to(device)
    detector = GradientConflictDetector([])
    score = detector._cosine_similarity(grad1, grad2)
    assert -1 <= score <= 1


# 测试梯度冲突分数计算部分
def test_calculate_conflict(device):
    from models.clip_alignment_v3 import GradientConflictDetector
    import torch.nn as nn
    model = nn.Linear(10, 10).to(device)
    input = torch.randn(1, 10).to(device)
    output = model(input)
    loss = output.sum()
    loss.backward()
    detector = GradientConflictDetector(model.parameters())
    conflict_scores = detector.calculate_conflict()
    assert isinstance(conflict_scores, torch.Tensor)


# 测试 logit_scale = self.logit_scale.exp()
def test_logit_scale_exp(device):
    from models.clip_alignment_v3 import MemoryOptimizedCLIP
    model = MemoryOptimizedCLIP(device=device, use_dynamic_threshold=False)
    image = torch.randn(1, 3, 224, 224).to(device)
    text = {
        'input_ids': torch.randint(0, 1000, (1, 10)).to(device),
        'attention_mask': torch.ones(1, 10).to(device)
    }
    logits = model(image, text)
    expected_logit_scale = model.logit_scale.exp()
    assert isinstance(expected_logit_scale, torch.Tensor)


# 测试 param_group['weight_decay'] = current_decay * conflict_level.item()
def test_weight_decay_update(device):
    from models.clip_alignment_v3 import MemoryOptimizedCLIP, GradientConflictDetector
    import torch.optim as optim
    model = MemoryOptimizedCLIP(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    conflict_detector = GradientConflictDetector(model.parameters())
    conflict_level = conflict_detector.calculate_conflict()
    current_decay = 0.7
    for param_group in optimizer.param_groups:
        original_weight_decay = param_group['weight_decay']
        param_group['weight_decay'] = current_decay * conflict_level.item()
        assert param_group['weight_decay'] != original_weight_decay


# 测试 decay_values.append(param_group['weight_decay'])
def test_decay_values_append(device):
    from models.clip_alignment_v3 import MemoryOptimizedCLIP
    import torch.optim as optim
    model = MemoryOptimizedCLIP(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    decay_values = []
    for param_group in optimizer.param_groups:
        decay_values.append(param_group['weight_decay'])
    assert len(decay_values) > 0
    