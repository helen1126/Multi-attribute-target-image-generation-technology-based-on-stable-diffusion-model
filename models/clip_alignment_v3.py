import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import unittest
import time
import math
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from collections import deque
import seaborn as sns
from transformers import ViTModel, BertModel, BertTokenizer, ViTConfig
warnings.filterwarnings('ignore')

# ------------------ 完整CLIP模型（保持原样）------------------
class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(**text)
        return image_features, text_features


# ------------------ 对齐度计算模块（修复路径处理）------------------
class CLIPAlignmentAnalyzer:
    def __init__(self, clip_model, tokenizer, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = clip_model.to(self.device)
        self.tokenizer = tokenizer
        self.clip_model.eval()

    def _prepare_text(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            max_length=77,  # CLIP标准长度
            truncation=True,
            return_token_type_ids=False
        ).to(self.device)

    def calculate_similarity(self, images, texts):
        """支持批量与单样本的灵活处理"""
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_inputs = self._prepare_text(texts)
            image_features = self.clip_model.image_encoder(images.to(self.device))
            text_features = self.clip_model.text_encoder(**text_inputs)

            logit_scale = self.clip_model.logit_scale.exp()
            similarity = logit_scale * image_features @ text_features.t()
        return similarity.cpu()

    def visualize_alignment(self, similarity_matrix, texts, figsize=(12, 10)):
        """优化可视化显示"""
        plt.figure(figsize=figsize, dpi=120)
        matrix = similarity_matrix.numpy() if torch.is_tensor(similarity_matrix) else similarity_matrix

        heatmap = plt.imshow(matrix, cmap='plasma', aspect='auto')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(texts)), texts, rotation=55, ha='right')
        plt.yticks(np.arange(len(texts)), [f"Image {i+1}" for i in range(len(texts))])
        plt.title("Cross-Modal Semantic Alignment", pad=20)
        plt.xlabel("Text Descriptions", labelpad=15)
        plt.ylabel("Visual Content", labelpad=15)
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """修复保存逻辑：创建专用tokenizer子目录"""
        # 确保父目录存在
        save_dir = os.path.dirname(path) or '.'  # 处理无目录的情况
        os.makedirs(save_dir, exist_ok=True)

        # 创建专用tokenizer子目录
        tokenizer_dir = os.path.join(save_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)

        # 保存模型时记录tokenizer路径
        torch.save({
            'clip_model': self.clip_model.state_dict(),
            'tokenizer_dir': tokenizer_dir,
        }, path)

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


# ------------------ 显存优化的动态阈值模块 ------------------
class MemoryEfficientDynamicThreshold(nn.Module):
    def __init__(self, window_size=1000, init_temp=0.07, momentum=0.9, 
                 update_freq=50, max_scale=5.0, device='cpu'):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.momentum = momentum
        self.update_freq = update_freq
        self.max_scale = max_scale
        
        self.base_scale = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32, device=device))
        self.register_buffer('running_mean', torch.zeros(1, dtype=torch.float32, device=device))
        self.register_buffer('running_std', torch.ones(1, dtype=torch.float32, device=device))
        
        self.scale_history = deque(maxlen=500)
        self.stability_warning_count = 0
        self.cm_alpha = 0.3
        self.update_counter = 0

    def _calculate_adaptive_scale(self, logits):
        with torch.cuda.amp.autocast():
            z_scores = (logits - self.running_mean) / (self.running_std + 1e-7)
            return torch.sigmoid(z_scores) * 2.0 + 0.5

    def update_statistics(self, similarities):
        if len(self.scale_history) >= 500:
            self.scale_history.popleft()
        self.scale_history.append(self.base_scale.exp().item())
        
        if self.update_counter % self.update_freq == 0:
            recent_scales = np.array(self.scale_history)
            if len(recent_scales) >= 100:
                current_std = np.std(recent_scales[-100:])
                current_mean = np.mean(recent_scales[-100:])
                if current_std > 0.15 * current_mean:
                    self.stability_warning_count += 1
                    print(f"[Warning] Threshold instability! STD={current_std:.4f}, Mean={current_mean:.4f}")
        self.update_counter += 1

    def forward(self, logits, cm_metrics=None):
        with torch.no_grad():
            base_scale = torch.clamp(self.base_scale.exp(), min=1e-4, max=self.max_scale)
            
            if self.training and cm_metrics is not None:
                f1_factor = 1.0 + (cm_metrics['f1'] - 0.5) * self.cm_alpha
                f1_factor = torch.clamp(f1_factor, 0.5, 2.0).to(self.device)
                adaptive_scale = self._calculate_adaptive_scale(logits)
                final_scale = base_scale * f1_factor * adaptive_scale
                self.update_statistics(final_scale)
                return final_scale
            
            return base_scale

# ------------------ 内存优化增强的混淆矩阵跟踪器 ------------------
class OptimizedConfusionMatrixTracker:
    def __init__(self, num_classes=2, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
        
        self.history_size = 1000
        self.precision_history = deque(maxlen=self.history_size)
        self.recall_history = deque(maxlen=self.history_size)
        self.f1_history = deque(maxlen=self.history_size)
        self.threshold_history = deque(maxlen=self.history_size)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, preds, labels, threshold=0.5):
        with torch.no_grad():
            preds = (preds >= threshold).float()
            labels = labels.float()
            
            tp = torch.sum((preds == 1) & (labels == 1)).item()
            fp = torch.sum((preds == 1) & (labels == 0)).item()
            fn = torch.sum((preds == 0) & (labels == 1)).item()
            tn = torch.sum((preds == 0) & (labels == 0)).item()
            
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn

            self.threshold_history.append(threshold)
            metrics = self.get_metrics()
            self.precision_history.append(metrics['precision'].item())
            self.recall_history.append(metrics['recall'].item())
            self.f1_history.append(metrics['f1'].item())

    def get_metrics(self):
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return {
            'precision': torch.tensor(precision, device=self.device),
            'recall': torch.tensor(recall, device=self.device),
            'f1': torch.tensor(f1, device=self.device)
        }

    def visualize(self, window_size=100):
        plt.figure(figsize=(12,6))
        plt.plot(list(self.precision_history)[-window_size:], label='Precision')
        plt.plot(list(self.recall_history)[-window_size:], label='Recall')
        plt.plot(list(self.f1_history)[-window_size:], label='F1')
        plt.plot(list(self.threshold_history)[-window_size:], label='Threshold', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Training Dynamics Monitoring')
        plt.legend()
        plt.show()


# ========== 1.梯度冲突检测模块 ======================================================
class GradientConflictDetector:
    """梯度冲突检测器，基于参数更新方向一致性"""
    def __init__(self, model_params, decay_init=0.7, decay_final=0.3):
        self.decay_init = decay_init
        self.decay_final = decay_final
        self.model_params = list(model_params)
        self.grad_cos_sim = []

    def _cosine_similarity(self, grad1, grad2):
        return F.cosine_similarity(grad1.flatten(), grad2.flatten(), dim=0)

    def calculate_conflict(self):
        """计算参数间平均梯度余弦相似度"""
        conflict_scores = []
        for param in self.model_params:
            if param.grad is None:
                continue
            # 分层采样关键参数
            if 'projection' in str(param) or 'logit_scale' in str(param):
                for other_param in self.model_params:
                    if other_param is param or other_param.grad is None:
                        continue
                    score = self._cosine_similarity(param.grad, other_param.grad)
                    conflict_scores.append(score.item())
        return torch.tensor(conflict_scores).mean() if conflict_scores else torch.tensor(0.0)

# ========== 2.模型编码器修改 ======================================================
class ImageEncoder(nn.Module):
    """视觉编码器，基于ViT实现，支持分层冻结"""
    def __init__(self, freeze_vit_layers=6):
        super().__init__()
        self.visual_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 冻结指定层
        for layer in self.visual_backbone.encoder.layer[:freeze_vit_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, image):
        features = self.visual_backbone(pixel_values=image).last_hidden_state[:, 0, :]
        return F.normalize(self.projection(features), dim=-1)

class TextEncoder(nn.Module):
    """文本编码器，基于BERT实现，支持分层冻结"""
    def __init__(self, freeze_bert_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # 冻结指定层
        for layer in self.bert.encoder.layer[:freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        features = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.projection(features), dim=-1)

# ========== 3.动态权重衰减集成 ====================================================
class MemoryOptimizedCLIP(nn.Module):
    """改进版CLIP模型，集成动态阈值优化"""
    def __init__(self, use_dynamic_threshold=True, freeze_vit=6, freeze_bert=3, device='cpu'):
        super().__init__()
        self.device = device
        self.image_encoder = ImageEncoder(freeze_vit_layers=freeze_vit)
        self.text_encoder = TextEncoder(freeze_bert_layers=freeze_bert)
        self.use_dynamic = use_dynamic_threshold
        self.threshold_module = MemoryEfficientDynamicThreshold(device=device) if use_dynamic_threshold else None
        self.logit_scale = nn.Parameter(torch.ones(1, device=device) * np.log(1 / 0.07))
        self.cm_tracker = OptimizedConfusionMatrixTracker(device=device)
        self.last_metrics = {}
        
        # 初始化梯度冲突检测器
        self.conflict_detector = GradientConflictDetector(
            self.parameters(),
            decay_init=0.7,
            decay_final=0.3
        )
        self.decay_scheduler = None  # 延迟初始化
        self.current_decay = 0.7

    def get_dynamics(self):
        return {
            'scale': self.threshold_module.base_scale.exp().item() if self.use_dynamic else self.logit_scale.exp().item(),
            **{k: v.item() for k, v in self.last_metrics.items()}
        }

    def forward(self, image, text_inputs, labels=None):
        with torch.cuda.amp.autocast():
            image_features = self.image_encoder(image)
            text_features = self.text_encoder(**text_inputs)
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            if self.use_dynamic and self.training:
                current_metrics = self.cm_tracker.get_metrics()
                logit_scale = self.threshold_module(logits_per_image.detach(), cm_metrics=current_metrics)
            else:
                logit_scale = self.logit_scale.exp()

            scaled_logits = logit_scale * logits_per_image

            if self.training:
                batch_size = image_features.size(0)
                target = torch.eye(batch_size, device=self.device)
                self.cm_tracker.update(scaled_logits.detach(), target, threshold=0.5)
                self.last_metrics = self.cm_tracker.get_metrics()

            return scaled_logits

# ========== 4.优化器与训练逻辑 ====================================================
class OptimizedCLIPLoss(nn.Module):
    """改进版CLIP损失函数，含动态惩罚项"""
    def __init__(self, model, alpha=0.5):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=self.model.device)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        base_loss = (loss_img + loss_txt) / 2

        dynamics = self.model.get_dynamics()
        conflict_level = self.model.conflict_detector.calculate_conflict()

        # 动态衰减逻辑
        current_step = self.model.decay_scheduler.last_epoch if self.model.decay_scheduler else 0
        scheduled_decay = self.model.decay_scheduler.get_last_lr()[0] if self.model.decay_scheduler else 0.7
        
        # 基于指标的自适应衰减
        f1_factor = (1.0 - dynamics['f1']) * 0.5
        adaptive_decay = scheduled_decay * (1 - f1_factor)

        # 最终衰减系数
        final_decay = torch.clamp(
            torch.tensor(adaptive_decay),
            min=0.3,
            max=0.7
        ).to(self.model.device)

        scale_penalty = F.mse_loss(
            torch.tensor(dynamics['scale'], device=self.model.device),
            torch.tensor(1.0, device=self.model.device)
        )
        metric_balance = (1.0 - dynamics['f1']) * self.alpha

        return base_loss + final_decay * scale_penalty + 0.3 * metric_balance

def optimized_train_example():
    """训练示例"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryOptimizedCLIP(device=device, use_dynamic_threshold=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    analyzer = CLIPAlignmentAnalyzer(model, tokenizer, device)
    loss_fn = OptimizedCLIPLoss(model, alpha=0.5)

    # 参数分组优化器
    projection_params = [
        {"params": model.image_encoder.projection.parameters(), "weight_decay": 0.7},
        {"params": model.text_encoder.projection.parameters(), "weight_decay": 0.7}
    ]
    other_params = [
        {"params": [p for n, p in model.named_parameters() if 'projection' not in n], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(projection_params + other_params, lr=1e-4)
    
    # 初始化学习率调度器
    model.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=100,
        eta_min=0.3
    )

    # 测试数据生成
    images = torch.randn(16, 3, 224, 224).to(device)
    texts = ["a photo of cat"] * 8 + ["a picture of dog"] * 8
    text_inputs = analyzer._prepare_text(texts)

    model.train()
    decay_values = []
    conflict_levels = []

    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(images, text_inputs)
        loss = loss_fn(logits, logits.t())
        loss.backward()
        optimizer.step()
        model.decay_scheduler.step()  # 正确的位置更新学习率

        # 动态调整权重衰减
        conflict_level = model.conflict_detector.calculate_conflict()
        current_decay = 0.7 - (0.7 - 0.3) * (epoch / 10)  # 线性衰减
        
        # 更新优化器参数
        for param_group in optimizer.param_groups:
            if 'projection' in str(param_group['params'][0]):
                param_group['weight_decay'] = current_decay * conflict_level.item()

        # 记录衰减值
        decay_values.append(current_decay)
        conflict_levels.append(conflict_level.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Decay: {current_decay:.3f}')

    # 可视化监控
    plt.figure(figsize=(10, 6))
    plt.plot(decay_values, label='Weight Decay')
    plt.plot(conflict_levels, label='Conflict Level', linestyle='--')
    plt.title("Dynamic Weight Decay Monitoring")
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('decay_dynamics.png')

# ========== 5.监控测试类 ========================================================
class EnhancedCLIPTest(unittest.TestCase):
    """增强型测试模块，集成双可视化功能"""
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MemoryOptimizedCLIP(device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.analyzer = CLIPAlignmentAnalyzer(self.model, self.tokenizer, self.device)
        self.test_images = torch.randn(4, 3, 224, 224).to(self.device)
        self.text_descriptions = [
            "a photo of cat", "a picture of dog",
            "an animal in forest", "a furry mammal"
        ]

    def test_decay_monitoring(self):
        """权重衰减动态监控测试"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loss_fn = OptimizedCLIPLoss(self.model, alpha=0.5)
        
        # 初始化调度器
        self.model.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
            eta_min=0.3
        )
        
        self.model.train()
        decay_values = []
        conflict_levels = []

        for epoch in range(5):
            optimizer.zero_grad()
            text_inputs = self.analyzer._prepare_text(self.text_descriptions)
            logits = self.model(self.test_images, text_inputs)
            loss = loss_fn(logits, logits.t())
            loss.backward()
            optimizer.step()
            self.model.decay_scheduler.step()

            # 记录衰减值
            for param_group in optimizer.param_groups:
                if 'projection' in str(param_group['params'][0]):
                    decay_values.append(param_group['weight_decay'])
            conflict_levels.append(self.model.conflict_detector.calculate_conflict().item())

        # 可视化验证
        plt.figure(figsize=(10, 6))
        plt.plot(decay_values, label='Weight Decay')
        plt.plot(conflict_levels, label='Conflict Level', linestyle='--')
        plt.title("Dynamic Weight Decay Test")
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('test_decay_dynamics.png')
        plt.close()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(EnhancedCLIPTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    optimized_train_example()
