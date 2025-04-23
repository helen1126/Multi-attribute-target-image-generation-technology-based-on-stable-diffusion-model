import torch
import torch.nn as nn
from models.u_net import SpatialSemanticAttention
from utils.dynamic_weights import load_conflicts, detect_conflict, handle_conflicts
from PIL import Image
from torchvision import transforms
from io import BytesIO

# 定义图片转换函数
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 假设我们已经有了一个处理冲突属性的函数
def optimize_conflict_localization(data, conflicts, fallback_strategy, temperature):
    # 处理属性冲突
    final_weights, conflict_report, adjustment_log = handle_conflicts(data, conflicts, fallback_strategy, temperature)

    if conflict_report["detected"]:
        # 获取冲突属性的索引
        conflict_indices = []
        for conflict in conflict_report["potential_conflicts"]:
            index1 = next((i for i, attr in enumerate(data['attributes']) if attr['value'] == conflict[0]), None)
            index2 = next((i for i, attr in enumerate(data['attributes']) if attr['value'] == conflict[1]), None)
            if index1 is not None and index2 is not None:
                conflict_indices.extend([index1, index2])

        # 加载示例图片
        try:
            img = Image.open('demo/image3.jpg')
            conflict_features = transform(img).unsqueeze(0)
            conflict_features = conflict_features.repeat(len(conflict_indices), 1, 1, 1)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, {
                "code": 42203,
                "message": "Failed to load example image",
                "solution": ["Check if image3.jpg exists or use a different image"]
            }, adjustment_log

        semantic_embedding = torch.randn(1, 768)  # 假设语义嵌入维度为768
        weights = [1.0] * 768  # 假设动态权重为全1

        # 初始化空间语义注意力模块
        in_channels = 3
        semantic_dim = 768
        attention = SpatialSemanticAttention(in_channels, semantic_dim)

        # 应用空间语义注意力模块
        enhanced_features = attention(conflict_features, semantic_embedding, weights)

        # 根据增强后的特征图更新冲突属性的权重
        # 这里简单地将增强后的特征图的均值作为权重调整的依据
        for i, index in enumerate(conflict_indices):
            feature_mean = enhanced_features[i].mean().item()
            final_weights[str(data['attributes'][index]['value'])] *= feature_mean

        # 重新检查权重总和是否超出阈值
        weight_sum = sum(final_weights.values())
        if weight_sum > 1.5:
            return None, {
                "code": 42202,
                "message": "权重总和超出阈值（Σweights>1.5）",
                "solution": ["降低初始权重或启用自动归一化"]
            }, adjustment_log

    return final_weights, conflict_report, adjustment_log

# 示例数据
data = {
    "base_prompt": "A dog on the grass",
    "attributes": [
        {
            "name": "object",
            "type": "text",
            "value": "dog",
            "initial_weight": 0.8
        },
        {
            "name": "background",
            "type": "text",
            "value": "grass",
            "initial_weight": 0.2
        }
    ]
}
conflicts = load_conflicts()
fallback_strategy = "balanced"
temperature = 1.2

# 调用优化函数
final_weights, conflict_report, adjustment_log = optimize_conflict_localization(data, conflicts, fallback_strategy, temperature)
print("Final weights:", final_weights)
print("Conflict report:", conflict_report)
print("Adjustment log:", adjustment_log)
    