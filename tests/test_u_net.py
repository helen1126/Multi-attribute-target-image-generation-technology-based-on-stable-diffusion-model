import torch
import pytest
from models.u_net import DoubleConv, SpatialSemanticAttention, UNet
from models.mha import TextEncoderWithMHA

# 测试 DoubleConv 类
def test_double_conv():
    in_channels = 3
    out_channels = 64
    double_conv = DoubleConv(in_channels, out_channels)
    x = torch.randn(1, in_channels, 128, 128)
    output = double_conv(x)
    assert output.shape[1] == out_channels

# 测试 SpatialSemanticAttention 类
def test_spatial_semantic_attention():
    in_channels = 64
    semantic_dim = 768
    attention = SpatialSemanticAttention(in_channels, semantic_dim)
    x = torch.randn(1, in_channels, 128, 128)
    semantic_embedding = torch.randn(1, semantic_dim)
    weights = [1.0] * semantic_dim
    output = attention(x, semantic_embedding, weights)
    assert output.shape == x.shape

# 测试 UNet 类
def test_unet():
    in_channels = 3
    out_channels = 3
    model = UNet(in_channels, out_channels)
    x = torch.randn(1, in_channels, 128, 128)
    text_encoder = TextEncoderWithMHA()
    text = ["dog", "grass"]
    result = text_encoder.encode_text(text)
    semantic_embedding = result["embeddings"]
    weights = [1.0] * len(text)
    output = model(x, semantic_embedding, weights)
    assert output.shape[1] == out_channels