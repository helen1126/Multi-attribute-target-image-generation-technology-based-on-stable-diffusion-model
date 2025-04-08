import torch
from models.u_net import UNet, DoubleConv, SpatialSemanticAttention

def test_double_conv():
    in_channels = 3
    out_channels = 64
    double_conv = DoubleConv(in_channels, out_channels)
    x = torch.randn(1, in_channels, 128, 128)
    output = double_conv(x)
    assert output.shape[1] == out_channels

def test_spatial_semantic_attention():
    in_channels = 64
    semantic_dim = 768
    attention = SpatialSemanticAttention(in_channels, semantic_dim)
    x = torch.randn(1, in_channels, 128, 128)
    semantic_embedding = torch.randn(1, semantic_dim)
    output = attention(x, semantic_embedding)
    assert output.shape == x.shape

def test_unet():
    in_channels = 3
    out_channels = 3
    model = UNet(in_channels, out_channels)
    x = torch.randn(1, in_channels, 128, 128)
    semantic_embedding = torch.randn(1, 768)
    output = model(x, semantic_embedding)
    assert output.shape[1] == out_channels