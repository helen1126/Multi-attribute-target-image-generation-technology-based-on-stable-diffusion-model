import torch
from models.mha import TextEncoderWithMHA, MultiHeadAttention

def test_multi_head_attention():
    d_model = 768
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)
    Q = torch.randn(1, 10, d_model)
    K = torch.randn(1, 10, d_model)
    V = torch.randn(1, 10, d_model)
    output, attention_weights = mha(Q, K, V)
    assert output.shape == (1, 10, d_model)
    assert attention_weights.shape == (1, num_heads, 10, 10)

def test_text_encoder():
    encoder = TextEncoderWithMHA(output_dim=768)
    text = ["A dog"]
    result = encoder.encode_text(text)
    assert "embeddings" in result
    assert "attention_weights" in result
    assert isinstance(result["embeddings"], torch.Tensor)
    assert isinstance(result["attention_weights"], torch.Tensor)