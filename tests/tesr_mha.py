import torch
from models.mha import TextEncoderWithMHA

def test_text_encoder():
    encoder = TextEncoderWithMHA(output_dim=768)
    text = ["A dog"]
    result = encoder.encode_text(text)
    assert "embeddings" in result
    assert "attention_weights" in result
    assert isinstance(result["embeddings"], torch.Tensor)
    assert isinstance(result["attention_weights"], torch.Tensor)