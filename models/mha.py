import torch
import torch.nn as nn
from transformers import BertTokenizer
from models.clip_alignment_v3 import MemoryOptimizedCLIP, TextEncoder

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力层

        :param d_model: 模型的维度
        :param num_heads: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        """
        前向传播函数

        :param Q: 查询张量 (batch_size, seq_len, d_model)
        :param K: 键张量 (batch_size, seq_len, d_model)
        :param V: 值张量 (batch_size, seq_len, d_model)
        :return: 输出张量 (batch_size, seq_len, d_model) 和注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention_weights



class TextEncoderWithMHA:
    def __init__(self, output_dim=768, device='cpu'):
        # 加载Bert分词器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = MemoryOptimizedCLIP(device=device)
        self.text_encoder = self.model.text_encoder

        d_model = 512
        num_heads = 8
        self.mha = MultiHeadAttention(d_model, num_heads)

        self.output_projection = nn.Linear(d_model, output_dim)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        with torch.no_grad():
            text_features = self.text_encoder(input_ids, attention_mask)

        attn_output, attention_weights = self.mha(text_features.unsqueeze(1), text_features.unsqueeze(1), text_features.unsqueeze(1))

        embeddings = attn_output[:, 0, :]

        embeddings = self.output_projection(embeddings)

        if len(text) > 1:
            embeddings = embeddings.mean(dim=0, keepdim=True)

        output = {"embeddings": embeddings, "attention_weights": attention_weights}

        return output