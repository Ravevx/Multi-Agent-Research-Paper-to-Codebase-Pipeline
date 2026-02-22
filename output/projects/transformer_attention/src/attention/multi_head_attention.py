import torch
import math

class MultiHeadAttention:
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Paper Eq. (1): Learnable projection matrices for queries, keys, values
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

        # Scaled dot-product attention weights
        self.scaled_attention_weights = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape

        # Paper Eq. (2): Project inputs through learned linear transformations
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Paper Eq. (3): Compute scaled dot-product attention
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.permute(0, 2, 1, 3)

        # Scaled by sqrt(head_dim) as per paper Eq. (5)
        scaled_attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Paper Eq. (4): Apply softmax to normalize attention weights
        self.scaled_attention_weights = torch.nn.functional.softmax(scaled_attention_weights, dim=-1)

        # Multiply attention weights with value vectors
        output = torch.matmul(self.scaled_attention_weights, V.permute(0, 2, 1, 3))
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return output