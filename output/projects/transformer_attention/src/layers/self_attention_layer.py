import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, dmodel: int = 512, nhead: int = 8, dropout: float = 0.1):
        super(SelfAttentionLayer, self).__init__()
        self.dmodel = dmodel
        self.nhead = nhead

        # Linear transformations for Q, K, V (Paper Eq. 3)
        self.W_q = nn.Linear(dmodel, dmodel)
        self.W_k = nn.Linear(dmodel, dmodel)
        self.W_v = nn.Linear(dmodel, dmodel)

        # Output projection
        self.W_o = nn.Linear(dmodel, dmodel)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention (Paper Eq. 3)."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

    def multi_head_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Split into multiple heads and concatenate results (Paper Eq. 3)."""
        batch_size = Q.size(0)
        head_dim = self.dmodel // self.nhead

        # Project to different heads
        Q_heads = Q.view(batch_size, -1, self.nhead, head_dim).transpose(1, 2)
        K_heads = K.view(batch_size, -1, self.nhead, head_dim).transpose(1, 2)
        V_heads = V.view(batch_size, -1, self.nhead, head_dim).transpose(1, 2)

        # Compute attention for each head
        attn_output_heads = [
            self.scaled_dot_product_attention(Q_h, K_h, V_h)
            for Q_h, K_h, V_h in zip(Q_heads, K_heads, V_heads)
        ]

        # Concatenate heads and project back
        concat_heads = torch.cat(attn_output_heads, dim=2).transpose(1, 2).contiguous()
        return self.W_o(concat_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complete self-attention layer with residual connection and normalization."""
        # Step 1: Linear transformations for Q, K, V (Paper Eq. 3)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Multi-head attention
        attn_output = self.multi_head_attention(Q, K, V)

        # Step 3: Residual connection (add original input)
        residual = x
        out = attn_output + residual

        # Step 4: Layer normalization
        out = self.layer_norm(out)

        return out