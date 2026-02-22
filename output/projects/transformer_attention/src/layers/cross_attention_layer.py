import torch
import torch.nn as nn
from typing import Tuple

class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 df_ff: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Paper Eq. (1): Scaled Dot-Product Attention
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)

        self.attention_weights = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        self.output_layer = nn.Linear(d_model, d_model)

        # Feed-forward network for multi-head attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, df_ff),
            nn.ReLU(),
            nn.Linear(df_ff, d_model)
        )

    def scaled_dot_product_attention(self,
                                    queries: torch.Tensor,
                                    keys: torch.Tensor,
                                    values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Paper Eq. (1): Scaled Dot-Product Attention
        d_k = self.head_dim
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        return output, attention_weights

    def multi_head_attention(self,
                           queries: torch.Tensor,
                           keys: torch.Tensor,
                           values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = queries.size(0)

        # Linear projections for each head
        q = self.query_layer(queries).view(batch_size, -1, self.n_heads, self.head_dim)
        k = self.key_layer(keys).view(batch_size, -1, self.n_heads, self.head_dim)
        v = self.value_layer(values).view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for matmul
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        output, attention_weights = self.scaled_dot_product_attention(q, k, v)
        output = output.transpose(2, 3).contiguous().view(batch_size, -1, self.d_model)

        return output, attention_weights

    def forward(self,
               decoder_queries: torch.Tensor,
               encoder_keys: torch.Tensor,
               encoder_values: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute scaled dot-product attention between decoder queries and encoder keys/values
        transformed_output, _ = self.multi_head_attention(decoder_queries, encoder_keys, encoder_values)

        # Step 2: Apply multi-head attention on top of that (implicitly handled in forward pass)
        residual = decoder_queries

        # Step 3: Add residual connection to original input
        transformed_output += residual

        # Step 4: Apply layer normalization
        output = nn.LayerNorm(self.d_model)(transformed_output)

        return output