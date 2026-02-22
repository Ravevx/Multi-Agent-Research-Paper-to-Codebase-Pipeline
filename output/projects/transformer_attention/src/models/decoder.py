import torch
import torch.nn as nn
from typing import Optional

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 df_ffn: int = 2048,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.ffn_layer1 = nn.Linear(d_model, df_ffn)
        self.ffn_layer2 = nn.Linear(df_ffn, d_model)
        self.norm1_self = nn.LayerNorm(d_model)
        self.norm1_cross = nn.LayerNorm(d_model)
        self.norm2_ffn = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                encoder_outputs: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with masking (Paper Eq. 3)
        self_attn_output, _ = self.self_attention(
            x,
            x,
            x,
            need_weights=False
        )
        self_attn_output = self.norm1_self(x + self_attn_output)

        # Cross-attention to encoder outputs
        cross_attn_output, _ = self.cross_attention(
            self_attn_output,
            encoder_outputs,
            encoder_outputs,
            need_weights=False
        )
        cross_attn_output = self.norm1_cross(cross_attn_output + self_attn_output)

        # Feed-forward network (Paper Eq. 2)
        ffn_input = self.ffn_layer1(cross_attn_output)
        ffn_output = torch.relu(ffn_input)
        ffn_output = self.ffn_layer2(ffn_output)
        ffn_output = self.norm2_ffn(cross_attn_output + ffn_output)

        return ffn_output

class DecoderStack(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 df_ffn: int = 2048,
                 num_layers: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, df_ffn, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                encoder_outputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_outputs)
        return x