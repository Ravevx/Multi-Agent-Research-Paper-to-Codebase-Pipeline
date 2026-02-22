import torch
import torch.nn as nn
import math

class EncoderLayer(nn.Module):
    def __init__(self, dmodel: int = 512, df_ffn: int = 2048) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model=dmodel, num_heads=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(dmodel, df_ffn),
            nn.ReLU(),
            nn.Linear(df_ffn, dmodel)
        )
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection and layer normalization (Paper Eq. 2)
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))  # Residual connection

        # Feed-forward network with residual connection and layer normalization (Paper Eq. 2)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class EncoderStack(nn.Module):
    def __init__(self, dmodel: int = 512, df_ffn: int = 2048, n_layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dmodel=dmodel, df_ffn=df_ffn)
                                    for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process entire input sequence through all encoder layers in parallel
        for layer in self.layers:
            x = layer(x)
        return x