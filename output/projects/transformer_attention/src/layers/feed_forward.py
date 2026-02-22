import torch
import torch.nn as nn
from typing import Tuple

class positionwise_feed_forward(nn.Module):
    def __init__(self, dmodel: int = 512, df: int = 2048) -> None:
        super().__init__()
        self.W1 = nn.Linear(dmodel, df)
        self.b1 = nn.Parameter(torch.zeros(df))
        self.W2 = nn.Linear(df, dmodel)
        self.b2 = nn.Parameter(torch.zeros(dmodel))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # Paper Eq. 2: FFN(x) = max(0, xW1 + b1)W2 + b2
        x_proj = self.W1(x)
        relu_output = torch.relu(x_proj + self.b1)
        output = self.W2(relu_output) + self.b2
        return output

def positionwise_feed_forward(
    input_tensor: torch.Tensor,
    dmodel: int = 512,
    df: int = 2048
) -> torch.Tensor:
    """Applies two-layer MLP with ReLU activation to input tensor.

    Args:
        input_tensor: (batch_size, seq_len, embed_dim)
        dmodel: Input/output dimension (dmodel = 512 per paper)
        df: Inner layer dimension (df = 2048 per paper)

    Returns:
        Output of FFN(x) as described in Eq. 2
    """
    W1 = torch.nn.Linear(dmodel, df)
    b1 = torch.zeros(df)
    W2 = torch.nn.Linear(df, dmodel)
    b2 = torch.zeros(dmodel)

    x_proj = W1(input_tensor)
    relu_output = torch.relu(x_proj + b1)
    output = W2(relu_output) + b2
    return output