import torch
import numpy as np

def compute_scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes scaled dot-product attention as described in the paper (Eq. 1).

    Args:
        queries: Shape (batch_size, heads, seq_len_q, embed_dim)
        keys: Shape (batch_size, heads, seq_len_k, embed_dim)
        values: Shape (batch_size, heads, seq_len_v, embed_dim)
        mask: Optional masking tensor for illegal connections

    Returns:
        Attention output with shape (batch_size, heads, seq_len_q, seq_len_v)
    """
    # Step 1: Compute raw dot-product between queries and keys
    dk = queries.shape[-1]
    scaled_dot_product = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    # Step 2: Apply masking if provided (setting illegal connections to -∞)
    if mask is not None:
        scaled_dot_product.masked_fill_(mask == 0, float('-inf'))

    # Step 3: Apply softmax along sequence dimension for each head
    attention_weights = torch.softmax(scaled_dot_product, dim=-1)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, values)
    return output