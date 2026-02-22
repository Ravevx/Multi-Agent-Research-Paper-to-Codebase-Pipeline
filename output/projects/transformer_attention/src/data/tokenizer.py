import numpy as np
import torch
from typing import List, Tuple

def generate_embeddings_with_pos_encoding(
    token_ids: torch.Tensor,
    embedding_dim: int = 512,
    max_position_embedding: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates embeddings with positional encoding using sinusoidal functions.
    # Paper Eq.3: PE(pos, 2i) = sin(pos/10000^(2i/dmodel)), PE(pos, 2i+1) = cos(...)
    """
    batch_size, seq_len = token_ids.shape
    embedding_matrix = torch.randn((embedding_dim, token_ids.max() + 1)) * (embedding_dim ** -0.5)

    # Initialize positional encoding matrix
    pos_encoding = torch.zeros(batch_size, seq_len, embedding_dim)
    for b in range(batch_size):
        for t in range(seq_len):
            pos = torch.tensor(t, dtype=torch.float32)
            for i in range(embedding_dim):
                if i % 2 == 0:
                    # Paper Eq.3: sin(pos/10000^(2i/dmodel))
                    pos_encoding[b, t, i] = torch.sin(pos / (max_position_embedding ** (i / embedding_dim)))
                else:
                    # Paper Eq.3: cos(pos/10000^(2i+1)/dmodel)
                    pos_encoding[b, t, i] = torch.cos(pos / (max_position_embedding ** ((i + 1) / embedding_dim)))

    return token_ids @ embedding_matrix, pos_encoding

def pad_sequence_to_max_len(
    sequences: List[torch.Tensor],
    max_length: int
) -> torch.Tensor:
    """
    Pads sequences to maximum length in batch for attention computation.
    """
    padded = torch.zeros(len(sequences), max_length)
    for i, seq in enumerate(sequences):
        if len(seq) < max_length:
            padded[i, :len(seq)] = seq
    return padded