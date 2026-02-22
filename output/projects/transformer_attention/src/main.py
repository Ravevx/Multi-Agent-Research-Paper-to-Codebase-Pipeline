import torch
from torch.nn import Linear
from transformers import TransformerModel, TransformerEncoderLayer, TransformerDecoderLayer

def main():
    # Step 2: Define config dict with all hyperparameters from the paper
    config = {
        "dmodel": 512,
        "df": 2,
        "N": 6,
        "vocab_size": 30522,  # Example placeholder; adjust based on actual tokenizer
        "max_seq_len": 100,
        "batch_size": 4,
    }

    # Step 3: Instantiate the model using config values
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, dmodel, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, dmodel)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dmodel, 2) * -(math.log(10000.0) / dmodel))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[:x.size(1)]

    import math
    positional_encoder = PositionalEncoding(config["dmodel"], config["max_seq_len"])

    # Create encoder and decoder stacks (N=6 layers)
    encoder_layers = TransformerEncoderLayer(d_model=config["dmodel"], nhead=8, dim_feedforward=2*config["dmodel"])
    decoder_layers = TransformerDecoderLayer(d_model=config["dmodel"], nhead=8, dim_feedforward=2*config["dmodel"])

    encoder_stack = torch.nn.ModuleList([encoder_layers] * config["N"])
    decoder_stack = torch.nn.ModuleList([decoder_layers] * config["N"])

    model = TransformerModel(
        d_model=config["dmodel"],
        nhead=8,
        num_encoder_layers=config["N"],
        dim_feedforward=2*config["dmodel"]
    )

    # Step 4: Generate sample input tensors matching the paper's expected dimensions
    # Input embeddings (batch_size, seq_len, dmodel)
    input_embeddings = torch.randn(config["batch_size"], config["max_seq_len"], config["dmodel"])

    # Apply positional encoding
    input_with_pos = positional_encoder(input_embeddings)

    # Step 5: Run the model forward pass
    outputs = model(
        src=input_with_pos,
        src_mask=None,
        memory_key_padding_mask=None,
        memory_attn_mask=None,
        memory_key_padding_mask=None,
        memory_value_padding_mask=None,
        memory_attn_mask=None,
        memory_key_padding_mask=None,
        memory_value_padding_mask=None
    )

    # Step 6: Print output shape to verify correctness
    print("Output shape:", outputs.last_hidden_state.shape)

if __name__ == "__main__":
    main()