import torch
import torch.nn as nn
import torch.nn.functional as F
# Part 3 8.1 Vanilla Transformer


class VanillaTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(VanillaTransformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, model_dim))  # Max sequence length
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.transformer_encoder(x)
        # Use the last time step's output for forecasting
        x = self.fc_out(x[:, -1, :])
        return x
