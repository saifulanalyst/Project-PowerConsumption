from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
# PatchTST


class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_length, num_patches, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_length = patch_length
        self.num_patches = num_patches
        self.model_dim = model_dim

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_length, model_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, num_patches, model_dim))

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(model_dim * num_patches, output_dim)

    def forward(self, x):
        # Reshape input into patches
        x = x.unfold(1, self.patch_length, self.patch_length).permute(0, 2, 1)
        x = self.patch_embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)  # Flatten patches
        x = self.fc_out(x)
        return x


# Hyperparameter Tuning
# Define hyperparameters
learning_rate = 1e-3
batch_size = 64
sequence_length = 96
num_heads = 8
ff_dim = 2048
num_layers = 4

# Initialize model
model = VanillaTransformer(input_dim=1, model_dim=64,
                           num_heads=num_heads, num_layers=num_layers, output_dim=1)

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=learning_rate)
# Reduce learning rate every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Adjust Dropout and Weight Initialization
# Add dropout to the Transformer encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=8, dim_feedforward=ff_dim, dropout=0.2
)

# Initialize weights


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


model.apply(init_weights)
