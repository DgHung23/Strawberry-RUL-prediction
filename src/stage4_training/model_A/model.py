import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class StrawberryRULModel(nn.Module):
    def __init__(self, gru_hidden_size=128, num_layers=1, dropout=0.2):
        super(StrawberryRULModel, self).__init__()
        
        # 1. Feature Extractor (EfficientNet B0)
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # We only need the features (remove classifier)
        # EfficientNet-B0 outputs feature map of size 1280
        self.feature_dim = 1280
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone? Optional. Unfreezing fine-tunes better, but freezing is faster.
        # We will freeze for faster training initially, except for last few layers.
        # Uncomment to freeze all:
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # 2. Environmental Features Dimension (temp, humidity)
        self.env_dim = 2
        
        # 3. GRU Temporal Model
        self.gru_input_size = self.feature_dim + self.env_dim
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 4. Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Output single RUL value
        )

    def forward(self, images_seq, env_seq):
        """
        Args:
            images_seq: (batch_size, seq_len, C, H, W)
            env_seq: (batch_size, seq_len, 2)
        Returns:
            rul: (batch_size, 1)
        """
        batch_size, seq_len, C, H, W = images_seq.size()
        
        # Reshape for EfficientNet: combine batch and seq_len
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract spatial features
        spatial_features = self.backbone(images_reshaped) # (batch_size * seq_len, 1280)
        spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)
        
        # Concatenate spatial and environmental features
        fused_features = torch.cat((spatial_features, env_seq), dim=2) # (batch_size, seq_len, 1280 + 2)
        
        # Pass through GRU
        gru_out, hidden = self.gru(fused_features) # gru_out: (batch_size, seq_len, gru_hidden_size)
        
        # We take the output of the last time step for RUL prediction
        last_out = gru_out[:, -1, :] # (batch_size, gru_hidden_size)
        
        # Regression
        rul = self.regressor(last_out) # (batch_size, 1)
        
        return rul

if __name__ == "__main__":
    # Test model shape
    model = StrawberryRULModel()
    dummy_images = torch.randn(2, 5, 3, 224, 224) # batch=2, seq=5
    dummy_envs = torch.randn(2, 5, 2)
    output = model(dummy_images, dummy_envs)
    print(f"Output shape: {output.shape}") # Expected: (2, 1)
