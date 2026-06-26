import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class StrawberryRULModelB(nn.Module):
    def __init__(self, lstm_hidden_size=128, num_layers=1, dropout=0.2):
        super(StrawberryRULModelB, self).__init__()
        
        # 1. Feature Extractor (MobileNetV2)
        weights = MobileNet_V2_Weights.DEFAULT
        self.backbone = mobilenet_v2(weights=weights)
        
        # MobileNetV2 outputs feature map of size 1280 before the classifier
        self.feature_dim = 1280
        # We replace the classifier with Identity so it just returns the pooled features
        self.backbone.classifier = nn.Identity()
        
        # 2. Environmental Features Dimension (temp, humidity)
        self.env_dim = 2
        
        # 3. LSTM Temporal Model
        self.lstm_input_size = self.feature_dim + self.env_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 4. Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
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
        
        # Reshape for MobileNet: combine batch and seq_len
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract spatial features
        spatial_features = self.backbone(images_reshaped) # (batch_size * seq_len, 1280)
        spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)
        
        # Concatenate spatial and environmental features
        fused_features = torch.cat((spatial_features, env_seq), dim=2) # (batch_size, seq_len, 1280 + 2)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(fused_features) # lstm_out: (batch_size, seq_len, lstm_hidden_size)
        
        # We take the output of the last time step for RUL prediction
        last_out = lstm_out[:, -1, :] # (batch_size, lstm_hidden_size)
        
        # Regression
        rul = self.regressor(last_out) # (batch_size, 1)
        
        return rul

if __name__ == "__main__":
    # Test model shape
    model = StrawberryRULModelB()
    dummy_images = torch.randn(2, 5, 3, 224, 224) # batch=2, seq=5
    dummy_envs = torch.randn(2, 5, 2)
    output = model(dummy_images, dummy_envs)
    print(f"Output shape: {output.shape}") # Expected: (2, 1)
