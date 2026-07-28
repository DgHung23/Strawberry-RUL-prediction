"""
Model B: MobileNetV2 + CBAM + LSTM + Regression Head

Architecture Pipeline:
  1. MobileNetV2 (conv features, pretrained ImageNet) → 1280-dim feature maps (7×7)
  2. CBAM (Channel + Spatial Attention) → refined feature maps
  3. Global Average Pooling → 1280-dim vector
  4. Concatenate with environmental features (temp, humidity) → 1282-dim
  5. LSTM (temporal modeling) → 128-dim hidden
  6. Regression Head (128→64→1) → RUL in hours
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Allow import from src/shared/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.shared.cbam import CBAM


class StrawberryRULModelB(nn.Module):
    """
    Hybrid model for strawberry RUL prediction.

    MobileNetV2 (spatial) → CBAM (attention) → LSTM (temporal) → Regression Head
    """

    def __init__(
        self,
        rnn_hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        cbam_reduction_ratio: int = 16,
        cbam_kernel_size: int = 7,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            rnn_hidden_size: Hidden size of the LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate for LSTM and regression head.
            cbam_reduction_ratio: Channel reduction ratio for CBAM.
            cbam_kernel_size: Spatial kernel size for CBAM.
            freeze_backbone: If True, freeze MobileNetV2 weights.
        """
        super(StrawberryRULModelB, self).__init__()

        # ---- 1. CNN Backbone: MobileNetV2 ----
        weights = MobileNet_V2_Weights.DEFAULT
        backbone = mobilenet_v2(weights=weights)

        self.feature_dim = 1280  # MobileNetV2 final conv channels
        self.cnn_features = backbone.features  # Conv layers → (B, 1280, 7, 7) @224×224
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)  # (B, 1280, 1, 1)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.cnn_features.parameters():
                param.requires_grad = False

        # ---- 2. CBAM Attention Module ----
        self.cbam = CBAM(
            in_channels=self.feature_dim,
            reduction_ratio=cbam_reduction_ratio,
            kernel_size=cbam_kernel_size,
        )

        # ---- 3. Environmental Features ----
        self.env_dim = 2  # temperature, humidity

        # ---- 4. LSTM Temporal Model ----
        self.rnn_input_size = self.feature_dim + self.env_dim  # 1280 + 2 = 1282
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ---- 5. Regression Head ----
        self.regressor = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Single RUL value in hours
        )

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from a batch of images using
        MobileNetV2 + CBAM.

        Args:
            images: (N, 3, 224, 224) — arbitrary batch of single frames.

        Returns:
            features: (N, 1280) — pooled feature vector per frame.
        """
        # Conv features: (N, 1280, 7, 7)
        feat_maps = self.cnn_features(images)

        # CBAM attention: (N, 1280, 7, 7)
        attended = self.cbam(feat_maps)

        # Global pooling: (N, 1280, 1, 1) → (N, 1280)
        pooled = self.cnn_pool(attended).flatten(1)

        return pooled

    def forward(
        self, images_seq: torch.Tensor, env_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images_seq: (batch_size, seq_len, 3, 224, 224)
            env_seq:    (batch_size, seq_len, 2)

        Returns:
            rul: (batch_size, 1) — predicted remaining useful life in hours.
        """
        batch_size, seq_len, C, H, W = images_seq.size()

        # ---- Step 1: Extract per-frame spatial features ----
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        spatial_features = self._extract_features(images_reshaped)  # (B*S, 1280)

        # Reshape back to sequences: (B, S, 1280)
        spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)

        # ---- Step 2: Fuse with environmental features ----
        fused_features = torch.cat((spatial_features, env_seq), dim=2)  # (B, S, 1282)

        # ---- Step 3: Temporal modeling with LSTM ----
        lstm_out, _ = self.lstm(fused_features)  # (B, S, hidden_size)

        # Take the last time step for RUL prediction
        last_out = lstm_out[:, -1, :]  # (B, hidden_size)

        # ---- Step 4: Regression ----
        rul = self.regressor(last_out)  # (B, 1)

        return rul


# ---------------------------------------------------------------------------
# Shape sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Model B: MobileNetV2 + CBAM + LSTM")
    model = StrawberryRULModelB()
    dummy_images = torch.randn(2, 5, 3, 224, 224)  # batch=2, seq=5
    dummy_envs = torch.randn(2, 5, 2)
    output = model(dummy_images, dummy_envs)
    print(f"  Input images: {dummy_images.shape}")
    print(f"  Input env:    {dummy_envs.shape}")
    print(f"  Output:       {output.shape}  (expected: [2, 1])")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:    {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print("Model B test passed!")
