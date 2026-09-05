"""IMU Denoising and Bias Correction Network.

Lightweight 1D-CNN regressing bias/noise residuals from raw smartphone IMU windows.
Compatible with ONNX Opset 14+ and TFLite.
"""

import torch
import torch.nn as nn

class IMUDenoiseNet(nn.Module):
    """1D Dilated Residual CNN for IMU bias and noise removal.
    
    Input shape:  (Batch, 6, window_size) -> [ax, ay, az, gx, gy, gz]
    Output shape: (Batch, 6) -> residual bias/noise offset to subtract
    """

    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, out_channels: int = 6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Dilated residual blocks for wide receptive field at low compute
        self.res1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu2 = nn.ReLU()

        self.res2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu3 = nn.ReLU()

        # Global average pooling + projection head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, L)
        h = self.relu1(self.conv1(x))
        h = self.relu2(h + self.res1(h))
        h = self.relu3(h + self.res2(h))
        
        pooled = self.gap(h).squeeze(-1)  # (B, hidden_dim)
        residual = self.head(pooled)      # (B, 6)
        return residual
