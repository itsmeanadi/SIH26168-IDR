"""Forward-Velocity Estimator Network.

Estimates vehicle forward speed from smartphone IMU time windows, replacing missing OBD-II wheel speed.
Explicitly rejects non-navigation motion (idle vibration, road potholes, mount shocks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityEstimatorNet(nn.Module):
    """Deep Odometry Speed Estimator with non-navigation motion gating.
    
    Input:  (B, 6, window_size) -> [ax, ay, az, gx, gy, gz] in vehicle body frame
    Output: (B, 1) -> estimated vehicle forward speed (m/s) >= 0
    """

    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        # Motion feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # GRU for temporal integration of acceleration to velocity
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Velocity regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Speed must be non-negative (m/s)
        )

        # Idle motion gate: classifies whether the vehicle is stationary (v=0)
        # to filter out engine vibrations, potholes, or phone handling
        self.stationary_gate = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output: 0 if stationary, 1 if moving
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, L)
        feat = self.feature_extractor(x)  # (B, hidden_dim, L)
        
        # Reshape for GRU: (B, L, hidden_dim)
        feat_seq = feat.permute(0, 2, 1)
        out_seq, h_n = self.gru(feat_seq)
        
        # Last time-step representation
        last_h = out_seq[:, -1, :]  # (B, hidden_dim)

        raw_speed = self.regressor(last_h)     # (B, 1)
        motion_prob = self.stationary_gate(last_h) # (B, 1)

        # Gated forward speed: if vehicle is stationary, output clamps to 0
        speed = raw_speed * motion_prob
        return speed
