"""Learned Residual Drift Corrector Network."""

import torch
import torch.nn as nn

class ResidualDriftNet(nn.Module):
    """Predicts residual trajectory correction [dx, dy, d_yaw] over a short integration horizon."""

    def __init__(self, in_features: int = 12, hidden_dim: int = 64):
        super().__init__()
        # in_features: [current_vel_x, current_vel_y, gyro_z, acc_x, acc_y, dt_outage, ...]
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [delta_x, delta_y, delta_yaw]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
