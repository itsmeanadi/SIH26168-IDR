"""2D Inertial Odometry Network (InertialOdomNet) with Heteroscedastic Uncertainty.

Predicts body-frame 2D displacement (dx, dy) over a sliding IMU window
together with aleatoric uncertainty (log-variance), trained via Gaussian NLL loss.
Supports extensive data augmentation (rotation, speed scaling, bias perturbation)
to prevent shortcut learning on highway datasets.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """1D Dilated Residual Block."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class InertialOdomNet(nn.Module):
    """Deep 2D Inertial Odometry Network.

    Input:  (B, 6, W) -> 6-axis IMU [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Output: (B, 4)    -> [dx_body, dy_body, log_var_x, log_var_y]
    """

    def __init__(self, in_channels: int = 6, window_size: int = 50, hidden_dim: int = 128):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Front-end stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Multi-scale temporal feature extractor (dilated convolutions)
        self.res1 = ResBlock1D(64, 64, dilation=1)
        self.res2 = ResBlock1D(64, 128, dilation=2)
        self.res3 = ResBlock1D(128, hidden_dim, dilation=4)

        # Recurrent sequence aggregator (causal)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output head: [dx, dy, log_var_x, log_var_y]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, 6, W) or (B, W, 6)
        Returns:
            (B, 4): [dx_body, dy_body, log_var_x, log_var_y]
        """
        if x.dim() == 3 and x.shape[1] != 6 and x.shape[2] == 6:
            x = x.transpose(1, 2)

        feat = self.stem(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)  # (B, hidden_dim, W)

        # Permute for GRU: (B, W, hidden_dim)
        gru_in = feat.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)
        last_hidden = gru_out[:, -1, :]  # (B, hidden_dim)

        out = self.head(last_hidden)  # (B, 4)
        # Clamp log-variance for numerical stability: log_var in [-6, 6] -> sigma in [0.05, 20.0]
        dx_dy = out[:, :2]
        log_var = torch.clamp(out[:, 2:], min=-6.0, max=6.0)
        return torch.cat([dx_dy, log_var], dim=-1)


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian Negative Log-Likelihood Loss.

    pred:   (B, 4) -> [dx_pred, dy_pred, log_var_x, log_var_y]
    target: (B, 2) -> [dx_true, dy_true]
    """
    mu = pred[:, :2]
    log_var = pred[:, 2:]
    precision = torch.exp(-log_var)
    diff_sq = (target - mu) ** 2
    loss = 0.5 * torch.mean(precision * diff_sq + log_var)
    return loss


def augment_imu_sample(
    imu_window: np.ndarray,
    target_displacement: np.ndarray,
    speed_scale_range: Tuple[float, float] = (0.5, 2.0),
    acc_bias_std: float = 0.25,
    gyro_bias_std: float = 0.02,
    noise_std_acc: float = 0.15,
    noise_std_gyro: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies physics-consistent augmentation to a 6-axis IMU window and displacement target.

    Args:
        imu_window: (6, W) [ax, ay, az, gx, gy, gz]
        target_displacement: (2,) [dx_body, dy_body]
    """
    win = imu_window.copy()
    target = target_displacement.copy()

    # 1. Speed scaling: scale velocities and linear accelerations by factor s
    s = float(np.random.uniform(speed_scale_range[0], speed_scale_range[1]))
    win[:3, :] *= s
    win[3:, :] *= s  # turning rate scales linearly with speed for same radius
    target *= s

    # 2. Additive random sensor bias
    b_acc = np.random.randn(3, 1) * acc_bias_std
    b_gyro = np.random.randn(3, 1) * gyro_bias_std
    win[:3, :] += b_acc
    win[3:, :] += b_gyro

    # 3. Additive Gaussian white noise
    win[:3, :] += np.random.randn(*win[:3, :].shape) * noise_std_acc
    win[3:, :] += np.random.randn(*win[3:, :].shape) * noise_std_gyro

    # 4. Small random planar heading perturbation (-10 to +10 degrees)
    theta = float(np.radians(np.random.uniform(-10.0, 10.0)))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R2 = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # Rotate horizontal accel
    win[:2, :] = R2 @ win[:2, :]
    # Rotate displacement target
    target = R2 @ target

    return win.astype(np.float32), target.astype(np.float32)
