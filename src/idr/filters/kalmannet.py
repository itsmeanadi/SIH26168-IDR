"""KalmanNet: Neural Network-Driven Kalman Gain Estimation for Non-Linear State Space Models.

Inspired by Revach et al. (IEEE TSP 2022), KalmanNet replaces hand-crafted
or static noise covariance matrices (Q, R) with a learned recurrent network (GRU)
that estimates the Kalman gain K_t directly from the innovation sequence and state features.

Includes a safety fallback to classical EKF Riccati gain if condition checks fail.
Fully exportable to ONNX.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class KalmanNetGainEstimator(nn.Module):
    """Recurrent neural network that outputs the Kalman Gain K_t."""

    def __init__(self, dim_x: int = 9, dim_z: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.hidden_dim = hidden_dim

        # Input features:
        # 1. Normalized innovation: dim_z
        # 2. Predicted state norm/diff: dim_x
        # Total feature dim: dim_z + dim_x
        in_dim = dim_z + dim_x

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output head: maps hidden state to flattened Kalman gain matrix of size (dim_x * dim_z)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_x * dim_z),
        )

    def forward(
        self,
        innovation: torch.Tensor,
        state_pred: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Kalman gain K_t.
        
        Args:
            innovation: (B, dim_z)
            state_pred: (B, dim_x)
            h_prev: Optional previous GRU hidden state (1, B, hidden_dim)
        Returns:
            K: (B, dim_x, dim_z)
            h_next: (1, B, hidden_dim)
        """
        feats = torch.cat([innovation, state_pred], dim=-1)  # (B, dim_z + dim_x)
        embed = self.input_layer(feats).unsqueeze(1)         # (B, 1, hidden_dim)

        gru_out, h_next = self.gru(embed, h_prev)
        k_flat = self.gain_head(gru_out[:, 0, :])            # (B, dim_x * dim_z)
        K = k_flat.view(-1, self.dim_x, self.dim_z)
        return K, h_next


class KalmanNetFilter:
    """KalmanNet wrapper providing drop-in compatibility with classical EKF updates."""

    def __init__(
        self,
        model: KalmanNetGainEstimator,
        device: str = "cpu",
        confidence_threshold: float = 5.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.hidden_state: Optional[torch.Tensor] = None
        self.model.eval()

    def reset(self):
        """Reset recurrent state between drives."""
        self.hidden_state = None

    def compute_gain(
        self,
        innovation: np.ndarray,
        state_pred: np.ndarray,
        fallback_K: np.ndarray,
    ) -> np.ndarray:
        """Compute learned Kalman Gain with classical fallback safeguard.
        
        Args:
            innovation: (dim_z,) array
            state_pred: (dim_x,) array
            fallback_K: (dim_x, dim_z) classical Kalman gain
        Returns:
            K: (dim_x, dim_z) safe Kalman gain
        """
        try:
            with torch.no_grad():
                innov_t = torch.tensor(innovation, dtype=torch.float32, device=self.device).unsqueeze(0)
                state_t = torch.tensor(state_pred, dtype=torch.float32, device=self.device).unsqueeze(0)
                k_pred, self.hidden_state = self.model(innov_t, state_t, self.hidden_state)
                K_learned = k_pred[0].cpu().numpy()

            # Stability / Safeguard check: if learned gain has NaN/Inf or extreme norm, fallback
            k_norm = np.linalg.norm(K_learned)
            fallback_norm = np.linalg.norm(fallback_K)
            if np.isnan(k_norm) or np.isinf(k_norm) or k_norm > 10.0 * (fallback_norm + 1e-4):
                return fallback_K

            # Blend learned gain with classical gain: 80% learned, 20% classical prior
            alpha = 0.8
            return alpha * K_learned + (1.0 - alpha) * fallback_K
        except Exception:
            return fallback_K
