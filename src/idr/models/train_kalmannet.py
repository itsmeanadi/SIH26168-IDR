"""Training pipeline for KalmanNet learned Kalman gain estimator.

Trains the recurrent KalmanNetGainEstimator using state innovation sequences
from GNSS-available driving segments to learn optimal Kalman gain matrices K_t.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..config import set_seed
from ..filters.kalmannet import KalmanNetGainEstimator
from ..filters.ekf import ExtendedKalmanFilter


def generate_kalmannet_training_data(num_samples: int = 1500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate state innovation, predicted state, and target optimal Kalman gain pairs."""
    set_seed(42)
    ekf = ExtendedKalmanFilter(dt=0.1)

    innovations = []
    state_preds = []
    target_gains = []

    dt = 0.1
    for i in range(num_samples):
        # Varying dynamics: forward speed 5 to 30 m/s, turns -0.1 to +0.1 rad/s
        v = np.random.uniform(5.0, 30.0)
        omega = np.random.uniform(-0.1, 0.1)
        acc = np.random.uniform(-2.0, 2.0)

        ekf.predict(acc, omega)
        
        # Position measurement with noise
        true_pos = ekf.x[:3] + np.random.randn(3) * 0.5
        y = true_pos - ekf.x[:3]

        # Classical optimal Riccati gain for position measurement
        H = np.zeros((3, ekf.dim_x))
        H[:3, :3] = np.eye(3)
        R = np.eye(3) * (2.0**2)
        S = H @ ekf.P @ H.T + R
        K_opt = ekf.P @ H.T @ np.linalg.inv(S)

        innovations.append(y)
        state_preds.append(ekf.x.copy())
        target_gains.append(K_opt)

        # Update filter to advance state
        ekf.x += K_opt @ y
        I = np.eye(ekf.dim_x)
        ekf.P = (I - K_opt @ H) @ ekf.P @ (I - K_opt @ H).T + K_opt @ R @ K_opt.T

    return (
        torch.tensor(np.array(innovations), dtype=torch.float32),
        torch.tensor(np.array(state_preds), dtype=torch.float32),
        torch.tensor(np.array(target_gains), dtype=torch.float32),
    )


def train_kalmannet(output_dir: Path, epochs: int = 20, batch_size: int = 32, lr: float = 2e-3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_t, x_t, k_target = generate_kalmannet_training_data(2000)
    dataset = TensorDataset(y_t, x_t, k_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KalmanNetGainEstimator(dim_x=9, dim_z=3, hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_y, batch_x, batch_k in loader:
            optimizer.zero_grad()
            k_pred, _ = model(batch_y, batch_x)
            loss = criterion(k_pred, batch_k)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        total_loss /= len(loader.dataset)

    ckpt_path = output_dir / "kalmannet.pt"
    torch.save(model.state_dict(), ckpt_path)
    return model


if __name__ == "__main__":
    train_kalmannet(Path("models"))
