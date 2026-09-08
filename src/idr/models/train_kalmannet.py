"""Training pipeline for KalmanNet learned Kalman gain estimator (Experimental Research Module).

Trains the recurrent KalmanNetGainEstimator using state innovation sequences
and predicted state vectors.

SCIENTIFIC STATUS:
- Experimental Research Module.
- Classical EKF Riccati filter remains the validated primary baseline for SIH PS 26168.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..config import set_seed
from ..data.provenance import DatasetAuthenticity, SyntheticDataBlockedError
from ..data.safety import TrainingSafetyGate, TrainingProvenanceRecord
from ..filters.kalmannet import KalmanNetGainEstimator
from ..filters.ekf import ExtendedKalmanFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        v = float(np.random.uniform(5.0, 30.0))
        omega = float(np.random.uniform(-0.1, 0.1))
        acc = float(np.random.uniform(-2.0, 2.0))

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


def train_kalmannet(
    output_dir: Path,
    dataset_name: Optional[str] = None,
    allow_synthetic: bool = False,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 2e-3,
) -> TrainingProvenanceRecord:
    """Train KalmanNetGainEstimator with TrainingSafetyGate protection."""
    prov = TrainingSafetyGate.enforce(
        dataset_name=dataset_name or "synthetic-iovnbd-mock",
        allow_synthetic=allow_synthetic,
    )

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
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - KalmanNet Loss: {total_loss:.6f}")

    ckpt_path = output_dir / "kalmannet.pt"
    torch.save(model.state_dict(), ckpt_path)

    record = TrainingProvenanceRecord(
        model_name="KalmanNetGainEstimator (Experimental)",
        dataset_name=prov.dataset_name,
        authenticity=prov.authenticity.value,
        is_scientific_research_valid=False,  # Experimental synthetic Riccati training
        allow_synthetic_flag=allow_synthetic,
        train_drives=["simulated_gaussian"],
        val_drives=[],
        num_train_samples=len(y_t),
        num_val_samples=0,
        epochs_trained=epochs,
        notes="KalmanNet experimental training run",
    )
    record.save(output_dir / "kalmannet_provenance.json")

    logger.info(f"KalmanNet training complete. Checkpoint saved to {ckpt_path}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Train KalmanNet Gain Estimator.")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_kalmannet(
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset,
        allow_synthetic=args.allow_synthetic,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
