"""Unified training script for IMU Denoising, Velocity Estimation, and Residual Drift networks."""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import CONFIG, set_seed
from ..io.preprocess import IDRWindowDataset
from .imu_denoise import IMUDenoiseNet
from .velocity_net import VelocityEstimatorNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, target_idx: int) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_vel, batch_imu in loader:
        optimizer.zero_grad()
        # target_idx == 0 -> velocity, target_idx == 1 -> imu
        target = batch_vel.unsqueeze(1) if target_idx == 0 else batch_imu
        pred = model(batch_x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
    return total_loss / len(loader.dataset)

def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, target_idx: int) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_vel, batch_imu in loader:
            target = batch_vel.unsqueeze(1) if target_idx == 0 else batch_imu
            pred = model(batch_x)
            loss = criterion(pred, target)
            total_loss += loss.item() * len(batch_x)
    return total_loss / len(loader.dataset)

def train_pipeline(data_dir: Path, output_dir: Path):
    set_seed(CONFIG["dataset"].window_size)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_npz = data_dir / "train_data.npz"
    val_npz = data_dir / "val_data.npz"

    if not train_npz.exists():
        logger.warning(f"{train_npz} not found. Synthesizing benchmark demonstration tensors.")
        # Create synthetic benchmark data if raw download has not run
        N_train = 500
        N_val = 100
        L = CONFIG["dataset"].window_size
        X_train = np.random.randn(N_train, 6, L).astype(np.float32)
        y_vel_train = np.abs(np.random.randn(N_train)).astype(np.float32) * 15.0  # 0-30 m/s
        y_imu_train = np.random.randn(N_train, 6).astype(np.float32) * 0.05

        X_val = np.random.randn(N_val, 6, L).astype(np.float32)
        y_vel_val = np.abs(np.random.randn(N_val)).astype(np.float32) * 15.0
        y_imu_val = np.random.randn(N_val, 6).astype(np.float32) * 0.05
    else:
        train_data = np.load(train_npz)
        val_data = np.load(val_npz) if val_npz.exists() else train_data
        X_train, y_vel_train, y_imu_train = train_data["windows"], train_data["targets_vel"], train_data["targets_imu"]
        X_val, y_vel_val, y_imu_val = val_data["windows"], val_data["targets_vel"], val_data["targets_imu"]

    train_ds = IDRWindowDataset(X_train, y_vel_train, y_imu_train)
    val_ds = IDRWindowDataset(X_val, y_vel_val, y_imu_val)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["model"].batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["model"].batch_size, shuffle=False)

    # 1. Train Velocity Estimator
    logger.info("--- Training Forward Velocity Estimator ---")
    vel_model = VelocityEstimatorNet()
    optimizer_v = torch.optim.AdamW(vel_model.parameters(), lr=CONFIG["model"].learning_rate)
    criterion_v = nn.HuberLoss()

    best_v_loss = float("inf")
    epochs = 15  # Fast screening run
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(vel_model, train_loader, optimizer_v, criterion_v, target_idx=0)
        va_loss = eval_epoch(vel_model, val_loader, criterion_v, target_idx=0)
        if va_loss < best_v_loss:
            best_v_loss = va_loss
            torch.save(vel_model.state_dict(), output_dir / "velocity_net.pt")
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

    # 2. Train IMU Denoise Net
    logger.info("--- Training IMU Denoise / Bias Net ---")
    denoise_model = IMUDenoiseNet()
    optimizer_d = torch.optim.AdamW(denoise_model.parameters(), lr=CONFIG["model"].learning_rate)
    criterion_d = nn.MSELoss()

    best_d_loss = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(denoise_model, train_loader, optimizer_d, criterion_d, target_idx=1)
        va_loss = eval_epoch(denoise_model, val_loader, criterion_d, target_idx=1)
        if va_loss < best_d_loss:
            best_d_loss = va_loss
            torch.save(denoise_model.state_dict(), output_dir / "imu_denoise_net.pt")
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

    logger.info(f"All models successfully trained and checkpoints saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train IDR models.")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()
    train_pipeline(Path(args.data_dir), Path(args.output_dir))

if __name__ == "__main__":
    main()
