"""Unified training script for IMU Denoising and Velocity Estimation networks with Safety Gate."""

import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import CONFIG, set_seed
from ..io.preprocess import IDRWindowDataset
from ..data.provenance import (
    DatasetAuthenticity,
    DatasetProvenance,
    SyntheticDataBlockedError,
)
from ..data.safety import TrainingSafetyGate, TrainingProvenanceRecord
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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


def train_pipeline(
    data_dir: Path,
    output_dir: Path,
    dataset_name: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    allow_synthetic: bool = False,
    epochs: int = 25,
    batch_size: int = 128,
) -> TrainingProvenanceRecord:
    """Execute training pipeline protected by TrainingSafetyGate."""
    import time
    import json

    t_start = time.time()

    # 1. Enforce dataset provenance & safety gate
    provenance = TrainingSafetyGate.enforce(
        dataset_name=dataset_name,
        manifest_path=manifest_path,
        allow_synthetic=allow_synthetic,
    )

    set_seed(42)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_npz = data_dir / "train_data.npz"
    val_npz = data_dir / "val_data.npz"

    if not train_npz.exists():
        raise FileNotFoundError(
            f"Required training data file not found at: {train_npz}. "
            f"Preprocess dataset using scripts/prepare_data.py before training."
        )

    train_data = np.load(train_npz)
    val_data = np.load(val_npz) if val_npz.exists() else train_data
    X_train, y_vel_train, y_imu_train = train_data["windows"], train_data["targets_vel"], train_data["targets_imu"]
    X_val, y_vel_val, y_imu_val = val_data["windows"], val_data["targets_vel"], val_data["targets_imu"]

    train_ds = IDRWindowDataset(X_train, y_vel_train, y_imu_train)
    val_ds = IDRWindowDataset(X_val, y_vel_val, y_imu_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {
        "velocity_net": [],
        "imu_denoise_net": [],
        "best_epoch_vel": 0,
        "best_val_loss_vel": float("inf"),
        "best_epoch_imu": 0,
        "best_val_loss_imu": float("inf"),
        "total_duration_sec": 0.0,
    }

    # 2. Train Velocity Estimator
    logger.info("=================================================================")
    logger.info(f"--- Training VelocityEstimatorNet on {len(X_train)} windows ({epochs} epochs, batch={batch_size}) ---")
    logger.info("=================================================================")
    vel_model = VelocityEstimatorNet()
    optimizer_v = torch.optim.AdamW(vel_model.parameters(), lr=CONFIG["model"].learning_rate, weight_decay=1e-4)
    scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_v, T_max=epochs, eta_min=1e-5)
    criterion_v = nn.HuberLoss(delta=1.0)

    best_v_loss = float("inf")
    best_v_epoch = 0

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        cur_lr = optimizer_v.param_groups[0]["lr"]
        tr_loss = train_epoch(vel_model, train_loader, optimizer_v, criterion_v, target_idx=0)
        va_loss = eval_epoch(vel_model, val_loader, criterion_v, target_idx=0)
        scheduler_v.step()
        ep_sec = time.time() - t_ep_start

        is_best = False
        if va_loss < best_v_loss:
            best_v_loss = va_loss
            best_v_epoch = epoch
            is_best = True
            torch.save(vel_model.state_dict(), output_dir / "velocity_net.pt")

        history["velocity_net"].append({
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "lr": float(cur_lr),
            "epoch_sec": float(ep_sec),
            "is_best": is_best,
        })

        best_tag = " [BEST]" if is_best else ""
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Huber Loss: {tr_loss:.4f} | Val Huber Loss: {va_loss:.4f} | LR: {cur_lr:.2e} | Time: {ep_sec:.1f}s{best_tag}")

    history["best_epoch_vel"] = best_v_epoch
    history["best_val_loss_vel"] = float(best_v_loss)

    # 3. Train IMU Denoise Net (Only if clean reference IMU is available in dataset)
    manifest_file = data_dir / "manifest.json"
    imu_denoise_available = True
    if manifest_file.exists():
        with open(manifest_file) as f:
            m_data = json.load(f)
            imu_denoise_available = m_data.get("imu_denoise_available", True)

    has_non_zero_imu_targets = bool(y_imu_train is not None and len(y_imu_train) > 0 and np.any(y_imu_train != 0))

    if imu_denoise_available and has_non_zero_imu_targets:
        logger.info("=================================================================")
        logger.info(f"--- Training IMUDenoiseNet on {len(X_train)} windows ({epochs} epochs, batch={batch_size}) ---")
        logger.info("=================================================================")
        denoise_model = IMUDenoiseNet()
        optimizer_d = torch.optim.AdamW(denoise_model.parameters(), lr=CONFIG["model"].learning_rate, weight_decay=1e-4)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=epochs, eta_min=1e-5)
        criterion_d = nn.MSELoss()

        best_d_loss = float("inf")
        best_d_epoch = 0

        for epoch in range(1, epochs + 1):
            t_ep_start = time.time()
            cur_lr = optimizer_d.param_groups[0]["lr"]
            tr_loss = train_epoch(denoise_model, train_loader, optimizer_d, criterion_d, target_idx=1)
            va_loss = eval_epoch(denoise_model, val_loader, criterion_d, target_idx=1)
            scheduler_d.step()
            ep_sec = time.time() - t_ep_start

            is_best = False
            if va_loss < best_d_loss:
                best_d_loss = va_loss
                best_d_epoch = epoch
                is_best = True
                torch.save(denoise_model.state_dict(), output_dir / "imu_denoise_net.pt")

            history["imu_denoise_net"].append({
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "val_loss": float(va_loss),
                "lr": float(cur_lr),
                "epoch_sec": float(ep_sec),
                "is_best": is_best,
            })

            best_tag = " [BEST]" if is_best else ""
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE Loss: {tr_loss:.6f} | Val MSE Loss: {va_loss:.6f} | LR: {cur_lr:.2e} | Time: {ep_sec:.1f}s{best_tag}")

        history["best_epoch_imu"] = best_d_epoch
        history["best_val_loss_imu"] = float(best_d_loss)
    else:
        logger.warning(
            "IMUDenoiseNet training SKIPPED: Dataset does not contain clean 6-axis IMU reference ground truth. "
            "Preventing generation of trivial/zero-target denoiser checkpoint."
        )
        history["imu_denoise_net_status"] = "SKIPPED_NO_CLEAN_IMU_REFERENCE"

    history["total_duration_sec"] = float(time.time() - t_start)

    # Save training history JSON
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 4. Write provenance audit record
    record = TrainingProvenanceRecord(
        model_name="VelocityEstimatorNet + IMUDenoiseNet",
        dataset_name=provenance.dataset_name,
        authenticity=provenance.authenticity.value,
        is_scientific_research_valid=(provenance.authenticity != DatasetAuthenticity.SYNTHETIC),
        allow_synthetic_flag=allow_synthetic,
        train_drives=list(CONFIG["dataset"].train_drives),
        val_drives=list(CONFIG["dataset"].val_drives),
        num_train_samples=len(X_train),
        num_val_samples=len(X_val),
        epochs_trained=epochs,
        notes="Authentic research training (Driver-Disjoint split)",
    )
    record.save(output_dir / "training_provenance.json")

    logger.info(f"All models trained in {history['total_duration_sec']:.1f}s. Checkpoints saved to {output_dir}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Train IDR models with dataset safety verification.")
    parser.add_argument("--data-dir", type=str, default="data/processed/authentic", help="Path to processed NPZ data")
    parser.add_argument("--output-dir", type=str, default="models/authentic", help="Destination directory for checkpoints")
    parser.add_argument("--dataset", type=str, default="authentic-iovnbd", help="Registered dataset name")
    parser.add_argument("--manifest", type=str, default=None, help="Direct path to dataset YAML/JSON manifest")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly permit training on synthetic mock data")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    args = parser.parse_args()

    train_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset,
        manifest_path=Path(args.manifest) if args.manifest else None,
        allow_synthetic=args.allow_synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
