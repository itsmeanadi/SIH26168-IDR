"""Training script for InertialOdomNet with physics-consistent data augmentation and Safety Gate.

Loads synchronized drive recordings, extracts body-frame 2D displacement
windows, applies physics-consistent data augmentation, and optimizes Gaussian NLL loss.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..config import CONFIG, set_seed
from ..io.loader import load_drive_pair
from ..data.provenance import DatasetAuthenticity, SyntheticDataBlockedError
from ..data.safety import TrainingSafetyGate, TrainingProvenanceRecord
from .inertial_odom import InertialOdomNet, gaussian_nll_loss, augment_imu_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AugmentedOdomDataset(Dataset):
    """Dataset for 2D Inertial Odometry with on-the-fly or precomputed augmentation."""

    def __init__(self, windows: np.ndarray, targets: np.ndarray, augment: bool = False, augment_factor: int = 4):
        self.augment = augment
        self.windows = windows.astype(np.float32)      # (N, 6, W)
        self.targets = targets.astype(np.float32)      # (N, 2) [dx, dy]
        self.augment_factor = augment_factor if augment else 1

    def __len__(self) -> int:
        return len(self.windows) * self.augment_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx % len(self.windows)
        win = self.windows[base_idx].copy()
        target = self.targets[base_idx].copy()

        if self.augment and (idx >= len(self.windows)):
            win, target = augment_imu_sample(win, target)

        return torch.tensor(win, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def extract_drive_windows(
    imu_data: np.ndarray,
    enu_pos: np.ndarray,
    headings: np.ndarray,
    window_size: int = 50,
    stride: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sliding windows of IMU data and corresponding body-frame displacement."""
    N = len(imu_data)
    windows = []
    targets = []

    for end in range(window_size, N, stride):
        start = end - window_size
        win = imu_data[start:end].T  # (6, W)

        # Total displacement in world ENU over this window
        de = enu_pos[end - 1, 0] - enu_pos[start, 0]
        dn = enu_pos[end - 1, 1] - enu_pos[start, 1]

        # Rotate into body frame at the midpoint of the window
        mid = (start + end) // 2
        psi = headings[mid]
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        dx_body = cos_p * de + sin_p * dn
        dy_body = -sin_p * de + cos_p * dn

        windows.append(win)
        targets.append([dx_body, dy_body])

    if len(windows) == 0:
        return np.empty((0, 6, window_size), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_all_drives(raw_dir: Path, window_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load categorized drives and build train/val splits adhering to frozen DriveSplit policy."""
    train_drives = list(CONFIG["dataset"].train_drives)
    val_drives = list(CONFIG["dataset"].val_drives)
    # Note: test_drives (e.g. Vf) is strictly held out and never loaded during training.
    drives = train_drives + val_drives

    train_wins, train_tgts = [], []
    val_wins, val_tgts = [], []

    R_earth = 6378137.0

    for d in drives:
        drive_path = raw_dir / d
        if not drive_path.exists():
            continue
        try:
            drive_data = load_drive_pair(drive_path, d)
            phone_imu, phone_gps, v_speed, t = drive_data.get_synced_data()
        except Exception as e:
            logger.warning(f"Could not load drive {d}: {e}")
            continue

        ref_lat, ref_lon = phone_gps[0, 0], phone_gps[0, 1]
        lat_rad = np.deg2rad(ref_lat)
        d_lat = np.deg2rad(phone_gps[:, 0] - ref_lat)
        d_lon = np.deg2rad(phone_gps[:, 1] - ref_lon)
        north = d_lat * R_earth
        east = d_lon * R_earth * np.cos(lat_rad)
        enu = np.column_stack([east, north])

        headings = np.zeros(len(enu))
        headings[1:] = np.arctan2(np.diff(north), np.diff(east))
        headings[0] = headings[1]

        wins, tgts = extract_drive_windows(phone_imu, enu, headings, window_size=window_size)
        if len(wins) == 0:
            continue

        if d in train_drives:
            train_wins.append(wins)
            train_tgts.append(tgts)
        else:
            val_wins.append(wins)
            val_tgts.append(tgts)

    if not train_wins:
        raise FileNotFoundError(
            f"No valid training drive CSV files found in {raw_dir}. "
            "Silent fallback to random tensors has been removed for scientific integrity. "
            "Ensure authentic or synthetic datasets are installed."
        )

    X_tr = np.concatenate(train_wins, axis=0)
    Y_tr = np.concatenate(train_tgts, axis=0)
    X_va = np.concatenate(val_wins, axis=0) if val_wins else X_tr[:50]
    Y_va = np.concatenate(val_tgts, axis=0) if val_tgts else Y_tr[:50]

    return X_tr, Y_tr, X_va, Y_va


def train_inertial_odom(
    raw_dir: Path,
    output_dir: Path,
    dataset_name: Optional[str] = None,
    allow_synthetic: bool = False,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    window_size: int = 50,
) -> TrainingProvenanceRecord:
    """Train InertialOdomNet protected by TrainingSafetyGate."""
    prov = TrainingSafetyGate.enforce(
        dataset_name=dataset_name,
        allow_synthetic=allow_synthetic,
    )

    set_seed(42)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset and building training windows...")
    X_tr, Y_tr, X_va, Y_va = prepare_all_drives(raw_dir, window_size=window_size)
    logger.info(f"Dataset extracted: Train={len(X_tr)} windows, Val={len(X_va)} windows (W={window_size} steps)")

    train_ds = AugmentedOdomDataset(X_tr, Y_tr, augment=True, augment_factor=5)
    val_ds = AugmentedOdomDataset(X_va, Y_va, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = InertialOdomNet(in_channels=6, window_size=window_size, hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    ckpt_path = output_dir / "inertial_odom.pt"

    logger.info("--- Starting InertialOdomNet Training (Gaussian NLL Loss) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            pred_dx_dy = out[:, :2]
            log_var = out[:, 2:]
            loss = gaussian_nll_loss(pred_dx_dy, log_var, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation MAE
        model.eval()
        val_errors = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x)
                pred_dx_dy = out[:, :2]
                err = torch.norm(pred_dx_dy - batch_y, dim=1)
                val_errors.extend(err.cpu().numpy().tolist())

        val_mae = float(np.mean(val_errors)) if val_errors else 0.0
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train NLL: {train_loss:.4f} | Val MAE: {val_mae:.3f} m")

    record = TrainingProvenanceRecord(
        model_name="InertialOdomNet",
        dataset_name=prov.dataset_name,
        authenticity=prov.authenticity.value,
        is_scientific_research_valid=(prov.authenticity != DatasetAuthenticity.SYNTHETIC),
        allow_synthetic_flag=allow_synthetic,
        train_drives=["M", "S", "Vf", "Vta", "Vtb"],
        val_drives=["Vw", "Y1"],
        num_train_samples=len(X_tr),
        num_val_samples=len(X_va),
        epochs_trained=epochs,
        notes="InertialOdomNet training run",
    )
    record.save(output_dir / "inertial_odom_provenance.json")

    logger.info(f"Training complete. Best model saved to {ckpt_path} (Val MAE: {best_val_mae:.3f} m)")
    return record


def main():
    parser = argparse.ArgumentParser(description="Train InertialOdomNet.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/categorised")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    train_inertial_odom(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset,
        allow_synthetic=args.allow_synthetic,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
