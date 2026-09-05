from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import CONFIG, DatasetConfig
from .loader import load_drive_pair, IOVNBDrive

logger = logging.getLogger(__name__)

class IDRWindowDataset:
    """PyTorch Dataset yielding IMU sliding windows and supervision targets."""

    def __init__(self, windows: np.ndarray, targets_vel: np.ndarray, targets_imu: np.ndarray):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.targets_vel = torch.tensor(targets_vel, dtype=torch.float32)
        self.targets_imu = torch.tensor(targets_imu, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets_vel[idx], self.targets_imu[idx]

def create_sliding_windows(
    imu_data: np.ndarray,
    speed_data: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
    clean_imu_ref: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice continuous time-series into overlapping sliding windows.
    
    Returns:
        X_windows: shape (N, 6, window_size) - channel first for 1D convolutions
        y_vel: shape (N,)
        y_imu: shape (N, 6)
    """
    N_samples = len(imu_data)
    if clean_imu_ref is None:
        clean_imu_ref = imu_data

    X_list = []
    y_vel_list = []
    y_imu_list = []

    for start in range(0, N_samples - window_size + 1, stride):
        end = start + window_size
        win = imu_data[start:end].T  # shape (6, window_size)
        
        # Target velocity is the velocity at the end of the window
        target_v = speed_data[end - 1]
        target_imu = clean_imu_ref[end - 1]

        X_list.append(win)
        y_vel_list.append(target_v)
        y_imu_list.append(target_imu)

    if not X_list:
        return np.empty((0, 6, window_size)), np.empty((0,)), np.empty((0, 6))

    return np.array(X_list, dtype=np.float32), np.array(y_vel_list, dtype=np.float32), np.array(y_imu_list, dtype=np.float32)

def preprocess_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: DatasetConfig = None,
) -> Dict[str, Path]:
    """Preprocess all categorized drives and export train/val/test NPZ files."""
    if config is None:
        config = CONFIG["dataset"]

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": config.train_drives,
        "val": config.val_drives,
        "test": config.test_drives,
    }

    saved_paths = {}

    for split_name, drive_ids in splits.items():
        all_X, all_y_vel, all_y_imu = [], [], []
        logger.info(f"Processing {split_name} split with drives: {drive_ids}")

        for drive_id in drive_ids:
            # Search candidate drive directories
            matching_dirs = list(raw_dir.glob(f"**/*{drive_id}*"))
            drive_folder = None
            for d in matching_dirs:
                if d.is_dir() and (list(d.glob("S-*.csv")) or list(d.glob("V-*.csv"))):
                    drive_folder = d
                    break

            if drive_folder is None:
                logger.warning(f"Drive {drive_id} not found in {raw_dir}, skipping.")
                continue

            try:
                drive = load_drive_pair(drive_folder, drive_id)
                phone_imu, phone_gps, v_speed, t = drive.get_synced_data()
                
                clean_ref = None
                if drive.vehicle_df is not None and len(drive.vehicle_df) > 0:
                    clean_ref = phone_imu  # Can be substituted with vehicle IMU if available

                X_w, y_v, y_i = create_sliding_windows(
                    phone_imu,
                    v_speed,
                    window_size=config.window_size,
                    stride=config.window_stride,
                    clean_imu_ref=clean_ref,
                )

                if len(X_w) > 0:
                    all_X.append(X_w)
                    all_y_vel.append(y_v)
                    all_y_imu.append(y_i)
                    logger.info(f"  Drive {drive_id}: generated {len(X_w)} windows.")
            except Exception as e:
                logger.error(f"Error processing drive {drive_id}: {e}")

        if all_X:
            concat_X = np.concatenate(all_X, axis=0)
            concat_y_vel = np.concatenate(all_y_vel, axis=0)
            concat_y_imu = np.concatenate(all_y_imu, axis=0)

            out_file = output_dir / f"{split_name}_data.npz"
            np.savez_compressed(
                out_file,
                windows=concat_X,
                targets_vel=concat_y_vel,
                targets_imu=concat_y_imu,
            )
            saved_paths[split_name] = out_file
            logger.info(f"Saved {split_name} split ({len(concat_X)} windows) to {out_file}")
        else:
            logger.warning(f"No data processed for {split_name} split.")

    return saved_paths
