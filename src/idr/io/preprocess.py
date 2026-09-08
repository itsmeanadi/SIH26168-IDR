import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import CONFIG, DatasetConfig
from .loader import load_drive_pair, IOVNBDrive

logger = logging.getLogger(__name__)

GAP_THRESHOLD_SEC = 0.35  # > 3.5x nominal 100ms dt


class IDRWindowDataset(Dataset):
    """PyTorch Dataset yielding IMU sliding windows and supervision targets."""

    def __init__(
        self,
        windows: Union[np.ndarray, torch.Tensor],
        targets_vel: Union[np.ndarray, torch.Tensor],
        targets_imu: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        if isinstance(windows, np.ndarray):
            self.windows = torch.tensor(windows, dtype=torch.float32)
        else:
            self.windows = windows.float()

        if isinstance(targets_vel, np.ndarray):
            self.targets_vel = torch.tensor(targets_vel, dtype=torch.float32)
        else:
            self.targets_vel = targets_vel.float()

        if targets_imu is not None:
            if isinstance(targets_imu, np.ndarray):
                self.targets_imu = torch.tensor(targets_imu, dtype=torch.float32)
            else:
                self.targets_imu = targets_imu.float()
        else:
            self.targets_imu = torch.zeros((len(self.windows), 6), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets_vel[idx], self.targets_imu[idx]


def detect_continuous_segments(
    timestamps: np.ndarray,
    gap_threshold_sec: float = GAP_THRESHOLD_SEC,
    min_segment_length: int = 50,
) -> List[Tuple[int, int]]:
    """Detect contiguous index intervals without recording gaps (> gap_threshold_sec).
    
    Returns:
        List of (start_idx, end_idx) tuples for slices where timestamps are continuous.
    """
    N = len(timestamps)
    if N < min_segment_length:
        return []

    dt = np.diff(timestamps)
    gap_indices = np.where(dt > gap_threshold_sec)[0]

    # Gap between index i and i+1 means segment 0 ends at i+1 (exclusive slice [0:i+1])
    # and segment 1 starts at i+1 (slice [i+1:end])
    seg_starts = [0] + (gap_indices + 1).tolist()
    seg_ends = (gap_indices + 1).tolist() + [N]

    segments = []
    for start, end in zip(seg_starts, seg_ends):
        if (end - start) >= min_segment_length:
            segments.append((start, end))

    return segments


def create_sliding_windows(
    imu_data: np.ndarray,
    speed_data: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
    clean_imu_ref: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    gap_threshold_sec: float = GAP_THRESHOLD_SEC,
    return_metadata: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]],
]:
    """Slice time-series into overlapping sliding windows STRICTLY within continuous segments.
    
    Guarantees that NO sliding window crosses a timestamp discontinuity.
    
    Returns:
        X_windows: shape (N, 6, window_size) - channel first for 1D convolutions
        y_vel: shape (N,) - forward speed in m/s at end of window
        y_imu: shape (N, 6) - sensor bias/noise offset
        (optional) window_metadata: list of dicts with segment index and relative window time bounds
    """
    N_samples = len(imu_data)
    has_valid_clean_ref = (
        clean_imu_ref is not None
        and clean_imu_ref is not imu_data
        and clean_imu_ref.shape == imu_data.shape
    )

    if timestamps is not None:
        segments = detect_continuous_segments(
            timestamps,
            gap_threshold_sec=gap_threshold_sec,
            min_segment_length=window_size,
        )
    else:
        segments = [(0, N_samples)] if N_samples >= window_size else []

    X_list = []
    y_vel_list = []
    y_imu_list = []
    meta_list = []

    for seg_id, (seg_start, seg_end) in enumerate(segments):
        for start in range(seg_start, seg_end - window_size + 1, stride):
            end = start + window_size
            win = imu_data[start:end].T  # shape (6, window_size)

            target_v = speed_data[end - 1]

            if has_valid_clean_ref:
                target_imu_bias = (imu_data[end - 1] - clean_imu_ref[end - 1]).astype(np.float32)
            else:
                target_imu_bias = np.zeros(6, dtype=np.float32)

            X_list.append(win)
            y_vel_list.append(target_v)
            y_imu_list.append(target_imu_bias)

            if return_metadata:
                meta_list.append({
                    "segment_id": seg_id,
                    "start_idx": start,
                    "end_idx": end,
                    "t_start": float(timestamps[start]) if timestamps is not None else start * 0.1,
                    "t_end": float(timestamps[end - 1]) if timestamps is not None else (end - 1) * 0.1,
                })

    empty_X = np.empty((0, 6, window_size), dtype=np.float32)
    empty_y_v = np.empty((0,), dtype=np.float32)
    empty_y_i = np.empty((0, 6), dtype=np.float32)

    if not X_list:
        if return_metadata:
            return empty_X, empty_y_v, empty_y_i, []
        return empty_X, empty_y_v, empty_y_i

    arr_X = np.array(X_list, dtype=np.float32)
    arr_y_v = np.array(y_vel_list, dtype=np.float32)
    arr_y_i = np.array(y_imu_list, dtype=np.float32)

    if return_metadata:
        return arr_X, arr_y_v, arr_y_i, meta_list
    return arr_X, arr_y_v, arr_y_i


def fit_training_scaler(X_train: np.ndarray) -> Dict[str, List[float]]:
    """Fit per-channel mean and std scaler STRICTLY on training windows.
    
    X_train: shape (N, 6, window_size)
    """
    assert len(X_train) > 0, "Cannot fit scaler on empty training set"
    # Channel-wise mean and std across all windows and time steps
    # Flatten spatial/temporal dimensions per channel: shape (N * window_size, 6)
    transposed = np.transpose(X_train, (0, 2, 1)).reshape(-1, 6)
    means = np.mean(transposed, axis=0).astype(float).tolist()
    stds = np.std(transposed, axis=0).astype(float).tolist()
    # Avoid zero division
    stds = [s if s > 1e-6 else 1.0 for s in stds]

    scaler = {
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "mean": means,
        "std": stds,
        "fitted_samples_count": int(transposed.shape[0]),
    }
    return scaler


def apply_scaler(X: np.ndarray, scaler: Dict[str, List[float]]) -> np.ndarray:
    """Normalize feature array using pre-fitted channel-wise scaler.
    
    X: shape (N, 6, window_size)
    """
    if len(X) == 0:
        return X
    means = np.array(scaler["mean"], dtype=np.float32).reshape(1, 6, 1)
    stds = np.array(scaler["std"], dtype=np.float32).reshape(1, 6, 1)
    return (X - means) / stds


def preprocess_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: Optional[DatasetConfig] = None,
    normalize_features: bool = False,
) -> Dict[str, Path]:
    """Preprocess categorized drives and export train/val/test NPZ files."""
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

    raw_split_data: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict]]] = {
        "train": ([], [], [], []),
        "val": ([], [], [], []),
        "test": ([], [], [], []),
    }

    manifest_records = []

    for split_name, drive_prefixes in splits.items():
        logger.info(f"Processing {split_name} split with drive prefixes: {drive_prefixes}")
        processed_folders = set()

        for prefix in drive_prefixes:
            matching_dirs = []
            for d in raw_dir.glob(f"**/*{prefix}*"):
                if d.is_dir() and (list(d.glob("S-*.csv")) or list(d.glob("s-*.csv"))):
                    matching_dirs.append(d)

            # De-duplicate
            unique_dirs = sorted(list(set(matching_dirs)))
            if not unique_dirs:
                logger.warning(f"No drive folders found matching prefix '{prefix}' in {raw_dir}")
                continue

            for drive_folder in unique_dirs:
                if drive_folder in processed_folders:
                    continue
                processed_folders.add(drive_folder)

                # Find drive ID from phone csv
                p_csvs = list(drive_folder.glob("S-*.csv")) or list(drive_folder.glob("s-*.csv"))
                if not p_csvs:
                    continue
                d_id = p_csvs[0].stem[2:]

                try:
                    drive = load_drive_pair(drive_folder, d_id)
                    phone_imu, phone_gps, v_speed, t = drive.get_synced_data()

                    clean_ref = None
                    if drive.vehicle_df is not None and "acc_x" in drive.vehicle_df.columns:
                        clean_ref = drive.vehicle_df[
                            ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
                        ].to_numpy(dtype=np.float32)
                        if len(clean_ref) < len(phone_imu):
                            clean_ref = np.pad(clean_ref, ((0, len(phone_imu) - len(clean_ref)), (0, 0)), mode="edge")
                        elif len(clean_ref) > len(phone_imu):
                            clean_ref = clean_ref[:len(phone_imu)]

                    X_w, y_v, y_i, meta = create_sliding_windows(
                        phone_imu,
                        v_speed,
                        window_size=config.window_size,
                        stride=config.window_stride,
                        clean_imu_ref=clean_ref,
                        timestamps=t,
                        gap_threshold_sec=GAP_THRESHOLD_SEC,
                        return_metadata=True,
                    )

                    if len(X_w) > 0:
                        raw_split_data[split_name][0].append(X_w)
                        raw_split_data[split_name][1].append(y_v)
                        raw_split_data[split_name][2].append(y_i)
                        raw_split_data[split_name][3].extend(meta)

                        manifest_records.append({
                            "drive_id": d_id,
                            "split": split_name,
                            "folder": str(drive_folder.relative_to(raw_dir)).replace("\\", "/"),
                            "num_samples": len(phone_imu),
                            "num_windows": len(X_w),
                            "duration_sec": float(t[-1] - t[0]) if len(t) > 0 else 0.0,
                        })
                        logger.info(f"  [{split_name}] Drive {d_id}: generated {len(X_w)} continuous windows.")
                except Exception as e:
                    logger.error(f"Error processing drive {d_id} in {drive_folder}: {e}")

    # Concatenate windows per split
    split_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ["train", "val", "test"]:
        X_list, y_v_list, y_i_list, _ = raw_split_data[split_name]
        if X_list:
            split_arrays[split_name] = {
                "windows": np.concatenate(X_list, axis=0),
                "targets_vel": np.concatenate(y_v_list, axis=0),
                "targets_imu": np.concatenate(y_i_list, axis=0),
            }
        else:
            split_arrays[split_name] = {
                "windows": np.empty((0, 6, config.window_size), dtype=np.float32),
                "targets_vel": np.empty((0,), dtype=np.float32),
                "targets_imu": np.empty((0, 6), dtype=np.float32),
            }

    # Fit scaler STRICTLY on training split
    scaler = None
    if len(split_arrays["train"]["windows"]) > 0:
        scaler = fit_training_scaler(split_arrays["train"]["windows"])
        scaler_file = output_dir / "train_scaler.json"
        with open(scaler_file, "w") as f:
            json.dump(scaler, f, indent=2)
        logger.info(f"Saved training-only scaler to {scaler_file}")

    # Optionally normalize features using the training scaler
    saved_paths = {}
    for split_name, arrs in split_arrays.items():
        if len(arrs["windows"]) > 0:
            windows_to_save = arrs["windows"]
            if normalize_features and scaler is not None:
                windows_to_save = apply_scaler(windows_to_save, scaler)

            out_file = output_dir / f"{split_name}_data.npz"
            np.savez_compressed(
                out_file,
                windows=windows_to_save,
                targets_vel=arrs["targets_vel"],
                targets_imu=arrs["targets_imu"],
            )
            saved_paths[split_name] = out_file
            logger.info(f"Saved {split_name} split ({len(windows_to_save)} windows) to {out_file}")

    # Check if clean IMU reference is available in any drive
    has_any_clean_ref = any(
        (arrs["targets_imu"] is not None and len(arrs["targets_imu"]) > 0 and np.any(arrs["targets_imu"] != 0))
        for arrs in split_arrays.values()
    )

    # Save preprocessing manifest
    manifest_file = output_dir / "manifest.json"
    manifest_data = {
        "preprocessing_version": "2.1.0_authentic_synchronized",
        "raw_dir": str(raw_dir).replace("\\", "/"),
        "window_size": config.window_size,
        "window_stride": config.window_stride,
        "gap_threshold_sec": GAP_THRESHOLD_SEC,
        "features": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "target_vel": "reference_vehicle_forward_speed_mps",
        "imu_denoise_available": bool(has_any_clean_ref),
        "split_policy": config.split_policy,
        "split_summary": {
            s: {
                "num_windows": int(len(split_arrays[s]["windows"])),
                "num_drives": sum(1 for r in manifest_records if r["split"] == s),
            }
            for s in ["train", "val", "test"]
        },
        "drives": manifest_records,
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)

    return saved_paths
