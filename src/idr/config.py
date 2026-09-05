"""Global configuration, hyperparameters, seed settings and constants."""

from pathlib import Path
from dataclasses import dataclass
import os
import random
import numpy as np

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OSM_DIR = DATA_DIR / "osm"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, OSM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hardware & Seeds
SEED = 42
SAMPLING_RATE_HZ = 10.0  # 10 Hz smartphone target
DT = 1.0 / SAMPLING_RATE_HZ

def set_seed(seed: int = SEED):
    """Ensure determinism across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass
class DatasetConfig:
    """IO-VNBD Dataset Split Configuration."""
    train_drives: tuple = ("M", "S", "Vta", "Vtb", "Vw")
    val_drives: tuple = ("Y1",)
    test_drives: tuple = ("Vf",)
    window_size: int = 50  # 5 seconds at 10 Hz
    window_stride: int = 10  # 1 second stride (80% overlap)

@dataclass
class ModelConfig:
    """Model hyperparameters."""
    # IMU Denoise Net
    denoise_in_channels: int = 6  # ax, ay, az, gx, gy, gz
    denoise_out_channels: int = 6
    denoise_hidden_dim: int = 64
    
    # Velocity Estimator Net
    vel_in_channels: int = 6
    vel_hidden_dim: int = 64
    vel_num_layers: int = 2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 30
    weight_decay: float = 1e-4

@dataclass
class FilterConfig:
    """EKF / UKF sensor fusion tuning parameters."""
    pos_noise: float = 5.0      # GNSS horizontal position standard deviation (m)
    vel_noise: float = 0.5      # GNSS velocity standard deviation (m/s)
    accel_noise: float = 0.2    # Accelerometer process noise (m/s^2)
    gyro_noise: float = 0.02    # Gyro process noise (rad/s)
    nhc_lat_noise: float = 0.05 # Non-holonomic lateral velocity pseudo-measurement noise (m/s)
    nhc_vert_noise: float = 0.05# Non-holonomic vertical velocity pseudo-measurement noise (m/s)

CONFIG = {
    "dataset": DatasetConfig(),
    "model": ModelConfig(),
    "filter": FilterConfig(),
}
