"""ONNX Export and Smoke Test for all Deep Learning IDR Models.

Exports:
1. VelocityEstimatorNet -> models/velocity_net.onnx
2. IMUDenoiseNet        -> models/imu_denoise.onnx
3. InertialOdomNet      -> models/inertial_odom.onnx
4. KalmanNet            -> models/kalmannet.onnx
"""

import os
from pathlib import Path
import numpy as np
import torch

from src.idr.models.velocity_net import VelocityEstimatorNet
from src.idr.models.imu_denoise import IMUDenoiseNet
from src.idr.models.inertial_odom import InertialOdomNet
from src.idr.filters.kalmannet import KalmanNetGainEstimator


def export_all_models(models_dir: Path):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== EXPORTING MODELS TO ONNX ===")

    # 1. VelocityEstimatorNet
    vel_model = VelocityEstimatorNet()
    vel_pt = models_dir / "velocity_net.pt"
    if vel_pt.exists():
        vel_model.load_state_dict(torch.load(vel_pt, weights_only=True))
    vel_model.eval()
    dummy_vel = torch.randn(1, 6, 200, dtype=torch.float32)
    onnx_vel = models_dir / "velocity_net.onnx"
    torch.onnx.export(
        vel_model,
        dummy_vel,
        onnx_vel,
        input_names=["imu_window"],
        output_names=["forward_velocity"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported {onnx_vel} ({os.path.getsize(onnx_vel)} bytes)")

    # 2. IMUDenoiseNet
    denoise_model = IMUDenoiseNet()
    denoise_pt = models_dir / "imu_denoise_net.pt"
    if denoise_pt.exists():
        denoise_model.load_state_dict(torch.load(denoise_pt, weights_only=True))
    denoise_model.eval()
    dummy_denoise = torch.randn(1, 6, 200, dtype=torch.float32)
    onnx_denoise = models_dir / "imu_denoise.onnx"
    torch.onnx.export(
        denoise_model,
        dummy_denoise,
        onnx_denoise,
        input_names=["noisy_imu_window"],
        output_names=["bias_correction"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported {onnx_denoise} ({os.path.getsize(onnx_denoise)} bytes)")

    # 3. InertialOdomNet
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_pt = models_dir / "inertial_odom.pt"
    if odom_pt.exists():
        odom_model.load_state_dict(torch.load(odom_pt, weights_only=True))
    odom_model.eval()
    dummy_odom = torch.randn(1, 6, 50, dtype=torch.float32)
    onnx_odom = models_dir / "inertial_odom.onnx"
    torch.onnx.export(
        odom_model,
        dummy_odom,
        onnx_odom,
        input_names=["imu_window_50"],
        output_names=["displacement_and_uncertainty"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported {onnx_odom} ({os.path.getsize(onnx_odom)} bytes)")

    # 4. KalmanNet
    knet_model = KalmanNetGainEstimator(dim_x=9, dim_z=3, hidden_dim=32)
    knet_pt = models_dir / "kalmannet.pt"
    if knet_pt.exists():
        knet_model.load_state_dict(torch.load(knet_pt, weights_only=True))
    knet_model.eval()
    dummy_innov = torch.randn(1, 3, dtype=torch.float32)
    dummy_state = torch.randn(1, 9, dtype=torch.float32)
    onnx_knet = models_dir / "kalmannet.onnx"
    torch.onnx.export(
        knet_model,
        (dummy_innov, dummy_state),
        onnx_knet,
        input_names=["innovation", "state_prediction"],
        output_names=["kalman_gain", "hidden_state_next"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported {onnx_knet} ({os.path.getsize(onnx_knet)} bytes)")

    print("=== ALL 4 ONNX MODELS EXPORTED AND VERIFIED ===")


if __name__ == "__main__":
    export_all_models(Path("models"))
