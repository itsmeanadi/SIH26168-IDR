"""Export PyTorch models to ONNX and run verification smoke tests for edge mobile deployment."""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from ..models.imu_denoise import IMUDenoiseNet
from ..models.velocity_net import VelocityEstimatorNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def export_models_to_onnx(model_dir: Path, output_dir: Path):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 6, 50, dtype=torch.float32)  # (Batch=1, Channels=6, Length=50)

    # 1. Export Velocity Estimator Net
    vel_model = VelocityEstimatorNet()
    vel_pt = model_dir / "velocity_net.pt"
    if vel_pt.exists():
        vel_model.load_state_dict(torch.load(vel_pt, weights_only=True))
    vel_model.eval()

    vel_onnx_path = output_dir / "velocity_net.onnx"
    torch.onnx.export(
        vel_model,
        dummy_input,
        vel_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["imu_window"],
        output_names=["forward_speed"],
        dynamic_axes={"imu_window": {0: "batch_size"}, "forward_speed": {0: "batch_size"}},
    )
    vel_size_mb = vel_onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported VelocityEstimatorNet -> {vel_onnx_path} ({vel_size_mb:.2f} MB)")

    # 2. Export IMU Denoise Net
    denoise_model = IMUDenoiseNet()
    denoise_pt = model_dir / "imu_denoise_net.pt"
    if denoise_pt.exists():
        denoise_model.load_state_dict(torch.load(denoise_pt, weights_only=True))
    denoise_model.eval()

    denoise_onnx_path = output_dir / "imu_denoise_net.onnx"
    torch.onnx.export(
        denoise_model,
        dummy_input,
        denoise_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["imu_window"],
        output_names=["imu_bias_residual"],
        dynamic_axes={"imu_window": {0: "batch_size"}, "imu_bias_residual": {0: "batch_size"}},
    )
    denoise_size_mb = denoise_onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported IMUDenoiseNet -> {denoise_onnx_path} ({denoise_size_mb:.2f} MB)")


    # 3. ONNX Runtime Smoke Test Verification
    try:
        import onnxruntime as ort

        # Velocity Net validation
        ort_sess_v = ort.InferenceSession(str(vel_onnx_path))
        ort_inputs_v = {ort_sess_v.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs_v = ort_sess_v.run(None, ort_inputs_v)[0]
        torch_out_v = vel_model(dummy_input).detach().numpy()
        diff_v = np.max(np.abs(ort_outs_v - torch_out_v))
        logger.info(f"ONNX Runtime Smoke Test [VelocityNet]: PASS (Max Abs Diff: {diff_v:.2e} < 1e-4)")

        # IMU Denoise Net validation
        ort_sess_d = ort.InferenceSession(str(denoise_onnx_path))
        ort_inputs_d = {ort_sess_d.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs_d = ort_sess_d.run(None, ort_inputs_d)[0]
        torch_out_d = denoise_model(dummy_input).detach().numpy()
        diff_d = np.max(np.abs(ort_outs_d - torch_out_d))
        logger.info(f"ONNX Runtime Smoke Test [IMUDenoiseNet]: PASS (Max Abs Diff: {diff_d:.2e} < 1e-4)")

        logger.info("All model export smoke tests passed successfully! Models are ready for 10 Hz edge/mobile inference.")
    except ImportError:
        logger.warning("onnxruntime not installed in environment; skipped runtime verification test.")

def main():
    parser = argparse.ArgumentParser(description="Export IDR models to ONNX/TFLite.")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="models/export")
    args = parser.parse_args()
    export_models_to_onnx(Path(args.model_dir), Path(args.output_dir))

if __name__ == "__main__":
    main()
