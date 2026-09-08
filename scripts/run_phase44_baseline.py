"""Phase 44 Experiment Runner: Physical Session Replay + Authentic Baseline Evaluation.

Executes:
1. Physical Session Replay of exp_20260908_172204_two_wheeler -> reports/PHASE44_PHYSICAL_SESSION_REPORT.json/.md
2. Authentic Test Baseline Evaluation (Drive Y1 / Driver D) -> reports/PHASE44_AUTHENTIC_TEST_BASELINE.json/.md
3. Authentic Validation Baseline Evaluation (Drive M / Driver B) -> reports/PHASE44_AUTHENTIC_VAL_BASELINE.json/.md
"""

from pathlib import Path
import logging

from idr.eval.experiment_runner import ExperimentHarness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    harness = ExperimentHarness(models_dir="models")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Physical Field Trial Session Replay
    session_csv = Path("data/sessions/exp_20260908_172204_two_wheeler/telemetry.csv")
    if session_csv.exists():
        logger.info("Running Physical Field Trial Replay (exp_20260908_172204_two_wheeler)...")
        res_phys = harness.run_physical_session_replay(
            session_csv_path=session_csv,
            experiment_id="exp_20260908_172204_two_wheeler",
            notes="Real Android GPS-denied walking session replay. Standstill -> ~10m walk -> ~10s standstill.",
        )
        res_phys.save_json(reports_dir / "PHASE44_PHYSICAL_SESSION_REPORT.json")
        res_phys.save_markdown(reports_dir / "PHASE44_PHYSICAL_SESSION_REPORT.md")
        logger.info("Saved Physical Session Report to reports/PHASE44_PHYSICAL_SESSION_REPORT.json/.md")

    # 2. Authentic Test Drive Y1 (Driver D)
    drive_y1_dir = Path("data/raw/categorised_authentic/Y (Driver D)/Y1")
    if drive_y1_dir.exists():
        logger.info("Running Authentic Baseline Evaluation on Test Set (Drive Y1 / Driver D)...")
        res_y1 = harness.run_authentic_drive_evaluation(
            driver_folder_path=drive_y1_dir,
            experiment_id="authentic_baseline_test_drive_y1",
            stride=10,
            dataset_role="TEST",
        )
        res_y1.save_json(reports_dir / "PHASE44_AUTHENTIC_TEST_BASELINE.json")
        res_y1.save_markdown(reports_dir / "PHASE44_AUTHENTIC_TEST_BASELINE.md")
        logger.info("Saved Authentic Test Baseline to reports/PHASE44_AUTHENTIC_TEST_BASELINE.json/.md")

    # 3. Authentic Validation Drive M (Driver B)
    drive_m_dir = Path("data/raw/categorised_authentic/M (Driver B)")
    if drive_m_dir.exists():
        logger.info("Running Authentic Baseline Evaluation on Validation Set (Drive M / Driver B)...")
        res_m = harness.run_authentic_drive_evaluation(
            driver_folder_path=drive_m_dir,
            experiment_id="authentic_baseline_val_drive_m",
            stride=10,
            dataset_role="VALIDATION",
        )
        res_m.save_json(reports_dir / "PHASE44_AUTHENTIC_VAL_BASELINE.json")
        res_m.save_markdown(reports_dir / "PHASE44_AUTHENTIC_VAL_BASELINE.md")
        logger.info("Saved Authentic Val Baseline to reports/PHASE44_AUTHENTIC_VAL_BASELINE.json/.md")

    print("\n=======================================================")
    print("  PHASE 44 EXPERIMENT HARNESS EXECUTION COMPLETE")
    print("=======================================================")
    if session_csv.exists():
        print(f"Physical Session DR Distance: {res_phys.dead_reckoning.total_dr_distance_m:.2f} m")
        print(f"Physical Session Peak EKF Speed: {res_phys.ekf_velocity.peak_pred_mps:.2f} m/s")
    if drive_y1_dir.exists():
        print(f"Drive Y1 Authentic MAE: {res_y1.ai_velocity.mae_mps:.3f} m/s ({res_y1.ai_velocity.mae_kmh:.2f} km/h)")
        print(f"Drive Y1 Authentic RMSE: {res_y1.ai_velocity.rmse_mps:.3f} m/s ({res_y1.ai_velocity.rmse_kmh:.2f} km/h)")
    if drive_m_dir.exists():
        print(f"Drive M Authentic MAE: {res_m.ai_velocity.mae_mps:.3f} m/s ({res_m.ai_velocity.mae_kmh:.2f} km/h)")
        print(f"Drive M Authentic RMSE: {res_m.ai_velocity.rmse_mps:.3f} m/s ({res_m.ai_velocity.rmse_kmh:.2f} km/h)")


if __name__ == "__main__":
    main()
