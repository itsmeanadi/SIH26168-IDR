"""Phase 45 Experiment Runner: Authentic IO-VNBD Model Evaluation + Physical Field Session Analysis.

Evaluates:
1. Untouched Final Test Set: Drive Y1 (Driver D) -> reports/PHASE45_AUTHENTIC_TEST_EVALUATION.json/.md
2. Validation Set: Drive M (Driver B) -> reports/PHASE45_AUTHENTIC_VAL_EVALUATION.json/.md
3. Qualitative Physical Field Replay: exp_20260908_172204_two_wheeler -> reports/PHASE45_PHYSICAL_SESSION_REPORT.json/.md
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

    print("\n=======================================================")
    print("  PHASE 45 EVALUATION ON NEW AUTHENTIC VELOCITY MODEL")
    print("=======================================================")

    # 1. Untouched Held-Out Test Set (Drive Y1 / Driver D)
    drive_y1_dir = Path("data/raw/categorised_authentic/Y (Driver D)/Y1")
    if drive_y1_dir.exists():
        logger.info("Evaluating on Untouched Test Set (Drive Y1 / Driver D)...")
        res_test = harness.run_authentic_drive_evaluation(
            driver_folder_path=drive_y1_dir,
            experiment_id="phase45_authentic_test_drive_y1",
            stride=10,
            dataset_role="TEST",
        )
        res_test.save_json(reports_dir / "PHASE45_AUTHENTIC_TEST_EVALUATION.json")
        res_test.save_markdown(reports_dir / "PHASE45_AUTHENTIC_TEST_EVALUATION.md")
        print("\n[1] TEST SET (Drive Y1 / Driver D - Untouched Held-Out):")
        print(f"    MAE:       {res_test.ai_velocity.mae_mps:.3f} m/s ({res_test.ai_velocity.mae_kmh:.2f} km/h)")
        print(f"    RMSE:      {res_test.ai_velocity.rmse_mps:.3f} m/s ({res_test.ai_velocity.rmse_kmh:.2f} km/h)")
        print(f"    Mean Bias: {res_test.ai_velocity.mean_bias_mps:.3f} m/s")
        print(f"    Low Speed MAE (<1 m/s):   {res_test.ai_velocity.regime_low_speed_mae:.3f} m/s")
        print(f"    Med Speed MAE (1-10 m/s): {res_test.ai_velocity.regime_med_speed_mae:.3f} m/s")
        print(f"    High Speed MAE (>10 m/s): {res_test.ai_velocity.regime_high_speed_mae:.3f} m/s")

    # 2. Validation Set (Drive M / Driver B)
    drive_m_dir = Path("data/raw/categorised_authentic/M (Driver B)")
    if drive_m_dir.exists():
        logger.info("Evaluating on Validation Set (Drive M / Driver B)...")
        res_val = harness.run_authentic_drive_evaluation(
            driver_folder_path=drive_m_dir,
            experiment_id="phase45_authentic_val_drive_m",
            stride=10,
            dataset_role="VALIDATION",
        )
        res_val.save_json(reports_dir / "PHASE45_AUTHENTIC_VAL_EVALUATION.json")
        res_val.save_markdown(reports_dir / "PHASE45_AUTHENTIC_VAL_EVALUATION.md")
        print("\n[2] VALIDATION SET (Drive M / Driver B):")
        print(f"    MAE:       {res_val.ai_velocity.mae_mps:.3f} m/s ({res_val.ai_velocity.mae_kmh:.2f} km/h)")
        print(f"    RMSE:      {res_val.ai_velocity.rmse_mps:.3f} m/s ({res_val.ai_velocity.rmse_kmh:.2f} km/h)")
        print(f"    Mean Bias: {res_val.ai_velocity.mean_bias_mps:.3f} m/s")
        print(f"    Low Speed MAE (<1 m/s):   {res_val.ai_velocity.regime_low_speed_mae:.3f} m/s")
        print(f"    Med Speed MAE (1-10 m/s): {res_val.ai_velocity.regime_med_speed_mae:.3f} m/s")
        print(f"    High Speed MAE (>10 m/s): {res_val.ai_velocity.regime_high_speed_mae:.3f} m/s")

    # 3. Physical Field Session Replay (exp_20260908_172204_two_wheeler)
    session_csv = Path("data/sessions/exp_20260908_172204_two_wheeler/telemetry.csv")
    if session_csv.exists():
        logger.info("Running Physical Field Session Replay...")
        res_phys = harness.run_physical_session_replay(
            session_csv_path=session_csv,
            experiment_id="phase45_physical_replay_exp_20260908_172204",
            notes="Qualitative out-of-domain evaluation on real Android GPS-denied walk with authentic checkpoint.",
        )
        res_phys.save_json(reports_dir / "PHASE45_PHYSICAL_SESSION_REPORT.json")
        res_phys.save_markdown(reports_dir / "PHASE45_PHYSICAL_SESSION_REPORT.md")
        init_speed = res_phys.diagnostics.get("initial_standstill_speed_mps", 0.0)
        final_speed = res_phys.diagnostics.get("final_stopping_speed_mps", 0.0)
        print("\n[3] PHYSICAL FIELD SESSION (exp_20260908_172204_two_wheeler - Real Android):")
        print(f"    Total DR Distance:      {res_phys.dead_reckoning.total_dr_distance_m:.2f} m")
        print(f"    Initial Standstill EKF: {init_speed:.2f} m/s")
        print(f"    Final Stopping EKF:     {final_speed:.2f} m/s")
        print(f"    AI Peak Speed:          {res_phys.ai_velocity.peak_pred_mps:.2f} m/s")
        print(f"    AI Mean Speed:          {res_phys.ai_velocity.mean_pred_mps:.2f} m/s")
        print(f"    ZUPT Intervals:         {res_phys.dead_reckoning.zupt_intervals_count}")


if __name__ == "__main__":
    main()
