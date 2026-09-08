import json

def main():
    with open("reports/PHASE46_BLACKOUT_BENCHMARK_RESULTS.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=========================================================================")
    print("                    PHASE 46 BLACKOUT BENCHMARK SUMMARY                  ")
    print("=========================================================================")

    for split_key, split_title in [
        ("validation_drive_m", "VALIDATION DRIVE M (DRIVER B)"),
        ("held_out_test_drive_y1", "HELDOUT TEST DRIVE Y1 (DRIVER D)"),
        ("combined_benchmark", "COMBINED (VALIDATION + TEST)"),
    ]:
        print(f"\n>>> {split_title} <<<")
        for k, v in data[split_key].items():
            lbl = v["config_label"]
            ov = v["overall"]
            print(f"\n--- {lbl} ---")
            print(f"    Overall Median Drift: {ov['median_drift_pct']}% | P90 Drift: {ov['p90_drift_pct']}% | Worst Drift: {ov['worst_drift_pct']}%")
            print(f"    Overall Median Pos Err: {ov['median_final_pos_err_m']} m | P90 Pos Err: {ov['p90_final_pos_err_m']} m")
            print(f"    Speed RMSE: {ov['speed_rmse_mps']} m/s | Heading RMSE: {ov['heading_rmse_deg']} deg | SIH Pass Rate: {ov['sih_pass_rate_pct']}%")
            for d, d_val in v.get("by_duration", {}).items():
                print(f"      [{d:>3s}] Med Drift: {d_val['median_drift_pct']:>6.1f}% | P90 Drift: {d_val['p90_drift_pct']:>6.1f}% | Med Pos Err: {d_val['median_final_pos_err_m']:>6.1f} m | P90 Pos Err: {d_val['p90_final_pos_err_m']:>6.1f} m | Spd RMSE: {d_val['speed_rmse_mps']:>5.2f} m/s")


if __name__ == "__main__":
    main()
