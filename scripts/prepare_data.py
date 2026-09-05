"""Dataset preparation script: synchronizes, windows, and splits IO-VNBD."""

import argparse
import logging
from pathlib import Path
from idr.io.preprocess import preprocess_dataset
from idr.io.loader import load_drive_pair
from idr.eval.plotting import generate_eda_plot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preprocess and window IO-VNBD dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory with raw CSVs")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for NPZ files")
    parser.add_argument("--eda-output", type=str, default="reports/figures/eda_sensor_timeseries.png",
                        help="Path for the sensor time-series EDA plot")
    args = parser.parse_args()

    logger.info("Starting dataset preprocessing...")
    saved_files = preprocess_dataset(Path(args.raw_dir), Path(args.output_dir))
    drive_dir = next((p for p in Path(args.raw_dir).glob("**/*")
                      if p.is_dir() and list(p.glob("S-*.csv"))), None)
    if drive_dir is not None:
        drive_id = drive_dir.name
        drive = load_drive_pair(drive_dir, drive_id)
        generate_eda_plot(drive.phone_df, Path(args.eda_output))
        logger.info("Saved EDA plot to %s", args.eda_output)
    logger.info(f"Preprocessing complete. Saved splits: {list(saved_files.keys())}")

if __name__ == "__main__":
    main()
