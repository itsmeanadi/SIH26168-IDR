# AI-ML Intelligent Dead Reckoning (IDR) System
**Smart India Hackathon Problem Statement 26168 (ISRO)**
*Seamless navigation during GNSS outage using smartphone-grade IMU sensor fusion, deep learning forward-velocity estimation, Non-Holonomic Constraints (NHC), and OpenStreetMap map-matching.*

---

## 🚀 One-Command Quickstart

Reproduce the full pipeline end-to-end:
```bash
make setup && make data && make train && make eval && make plots
```

Or step-by-step:
```bash
# 1. Install pinned dependencies & local package
make setup

# 2. Download IO-VNBD dataset, validate both S/V schemas, preprocess into sliding windows,
#    and generate reports/figures/eda_sensor_timeseries.png
make data

# 3. Train IMU Denoise Net, Forward-Velocity Net, and Residual Drift Corrector
make train

# 4. Evaluate GNSS blackout scenarios (DR + NHC + HMM Map-matching)
make eval

# 5. Generate publication-ready position drift plots & report
make plots

# 6. Run ONNX / TFLite export smoke tests (10 Hz on-device mobile ready)
make export
```

---

## 🎯 Hard Constraints & Key Specifications

- **Drift Requirement**: < 10% of distance travelled during GNSS outage (e.g. < 100 m over 1 km @ ~60 km/h; < 5 m over 50 m).
- **Position Update Rate**: 10 Hz nominal design target (smartphone Android `SensorManager`), scalable to ~200 Hz edge/tactical IMU.
- **Offline Capable**: Zero live API calls at inference time. Bundled/cached OSM graph (GraphML).
- **Lightweight Models**: Small parameter count (< 2 MB footprint), compatible with ONNX Runtime and TFLite (no unsupported ops).
- **Deterministic**: Fixed RNG seeds (`seed=42`), track-based dataset split without cross-window leakage.

---

## 📊 Benchmark Results Preview

Evaluated on **443 distinct GNSS blackout scenarios** across 7 real-world drives from the IO-VNBD dataset.

| Architecture Configuration | Outage Dist (m) | Final Drift (m) | Drift (% Dist) | CEP 50% (m) | Compliance (<10%) |
|---|---|---|---|---|---|
| **Raw IMU Mechanization (Baseline)** | 1153.11 | 27.97 | **2.43%** | 5.97 | **PASSED** |
| **EKF + AI-Velocity + NHC** | 1153.11 | 46.32 | **4.02%** | 13.15 | **PASSED** |
| **EKF + AI-Velocity + NHC + OSM Snap** | 1153.11 | 29.19 | **2.53%** | 12.78 | **PASSED** |

<p align="center">
  <img src="results/figures/trajectory_map.png" alt="Trajectory Map" width="85%"/>
</p>

👉 **Full experimental results, plots, and scenario breakdowns: [results/README.md](results/README.md) & [results/RESULTS.md](results/RESULTS.md).**

---

## 📁 Repository Layout

```
idr-system/
├── data/                          # Gitignored raw & processed datasets
│   ├── raw/                       # Extracted IO-VNBD CSVs (V-*.csv, S-*.csv)
│   └── processed/                 # Resampled (10 Hz), aligned, windowed folds
├── models/                        # Checkpoints (.pt) and exports (.onnx, .tflite)
├── results/                       # Evaluated metrics, benchmark tables & visual PNG figures
│   ├── figures/                   # Trajectory maps, drift CDFs, boxplots, velocity curves
│   ├── RESULTS.md                 # Detailed benchmark report across 443 blackout scenarios
│   └── README.md                  # Visual gallery with embedded plots
├── reports/                       # Generated evaluation outputs & figures
│   ├── RESULTS.md                 # Summary tables across drive scenarios & metrics
│   └── figures/                   # Trajectory maps, drift-vs-distance, velocity curves
├── scripts/
│   ├── download_data.py           # Ingestion script with schema auto-detection & LFS/zip fallback
│   └── prepare_data.py            # Synchronization, coordinate frame transform, windowing
├── src/idr/
│   ├── config.py                  # Pinned configuration, hyperparameters, seed settings
│   ├── io/                        # Robust CSV schema detection, loaders, window generators
│   ├── calib/                     # Phone-to-vehicle attitude alignment (gravity + heading)
│   ├── models/                    # IMU Denoise (1D-CNN), Velocity Estimator (TCN/GRU), ResidualNet
│   ├── filters/                   # EKF / UKF sensor fusion, NHC (Non-Holonomic Constraints)
│   ├── mapmatch/                  # OSM road graph ingestion, HMM / Viterbi snapping
│   ├── eval/                      # Blackout scenario generator, CEP/RMSE/drift metrics, plotting
│   └── export/                    # ONNX and TFLite conversion & verification
├── tests/                         # Unit tests for filter math, schemas, model forward passes
├── DECISIONS.md                   # Architecture & methodology decision log
├── requirements.txt               # Pinned Python dependencies
├── Makefile                       # Workflow orchestration targets
└── README.md
```

To validate an already-ingested dataset without downloading it again:

```bash
python scripts/download_data.py --output-dir data/raw --validate-only
```

---

## 📊 Dataset: IO-VNBD
Dataset reference: *Onyekpe, U., Palade, V., Kanarachos, S., & Szkolnik, A. (2021). IO-VNBD: Inertial and odometry benchmark dataset for ground vehicle positioning. Data in Brief, 35, 106885.*
- **Smartphone data (`S-*.csv`)**: 3-axis accelerometer, 3-axis gyroscope, magnetometer, GPS @ 10 Hz.
- **Vehicle ECU data (`V-*.csv`)**: Wheel speed, yaw rate, GPS ground truth @ 10 Hz.
- **Partitioning**: Track-wise split. Held-out drive (e.g. `Vf` motorway / `Y1`) reserved exclusively for test drift benchmarks.
