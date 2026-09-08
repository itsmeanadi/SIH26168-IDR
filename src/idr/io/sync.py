"""Temporal synchronization and offline interpolation module for IO-VNBD drive pairs.

Aligns smartphone IMU recordings with vehicle ECU CAN/OBD speed streams using
deterministic timestamp parsing, offline cross-correlation offset optimization, and
offline timestamp-based interpolation onto smartphone timestamps within the valid temporal overlap.
"""

from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_phone_date_to_seconds(date_series: pd.Series) -> np.ndarray:
    """Parse 'DATE (YYYY-MO-DD HH-MI-SS_SSS)' to absolute seconds of day.
    
    Example input: '2019-08-30 18:12:45:703' -> 65565.703 seconds
    """
    times = date_series.astype(str).str.split(' ').str[1]
    parts = times.str.replace('_', ':').str.replace('-', ':').str.split(':')
    h = parts.str[0].astype(float)
    m = parts.str[1].astype(float)
    s = parts.str[2].astype(float)
    ms = parts.str[3].astype(float)
    return (h * 3600.0 + m * 60.0 + s + ms / 1000.0).to_numpy(dtype=np.float64)


def estimate_temporal_offset(
    t_phone: np.ndarray,
    t_vehicle: np.ndarray,
    phone_speed: np.ndarray,
    vehicle_speed: np.ndarray,
    search_window_sec: float = 60.0,
    search_step_sec: float = 0.1,
) -> Tuple[float, float, float]:
    """Estimate temporal offset tau such that t_vehicle_query = t_phone - tau.
    
    Returns:
        best_tau: total time offset (seconds)
        clock_diff: baseline nominal clock difference t_phone[0] - t_vehicle[0]
        max_corr: Pearson correlation achieved at best_tau
    """
    if len(t_phone) == 0 or len(t_vehicle) == 0:
        return 0.0, 0.0, 0.0

    tau_0 = float(t_phone[0] - t_vehicle[0])
    
    # If speed signals are not valid for cross-correlation, return nominal clock difference
    valid_p = ~np.isnan(phone_speed)
    valid_v = ~np.isnan(vehicle_speed)
    if np.sum(valid_p) < 50 or np.sum(valid_v) < 50 or np.std(phone_speed[valid_p]) < 1e-3:
        return tau_0, tau_0, 1.0

    # Grid search tau around tau_0
    best_c = -1.0
    best_tau = tau_0
    
    # Evaluate raw correlation at tau_0
    t_v_q0 = t_phone - tau_0
    m0 = (t_v_q0 >= t_vehicle[0]) & (t_v_q0 <= t_vehicle[-1]) & valid_p
    if np.sum(m0) > 50:
        v_interp0 = np.interp(t_v_q0[m0], t_vehicle, vehicle_speed)
        std_v0 = float(np.std(v_interp0))
        if std_v0 > 1e-4:
            c0 = float(np.corrcoef(phone_speed[m0], v_interp0)[0, 1])
            if not np.isnan(c0):
                best_c = c0

    # Coarse/fine search window
    min_tau = tau_0 - search_window_sec
    max_tau = tau_0 + search_window_sec
    
    for tau in np.arange(min_tau, max_tau + search_step_sec, search_step_sec):
        t_v_q = t_phone - tau
        m = (t_v_q >= t_vehicle[0]) & (t_v_q <= t_vehicle[-1]) & valid_p
        if np.sum(m) > 50:
            v_interp = np.interp(t_v_q[m], t_vehicle, vehicle_speed)
            std_v = float(np.std(v_interp))
            if std_v > 1e-4:
                c = float(np.corrcoef(phone_speed[m], v_interp)[0, 1])
                if not np.isnan(c) and c > best_c:
                    best_c = c
                    best_tau = float(tau)

    return best_tau, tau_0, best_c


def synchronize_and_interpolate_drive(
    phone_df: pd.DataFrame,
    vehicle_df: Optional[pd.DataFrame],
    phone_speed_col: str = "speed",
    vehicle_speed_col: str = "wheel_speed",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, float]]:
    """Synchronize phone DataFrame with vehicle ECU speeds via offline timestamp-based interpolation onto smartphone timestamps within the valid temporal overlap.
    
    Returns:
        aligned_phone_df: filtered phone DataFrame strictly within valid temporal overlap
        aligned_v_speed: vehicle speed array in m/s interpolated onto phone sample timestamps
        sync_meta: metadata dictionary containing offset and alignment QC metrics
    """
    N_phone = len(phone_df)
    if vehicle_df is None or len(vehicle_df) == 0:
        # No vehicle file: fallback to phone speed or zeros
        v_speed = phone_df[phone_speed_col].to_numpy(dtype=np.float32) if phone_speed_col in phone_df.columns else np.zeros(N_phone, dtype=np.float32)
        meta = {
            "total_tau_sec": 0.0,
            "nominal_clock_diff_sec": 0.0,
            "alignment_corr": 1.0,
            "valid_overlap_samples": N_phone,
            "overlap_ratio": 1.0,
        }
        return phone_df.copy(), v_speed, meta

    t_phone = phone_df["timestamp"].to_numpy(dtype=np.float64)
    t_vehicle = vehicle_df["timestamp"].to_numpy(dtype=np.float64)
    
    p_spd = phone_df[phone_speed_col].to_numpy(dtype=np.float64) if phone_speed_col in phone_df.columns else np.zeros(N_phone)
    v_spd = vehicle_df[vehicle_speed_col].to_numpy(dtype=np.float64) if vehicle_speed_col in vehicle_df.columns else np.zeros(len(vehicle_df))

    best_tau, tau_0, max_corr = estimate_temporal_offset(t_phone, t_vehicle, p_spd, v_spd)
    
    # Query timestamps on vehicle timeline
    t_v_query = t_phone - best_tau
    
    # Strictly inside vehicle recording boundaries (NO extrapolation)
    valid_overlap = (t_v_query >= t_vehicle[0]) & (t_v_query <= t_vehicle[-1])
    n_valid = int(np.sum(valid_overlap))
    
    if n_valid == 0:
        logger.warning(f"No temporal overlap found with tau={best_tau:.2f}s! Using full length fallback.")
        aligned_phone_df = phone_df.copy()
        aligned_v_speed = np.zeros(len(aligned_phone_df), dtype=np.float32)
        meta = {
            "total_tau_sec": best_tau,
            "nominal_clock_diff_sec": tau_0,
            "alignment_corr": float(max_corr),
            "valid_overlap_samples": 0,
            "overlap_ratio": 0.0,
        }
        return aligned_phone_df, aligned_v_speed, meta

    # Filter phone dataframe to valid overlap interval
    aligned_phone_df = phone_df.loc[valid_overlap].copy().reset_index(drop=True)
    t_query_valid = t_v_query[valid_overlap]
    
    # Interpolate vehicle forward speed onto valid phone samples
    aligned_v_speed = np.interp(t_query_valid, t_vehicle, v_spd).astype(np.float32)
    
    # Zero-base timestamps
    t_start = aligned_phone_df["timestamp"].iloc[0]
    aligned_phone_df["timestamp"] = aligned_phone_df["timestamp"] - t_start

    meta = {
        "total_tau_sec": float(best_tau),
        "nominal_clock_diff_sec": float(tau_0),
        "alignment_corr": float(max_corr),
        "valid_overlap_samples": n_valid,
        "overlap_ratio": float(n_valid / N_phone),
    }
    
    return aligned_phone_df, aligned_v_speed, meta
