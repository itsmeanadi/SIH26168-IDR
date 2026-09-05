"""Non-Holonomic Constraints (NHC) for ground vehicle dead reckoning.

Ground vehicles satisfy two kinematic constraints in the body frame:
1. No lateral slip: v_lateral ≈ 0 (vehicle doesn't slide sideways)
2. No vertical velocity: v_vertical ≈ 0 (vehicle stays on ground plane)

These are formulated as EKF pseudo-measurement updates that provide
heading observability through the cross-covariance d(v_lat)/d(ψ).
"""

import numpy as np
from .ekf import ExtendedKalmanFilter


def apply_nhc_update(
    ekf: ExtendedKalmanFilter,
    sigma_lat: float = 0.05,
    sigma_vert: float = 0.05,
):
    """Apply non-holonomic constraint pseudo-measurements.

    The lateral velocity in body frame is:
        v_lat = -sin(ψ) * vE + cos(ψ) * vN

    Its Jacobian with respect to heading ψ is:
        d(v_lat)/d(ψ) = -cos(ψ) * vE - sin(ψ) * vN

    This term is CRITICAL: it couples the NHC innovation to the heading
    state, providing heading observability during GNSS denial. Without it,
    heading error accumulates unchecked and dominates position drift.
    """
    psi = ekf.x[6]
    ve = ekf.x[3]
    vn = ekf.x[4]
    vu = ekf.x[5]

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Predicted body-frame velocities
    v_lat_pred = -sin_psi * ve + cos_psi * vn
    v_vert_pred = vu

    # Innovation: measurement (0) - prediction
    y = np.array([-v_lat_pred, -v_vert_pred])

    # Measurement Jacobian H (2 x dim_x)
    H = np.zeros((2, ekf.dim_x), dtype=np.float64)

    # d(v_lat)/d(vE) = -sin(ψ)
    H[0, 3] = -sin_psi
    # d(v_lat)/d(vN) = cos(ψ)
    H[0, 4] = cos_psi
    # d(v_lat)/d(ψ) = -cos(ψ)*vE - sin(ψ)*vN  ← heading observability term
    H[0, 6] = -cos_psi * ve - sin_psi * vn

    # d(v_vert)/d(vU) = 1
    H[1, 5] = 1.0

    R = np.diag([sigma_lat**2, sigma_vert**2])

    S = H @ ekf.P @ H.T + R
    K = ekf.P @ H.T @ np.linalg.inv(S)

    # Full Kalman update — no subspace restriction.
    # The heading Jacobian term in H[0,6] ensures the Kalman gain
    # correctly distributes the NHC innovation across velocity AND heading.
    ekf.x = ekf.x + K @ y
    ekf.x[6] = (ekf.x[6] + np.pi) % (2 * np.pi) - np.pi  # normalize heading

    I = np.eye(ekf.dim_x)
    ekf.P = (I - K @ H) @ ekf.P @ (I - K @ H).T + K @ R @ K.T
