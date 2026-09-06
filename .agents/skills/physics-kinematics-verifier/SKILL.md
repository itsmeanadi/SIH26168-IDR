---
name: physics-kinematics-verifier
description: Kinematic consistency and coordinate frame verification runbook (Phone Frame vs Vehicle Body Frame vs World ENU Frame, Two-Wheeler Lean angles, and Euler transformation matrices).
---

# Physics & Kinematics Verifier Skill

## Coordinate Frame Reference

### 1. Smartphone Frame (DeviceMotion API)
- $X_{phone}$: Lateral (Right)
- $Y_{phone}$: Longitudinal (Up / Screen Top)
- $Z_{phone}$: Normal (Out of screen toward user)

### 2. Vehicle Body Frame (Forward / Lateral / Vertical)
- $X_v$: Vehicle Forward (direction of forward travel)
- $Y_v$: Vehicle Lateral (Left)
- $Z_v$: Vehicle Vertical (Upward, opposite to gravity)

### 3. World Navigation Frame (Local ENU)
- $X_{world}$: East
- $Y_{world}$: North
- $Z_{world}$: Up
- Yaw Angle $\psi$: Counter-clockwise from East (radians)
- Compass Heading: $\theta = (90^\circ - \text{rad2deg}(\psi)) \pmod{360^\circ}$

## Two-Wheeler Lean Formula
$$\tan(\phi) \approx \frac{a_{centripetal}}{g} = \frac{v \cdot \omega_{yaw}}{9.80665}$$
$$\sigma_{lat}(\phi) = \sigma_{lat,0} \cdot (1 + 2\sin|\phi|)$$
Widening lateral tolerance $\sigma_{lat}$ during high-lean cornering prevents false lateral slip penalties in the EKF.
