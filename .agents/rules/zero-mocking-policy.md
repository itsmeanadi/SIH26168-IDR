---
trigger: always_on
description: Forbids placeholder functions, synthetic mocks posing as real data, and stubbed logic.
---

# Zero Mocking & Physical Fidelity Policy

## 1. No Fake Placeholders
- Never write placeholder functions that return hardcoded numbers (e.g. `return 0.95` or `Math.random()`) and represent them as working sensor models or real EKF state.
- Implement genuine mathematics, physics equations, or real model forward passes.

## 2. Respect Real Dataset Formats
- Always parse real CSV / telemetry structures (e.g., IO-VNBD datasets, NMEA GNSS sentences, IMU sensor frames).
- When generating simulation data for testing, ensure it follows physically consistent kinematics (acceleration $\to$ velocity $\to$ position $\to$ turn radius).
