---
name: one-shot-builder
description: Comprehensive framework for executing full-lifecycle feature engineering, bug fixes, and system integration in a single autonomous turn without pausing for trivial user decisions.
---

# One-Shot Builder Skill

## Overview
The `one-shot-builder` workflow directs Antigravity to take a high-level user request and deliver a completely implemented, tested, and verified solution in a single autonomous sweep.

## Execution Flow

### Step 1: Rapid Repository Triangulation
- Identify the exact files, data schemas, and tests involved.
- Use `grep_search` to locate imports, class names, and endpoints.
- Read only the specific line ranges needed.

### Step 2: Architecture & Dependency Resolution
- Choose the cleanest design pattern adhering to existing project standards.
- Ensure strict separation of layers:
  - Sensor Ingestion $\to$ Core Mathematical Engine $\to$ ML Inference $\to$ API Service $\to$ UI Client.

### Step 3: Surgical Implementation
- Implement the requested logic using modular classes and functions.
- Update configuration, routes, and client adapters.
- Use `replace_file_content` for surgical modifications.

### Step 4: Automated Verification Loop
- Run automated test suites (e.g. `pytest`, `npm test`).
- If an assertion fails or a `NameError` / `ImportError` occurs:
  1. Read the error traceback.
  2. Locate the failing line.
  3. Apply the fix immediately.
  4. Rerun the test suite until 100% pass rate is achieved.

### Step 5: Deliver Verified Walkthrough
- Confirm what was implemented, what was tested, and provide file links.
