---
trigger: always_on
description: Mandatory rule to automatically accept all execution steps and emit 3 distinct beeps on full task completion.
---

# Autonomous Execution & Task Completion Notification Rule

## 1. Full Auto-Approval & Uninterrupted Execution
- Every task, tool call, command execution, and file modification is automatically approved and executed without manual interruptions or trivial confirmation modals.
- Execute complete end-to-end workflows (investigate -> modify -> test -> fix -> verify) autonomously in a single continuous stream.

## 2. Mandatory 3-Beep Completion Notification
- **WHEN**: ONLY upon the full, final completion of the entire user task / assignment.
- **NEVER**: Do NOT play any beeps or chimes in-between steps, during iterations, or on partial progress.
- **ACTION**: At the very last step before presenting the final response to the user, the agent MUST trigger the 3-beep notification:
  ```powershell
  powershell -Command "[Console]::Beep(1000, 250); Start-Sleep -Milliseconds 100; [Console]::Beep(1000, 250); Start-Sleep -Milliseconds 100; [Console]::Beep(1000, 250)"
  ```
