# Fully Autonomous Execution & Triple Completion Notification Rule

## 1. Operating Paradigm
- **Full Autonomy**: When a task prompt is received, execute completely end-to-end without pausing.
- **Zero Interruption**: Do NOT halt execution for implementation plans, permission prompts, or intermediate confirmation.
- **Auto-Execution**: Directly inspect, edit, test, run benchmarks, and save artifacts autonomously.

## 2. Audio Notification Rule
- **NO Intermediate Sounds**: Never trigger beeps or audio signals during intermediate steps, subagents, or tool executions.
- **Triple Completion Beep (3x)**: ONLY upon complete task finish, play exactly 3 distinct audible beeps:
  ```powershell
  powershell -Command "1..3 | ForEach-Object { [System.Console]::Beep(1000, 200); Start-Sleep -Milliseconds 150 }"
  ```
- **Direct Deliverable**: Return the final structured report immediately with no extraneous questions.
