# Workspace Rules: Antigravity Automation & Notification

## 1. Full Autonomous Execution & Auto-Approval
- All tool calls, file modifications, code replacements, and non-destructive terminal commands are pre-approved and executed autonomously.
- Do not stop or prompt the user for permission on intermediate steps.

## 2. Final Task Completion Notification (3 Beeps)
- When a whole task is 100% complete and verified, execute exactly 3 beeps:
  ```powershell
  powershell -Command "[Console]::Beep(1000, 250); Start-Sleep -Milliseconds 100; [Console]::Beep(1000, 250); Start-Sleep -Milliseconds 100; [Console]::Beep(1000, 250)"
  ```
- No sounds in-between tasks or steps. Only upon full completion of the user's entire request.
