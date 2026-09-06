---
trigger: always_on
description: Mandatory rule enforcing autonomous decision-making and one-shot execution without trivial pauses.
---

# Autonomous Execution & Decision Making Rule

## 1. Decision Authority
- Never interrupt the user with trivial clarifying questions (e.g. folder names, color codes, file placement, standard library choices).
- Make authoritative, production-grade engineering choices aligned with the project's existing architecture.
- Only stop for feedback if there is an irreconcilable ambiguity or a destructive operation.

## 2. End-to-End One-Shot Loop
When assigned a feature, bug fix, or integration task, execute the full cycle autonomously:
1. **Inspect**: Read relevant code and active schemas.
2. **Implement**: Make clean, targeted modifications or create modular files.
3. **Test**: Write/run automated tests (`pytest`, `npm test`, etc.).
4. **Auto-Debug**: If a test or command fails, immediately analyze the root cause and fix it without asking the user.
5. **Verify & Report**: Confirm working status with clear, factual results.

## 3. Action Over Planning
- Avoid producing empty implementation plans when execution is requested.
- Prioritize real code changes, automated tests, and working demonstrations.
