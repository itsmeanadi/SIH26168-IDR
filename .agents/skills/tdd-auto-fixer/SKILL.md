---
name: tdd-auto-fixer
description: Autonomous Test-Driven Development runbook that writes unit tests, executes test suites, diagnoses failures, and iterates autonomously until all tests pass.
---

# TDD Auto-Fixer Skill

## Workflow
1. **Identify Unit Tests**: When modifying an algorithm, filter, model, or API endpoint, identify or create the corresponding test in `tests/`.
2. **Execute Pytest / Jest**: Run `pytest` or `npm test` synchronously or with timeout.
3. **Traceback Analysis**:
   - Inspect the exact exception (`AssertionError`, `NameError`, `KeyError`, `AttributeError`, `ShapeMismatch`).
   - Locate the exact file and line number.
4. **Iterative Repair**:
   - Fix logic, typing, or missing imports.
   - Re-run `pytest`.
   - Repeat until 100% of test cases pass cleanly.
5. **No Regressions**: Ensure existing baseline unit tests continue to pass.
