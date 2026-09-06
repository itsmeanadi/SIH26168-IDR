---
name: token-optimizer
description: Runbook for minimizing token consumption, optimizing prompt budgets, and keeping Antigravity fast and responsive over long coding conversations.
---

# Token Optimizer Skill

## Strategy

### 1. The 80/20 Rule for Tool Calls
- **Grep vs List**: Use `grep_search` with specific identifiers instead of listing large folder structures.
- **Slice vs Dump**: Never view an entire 500+ line file if only 20 lines around a function are relevant. Specify `StartLine` and `EndLine`.
- **Targeted Diff**: Use `replace_file_content` targeting small, exact character blocks. Never overwrite an entire 300-line file when editing 5 lines.

### 2. Context Pruning
- Do not repeat file contents or lengthy schemas in assistant chat responses.
- Point to files via clickable links (`[filename.py](file:///path/to/filename.py#L20-L40)`).

### 3. Progressive Loading
- Leverage progressive disclosure: Skills and rules are only expanded into context when actively triggered.
