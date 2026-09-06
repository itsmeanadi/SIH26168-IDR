---
trigger: always_on
description: Strict guidelines to maximize token efficiency, conserve context window, and accelerate turn latency.
---

# Token Optimization & Context Efficiency Rule

## 1. Targeted Code Inspection
- Use `grep_search` with specific query patterns or file extension filters rather than reading entire directory trees.
- When viewing files (`view_file`), always specify `StartLine` and `EndLine` slices rather than dumping hundreds of irrelevant lines.
- Never view binary, cache, or auto-generated files unless strictly necessary.

## 2. Surgical Code Modifications
- Use `replace_file_content` targeting small, exact character blocks instead of rewriting entire files with `write_to_file`.
- Avoid full file rewrites: modifying 10 lines in a 500-line file via targeted replacement consumes <5% of the tokens compared to an overwrite.

## 3. Compact Communication
- Be direct and concise.
- Avoid repeating entire code blocks or verbatim summaries in chat responses when artifacts or files already contain the code.
- Provide clickable markdown file links ([`path/to/file`](file:///path/to/file#L10-L20)).
