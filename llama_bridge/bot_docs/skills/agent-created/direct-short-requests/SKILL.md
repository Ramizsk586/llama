---
name: direct-short-requests
description: Agent-created procedural memory for direct request handling patterns observed 16 times.
version: 1.0.0
metadata:
  llama:
    category: agent-created
    signals: 16
---

# Direct Request Handling

## When to Use

Use this when a Telegram request matches the recurring direct request handling workflow.

## Procedure

1. Infer conservative defaults from local context.
2. Avoid unnecessary confirmation.
3. Keep responses concise and implementation-first.

## Verification

- Confirm the requested artifact, answer, or autonomous update was actually produced.
- Mention any blocked tool, missing dependency, or safety confirmation needed.
