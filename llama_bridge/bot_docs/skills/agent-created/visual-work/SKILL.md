---
name: visual-work
description: Agent-created procedural memory for visual workflow patterns observed 4 times.
version: 1.0.0
metadata:
  llama:
    category: agent-created
    signals: 4
---

# Visual Workflow

## When to Use

Use this when a Telegram request matches the recurring visual workflow workflow.

## Procedure

1. Prefer image candidates with provenance.
2. Download safe HTTP image assets when possible.
3. Send local files to Telegram when upload succeeds.
4. Fallback to remote image URLs only when download is unavailable.

## Verification

- Confirm the requested artifact, answer, or autonomous update was actually produced.
- Mention any blocked tool, missing dependency, or safety confirmation needed.
