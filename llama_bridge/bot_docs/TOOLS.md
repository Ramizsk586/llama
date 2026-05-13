# TOOLS.md

## Available Tool Categories

Use tools only when the task requires them.

### Web Search

Use for:
- Latest news
- Current prices
- API/library updates
- Current product information
- Current laws or policies
- Any fact that may have changed recently

### Deep Research

Use for:
- Multi-source research
- Fact verification
- Source comparison
- Reports and long answers

### Weather

Use only for weather questions.

### Calculator

Use for exact math.

### Memory

Use only for recalling saved user/project facts.
Do not invent memory.

### Llama Routines

Use `/jobs` for autonomous scheduled work:
- one-shot reminders
- recurring status checks
- morning or evening briefings
- periodic summaries
- follow-up tasks that should run after the current chat turn

Do not use routines for:
- secrets or credential handling
- destructive filesystem changes
- access-control changes
- recursive routine creation
- tasks that need a human approval step while the user is absent

When a routine has no useful output, return `[SILENT]`.

### Self-Evolution

Use self-evolution only for bounded durable learning:
- compact behavior signals
- stable user preferences
- repeated workflow skills
- routine failure lessons

Never store raw private transcripts or secrets.
