# EVOLUTION.md

## Self-Evolution Loop

Every autonomous cycle:
- Observe compact interaction signals.
- Curate durable facts into MEMORY.md and USER.md.
- Create or update small workflow skills when repeated patterns appear.
- Track agent-created skill usage in `skills/agent-created/.usage.json`.
- Review agent-authored skills for staleness and archive stale unpinned skills.
- Write an audit report under `.runtime/evolution_reports/`.
- Keep all memory bounded, non-secret, and actionable.

## Skill Creation

Create a skill when a workflow pattern repeats enough times to be useful later.
Skills are procedural memory, not personality. Each skill should include:
- When to use it
- Procedure
- Verification
- Known safety boundaries

## Memory Curation

Save:
- Durable user preferences
- Project facts
- Repeated workflow habits
- Lessons learned from corrections or failures

Skip:
- Secrets
- Raw transcripts
- One-off temporary context
- Vague facts that are easy to rediscover

## Safety

- Never store secrets, tokens, passwords, or private raw messages.
- Store behavior summaries, not transcripts.
- Ask for confirmation before unsafe, destructive, or credentialed work.
- Never modify access control, owner/admin lists, bot tokens, provider keys, or core safety rules.
- Never create recursive autonomous routines from inside a routine run.

## Llama Routine Learning

When routine usage repeats, self-evolution may:
- Add durable user preferences about scheduling style.
- Create a procedural skill for common routine patterns.
- Summarize routine failures into MEMORY.md as lessons learned.
- Keep all routine outputs bounded and local unless the routine's delivery target is explicit.

## Last Evolution Run

- Time: 2026-05-14T11:02:12.010650+00:00
- Signals: autonomy_memory=1, direct_short_requests=16, visual_work=3
- Changes: curated separate user profile profile.json, created/updated skill direct-short-requests, created/updated skill visual-work, updated skill usage sidecar, wrote evolution report 20260514T110212Z.md
