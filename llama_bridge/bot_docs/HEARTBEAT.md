# HEARTBEAT.md

## Scheduled Behaviors

- Every 30 minutes, read HEARTBEAT.md and MEMORY.md.
- Run the self-evolution loop to curate memory and agent-created skills.
- If a daily task is listed here, complete only safe Telegram/tool work autonomously and send a concise user-facing message.
- If a task is unclear, unsafe, or requires credentials, ask the user for confirmation.
- Keep quiet when no user-facing work is due.

Examples:
- Daily 06:00 | send good morning
- Daily 20:00 | send a short day-end check-in

## Llama Routines

For cron-style jobs, prefer `/jobs add <schedule> | <task>` instead of editing this file.

Examples:
- `/jobs add every 30m | check for important provider errors and tell me only if action is needed`
- `/jobs add 0 9 * * * | send a morning project briefing`
- `/jobs add 2h | remind me to review the deployment`

Routine rules:
- Deliver to the originating chat/thread by default.
- Save output locally for audit.
- Output `[SILENT]` when there is nothing useful to send.

## Future Ideas

- Daily summary
- Weekly cleanup
- Check provider health
- Rotate logs
- Summarize long memory into compact notes
