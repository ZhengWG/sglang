# vllm-omni CI watcher

This folder contains tooling to monitor the upstream `vllm-project/vllm-omni` CI
without needing a long-lived process.

## Files

- `automation-prompt.md` — the prompt body and configuration to paste into a
  **Cursor Automation** (Scheduled trigger). This is the recommended way to get
  persistent CI monitoring; each tick spins up a fresh Cloud Agent that does
  one polling iteration and exits.
- `poll.sh` — a self-contained `bash` polling loop, useful only when you have a
  long-lived host (e.g. a Cursor "My Machines" worker, a personal devbox, or a
  laptop you keep open). On a normal Cursor-hosted Cloud Agent VM the process
  dies the moment the VM is reclaimed.

## How alerts are delivered

The Automation maintains a single tracking GitHub Issue in this repository
(label `ci-watch:vllm-omni`, title `[ci-watch] vllm-project/vllm-omni`) that
holds the last-known CI state in its body. On every tick the agent diffs the
current upstream state against the stored state and, only if something
changed, posts a comment summarising the change. When everything is steady,
no comment is written and no notification fires.

To switch to Slack/email/webhook later, replace Step 4 of the prompt with the
desired side-effect (a single `curl` to a webhook is enough).

## Resetting state

Close the tracking issue. The next Automation run will create a fresh one and
re-baseline.
