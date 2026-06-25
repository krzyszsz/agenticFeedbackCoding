# Agentic Harness Principles

This repository builds a general-purpose AI harness. The harness must not solve
user tasks itself, hard-code benchmark answers, or steer prompts toward one
historical problem. Its job is to manage context, iteration, tools, evidence,
and verification so the configured models can solve arbitrary tasks.

## Non-Negotiables

- Keep prompts domain-neutral unless the active user request supplies the domain.
- Treat every task as a new problem requiring analysis before planning.
- Preserve alternatives, assumptions, failed attempts, evidence, and repair
  history so later model turns can make autonomous decisions.
- Let models choose repairs. The harness should provide failure evidence,
  boundaries, and prior context, not a predetermined patch.
- Verify model-requested tool calls before execution, especially destructive
  commands, path-sensitive file operations, shell quoting, and long-running work.
- Bound tool output and timeouts at the harness boundary before data reaches the
  model context.
- Compact history by preserving the initial user request, recent turns,
  current runbook state, important discovered facts, and unresolved risks.
- Update the runbook when a plan becomes stale, impossible, or superseded.
- After final verification, explicitly decide whether the chosen approach was
  appropriate or whether another approach should be attempted.

## Anti-Patterns

- Do not add prompt text that targets only one benchmark or one past failure if
  it would make unrelated tasks worse.
- Do not convert test tasks into harness-side deterministic solutions.
- Do not hide package installation, server startup, or long waits inside
  unrelated validation commands.
- Do not accept model claims without independent evidence when tools are
  available.
- Do not let a single huge tool response dominate context; truncate with a clear
  marker and preserve enough tail/head evidence to diagnose failures.
