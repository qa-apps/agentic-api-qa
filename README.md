# Agentic API QA

The workflow combines API Explorer and Adversary agents, a Playwright MCP UI
Explorer, an independent Judge, a bounded Healer, and a strict pull-request
Reviewer. A confirmed defect can produce a draft PR, but the workflow has no
merge capability and always ends with a human-review handoff.

The Reviewer runs deterministic secret, static-security, dependency, and
deletion/scope gates before using a high-reasoning LLM review. A rejected fix is
returned to the Healer for at most two incremental revisions. Only the exact
reviewed commit can receive the automated first-pass approval; human approval is
still mandatory.

Interview rehearsal for the ECI agentic QA challenge, mapped to
`alexpavsky.com` while keeping the orchestration target-agnostic.

```text
START
  -> Safety Governor
  -> Explorer LLM -> HTTP Tool -> LLM Observe/Adapt (bounded loop)
  -> Adversary LLM -> HTTP Tool -> LLM Observe/Adapt (bounded loop)
  -> UI Explorer LLM -> Playwright MCP Tool -> UI Observe/Adapt (bounded loop)
  -> Multimodal Design Evaluator
  -> Independent skeptical Judge LLM
       -> no confirmed bug -> Reporter
       -> confirmed design-only/inconclusive fix -> Human Review
       -> confirmed functional bug -> Healer -> Draft PR
          -> Security/Deletion Gates -> Reviewer LLM
             -> APPROVE -> Reviewer PR Comment -> Human Review
             -> CHANGES_REQUESTED -> Healer Revision (max 2) -> Reviewer
             -> ESCALATE/limit reached -> Reviewer PR Comment -> Human Review
  -> Deterministic JSON Reporter
  -> END
```

Explorer and Adversary are Z.AI/GLM agents: each independently selects its next
HTTP check, observes sanitized evidence, records a concise reflection, and adapts
its next choice. HTTP execution is a shared tool, not an agent. The deterministic
policy and JSON Reporter remain outside the LLM boundary. The independent Judge
uses a separately configurable, maximum-reasoning model to challenge every claimed
defect before it can reach repair. UI Explorer
uses the LLM to prioritize a fixed safe catalog, but only the MCP tool is allowed
to operate the browser.

## Browser MCP and visual QA

The graph launches the official `@playwright/mcp@0.0.81` server over stdio. UI
Explorer chooses one test at a time from 30 production-safe candidates covering
desktop rendering, navigation, themes, breakpoints, login/register, chat consent
and terms, forms, content sections, and keyboard focus. After each result, it may
choose another case or finish. Every executed case records:

- the exact MCP tools called;
- an accessibility snapshot;
- console errors;
- a viewport screenshot;
- a deterministic pass/fail reason;
- a WebM video retained only when the case fails.

Screenshots are mandatory evidence for both PASS and FAIL. The MCP adapter attempts
snapshot, console, screenshot, and video finalization even after an interaction
fails. `QA_UI_VIDEO=retain-on-failure` is the default; `off` and `on` are also
supported. The CLI writes both JSON and a visual HTML report whose UI cards embed
every screenshot and link retained failure videos.

The default soft limit is 20 executed cases per agent. Past 20, an agent can
continue only when a HIGH/CRITICAL failure already exists and the next decision
explicitly cites that case as its investigation reason. The hard limit is always
30. Unselected candidates are recorded as skipped, not silently lost.

The separate `design_evaluator` sends up to six captured UI states to the
multimodal judge and scores visual hierarchy, readability, consistency,
responsive layout, accessibility cues, and interaction clarity from 1 to 5.
Screenshots, retained failure videos, and MCP logs are stored under
`artifacts/ui/<run-id>/`; report-ready video copies are written under
`reports/videos/<run-id>/`. The final JSON report includes both smoke results and
the design evaluation.

## QA knowledge pack

`knowledge/target_contract.json` describes the known API/UI capabilities,
request/response shape hints, test-data rules, and safety constraints for
`alexpavsky.com`. `knowledge/examples/` contains positive, adversarial,
false-positive, and future UI examples.

Only five rotating, relevant examples are placed in an agent prompt. They are
few-shot references, not a fixed test plan. The agent still chooses its own next
action from current evidence. Response evidence records status, content type,
JSON parseability, top-level keys, truncation, secret markers, and stack-trace
markers before the LLM assigns a semantic verdict.

## Safety

- Same-origin HTTP requests only.
- Production is read-only by default.
- Only allowlisted nonpersistent chat calls are enabled in production.
- Persistent mutations require a scoped, unexpired approval.
- Every policy decision and tool call is audited.
- Target responses are untrusted data and cannot instruct the agents.
- API keys never enter `QAState`, reports, or LangSmith tool payloads.
- Explorer and Adversary have separate iteration and tool-call limits.
- The Judge must falsify each claim and may return CONFIRMED, REJECTED, or INCONCLUSIVE.
- Only a functional CONFIRMED finding with direct evidence and confidence >= 0.85 can reach Healer.
- Design opinions are reported for human review and are never auto-patched.
- Healer edits only a new worktree created from `origin/main`; it never touches the user's dirty checkout.
- The only publication tool creates a **draft PR**. No merge command or merge API exists in the project.
- Slack `#human-review` receives investigation, fix, validation, PR, and before/after screenshots.

## Judge, Healer, and human-review gate

The Judge is intentionally not a pass-through. It tests alternative explanations,
checks expected versus actual behavior, validates evidence IDs/screenshots, and
separates product bugs from test or environment failures. A Judge failure is
fail-closed: Healer is not allowed to run.

For one highest-severity eligible finding per run, Healer creates an isolated
`agent-fix/<run-id>-<finding-id>` branch, asks its LLM for a minimal unified diff,
blocks protected paths and oversized patches, runs fixed allowlisted validation
profiles, captures post-fix UI evidence through Playwright MCP when applicable,
then opens a draft PR. Multiple unrelated findings are not bundled into one patch;
the remaining findings stay visible in the report for later runs.

The final HTML/JSON report always records the Judge's skeptical challenge,
confidence, investigation reason/result, proposed fix, validation output, PR URL,
and available before/after screenshots. Slack notification is best-effort; if its
token is missing, the report records that condition without losing the review item.

## Install and test

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,dev]'
pytest
npx --yes @playwright/mcp@0.0.81 --help
```

## See the graph in LangSmith Studio

```bash
source .venv/bin/activate
langgraph dev --no-browser
```

The local Agent Server normally listens on `http://127.0.0.1:2024`. Connect
the open LangSmith Studio to that address and select `qa_agent`.

## Inspect agent communication in Langfuse

LangSmith remains the live graph debugger. Langfuse is an additional
observability backend: every workflow `run_id` becomes one deterministic trace
and every graph handoff is a typed observation. Agent decisions use `agent`, HTTP
and Playwright MCP executions use `tool`, the Judge and design review use
`evaluator`, and every real Z.AI request uses `generation` with model, latency,
and token usage. Hidden provider reasoning, API keys, cookies, and authorization
headers are never recorded.

Configure the `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and regional
`LANGFUSE_BASE_URL` values from `.env.example`. The CLI flushes observations at
shutdown and prints a direct trace link. It also publishes trace scores for the
overall verdict, UI pass rate, screenshot coverage, guardrail hold rate, and
design quality. Runs are grouped into Langfuse Sessions by `run_id` and tagged by
target, actor, and environment.

## Run the live rehearsal

```bash
agentic-api-qa
```

JSON and visual HTML reports are written under `reports/`. A non-PASS Judge result
produces a non-zero process exit code.

Open `reports/index.html` for the latest Playwright-style visual report. It is
updated by the final Reporter node after every completed CLI or LangSmith Studio
run and contains the executed test cases, verdicts, tool evidence, console output,
design evaluation, embedded screenshots, and retained failure videos. Screenshots
remain self-contained; share the adjacent `reports/videos/` directory when failure
videos are present. To open it from the terminal, run `agentic-api-qa-report`.
This starts a localhost-only report server and opens `http://127.0.0.1:9324/`,
matching the Playwright HTML reporter
experience.

Required local settings are documented in `.env.example`. Put `ZAI_API_KEY`
only in the gitignored `.env`; the default agent model is `glm-5.3-flash` with
high reasoning effort.

## DeepEval agent evaluations

The separate `evals/` suite runs real Explorer, Adversary, and UI Explorer decisions from frozen `QAState` fixtures. It does not test prompt routing: each case supplies completed checks, evidence, findings, previous actions, and remaining budget, then scores the agent's next decision.

Install this layer explicitly; the normal test extra does not install DeepEval:

```bash
python -m pip install -e '.[test,evals]'
```

The fast PR command remains unpaid and collects only `tests/`:

```bash
pytest
```

Run decision contracts without the LLM judge, or run the full nightly-equivalent suite:

```bash
python -m evals.run_evals --suite decisions --no-deepeval --no-slack
python -m evals.run_evals --suite all --mutations 3 --seed 1337
```

DeepEval uses `QA_EVAL_JUDGE_MODEL=glm-5.3` through Z.AI by default, while the agents use `glm-5.3-flash`; the harness rejects self-judging unless explicitly overridden. Deterministic checks own tool selection, parameters, same-origin policy, mutations, budgets, repeats, finish behavior, and evidence presence. DeepEval G-Eval owns next-action reasonableness, role adherence, priority, finding quality, evidence grounding, severity, and final-report completeness. Promptfoo is intentionally not included.

`evals/datasets/seeded_bugs.jsonl` is the closed manifest for eight defect profiles plus one clean control. Each profile is applied to a temporary localhost copy loaded from `QA_EVAL_SITE_SOURCE`; the manifest is never sent to the agents. The suite reports bug recall, false-positive rate, actions-to-detect, tool calls, policy violations, evidence grounding, severity accuracy, DeepEval score, tokens, estimated cost when price variables are set, and latency.

Reports are written to `reports/evals/`. Each result and score is attached to its Langfuse trace. A failure posts to `#alexpavsky-daily-audit` (`C0BPQTUS2DR`) when `SLACK_BOT_TOKEN` is configured. `.github/workflows/nightly-evals.yml` runs the 18 fixed scenarios against an isolated site copy and a local Postgres/Qdrant/RAG stack, then adds three reproducible non-blocking random mutation profiles. It never targets production with seeded bugs.

## GitHub automation ownership

`.github/workflows/ci.yml` runs only unpaid deterministic pytest checks on every PR and push to `main`. `.github/workflows/nightly.yml` runs the read-only Explorer + Adversary scan against production every night. `.github/workflows/nightly-evals.yml` owns scheduled real-agent evaluation against the local stack. `.github/workflows/production-smoke.yml` checks only the homepage and health endpoint after a successful production deployment. Seeded bugs and paid DeepEval judging never target production.

`.github/workflows/agentic-qa-healer.yml` remains manual-only. When explicitly dispatched, it preserves reports, screenshots, and retained failure videos as artifacts and can push only a Healer branch plus a draft PR to `qa-apps/alexpavsky`; it never receives merge permission.

Configure these repository secrets in `qa-apps/agentic-api-qa` before enabling automation: `ZAI_API_KEY`, `LANGSMITH_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `SLACK_BOT_TOKEN`,
`HUMAN_REVIEW_CHANNEL_ID`, and a scoped `HEALER_GITHUB_TOKEN` with access to open
branches/PRs in `qa-apps/alexpavsky`. GitHub does not expose existing secret values,
so they must be copied through repository settings rather than read by this code.

## Interview switch

For the interviewer-provided local P2P API, replace the target profile and use:

```dotenv
API_BASE_URL=http://localhost:8000
QA_ENVIRONMENT=local
QA_PRODUCTION_READ_ONLY=false
QA_REQUIRE_MUTATION_APPROVAL=false
```
