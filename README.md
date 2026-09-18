# Agentic API QA

[![CI](https://github.com/qa-apps/agentic-api-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/qa-apps/agentic-api-qa/actions/workflows/ci.yml)
[![Nightly scan](https://github.com/qa-apps/agentic-api-qa/actions/workflows/nightly.yml/badge.svg)](https://github.com/qa-apps/agentic-api-qa/actions/workflows/nightly.yml)

A LangGraph-orchestrated framework in which two autonomous LLM agents — an
Explorer and an Adversary — probe a live HTTP API from behind a deterministic
safety governor, while a deterministic Judge turns their evidence into a
machine-readable verdict that can gate a pipeline.

The reference target is `alexpavsky.com`. The orchestration is target-agnostic:
point it at another API with a profile and a handful of environment variables.

```text
START
  -> Safety Governor
  -> Explorer LLM -> HTTP Tool -> LLM Observe/Adapt (bounded loop)
  -> Adversary LLM -> HTTP Tool -> LLM Observe/Adapt (bounded loop)
  -> Judge
  -> Deterministic JSON Reporter
  -> END
```

Explorer and Adversary are Z.AI/GLM agents: each independently selects its next
HTTP check, observes sanitized evidence, records a concise reflection, and adapts
its next choice. HTTP execution is a shared tool, not an agent.

The deterministic policy, Judge, and JSON Reporter stay **outside** the LLM
boundary. That is the central design decision: the verdict that gates a pipeline
is never authored by the same model that produced the evidence.

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

The prompt-injection stance is the reason the evidence layer exists. A target
can return anything, including text shaped like instructions, so a response is
reduced to structural facts — status, content type, JSON parseability, top-level
keys, truncation, secret and stack-trace markers — before any model is allowed
to interpret it.

## Install and test

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,dev]'
pytest
```

## See the graph in LangSmith Studio

```bash
source .venv/bin/activate
langgraph dev --no-browser
```

The local Agent Server normally listens on `http://127.0.0.1:2024`. Connect
the open LangSmith Studio to that address and select `qa_agent`.

## Run a live scan

```bash
agentic-api-qa
```

Reports are written under `reports/`. A non-PASS Judge result produces a
non-zero process exit code, so the same command works as a quality gate.

Required local settings are documented in `.env.example`. Put `ZAI_API_KEY`
only in the gitignored `.env`; the default agent model is `glm-5.3-flash` with
high reasoning effort.

## Report format

Every run emits one JSON document validated by `FinalReport` in
`src/agentic_api_qa/models.py`: the run envelope and overall result, per-step
happy-path decisions, per-rule guardrail decisions with severity, aggregate
metrics including token usage, and the full evidence and audit trail.

`reports/sample-report.json` is an illustrative, schema-valid example showing
that shape. It is generated from the models, not captured from a live run, and
is there so the output can be read without credentials.

## Continuous scanning

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | push, pull request | Installs on Python 3.11 and runs `pytest`. No secrets needed. |
| `nightly.yml` | 03:00 America/New_York, manual | Runs a live read-only scan and uploads the JSON report as an artifact. Fails the run on a non-PASS verdict. |

The nightly scan needs a `ZAI_API_KEY` repository secret. When the secret is
absent the job reports that it was skipped rather than failing, so a fork or a
fresh clone stays green.

## Point it at a different target

Replace the target profile and override the environment. For a local API with
mutations enabled:

```dotenv
API_BASE_URL=http://localhost:8000
QA_ENVIRONMENT=local
QA_PRODUCTION_READ_ONLY=false
QA_REQUIRE_MUTATION_APPROVAL=false
```

The same overrides are available as CLI flags:

```bash
agentic-api-qa --base-url http://localhost:8000 --environment local
```
