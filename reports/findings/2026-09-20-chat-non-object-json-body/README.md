# Finding — 2026-09-20 — non-object JSON body crashes the handler (502)

The first live run of this harness against `alexpavsky.com` found a real,
reproducible input-validation defect in production. This directory keeps the
three consecutive scan reports that establish it, because the GitHub Actions
artifacts they came from expire after 30 days.

| File | Run | Verdict | What it shows |
| --- | --- | --- | --- |
| `01-initial-scan-ERROR.json` | [35543103265](https://github.com/qa-apps/agentic-api-qa/actions/runs/35543103265) | `ERROR` | The finding: 2 breached guardrails, 1 failed happy path |
| `02-after-server-fix.json` | [35543604363](https://github.com/qa-apps/agentic-api-qa/actions/runs/35543604363) | `ERROR` | Guardrails closed (0 breached), but a harness bug surfaced |
| `03-after-harness-fix-PASS.json` | [35543996889](https://github.com/qa-apps/agentic-api-qa/actions/runs/35543996889) | `PASS` | 8/8 happy path, 7/7 guardrails held, 0 pipeline errors |

## The defect

`chat_server.py` parsed every POST body with:

```python
body = json.loads(self.rfile.read(length).decode()) if length > 0 else {}
```

`json.loads` accepts **any** valid JSON, not only objects. A body of `"ping"`,
`[1,2,3]`, `42` or `null` parsed into a `str`/`list`/`int`/`None`, and the next
line called `body.get(...)`:

```
File "/var/www/alexpavsky.com/html/chat_server.py", line 3793, in do_POST
    message = _clean(body.get("message"), 12000)
AttributeError: 'str' object has no attribute 'get'
```

The exception escaped `do_POST`, the `http.server` worker thread died **without
writing a response**, and the gateway turned the dropped connection into a
**502 Bad Gateway** serving an nginx HTML error page.

All six body-parse sites were affected: chat, maintenance auth, agent-reports,
register, login, and forum posts.

## How the Adversary established it

The value of report `01` is not that a request failed — it is the reasoning
that separated a real defect from a blip:

1. `POST /api/chat` with an empty JSON body → `502` + nginx HTML.
   Recorded as a **breached** `MALFORMED_INPUT` guardrail, with the note that
   *"a 502 indicates the request never reached the validation layer"*.
2. The agent then retried a connectivity probe. That returned a controlled
   `400 {"error": "empty_message"}` — so the route and upstream were reachable,
   and the 502 was **not** a routing failure.
3. It retried once more with a minimal valid body. `502` again — and concluded
   *"the upstream chat backend is consistently failing — this is not a transient
   one-off"*.

Three tool calls, an adapted hypothesis between each, and a correct conclusion.

The same run also confirmed the SSRF guardrails **held**: the article proxy
refused both the cloud metadata address `169.254.169.254` and loopback
`127.0.0.1`, returning structured errors with `title`/`image` null rather than
proxying internal content back.

## The fix

Each of the six parse sites already sat inside a `try` whose `except` answers
`400`, so validating the parsed type at the parse site converts the crash into
a controlled 400 without touching any handler logic.

Fixed in [qa-apps/alexpavsky#8](https://github.com/qa-apps/alexpavsky/pull/8).
Applied to the live server the same day; production backup at
`/root/alexpavsky-backups/chat_server.py.pre-jsonobject-20260920-230412`.

Verified through the public edge:

| Body | Before | After |
| --- | --- | --- |
| `"ping"` (bare JSON string) | **502** | **400** |
| `[1,2,3]` (JSON array) | **502** | **400** |
| `{"message":"ping"}` | 200 | 200 |
| `{}` | 400 | 400 |
| `{nope` (malformed) | 400 | 400 |

## The harness bug this exposed

Report `02` shows the guardrails closing but the *happy path* failing — the
chat endpoint answering `400 invalid_json` to what the report displayed as a
valid object.

The endpoint was healthy. `execute_request` built the request two different
ways: the evidence excerpt came from `_serialize(json_body)`, which returns a
`str` unchanged, while the wire body came from httpx's `json=`, which encoded
that same string a **second** time. When an agent emitted `json_body` as JSON
*text*, the target received a JSON string where an object was intended — and
the evidence record showed the intent rather than the bytes, hiding it.

So the harness was itself sending the malformed bodies that exposed the server
defect. The defect was real either way; the reason it kept being hit was not.

Fixed in [qa-apps/agentic-api-qa#1](https://github.com/qa-apps/agentic-api-qa/pull/1),
with `tests/test_client.py` covering the decode rules.

## What was deliberately *not* changed

`EXPLORER_BUDGET_EXHAUSTED` forced the verdict to `ERROR` on every run.
Marking it non-blocking was attempted and correctly rejected by the existing
`test_budget_exhaustion_is_not_reported_as_pass`: an incomplete scan must not
report `PASS`, because an absence of findings may only mean the agent never
looked far enough.

The safety property stands. The agents' budgets were raised in `nightly.yml`
instead, so both can reach their own finish decision — which is what report
`03` shows, with `pipeline_errors: 0`.
