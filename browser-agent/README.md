# browser-agent

Agent side of the A2E browser environment: drives a `browser-env` host over the
`env/*` + `tools/*` capabilities and owns the REACT loop.

**Dependency rule:** this repo imports only the a2e protocol SDK. It never
imports `browser_env` — the environment is a container it reaches over HTTP
(spawned via Docker when nothing is listening). Enforced by
`tests/test_protocol_parity.py::test_agent_code_never_imports_browser_env`.

```
BrowserAgentRuntime ──(HTTP)──> browser-env host
        │                            │
        ├── REACT loop (loop.py)     ├── BrowserEnvPlugin  (env/*)
        ├── providers (stub /        └── BrowserToolPlugin (tool/*)
        │   OpenAI-compatible)
        └── browser_tools.py (schemas + tool→action map)
```

## Install / run

```bash
make venv
make test          # spawns browser-env as a subprocess; no LLM, no network

# with a live host (make -C ../browser-env run):
python -m browser_agent.cli tasks
python -m browser_agent.cli run  --task add-widget-to-cart   # fixed policy, no LLM
python -m browser_agent.cli ask  --task add-widget-to-cart "do the task"  # LLM turn
python -m browser_agent.cli tools                            # print the tool schemas
```

## How the agent drives the browser

Every `browser_*` tool is routed through **`env/step`**, so the host's step
counter, reward and `done` stay authoritative — an agent can never act outside
the episode's scoring. Two exceptions:

- `browser_text` → `env/observe` (re-reading the page does not consume a step)
- `env_*` / `browser_tasks` / `browser_score` → episode control, not actions

`browser_text` returns the indexed element list; elements carry a stable
`index` (`data-a2e-eid`) that should be preferred over a CSS selector.

The executor derives its status from the *observation*, not from exceptions —
the host reports a failed action inside `state.last_error` +
`metadata.action_ok = False`. If it didn't, the LLM would see `status: success`
for an action that did nothing.

## Tool surface

Episode control: `env_start`, `env_finish`, `env_observe`.
Catalogue/meta: `browser_tasks`, `browser_score`.

Browser actions (all consume an episode step):

`browser_navigate`, `browser_click`, `browser_fill`, `browser_press`,
`browser_scroll`, `browser_back`, `browser_forward`, `browser_reload`,
`browser_extract`, `browser_wait`.

Capture surfaces — the four things an agent needs when it cannot see the page:

| tool | what it returns |
|---|---|
| `browser_dom` | style-flattened DOM + all stylesheets + `checksum`; `mode="diff"` instead reports **what changed** vs the previous capture (added/removed elements, CSS edits, unified diff). Use it to answer "did my click do anything?" |
| `browser_console` | the page's `console.*` lines and uncaught JS errors for this episode — usually where "the click did nothing" is explained |
| `browser_screenshot` | base64 PNG |
| `browser_record` | start/stop/status a JSONL trajectory artifact on the host (auto-finalized when the episode ends; optional per-step screenshot frames) |

New console lines also arrive **automatically** with the step that produced them
(`data.console.new`), so a JS error surfaces without an extra call.

## Config

`browser_agent/runtime/runtime_config.yaml` — agent-side only (LLM, `agent_caps`,
system prompt, host URL). Host config lives in `browser-env/` and the two are
linked **in code**, never merged:

```yaml
llm:
  provider_name: stub          # offline; any other name = OpenAI-compatible
  api_credentials: "${OPENROUTER_API_KEY}"   # ${VAR} resolved at load time
agent_caps: [env, tools]
system_prompt: >- ...
max_steps: 12
host:
  base_url: "http://localhost:8770"
  docker_image: "browser-env:latest"
```

`build_runtime(config_path, orchestrate_host=True)` loads it, resolves
`${ENV_VAR}` (strict for credentials — a missing key fails at startup rather
than producing a confusing 401 later), optionally spawns the container, and
returns an unconnected runtime.

## Tests

```bash
python -m pytest -q     # 62 tests
```

| file | covers |
|---|---|
| `test_loop.py` | REACT loop, provider normalization, tool schemas + tool→action mapping (offline) |
| `test_runtime.py` | episode control, tool routing, direct tool calls, scripted-LLM turns (real host) |
| `test_capture.py` | DOM / console / screenshot / record over the wire (real host) |
| `test_protocol_parity.py` | protocol field-level parity with the host copy (read as **text**, not imported), no reverse coupling, config separation |

The host fixture (`tests/conftest.py`) spawns `browser-env` as a subprocess on a
free port and tears it down with `os.killpg` — a plain `terminate()` leaves
uvicorn's SSE reader alive and hangs the suite.

## Layout

```
browser_agent/
├── cli.py                        `browser-agent run|ask|tasks|tools`
└── runtime/
    ├── orchestrator.py           BrowserAgentRuntime: connect, episode control,
    │                             tool routing, observation shaping
    ├── loop.py                   provider-independent REACT loop
    ├── providers.py              LLMProvider, StubProvider, OpenAICompatProvider
    ├── browser_tools.py          tool schemas + tool→env-action map
    ├── runtime_protocol.py       the 3 browser-specific env/* pairs (agent copy)
    ├── bootstrap.py              config load → host ensure → runtime
    ├── docker.py                 ensure_env_host (Docker-only coupling)
    └── runtime_config.yaml
```