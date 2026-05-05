# llama

`llama` is a Python bridge CLI that exposes Anthropic-compatible endpoints on port `8089` and forwards requests to configured providers such as Ollama Local, Ollama Cloud, NVIDIA NIM, Groq, Gemini, OpenAI, Cohere, Mistral, DeepSeek, OpenRouter, or any OpenAI-compatible endpoint.

## Features

- Anthropic-compatible `/v1/messages`, `/v1/models`, `/v1/messages/count_tokens`, `/v1/messages/batches`, `/v1/files`, `/v1/skills`, `/v1/organizations/*`, and legacy `/v1/complete`
- OpenAI-compatible `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/embeddings`, and `/v1/models`
- OpenAI Assistants, Threads, Files, Fine-tuning, Moderations, and media endpoint compatibility
- Cohere-compatible `/v1/chat` and `/v1/embed`
- Gemini-compatible `generateContent` and `streamGenerateContent`
- Ollama-compatible generation, embeddings, model-management, version, and blob endpoints
- Streaming and non-streaming responses
- Configurable provider registry in `env.yml`
- Built-in provider types for `ollama_local`, `ollama_cloud`, `lm_studio`, `nvidia_nim`, `groq`, `gemini`, `openai`, `cohere`, `mistral`, `deepseek`, `openrouter`, and generic `openai_compatible` servers
- Anthropic model aliases like `haiku`, `sonnet`, and `opus`
- Claude model-name fallback, so requests for Claude Haiku/Sonnet/Opus IDs can route to your configured aliases
- Tool-call translation for Claude Code workflows
- Background process management with `serve`, `stop`, `logs`, and `status`

## Install

```powershell
pip install -e .
```

## Windows setup.exe

Build the installer bootstrapper:

```powershell
.\scripts\build_setup_exe.ps1
```

This creates `dist\llama setup.exe`. When a user runs it, setup checks for Git
and Python 3.11+, installs them with `winget` when possible, clones this repo,
downloads the Python dependencies, builds a packaged `llama.exe`, copies only
the packaged runtime files to `%LOCALAPPDATA%\Programs\llama`, adds that folder
to the user `PATH`, sets `LLAMA_HOME`, and deletes the temporary source clone.

## Initialize config

```powershell
llama init
llama configure
```

The first command that needs config creates `config.example.yml` if no config exists.
Edit that file with your API keys and models, then rename it to `env.yml`.
When an `env.yml` exists in the current directory, llama uses that project config.
When using `llama.exe`, the default `env.yml`, `Api.json`, pid file, and log file
are kept in the same directory as the exe.
Opening `llama.exe` with no command runs setup. On Windows, setup saves that
directory as the user `LLAMA_HOME` environment variable and adds it to the user
`PATH`, so new terminals can run `llama` from anywhere.

If `env.yml` already exists, `llama init` merges in any newly added default
sections and preserves your existing providers, API keys, models, and aliases.

`llama configure` opens an interactive terminal wizard that updates `env.yml`
in place. It can set provider auth, choose default models, update aliases, and
configure the restricted Telegram bot.

To scan the device, create config, install missing Python packages, and check local
provider tools:

```powershell
llama setup
```

`llama setup` can try to install Ollama through `winget` on Windows when an active
model alias uses the local Ollama provider. Use `--no-install-system` to only check
and report system tools.

Connect Claude Code with the generated settings file:

```powershell
claude --settings .\Api.json
```

Or let llama launch Claude Code with that file:

```powershell
llama claude
```

If the llama server is not running, `llama claude` starts it automatically. Servers
started this way shut down after 3 minutes with no requests. If you already started
the server yourself, `llama claude` uses the existing server and leaves it running.
If Claude Code is not installed, `llama claude` installs Node.js/npm through the
available OS package manager (`winget`, `brew`, or `pacman`) and then installs
Claude Code with `npm install -g @anthropic-ai/claude-code`. Use
`--no-install-claude` to disable automatic Claude Code installation.

Pass Claude Code arguments after `--`:

```powershell
llama claude -- --help
```

## Run

```powershell
llama serve
llama status
llama start --forever
llama api status
llama api --limits
llama logs
llama cli --list
llama cli --support
llama configure
llama pi
llama claude
llama codex
llama copilot
llama opencode
llama openclaw
llama poolside
llama bot
llama telegram status
llama stop
```

## Telegram Agent

Teligram is the standalone Telegram AI agent runtime. It reads `env.yml` through
the same bridge config loader, uses the configured provider/model, and composes
its system prompt from workspace Markdown files in `llama_bridge/bot_docs`, such as
`SOUL.md`, `AGENTS.md`, `IDENTITY.md`, `USER.md`, `TOOLS.md`, `MEMORY.md`, and
`HEARTBEAT.md`.

Setup checklist:

```powershell
llama bot setup
```

1. Create a bot with BotFather.
2. Copy the token into `TELEGRAM_BOT_TOKEN` or `telegram.bot_token`.
3. Set `telegram.enabled: true` in `env.yml`.
4. Choose `telegram.provider` and `telegram.model`.
5. Set `telegram.allowed_chat_ids` if you want to restrict access.
6. Customize the workspace Markdown files in `llama_bridge/bot_docs`.
7. Run the bot:

```powershell
llama bot run
```

Useful checks:

```powershell
llama telegram status
python -m llama_bridge.teligram --config .\env.yml --workspace .\llama_bridge\bot_docs
```

Example `env.yml` section:

```yaml
telegram:
  enabled: true
  bot_token: ${TELEGRAM_BOT_TOKEN}
  allowed_chat_ids: []
  provider: ollama_cloud
  model: qwen3.5:cloud
  system_prompt: "You are Teligram, a safe Telegram AI agent powered by Llama Bridge."
  max_input_chars: 4000
  max_output_tokens: 700
  poll_interval_seconds: 2.0
  response_timeout_seconds: 180.0
```

After launch, send `/status` in Telegram, then send a normal test message. The
runtime does not print bot tokens, API keys, private prompts, or message content
in normal logs.

`llama api status` checks only the model aliases saved under `anthropic_models`
in `env.yml` and reports whether each provider/model API responds.

`llama api --limits` opens a terminal view of configured providers and shows
their tracked quota windows such as hourly, weekly, and monthly limits. It reads
`providers.<name>.usage_limits` and `providers.<name>.model_limits` from
`env.yml`, groups providers by period, and shows limit, used, and remaining
values. `llama api limits` works too.

`llama logs` follows the live log until you press `Ctrl+C`. Use
`llama logs --no-follow` to print the current log and exit.

`llama cli --list` shows which llama-managed CLI tools are currently usable and
where they are installed. `llama cli --support` shows all CLI tools supported by
llama bridge, even when they are not installed. `llama cli --rm` lists the
installed tools and lets you choose one to remove, or you can pass a name
directly such as `llama cli --rm opencode`.

`llama pi` configures and launches the Pi coding agent. If Pi is missing, llama
installs Node.js/npm through the available OS package manager (`winget`, `brew`,
or `pacman`) and then installs Pi with
`npm install -g @mariozechner/pi-coding-agent`. It also writes local Pi
extensions under `~/.pi/agent/extensions/`: `llama_bridge_web_tools` for
`web_search` and `web_fetch`, and `llama_bridge_deep_research` for Pi-only deep
research. The web tools call the llama bridge, which proxies to Ollama's current
`/api/web_search` and `/api/web_fetch` endpoints. Configure the provider and
model in `env.yml`:
If the llama server is not running, `llama pi` starts it automatically and lets
it stop after 3 minutes with no requests once Pi closes.
Pi is configured to talk to the local llama bridge, so requests appear in
`llama logs` as `/v1/chat/completions` instead of going directly to the provider.

The `llama_bridge_deep_research` extension registers a Pi-only `deep_research`
tool. It is not advertised through `/v1/tools` or `/api/tools`; only Pi gets it
from the local extension. `deep_research` runs at least 10 SerpAPI search passes,
5 Tavily search passes when Tavily is enabled, then 6 `web_search`
verification/recheck passes before it deduplicates sources, fetches top pages,
and returns a structured evidence brief that Pi can synthesize from. The
`/deep_research` command then instructs Pi to verify important claims with
separate available search tools, collect 2-3 compact image candidates with
`image_research`, and create a `report.md` file in the current directory with
citations, images, and an end-of-report reliability warning. Generated reports
are structured as prepared research briefs: title, executive summary, numbered
analytical sections, evidence gaps/limitations, synthesis conclusion, and
numbered references. Images are included only when useful and are embedded as
compact sourced figures.

```yaml
pi:
  provider: ollama_cloud
  model: qwen3.5:cloud
  api: openai-completions
  config_dir: ~/.pi/agent
  web_search: true
```

You can override the configured model for one launch:

```powershell
llama pi --model qwen3.5:cloud
```

To use NVIDIA NIM instead, set `provider: nvidia_nim` and choose a model from
that provider.

`llama claude` launches Claude Code with `Api.json` and also writes a local
Claude Code plugin under `plugins/llama_bridge_tools_claude` next to your
`env.yml`. The plugin is loaded with `--plugin-dir` and exposes the same bridge
tools through an MCP server named `llama_bridge_tools`. It also adds Claude
user slash commands such as `/serp`, `/web`, `/fetch`, `/image`, `/image_search`,
`/image-sarch`, `/deep`, `/deep_research`, and `/deep-research` that prompt
Claude to use the corresponding bridge MCP tools without the long plugin
namespace.

`llama codex` configures and launches OpenAI Codex CLI with the local llama
bridge. If Codex is missing, llama installs Node.js/npm through the available OS
package manager and then installs Codex with `npm install -g @openai/codex`.
It updates only the `llama_bridge` provider and profile in
`~/.codex/config.toml`, preserving your existing Codex settings:

```toml
[model_providers.llama_bridge]
name = "llama bridge"
base_url = "http://127.0.0.1:8089/v1"
env_key = "LLAMA_BRIDGE_API_KEY"
wire_api = "responses"

[profiles.llama_bridge]
model = "qwen3.5:cloud"
model_provider = "llama_bridge"
model_context_window = 65536
model_catalog_json = "C:\\Users\\you\\.codex\\llama_bridge_models.json"

[mcp_servers.llama_bridge_tools]
command = "llama"
args = ["mcp-tools"]
env = { LLAMA_BRIDGE_BASE_URL = "http://127.0.0.1:8089", LLAMA_BRIDGE_API_KEY = "change-me" }
startup_timeout_sec = 30
tool_timeout_sec = 300
enabled = true
```

It also writes a Codex plugin bundle under `~/.codex/plugins/llama_bridge_tools`
with a companion skill and MCP manifest for Codex plugin-aware surfaces.

Pass Codex arguments after `--`:

```powershell
llama codex -- --help
```

`llama copilot` configures and launches GitHub Copilot CLI with the local llama
bridge. If Copilot CLI is missing, llama installs Node.js/npm through the
available OS package manager and then installs Copilot CLI with
`npm install -g @github/copilot`. It also writes
`~/.copilot/mcp-config.json` so Copilot CLI sees the bridge tools as the
`llama_bridge_tools` MCP server. Configure the provider and model in `env.yml`:

```yaml
copilot_cli:
  provider: ollama_cloud
  model: qwen3.5:cloud
  wire_api: responses
  max_prompt_tokens: 65536
  max_output_tokens: 2048
  install_package: "@github/copilot"
```

`llama opencode` launches OpenCode against the local llama bridge and injects a
runtime OpenCode config through `OPENCODE_CONFIG_CONTENT`. The generated config
uses OpenCode's current `provider` schema and registers the bridge as an
OpenAI-compatible provider at `http://127.0.0.1:8089/v1`.

`llama openclaw` launches OpenClaw through Ollama's integration while writing a
separate managed OpenClaw config at `~/.openclaw/llama-openclaw.json`. That
managed config enables OpenClaw sandboxing for all sessions, uses a dedicated
workspace, disables host bind mounts, keeps Docker sandbox networking at `none`,
forces exec into the sandbox, and keeps patch writes workspace-contained. By
default the sandbox has `workspaceAccess: none`, so OpenClaw tools cannot write
to your normal drives or project files unless you explicitly opt in:

```yaml
openclaw:
  provider: ollama_cloud
  model: qwen3.5:cloud
  config_path: ~/.openclaw/llama-openclaw.json
  workspace: ~/.openclaw/llama-workspace
  workspace_access: none
  sandbox_backend: docker
  install_package: "openclaw"
```

Useful OpenClaw commands:

```powershell
llama openclaw
llama openclaw --model qwen3.5:cloud
llama openclaw status
llama openclaw configure
llama openclaw channels
llama openclaw sandbox-explain
llama openclaw stop
```

`llama poolside` installs and launches Poolside Agent CLI. By default it uses
Poolside's official installer: `curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh`
on macOS/Linux, and on Windows it tries that installer through Git Bash when Git
is installed. If the shell installer is unavailable or rejects the Windows
environment, it falls back to `irm https://downloads.poolside.ai/pool/install.ps1 | iex`. It now follows the same provider/model pattern as the other llama
CLI launchers, writes Poolside `settings.yaml` when possible, passes configured
Poolside auth through environment variables, and wires in the bridge MCP tools.
You can configure it in
`env.yml`:

```yaml
poolside:
  provider: ollama_cloud
  model: qwen3.5:cloud
  api_url: null  # Optional Poolside deployment URL; leave null for saved/default Poolside setup.
  api_key: ${POOLSIDE_API_KEY}
  token: ${POOLSIDE_TOKEN}
  config_path: ~/.config/poolside/settings.yaml
  install_command: "curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh"
  windows_install_command: "irm https://downloads.poolside.ai/pool/install.ps1 | iex"
```

When `poolside.api_key` or `poolside.token` is configured, `llama poolside`
exports it as `POOLSIDE_API_KEY` or `POOLSIDE_TOKEN` so Poolside can authenticate
without prompting. When `poolside.api_url` is configured, it also exports
`POOLSIDE_STANDALONE_BASE_URL` in addition to `POOLSIDE_API_URL`.
Leave these unset to use credentials saved by `pool setup` or `pool login`.

Use `llama poolside --provider <name> --model <model>` to override the saved
provider or model for one launch.

Pass Copilot CLI arguments after `--`:

```powershell
llama copilot -- -p "how does this repository work?"
```

## Bridge tools

The bridge can expose its own HTTP tools for clients and external LLM agents that
can call the bridge directly:

```text
GET  /v1/tools
POST /v1/tools/call
POST /v1/tools/{name}
GET  /api/tools
POST /api/tools/{name}
```

Built-in tools include `datetime_now`, `wikipedia_search`, `wikipedia_page`,
`weather_current`, `serpapi_search`, `tavily_search`, `source_research`,
`image_research`, and `verify_sources`. `source_research` combines SerpAPI and
Tavily results, then runs parallel source-verifier workers so models can cite
only reachable, relevant sources. `image_research` returns 2-3 image URLs with
source pages plus compact Markdown/CSS snippets for report generation. When
Tavily image results are available, SerpAPI image search is only used as a
fallback. Configure them in `env.yml`:

```yaml
tools:
  enabled: true
  expose_http: true
  require_auth: true
  include:
    - datetime_now
    - wikipedia_search
    - wikipedia_page
    - weather_current
    - serpapi_search
    - tavily_search
    - source_research
    - image_research
    - verify_sources
  max_exposed: 8
  serpapi:
    enabled: false
    api_key: ${SERPAPI_API_KEY}
  tavily:
    enabled: false
    api_key: ${TAVILY_API_KEY}
  weather:
    enabled: true
  wikipedia:
    enabled: true
```

When bridge tools are enabled, chat-style endpoints automatically advertise them
to upstream tool-capable models. Non-streaming OpenAI (`/v1/chat/completions`),
Responses (`/v1/responses`), Anthropic (`/v1/messages`), Ollama (`/api/chat`),
Cohere (`/v1/chat`), and Gemini (`:generateContent`) calls execute bridge-owned
tool calls on the server and send the result back to the model before returning
the final answer. External apps can also call tools directly through the HTTP
tool endpoints above.

Claude Code, Codex, and Copilot CLI get those tools through the `llama mcp-tools`
stdio MCP adapter when launched by `llama claude`, `llama codex`, or
`llama copilot`. The adapter lists the enabled bridge tools from `/api/tools`
and calls `/api/tools/{name}` with the configured local bridge API key.

## GitHub Copilot

The bridge can also stand in for an Ollama or LM Studio server for tools that
use OpenAI-compatible endpoints. Start llama first:

```powershell
llama serve
```

For VS Code Copilot extensions or other clients that offer an Ollama-compatible
base URL, use:

```text
http://127.0.0.1:8089
```

The bridge reports Ollama `0.18.3` compatibility because VS Code requires
Ollama `0.18.3+`, VS Code `1.113+`, and GitHub Copilot Chat `0.41.0+`.
In the Copilot Chat panel, keep `Local` selected after adding Ollama models.

Ollama-compatible `/api/*` endpoints intentionally accept unauthenticated local
requests because many Ollama clients, including VS Code integrations, probe
`/api/version` without sending an API key. If a client accidentally appends
Ollama paths to a `/v1` base URL, the bridge also accepts `/v1/api/*` aliases.

Choose which models VS Code Copilot sees in `env.yml`. Add at least 1 and at
most 3 entries:

```yaml
vs_copilot:
  models:
    - name: gemma4:31b
      provider: ollama_cloud
      model: gemma4:31b
      context_size: 65536
      modified_at: "2026-04-02T09:00:00-08:00"
      size: 62546177752
      digest: 221b330d11a8
    - name: local-qwen
      provider: lm_studio
      model: qwen3.5
      context_size: 65536
```

The `name` is what appears in VS Code. The `provider` must match a provider
under `providers`, and `model` is the upstream model sent to that provider.
`modified_at`, `size`, and `digest` are optional metadata used to make
`/api/tags` look like Ollama's model list.

For clients that ask for an OpenAI-compatible or LM Studio-style base URL, use:

```text
http://127.0.0.1:8089/v1
```

Use any model id returned by `http://127.0.0.1:8089/api/tags` or
`http://127.0.0.1:8089/v1/models`. The API key is the `server.auth_token` value
from `env.yml`; the default example uses `change-me`.

Implemented Ollama-compatible endpoints:

```text
POST   /api/generate
POST   /api/chat
POST   /api/embed
POST   /api/embeddings
GET    /api/tags
GET    /api/ps
POST   /api/show
POST   /api/create
POST   /api/pull
POST   /api/push
POST   /api/copy
DELETE /api/delete
GET    /api/version
HEAD   /api/blobs/{digest}
POST   /api/blobs/{digest}
```

Generation and embeddings are translated to the configured upstream provider.
Model-management and blob routes are compatibility shims for clients that probe
or launch through Ollama; they do not download model weights or edit `env.yml`.

Implemented Anthropic-compatible endpoints:

```text
POST   /v1/messages
POST   /v1/messages/count_tokens
POST   /v1/messages/batches
GET    /v1/messages/batches
GET    /v1/messages/batches/{id}
POST   /v1/messages/batches/{id}/cancel
GET    /v1/messages/batches/{id}/results
GET    /v1/models
POST   /v1/files
GET    /v1/files
GET    /v1/files/{id}
DELETE /v1/files/{id}
GET    /v1/files/{id}/content
POST   /v1/skills
GET    /v1/skills
GET    /v1/skills/{id}
POST   /v1/skills/{id}
DELETE /v1/skills/{id}
GET    /v1/organizations/{path}
POST   /v1/organizations/{path}
DELETE /v1/organizations/{path}
POST   /v1/complete
```

Messages, token counting, model listing, and legacy completions are translated to
the configured provider. Batches, Files, Skills, and Admin API routes are
compatibility shims backed by the current bridge process; they are meant for
client compatibility and local testing, not persistent Anthropic account
administration.

Additional provider-compatible endpoints:

```text
POST   /v1/moderations
GET    /v1/assistants
POST   /v1/assistants
GET    /v1/assistants/{id}
POST   /v1/assistants/{id}
DELETE /v1/assistants/{id}
GET    /v1/threads
POST   /v1/threads
GET    /v1/threads/{id}
POST   /v1/threads/{id}
DELETE /v1/threads/{id}
GET    /v1/threads/{id}/messages
POST   /v1/threads/{id}/messages
GET    /v1/threads/{id}/messages/{message_id}
GET    /v1/threads/{id}/runs
POST   /v1/threads/{id}/runs
GET    /v1/threads/{id}/runs/{run_id}
POST   /v1/threads/{id}/runs/{run_id}
POST   /v1/threads/{id}/runs/{run_id}/cancel
POST   /v1/threads/{id}/runs/{run_id}/submit_tool_outputs
GET    /v1/threads/{id}/runs/{run_id}/steps
GET    /v1/fine_tuning/jobs
POST   /v1/fine_tuning/jobs
GET    /v1/fine_tuning/jobs/{id}
POST   /v1/fine_tuning/jobs/{id}/cancel
GET    /v1/fine_tuning/jobs/{id}/events
POST   /v1/images/generations
POST   /v1/audio/transcriptions
POST   /v1/audio/speech
POST   /v1/chat
POST   /v1/embed
POST   /v1beta/models/{model}:generateContent
POST   /v1beta/models/{model}:streamGenerateContent
POST   /v1/models/{model}:generateContent
POST   /v1/models/{model}:streamGenerateContent
```

Cohere chat/embed and Gemini content generation translate to configured upstream
providers. Assistants, Threads, Fine-tuning, Moderations, and media routes are
local compatibility surfaces; image and audio generation return clear
`not_supported_error` responses instead of silently pretending to create media.

GitHub Copilot CLI can use the bridge directly with `llama copilot`, or through
the same environment variables it uses for Ollama:

```powershell
$env:COPILOT_PROVIDER_BASE_URL="http://127.0.0.1:8089/v1"
$env:COPILOT_PROVIDER_API_KEY="change-me"
$env:COPILOT_PROVIDER_WIRE_API="responses"
$env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS="65536"
$env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS="2048"
$env:COPILOT_MODEL="qwen3.5:cloud"
copilot
```

Headless example:

```powershell
$env:COPILOT_PROVIDER_BASE_URL="http://127.0.0.1:8089/v1"
$env:COPILOT_PROVIDER_API_KEY="change-me"
$env:COPILOT_PROVIDER_WIRE_API="responses"
$env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS="65536"
$env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS="2048"
$env:COPILOT_MODEL="qwen3.5:cloud"
copilot -p "how does this repository work?"
```

Foreground mode:

```powershell
llama serve --foreground
```

## Test and benchmark

Start the bridge, then run the full test and benchmark suite:

```powershell
llama serve --foreground
python .\test.py
```

The plain command runs unit tests, live agent-shaped requests, advanced benchmark
cases, and parallel stress checks three times. It exercises
`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/api/chat`,
`/api/generate`, model listing, health, and token counting with messages and
tool schemas/results, so the server access log should show `200 OK` lines like
normal Pi, Codex, Copilot, or Claude Code traffic.

Use `--model your-model-id` to force a specific model; otherwise the runner uses
the first model returned by `/v1/models`. Use `--no-live`, `--basic`, or
`--no-stress` only when debugging the runner itself.

## Claude Code settings

Point Claude Code at the bridge:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8089",
    "ANTHROPIC_AUTH_TOKEN": "change-me",
    "ANTHROPIC_SMALL_FAST_MODEL": "haiku",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "opus"
  }
}
```

## Config

Run `llama setup` to generate `config.example.yml` when `env.yml` is missing.
After editing it with your providers, API keys, and models, rename it to `env.yml`.

Provider notes:

- `ollama_local` targets a local Ollama server, usually `http://127.0.0.1:11434/v1`
- `lm_studio` targets LM Studio's local OpenAI-compatible server, usually `http://127.0.0.1:1234/v1`
- `ollama_cloud` targets Ollama's hosted API and should use `OLLAMA_API_KEY`
- `nvidia_nim` defaults to `z-ai/glm4.7`, but you can override any alias with a different `model`
- `groq` targets `https://api.groq.com/openai/v1` and should use `GROQ_API_KEY`
- `gemini` targets Google's OpenAI-compatible Gemini endpoint and should use `GEMINI_API_KEY`
- `openai` targets `https://api.openai.com/v1` and should use `OPENAI_API_KEY`
- `cohere` targets Cohere's OpenAI compatibility endpoint and should use `COHERE_API_KEY`
- `mistral` targets `https://api.mistral.ai/v1` and should use `MISTRAL_API_KEY`
- `deepseek` targets `https://api.deepseek.com` and should use `DEEPSEEK_API_KEY`
- `openrouter` targets `https://openrouter.ai/api/v1` and should use `OPENROUTER_API_KEY`
- `openai_compatible` works for any OpenAI-style endpoint, including custom gateways
- Set `supports_tools: false` only for chat-only models; Claude Code works best with a model/provider that supports OpenAI-compatible tool calls
- Add `extra_body` under a provider when a server needs fixed request options, such as `reasoning_effort`, `num_ctx`, or vendor-specific flags

Claude Code should normally keep using `haiku`, `sonnet`, `opus`, and `small_fast`. To use a different provider, point one of those aliases at that provider:

```yaml
anthropic_models:
  sonnet:
    provider: groq
    model: openai/gpt-oss-20b
```

Other examples:

```yaml
anthropic_models:
  sonnet:
    provider: gemini
    model: gemini-3-flash-preview
  opus:
    provider: openrouter
    model: openrouter/auto
  small_fast:
    provider: deepseek
    model: deepseek-v4-flash
```

LM Studio example:

```yaml
providers:
  lm_studio:
    type: lm_studio
    base_url: http://127.0.0.1:1234/v1
    supports_tools: true
    default_model: local-model

anthropic_models:
  sonnet:
    provider: lm_studio
    model: local-model
```

Ollama local example:

```yaml
providers:
  ollama_local:
    type: ollama_local
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
    supports_tools: true

anthropic_models:
  haiku:
    provider: ollama_local
    model: llama3.1:8b
```

Example override:

```yaml
anthropic_models:
  opus:
    provider: nvidia_nim
    model: meta/llama-3.1-70b-instruct
```
