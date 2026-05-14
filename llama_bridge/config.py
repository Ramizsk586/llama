from __future__ import annotations

import io
import importlib.util
import os
import re
import shutil
import sys
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_data_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    package_root = Path(__file__).resolve().parent.parent

    # Prefer CWD if it's the project root (has pyproject.toml)
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd

    # Use package root if it's the project root (has pyproject.toml)
    if (package_root / "pyproject.toml").exists():
        return package_root

    # Fallback: launcher directory (pip-installed script)
    if sys.argv and sys.argv[0]:
        launcher_path = Path(sys.argv[0])
        if launcher_path.name.lower() in {"llama", "llama.exe"}:
            resolved_launcher = launcher_path
            if not resolved_launcher.exists():
                found_launcher = shutil.which(sys.argv[0])
                if found_launcher:
                    resolved_launcher = Path(found_launcher)
            if resolved_launcher.exists():
                return resolved_launcher.resolve().parent

    return package_root


DEFAULT_CONFIG_DIR = _default_data_dir()
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "env.yml"
DEFAULT_EXAMPLE_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.example.yml"
DEFAULT_API_SETTINGS_PATH = DEFAULT_CONFIG_DIR / "Api.json"
DEFAULT_PID_PATH = DEFAULT_CONFIG_DIR / "llama.pid"
DEFAULT_LOG_PATH = DEFAULT_CONFIG_DIR / "llama.log"
DEFAULT_NGROK_PID_PATH = DEFAULT_CONFIG_DIR / "llama.ngrok.pid"
DEFAULT_NGROK_LOG_PATH = DEFAULT_CONFIG_DIR / "llama.ngrok.log"

DEFAULT_CONFIG_TEMPLATE = """# =============================================================================
#                       llama -- config.example.yml
# =============================================================================
#  Copy this file to env.yml and customize it for your setup:
#
#      cp config.example.yml env.yml
#
#  Rules:
#    - Lines starting with # are comments (ignored)
#    - Values with ${VAR_NAME} are read from environment variables
#    - Use ${VAR_NAME} for API keys and secrets (never hardcode them)
#    - Indentation uses 2 spaces (YAML standard)
# =============================================================================


# =============================================================================
#                           SERVER CONFIG
# =============================================================================
#  The bridge HTTP server that proxies AI requests.
# =============================================================================

server:
  # Listen address (use 0.0.0.0 to accept remote connections)
  host: 127.0.0.1

  # Listen port
  port: 8089

  # Bearer token required by clients (Authorization: Bearer <token>)
  auth_token: change-me

  # Auto-started servers stop after this many idle seconds.
  # Set to 0 to keep running until `llama stop`.
  idle_timeout_seconds: 180

  # Optional second port without built-in tools (for Open Web UI etc.)
  # openwebui_port: 8090


# =============================================================================
#                           NGROK CONFIG
# =============================================================================
#  Used by `llama start --online` to publish the local bridge through ngrok.
#  Create a token at https://dashboard.ngrok.com/get-started/your-authtoken
# =============================================================================

ngrok:
  auth_token: ${NGROK_AUTHTOKEN}
  region: null


# =============================================================================
#                          AI PROVIDERS
# =============================================================================
#  Each block defines a connection to an AI model backend.
#  The "type" field selects the API protocol:
#
#    openai | ollama_local | ollama_cloud | groq | gemini | cohere
#    mistral | deepseek | openrouter | lm_studio | nvidia_nim
#    openai_compatible | sarvamai | kilo | opencode
#
#  Use ${VAR_NAME} for sensitive values (API keys, tokens).
# =============================================================================

providers:

  # -- Local Providers (no API key needed) --------------------------------

  ollama_local:
    type: ollama_local
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
    supports_tools: true
    default_model: llama3.1:8b

  lm_studio:
    type: lm_studio
    base_url: http://127.0.0.1:1234/v1
    api_key: lm-studio
    supports_tools: true
    default_model: local-model

  # -- Cloud Providers (API key via environment variable) -----------------

  ollama_cloud:
    type: ollama_cloud
    base_url: https://ollama.com
    api_key: ${OLLAMA_API_KEY}
    supports_tools: true
    default_model: gemma4:31b
    usage_limits:
      hourly:
        limit: 1000
        used: 0
        unit: requests
      monthly:
        limit: 50000
        used: 0
        unit: requests

  nvidia_nim:
    type: nvidia_nim
    base_url: https://integrate.api.nvidia.com/v1
    api_key: ${NVIDIA_API_KEY}
    supports_tools: true
    default_model: z-ai/glm4.7
    extra_body:
      chat_template_kwargs:
        enable_thinking: false

  groq:
    type: groq
    base_url: https://api.groq.com/openai/v1
    api_key: ${GROQ_API_KEY}
    supports_tools: true
    default_model: openai/gpt-oss-20b

  gemini:
    type: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    api_key: ${GEMINI_API_KEY}
    supports_tools: true
    default_model: gemini-3-flash-preview

  openai:
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    supports_tools: true
    default_model: gpt-5.1

  cohere:
    type: cohere
    base_url: https://api.cohere.com/compatibility/v1
    api_key: ${COHERE_API_KEY}
    supports_tools: true
    default_model: command-a-03-2025

  mistral:
    type: mistral
    base_url: https://api.mistral.ai/v1
    api_key: ${MISTRAL_API_KEY}
    supports_tools: true
    default_model: mistral-large-latest

  deepseek:
    type: deepseek
    base_url: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}
    supports_tools: true
    default_model: deepseek-chat

  openrouter:
    type: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    supports_tools: true
    default_model: openrouter/auto

  # -- Generic / Custom Providers ----------------------------------------

  openai_compatible:
    type: openai_compatible
    base_url: https://your-provider.example.com/v1
    api_key: ${CUSTOM_PROVIDER_API_KEY}
    supports_tools: true
    default_model: your-model

  sarvamai:
    type: sarvamai
    base_url: https://api.sarvam.ai/v1
    api_key: ${SARVAM_API_KEY}
    supports_tools: true
    default_model: sarvam-30b

  kilo:
    type: kilo
    base_url: https://api.kilo.ai/api/gateway
    api_key: ${KILO_API_KEY}
    supports_tools: true
    default_model: kilo-auto/free

  opencode:
    type: opencode
    base_url: https://opencode.ai/zen
    api_key: ${OPENCODE_API_KEY}
    supports_tools: true
    default_model: opencode/auto


# =============================================================================
#                       ANTHROPIC MODEL ALIASES
# =============================================================================
#  Maps Anthropic-style model names to bridge providers/models.
#  Used by clients that request Anthropic model names.
# =============================================================================

anthropic_models:
  haiku:
    provider: ollama_cloud
    model: gemma4:31b
  sonnet:
    provider: ollama_cloud
    model: gemma4:31b
  opus:
    provider: ollama_cloud
    model: gemma4:31b


# =============================================================================
#                        INTEGRATION: PI
# =============================================================================
#  Configuration for the Pi coding agent.
# =============================================================================

pi:
  provider: ollama_cloud
  model: gemma4:31b
  api: openai-completions
  config_dir: ~/.pi/agent
  install_package: "@mariozechner/pi-coding-agent"
  web_search: true


# =============================================================================
#                       INTEGRATION: CODEX
# =============================================================================
#  Configuration for OpenAI Codex CLI.
# =============================================================================

codex:
  provider: ollama_cloud
  model: gemma4:31b
  config_path: ~/.codex/config.toml
  profile: llama_bridge
  install_package: "@openai/codex"


# =============================================================================
#                     INTEGRATION: COPILOT CLI
# =============================================================================
#  Configuration for GitHub Copilot CLI (@github/copilot).
# =============================================================================

copilot_cli:
  provider: ollama_cloud
  model: gemma4:31b
  wire_api: responses
  max_prompt_tokens: 65536
  max_output_tokens: 2048
  install_package: "@github/copilot"


# =============================================================================
#                    INTEGRATION: VS CODE COPILOT LOCAL
# =============================================================================
#  Models advertised through Ollama-compatible /api/tags for VS Code Copilot.
#  VS Code allows up to 3 local Ollama models.
# =============================================================================

vs_copilot:
  models:
    - name: gemma4:31b
      provider: ollama_cloud
      model: gemma4:31b
      context_size: 65536
    - name: minimax
      provider: opencode
      model: minimax-m2.5-free
      context_size: 65536


# =============================================================================
#                      INTEGRATION: OPENCODE
# =============================================================================
#  Configuration for the opencode-ai CLI tool.
# =============================================================================

opencode:
  provider: ollama_cloud
  model: gemma4:31b
  install_package: "opencode-ai"


# =============================================================================
#                      INTEGRATION: OPENCLAW
# =============================================================================
#  Configuration for the OpenClaw sandboxed agent.
# =============================================================================

openclaw:
  provider: ollama_cloud
  model: gemma4:31b
  config_path: ~/.openclaw/llama-openclaw.json
  workspace: ~/.openclaw/llama-workspace
  workspace_access: none
  sandbox_backend: docker
  install_package: "openclaw"


# =============================================================================
#                      INTEGRATION: POOLSIDE
# =============================================================================
#  Configuration for Poolside AI assistant.
# =============================================================================

poolside:
  provider: ollama_cloud
  model: gemma4:31b
  api_url: http://127.0.0.1:8089
  api_key: null
  token: ${POOLSIDE_TOKEN}
  config_path: ~/.config/poolside/settings.yaml
  install_command: "curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh"
  windows_install_command: "irm https://downloads.poolside.ai/pool/install.ps1 | iex"


# =============================================================================
#                        TELEGRAM BOT
# =============================================================================
#  Optional Telegram bot integration. When enabled, the bridge can
#  receive and respond to messages via Telegram.
# =============================================================================

telegram:
  enabled: false
  bot_token: ${TELEGRAM_BOT_TOKEN}
  allowed_chat_ids: []
  owner_chat_ids: []
  admin_chat_ids: []
  allow_all_chats: false
  admin_pin_hash: null
  core_editing_enabled: false
  require_owner_approval_for_core_changes: true
  provider: ollama_cloud
  model: gemma4:31b
  system_prompt: "You are a restricted Telegram bot powered by llama bridge. Answer helpfully, keep replies concise, use bridge tools only when clearly needed, and refuse unsafe or privileged actions."
  max_input_chars: 4000
  max_output_tokens: 512
  poll_interval_seconds: 2.0
  response_timeout_seconds: 180.0
  autonomous_enabled: true
  autonomous_interval_seconds: 1800
  self_evolution_enabled: true
  self_evolution_min_events: 3

  # -- Command permissions ------------------------------------------------
  command_policy:
    help:
      enabled: true
      visible: true
      permission: everyone
    web:
      enabled: true
      visible: true
      permission: everyone
    deep:
      enabled: true
      visible: true
      permission: everyone
    allow:
      enabled: true
      visible: false
      permission: owner
    admin:
      enabled: true
      visible: false
      permission: owner
    core:
      enabled: true
      visible: false
      permission: admin

  # -- AI tool policy ----------------------------------------------------
  tool_policy:
    ai_auto_tools:
      - datetime_now
      - weather_current
      - wikipedia_search
      - tavily_search
      - source_research
      - ddg_image_download
    command_tools:
      - ddg_image_download
      - image_research
      - image_download
      - verify_sources
    blocked_tools:
      - shell.execute
    user_visible_tools:
      - weather_current
      - wikipedia_search
      - tavily_search
      - ddg_image_download
      - image_research
      - image_download
    require_admin_for:
      - shell.execute
      - manim_render
    require_owner_for:
      - shell.execute

  # -- Help menu settings ------------------------------------------------
  telegram_help:
    show_disabled_commands: false
    show_tool_list: false
    show_admin_commands_to_admins_only: true


# =============================================================================
#                            TOOL SYSTEM
# =============================================================================
#  Controls the built-in tool management system.
#  Tools allow AI models to perform actions (web search, file ops, etc.).
# =============================================================================

tools:
  enabled: true
  expose_http: true
  require_auth: true
  country: India

  # -- Available tool list ------------------------------------------------
  include:
    - shell.execute
    - datetime_now
    - manim_render
    - wikipedia_search
    - wikipedia_page
    - weather_current
    - serpapi_search
    - tavily_search
    - source_research
    - ddg_image_download
    - image_research
    - image_download
    - verify_sources

  # -- Tool selection & filtering ----------------------------------------
  max_exposed: 8
  relevance_filter: true
  force_for_keywords: true
  confidence_threshold: 0.5
  log_outputs: false
  default_search_provider: tavily
  cache_enabled: true
  cache_ttl_seconds: 300
  tool_system_instructions: null
  pi_system_instructions: null

  # -- External tool providers -------------------------------------------

  # SerpAPI (web search via Google)
  serpapi:
    enabled: false
    api_key: ${SERPAPI_API_KEY}
    base_url: https://serpapi.com/search
    defaults:
      engine: google
      num: 5

  # Tavily (AI-native search)
  tavily:
    enabled: false
    api_key: ${TAVILY_API_KEY}
    base_url: https://api.tavily.com/search
    defaults:
      search_depth: basic
      max_results: 5

  # Weather (free, no API key needed)
  weather:
    enabled: true
    provider: open_meteo
    base_url: https://api.open-meteo.com/v1/forecast
    geocoding_url: https://geocoding-api.open-meteo.com/v1/search

  # Wikipedia
  wikipedia:
    enabled: true
    language: en
    base_url: https://en.wikipedia.org


"""


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8089
    auth_token: str = "change-me"
    idle_timeout_seconds: int = 180
    openwebui_port: int | None = None


@dataclass(slots=True)
class NgrokConfig:
    auth_token: str | None = None
    region: str | None = None


@dataclass(slots=True)
class ProviderConfig:
    name: str
    type: str
    base_url: str
    api_key: str | None = None
    default_model: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 300.0
    supports_tools: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    usage_limits: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_limits: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ModelAlias:
    alias: str
    provider: str
    model: str | None = None


@dataclass(slots=True)
class PiConfig:
    provider: str = "ollama_cloud"
    model: str | None = None
    api: str = "openai-completions"
    config_dir: str = "~/.pi/agent"
    install_package: str = "@mariozechner/pi-coding-agent"
    web_search: bool = True


@dataclass(slots=True)
class CodexConfig:
    provider: str = "ollama_cloud"
    model: str | None = None
    config_path: str = "~/.codex/config.toml"
    profile: str = "llama_bridge"
    install_package: str = "@openai/codex"


@dataclass(slots=True)
class CopilotCliConfig:
    provider: str = "ollama_cloud"
    model: str | None = None
    wire_api: str = "responses"
    max_prompt_tokens: int = 65536
    max_output_tokens: int = 2048
    install_package: str = "@github/copilot"


@dataclass(slots=True)
class OpenCodeConfig:
    provider: str = "ollama_cloud"
    model: str | None = None
    config_path: str = "~/.config/opencode/opencode.json"
    provider_id: str = "llama-bridge"
    provider_name: str = "Llama Bridge"
    install_package: str = "opencode-ai"
    context_size: int = 65536
    output_tokens: int = 8192
    small_model: str | None = None
    write_project_config: bool = False


@dataclass(slots=True)
class OpenClawConfig:
    provider: str = "ollama_cloud"
    model: str | None = None
    config_path: str = "~/.openclaw/llama-openclaw.json"
    workspace: str = "~/.openclaw/llama-workspace"
    workspace_access: str = "none"
    sandbox_backend: str = "docker"
    install_package: str = "openclaw"


@dataclass(slots=True)
class PoolsideConfig:
    provider: str = "ollama_cloud"
    model: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    token: str | None = None
    config_path: str = "~/.config/poolside/settings.yaml"
    install_command: str = "curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh"
    windows_install_command: str = "irm https://downloads.poolside.ai/pool/install.ps1 | iex"


@dataclass(slots=True)
class CommandPolicy:
    enabled: bool = True
    visible: bool = True
    permission: str = "everyone"  # everyone, allowed, admin, owner


@dataclass(slots=True)
class ToolPolicy:
    ai_auto_tools: list[str] = field(default_factory=lambda: [
        "datetime_now", "weather_current", "wikipedia_search",
        "tavily_search", "source_research", "ddg_image_download",
    ])
    command_tools: list[str] = field(default_factory=lambda: [
        "ddg_image_download", "image_research", "image_download", "verify_sources",
    ])
    blocked_tools: list[str] = field(default_factory=lambda: ["shell.execute"])
    user_visible_tools: list[str] = field(default_factory=lambda: [
        "weather_current", "wikipedia_search",
        "tavily_search", "ddg_image_download", "image_research", "image_download",
    ])
    require_admin_for: list[str] = field(default_factory=lambda: ["shell.execute", "manim_render"])
    require_owner_for: list[str] = field(default_factory=lambda: ["shell.execute"])


DEFAULT_COMMAND_POLICIES: dict[str, CommandPolicy] = {
    "help": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "status": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "clear": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "reload": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "whoami": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "memory": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "remember": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "docs": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "editdoc": CommandPolicy(enabled=True, visible=True, permission="admin"),
    "image": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "file": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "schedule": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "evolve": CommandPolicy(enabled=True, visible=True, permission="admin"),
    "poll": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "web": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "deep": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "summarize": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "explain": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "myid": CommandPolicy(enabled=True, visible=True, permission="everyone"),
    "allowlist": CommandPolicy(enabled=True, visible=True, permission="admin"),
    "allow": CommandPolicy(enabled=True, visible=True, permission="owner"),
    "admin": CommandPolicy(enabled=True, visible=True, permission="owner"),
    "owner": CommandPolicy(enabled=True, visible=True, permission="owner"),
    "core": CommandPolicy(enabled=True, visible=True, permission="admin"),
    "project": CommandPolicy(enabled=True, visible=True, permission="admin"),
    "tools": CommandPolicy(enabled=True, visible=True, permission="admin"),
}


@dataclass(slots=True)
class TelegramBotConfig:
    enabled: bool = False
    bot_token: str | None = None
    allowed_chat_ids: list[str] = field(default_factory=list)
    owner_chat_ids: list[str] = field(default_factory=list)
    admin_chat_ids: list[str] = field(default_factory=list)
    allow_all_chats: bool = False
    admin_pin_hash: str | None = None
    core_editing_enabled: bool = False
    require_owner_approval_for_core_changes: bool = True
    provider: str = "ollama_cloud"
    model: str | None = None
    system_prompt: str = (
        "You are a restricted Telegram bot powered by llama bridge. "
        "Answer helpfully, keep replies concise, use bridge tools only when clearly needed, "
        "and refuse unsafe or privileged actions."
    )
    max_input_chars: int = 4000
    max_output_tokens: int = 512
    poll_interval_seconds: float = 2.0
    response_timeout_seconds: float = 180.0
    autonomous_enabled: bool = True
    autonomous_interval_seconds: float = 1800.0
    self_evolution_enabled: bool = True
    self_evolution_min_events: int = 3
    command_policy: dict[str, CommandPolicy] = field(default_factory=lambda: dict(DEFAULT_COMMAND_POLICIES))
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    telegram_help: dict[str, bool] = field(default_factory=lambda: {
        "show_disabled_commands": False,
        "show_tool_list": False,
        "show_admin_commands_to_admins_only": True,
    })


@dataclass(slots=True)
class VsCopilotModel:
    name: str
    provider: str
    model: str | None = None
    context_size: int = 65536
    modified_at: str | None = None
    size: int | None = None
    digest: str | None = None


@dataclass(slots=True)
class ExternalToolProviderConfig:
    enabled: bool = False
    api_key: str | None = None
    base_url: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolConfig:
    enabled: bool = True
    expose_http: bool = True
    require_auth: bool = True
    country: str | None = None
    include: list[str] = field(default_factory=list)
    serpapi: ExternalToolProviderConfig = field(default_factory=ExternalToolProviderConfig)
    tavily: ExternalToolProviderConfig = field(default_factory=ExternalToolProviderConfig)
    weather: ExternalToolProviderConfig = field(default_factory=ExternalToolProviderConfig)
    wikipedia: ExternalToolProviderConfig = field(default_factory=ExternalToolProviderConfig)
    # Tool selection and filtering options
    max_exposed: int = 8  # Maximum tools to expose in a single request (after filtering)
    relevance_filter: bool = True  # Enable relevance-based tool filtering
    force_for_keywords: bool = True  # Strongly prefer obvious tools for weather/time/latest/source queries
    confidence_threshold: float = 0.5  # Minimum score to include a tool (0.0-10.0)
    log_outputs: bool = False  # Log tool results and arguments
    default_search_provider: str = "tavily"  # Preferred search tool: "tavily" or "serpapi"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    tool_system_instructions: str | None = None
    # Pi CLI specific instructions
    pi_system_instructions: str | None = None  # Custom system instructions for Pi CLI
    # Tool management system options
    management_enabled: bool = True
    compact_manifest_enabled: bool = True
    compact_manifest_max_tools: int = 20
    always_expose_management_tools: bool = True
    expose_full_schema_policy: str = "relevant"  # one of: always, relevant, on_demand, never
    schema_memory_enabled: bool = True
    schema_memory_ttl_seconds: int = 86400
    schema_fetch_max_tools: int = 3
    compact_summary_max_chars: int = 160
    full_schema_token_budget: int = 5000
    fallback_to_full_schemas_for_unsupported_clients: bool = True


@dataclass(slots=True)
class OpenWebUIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    bridge_tools_port: int | None = None
    bridge_llm_only_port: int | None = None
    auth_enabled: bool = False
    auto_login: bool = True
    web_search_enabled: bool = True
    web_search_provider: str = "ollama"
    web_search_providers: dict[str, ExternalToolProviderConfig] = field(default_factory=dict)
    search_result_count: int = 3
    concurrent_requests: int = 1
    bypass_embedding_and_retrieval: bool = False
    bypass_web_loader: bool = False
    hf_token: str | None = None
    openai_base_url_mode: str = "llm_only"
    openwebui_data_dir: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    # Discovery config
    preferred_env_name: str = "omx-open-webui"
    preferred_python: str | None = None
    preferred_command: str | None = None
    auto_discover: bool = True


@dataclass(slots=True)
class BridgeConfig:
    server: ServerConfig
    ngrok: NgrokConfig
    providers: dict[str, ProviderConfig]
    anthropic_models: dict[str, ModelAlias]
    pi: PiConfig
    codex: CodexConfig
    copilot_cli: CopilotCliConfig
    opencode: OpenCodeConfig
    openclaw: OpenClawConfig
    poolside: PoolsideConfig
    telegram: TelegramBotConfig
    vs_copilot_models: list[VsCopilotModel]
    tools: ToolConfig
    openwebui: OpenWebUIConfig
    source_path: Path


def resolve_pi_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or config.pi.provider
    provider = config.providers[selected_provider]
    if model_override:
        return model_override
    if config.pi.model:
        return config.pi.model
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return None


def pi_model_error(config: BridgeConfig, provider_name: str | None = None) -> str:
    selected_provider = provider_name or config.pi.provider
    return (
        "Pi model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set pi.model, set that provider's default_model, configure a model alias "
        "for that provider, or pass `llama pi --model ...`."
    )


def resolve_codex_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or config.codex.provider
    provider = config.providers[selected_provider]
    if model_override:
        return model_override
    if config.codex.model:
        return config.codex.model
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return None


def codex_model_error(config: BridgeConfig, provider_name: str | None = None) -> str:
    selected_provider = provider_name or config.codex.provider
    return (
        "Codex model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set codex.model, set that provider's default_model, configure a model alias "
        "for that provider, or pass `llama codex --model ...`."
    )


def resolve_copilot_cli_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or config.copilot_cli.provider
    provider = config.providers[selected_provider]
    if model_override:
        return model_override
    if config.copilot_cli.model:
        return config.copilot_cli.model
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return None


def copilot_cli_model_error(
    config: BridgeConfig, provider_name: str | None = None
) -> str:
    selected_provider = provider_name or config.copilot_cli.provider
    return (
        "Copilot CLI model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set copilot_cli.model, set that provider's default_model, configure a model "
        "alias for that provider, or pass `llama copilot --model ...`."
    )


def resolve_opencode_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or config.opencode.provider
    provider = config.providers[selected_provider]
    if model_override:
        return model_override
    if config.opencode.model:
        return config.opencode.model
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return None


def opencode_model_error(config: BridgeConfig, provider_name: str | None = None) -> str:
    selected_provider = provider_name or config.opencode.provider
    return (
        "OpenCode model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set opencode.model, set that provider's default_model, configure a model "
        "alias for that provider, or pass `llama opencode --model ...`."
    )


def resolve_openclaw_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or config.openclaw.provider
    provider = config.providers[selected_provider]
    if model_override:
        return model_override
    if config.openclaw.model:
        return config.openclaw.model
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return None


def openclaw_model_error(config: BridgeConfig, provider_name: str | None = None) -> str:
    selected_provider = provider_name or config.openclaw.provider
    return (
        "OpenClaw model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set openclaw.model, set that provider's default_model, configure a model "
        "alias for that provider, or pass `llama openclaw --model ...`."
    )


def resolve_kilo_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or "kilo"
    provider = config.providers.get(selected_provider)
    if not provider:
        return None
    if model_override:
        return model_override
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return "kilo-auto/free"


def kilo_model_error(config: BridgeConfig, provider_name: str | None = None) -> str:
    selected_provider = provider_name or "kilo"
    return (
        "Kilo model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set the provider's default_model in env.yml, configure a model alias "
        "for that provider, or pass `--model kilo-auto/free` (or any other Kilo model)."
    )


def resolve_openai_model(
    config: BridgeConfig,
    provider_name: str | None = None,
    model_override: str | None = None,
) -> str | None:
    selected_provider = provider_name or config.openai.provider
    provider = config.providers[selected_provider]
    if model_override:
        return model_override
    if config.openai.model:
        return config.openai.model
    if provider.default_model:
        return provider.default_model

    preferred_aliases = ("sonnet", "opus", "haiku")
    for alias_name in preferred_aliases:
        alias = config.anthropic_models.get(alias_name)
        if alias and alias.provider == selected_provider and alias.model:
            return alias.model

    for alias in config.anthropic_models.values():
        if alias.provider == selected_provider and alias.model:
            return alias.model

    return None


def openai_model_error(config: BridgeConfig, provider_name: str | None = None) -> str:
    selected_provider = provider_name or config.openai.provider
    return (
        "OpenCode model is not configured. "
        f"Config: {config.source_path}. Provider: {selected_provider}. "
        "Set openai.model, set that provider's default_model, configure a model "
        "alias for that provider, or pass `llama openai --model ...`."
    )


def resolve_vs_copilot_model(
    config: BridgeConfig,
    requested_model: str,
) -> tuple[VsCopilotModel, str]:
    for entry in config.vs_copilot_models:
        if requested_model not in {entry.name, entry.model}:
            continue
        provider = config.providers[entry.provider]
        upstream_model = entry.model or provider.default_model
        if not upstream_model:
            raise KeyError(
                f"VS Copilot model '{entry.name}' has no model and provider "
                f"'{entry.provider}' has no default_model configured"
            )
        return entry, upstream_model
    available = ", ".join(model.name for model in config.vs_copilot_models)
    raise KeyError(f"Unknown VS Copilot model '{requested_model}'. Available models: {available}")


def write_claude_api_settings(
    path: Path = DEFAULT_API_SETTINGS_PATH,
    server: ServerConfig | None = None,
    aliases: dict[str, ModelAlias] | None = None,
    force: bool = False,
) -> Path:
    ensure_default_dirs(path.parent)
    if path.exists() and not force:
        return path

    server = server or ServerConfig()
    aliases = aliases or {}
    haiku_model = "haiku"
    sonnet_model = "sonnet" if "sonnet" in aliases else haiku_model
    opus_model = "opus" if "opus" in aliases else sonnet_model

    settings = _read_json_object(path) if path.exists() else {}
    env = dict(settings.get("env") or {})
    env.pop("ANTHROPIC_SMALL_FAST_MODEL", None)
    env.update(
        {
            "ANTHROPIC_BASE_URL": f"http://{server.host}:{server.port}",
            "ANTHROPIC_AUTH_TOKEN": server.auth_token,
            "API_TIMEOUT_MS": "3000000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": haiku_model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": sonnet_model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": opus_model,
        }
    )
    settings["env"] = env
    settings.setdefault("enabledPlugins", {})
    settings.setdefault("effortLevel", "medium")
    settings["model"] = sonnet_model
    settings.setdefault("theme", "dark-ansi")
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path


def _read_json_object(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
FULL_BRACED_VALUE_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


def _expand_env_string(value: str) -> str:
    full_match = FULL_BRACED_VALUE_PATTERN.match(value)
    if full_match:
        inner = full_match.group(1)
        env_value = os.environ.get(inner)
        if env_value is not None:
            return env_value
        if inner != inner.upper():
            return inner
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner):
            return inner

    expanded = os.path.expandvars(value)

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return ENV_VAR_PATTERN.sub(replace, expanded)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, dict):
        return {key: _expand_env(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def ensure_default_dirs(path: Path | None = None) -> None:
    target = path or DEFAULT_CONFIG_DIR
    target.mkdir(parents=True, exist_ok=True)


def write_default_config(
    path: Path = DEFAULT_EXAMPLE_CONFIG_PATH,
    force: bool = False,
) -> Path:
    ensure_default_dirs(path.parent)
    if path.exists() and not force:
        return path
    try:
        content = _default_config_template_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        content = DEFAULT_CONFIG_TEMPLATE
    path.write_text(content, encoding="utf-8")
    return path


def merge_missing_config_fields(path: Path) -> tuple[Path, bool]:
    try:
        if importlib.util.find_spec("ruamel.yaml") is None:
            raise ImportError
        _merge_missing_config_fields_ruamel(path)
        return path, True
    except ImportError:
        _merge_missing_config_fields_yaml(path)
        return path, True


def _merge_missing_config_fields_ruamel(path: Path) -> None:
    """Merge missing config fields while preserving comments using ruamel.yaml."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    ensure_default_dirs(path.parent)

    try:
        template_text = _default_config_template_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        template_text = DEFAULT_CONFIG_TEMPLATE

    yaml_parser = YAML()
    yaml_parser.preserve_quotes = True
    yaml_parser.default_flow_style = False
    yaml_parser.indent(mapping=2, sequence=4, offset=2)

    if path.exists():
        existing_text = path.read_text(encoding="utf-8")
        existing_data = yaml_parser.load(existing_text)
        template_data = yaml_parser.load(template_text)

        if existing_data is None:
            existing_data = CommentedMap()
        if template_data is None:
            template_data = CommentedMap()

        comment_line_count = sum(
            1 for line in existing_text.splitlines() if line.lstrip().startswith("#")
        )
        template_comment_line_count = sum(
            1 for line in template_text.splitlines() if line.lstrip().startswith("#")
        )
        if comment_line_count < max(10, template_comment_line_count):
            _overlay_config_values(template_data, existing_data)
            output = io.StringIO()
            yaml_parser.dump(template_data, output)
            path.write_text(output.getvalue(), encoding="utf-8")
            return

        _merge_commented_maps(existing_data, template_data)

        output = io.StringIO()
        yaml_parser.dump(existing_data, output)
        path.write_text(output.getvalue(), encoding="utf-8")
    else:
        path.write_text(template_text, encoding="utf-8")


def _merge_commented_maps(existing: Any, template: Any) -> None:
    """Recursively merge template values into existing, preserving existing keys and comments."""
    if not isinstance(existing, dict) or not isinstance(template, dict):
        return

    for key, template_value in template.items():
        if key not in existing:
            existing[key] = template_value
        elif isinstance(template_value, dict) and isinstance(existing.get(key), dict):
            _merge_commented_maps(existing[key], template_value)


def _overlay_config_values(template: Any, existing: Any) -> Any:
    """Overlay existing values onto the commented template, keeping template comments."""
    if not isinstance(template, dict) or not isinstance(existing, dict):
        return existing

    for key, existing_value in existing.items():
        if key in template and isinstance(template.get(key), dict) and isinstance(existing_value, dict):
            _overlay_config_values(template[key], existing_value)
        else:
            template[key] = existing_value
    return template


def _merge_missing_config_fields_yaml(path: Path) -> None:
    """Fallback merge using standard pyyaml (loses comments)."""
    import yaml

    ensure_default_dirs(path.parent)
    try:
        template_text = _default_config_template_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        template_text = DEFAULT_CONFIG_TEMPLATE
    template_raw = yaml.safe_load(template_text) or {}

    if path.exists():
        existing_raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        merged = _merge_missing_values(existing_raw, template_raw)
        changed = merged != existing_raw
        if changed:
            path.write_text(
                yaml.safe_dump(merged, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
        return

    path.write_text(
        yaml.safe_dump(template_raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _merge_missing_values(existing: Any, template: Any) -> Any:
    if isinstance(existing, dict) and isinstance(template, dict):
        merged = dict(existing)
        for key, template_value in template.items():
            if key in merged:
                merged[key] = _merge_missing_values(merged[key], template_value)
            else:
                merged[key] = template_value
        return merged
    return existing


def write_config_data(path: Path, data: dict[str, Any]) -> Path:
    """Write env.yml data while preserving the commented template when possible."""
    ensure_default_dirs(path.parent)
    try:
        if importlib.util.find_spec("ruamel.yaml") is None:
            raise ImportError
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap

        try:
            template_text = _default_config_template_path().read_text(encoding="utf-8")
        except FileNotFoundError:
            template_text = DEFAULT_CONFIG_TEMPLATE

        existing_text = path.read_text(encoding="utf-8") if path.exists() else ""
        comment_line_count = sum(
            1 for line in existing_text.splitlines() if line.lstrip().startswith("#")
        )
        template_comment_line_count = sum(
            1 for line in template_text.splitlines() if line.lstrip().startswith("#")
        )
        source_text = existing_text if comment_line_count >= template_comment_line_count else template_text

        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True
        yaml_parser.default_flow_style = False
        yaml_parser.indent(mapping=2, sequence=4, offset=2)
        base = yaml_parser.load(source_text) if source_text.strip() else CommentedMap()
        if base is None:
            base = CommentedMap()
        _sync_commented_config(base, data)

        output = io.StringIO()
        yaml_parser.dump(base, output)
        path.write_text(output.getvalue(), encoding="utf-8")
        return path
    except ImportError:
        import yaml

        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        return path


def _sync_commented_config(target: Any, source: Any) -> None:
    if not isinstance(target, dict) or not isinstance(source, dict):
        return

    for key, source_value in source.items():
        if key in target and isinstance(target.get(key), dict) and isinstance(source_value, dict):
            _sync_commented_config(target[key], source_value)
        else:
            target[key] = source_value


def load_config(path: Path | None = None) -> BridgeConfig:
    import yaml

    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        example_path = config_path.parent / DEFAULT_EXAMPLE_CONFIG_PATH.name
        write_default_config(example_path)
        raise FileNotFoundError(
            f"Missing {config_path.name}. Created {example_path.name}; edit it, "
            f"add your API keys/models, then rename it to {config_path.name}."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _expand_env(raw)

    server_raw = raw.get("server", {})
    raw_openwebui_port = server_raw.get("openwebui_port")
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=int(server_raw.get("port", 8089)),
        auth_token=server_raw.get("auth_token", "change-me"),
        idle_timeout_seconds=max(0, int(server_raw.get("idle_timeout_seconds", 180))),
        openwebui_port=int(raw_openwebui_port) if raw_openwebui_port is not None else None,
    )
    if server.auth_token == "change-me" and server.host not in {"127.0.0.1", "localhost", "::1"}:
        warnings.warn(
            "server.auth_token is still 'change-me' while server.host is not localhost.",
            RuntimeWarning,
            stacklevel=2,
        )

    ngrok_raw = raw.get("ngrok", {}) or {}
    ngrok = NgrokConfig(
        auth_token=ngrok_raw.get("auth_token"),
        region=ngrok_raw.get("region"),
    )

    providers: dict[str, ProviderConfig] = {}
    for name, value in (raw.get("providers", {}) or {}).items():
        providers[name] = ProviderConfig(
            name=name,
            type=value.get("type", "openai_compatible"),
            base_url=value["base_url"].rstrip("/"),
            api_key=value.get("api_key"),
            default_model=value.get("default_model"),
            headers=value.get("headers", {}) or {},
            timeout=float(value.get("timeout", 300)),
            supports_tools=bool(value.get("supports_tools", True)),
            extra_body=value.get("extra_body", {}) or {},
            usage_limits=value.get("usage_limits", {}) or {},
            model_limits=value.get("model_limits", {}) or {},
        )

    aliases: dict[str, ModelAlias] = {}
    for alias, value in (raw.get("anthropic_models", {}) or {}).items():
        aliases[alias] = ModelAlias(
            alias=alias,
            provider=value["provider"],
            model=value.get("model"),
        )

    if not providers:
        raise ValueError("No providers configured in env.yml")
    if not aliases:
        raise ValueError("No anthropic_models configured in env.yml")

    missing = [
        alias.alias
        for alias in aliases.values()
        if alias.provider not in providers
    ]
    if missing:
        raise ValueError(
            f"Unknown providers referenced by anthropic_models: {', '.join(missing)}"
        )

    missing_models = [
        alias.alias
        for alias in aliases.values()
        if not alias.model and not providers[alias.provider].default_model
    ]
    if missing_models:
        raise ValueError(
            "Missing model for aliases without provider defaults: "
            + ", ".join(missing_models)
        )

    pi_raw = raw.get("pi", {}) or {}
    pi = PiConfig(
        provider=pi_raw.get("provider", "ollama_cloud"),
        model=pi_raw.get("model"),
        api=pi_raw.get("api", "openai-completions"),
        config_dir=pi_raw.get("config_dir", "~/.pi/agent"),
        install_package=pi_raw.get("install_package", "@mariozechner/pi-coding-agent"),
        web_search=bool(pi_raw.get("web_search", True)),
    )
    if pi.provider not in providers:
        raise ValueError(f"Unknown provider referenced by pi.provider: {pi.provider}")

    codex_raw = raw.get("codex", {}) or {}
    codex = CodexConfig(
        provider=codex_raw.get("provider", pi.provider),
        model=codex_raw.get("model"),
        config_path=codex_raw.get("config_path", "~/.codex/config.toml"),
        profile=codex_raw.get("profile", "llama_bridge"),
        install_package=codex_raw.get("install_package", "@openai/codex"),
    )
    if codex.provider not in providers:
        raise ValueError(f"Unknown provider referenced by codex.provider: {codex.provider}")

    copilot_raw = raw.get("copilot_cli", {}) or {}
    copilot_cli = CopilotCliConfig(
        provider=copilot_raw.get("provider", codex.provider),
        model=copilot_raw.get("model"),
        wire_api=copilot_raw.get("wire_api", "responses"),
        max_prompt_tokens=int(copilot_raw.get("max_prompt_tokens", 65536)),
        max_output_tokens=int(copilot_raw.get("max_output_tokens", 2048)),
        install_package=copilot_raw.get("install_package", "@github/copilot"),
    )
    if copilot_cli.provider not in providers:
        raise ValueError(
            f"Unknown provider referenced by copilot_cli.provider: {copilot_cli.provider}"
        )

    opencode_raw = raw.get("opencode", {}) or {}
    opencode = OpenCodeConfig(
        provider=opencode_raw.get("provider", codex.provider),
        model=opencode_raw.get("model"),
        config_path=opencode_raw.get("config_path", "~/.config/opencode/opencode.json"),
        provider_id=opencode_raw.get("provider_id", "llama-bridge"),
        provider_name=opencode_raw.get("provider_name", "Llama Bridge"),
        install_package=opencode_raw.get("install_package", "opencode-ai"),
        context_size=int(opencode_raw.get("context_size", 65536)),
        output_tokens=int(opencode_raw.get("output_tokens", 8192)),
        small_model=opencode_raw.get("small_model"),
        write_project_config=bool(opencode_raw.get("write_project_config", False)),
    )
    if opencode.provider not in providers:
        raise ValueError(
            f"Unknown provider referenced by opencode.provider: {opencode.provider}"
        )

    openclaw_raw = raw.get("openclaw", {}) or {}
    openclaw = OpenClawConfig(
        provider=openclaw_raw.get("provider", codex.provider),
        model=openclaw_raw.get("model"),
        config_path=openclaw_raw.get("config_path", "~/.openclaw/llama-openclaw.json"),
        workspace=openclaw_raw.get("workspace", "~/.openclaw/llama-workspace"),
        workspace_access=str(openclaw_raw.get("workspace_access", "none")),
        sandbox_backend=str(openclaw_raw.get("sandbox_backend", "docker")),
        install_package=openclaw_raw.get("install_package", "openclaw"),
    )
    if openclaw.provider not in providers:
        raise ValueError(
            f"Unknown provider referenced by openclaw.provider: {openclaw.provider}"
        )

    poolside_raw = raw.get("poolside", {}) or {}
    poolside = PoolsideConfig(
        provider=poolside_raw.get("provider", codex.provider),
        model=poolside_raw.get("model"),
        api_url=poolside_raw.get("api_url"),
        api_key=poolside_raw.get("api_key"),
        token=poolside_raw.get("token"),
        config_path=poolside_raw.get("config_path", "~/.config/poolside/settings.yaml"),
        install_command=poolside_raw.get(
            "install_command",
            "curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh",
        ),
        windows_install_command=poolside_raw.get(
            "windows_install_command",
            "irm https://downloads.poolside.ai/pool/install.ps1 | iex",
        ),
    )
    if poolside.provider not in providers:
        raise ValueError(
            f"Unknown provider referenced by poolside.provider: {poolside.provider}"
        )

    def _load_command_policy(raw_policy: dict[str, Any]) -> dict[str, CommandPolicy]:
        policies = dict(DEFAULT_COMMAND_POLICIES)
        for cmd_name, entry in raw_policy.items():
            if isinstance(entry, dict):
                policies[cmd_name] = CommandPolicy(
                    enabled=bool(entry.get("enabled", True)),
                    visible=bool(entry.get("visible", True)),
                    permission=str(entry.get("permission", "everyone")),
                )
        return policies

    def _load_tool_policy(raw_policy: dict[str, Any]) -> ToolPolicy:
        return ToolPolicy(
            ai_auto_tools=list(raw_policy.get("ai_auto_tools", ToolPolicy().ai_auto_tools)),
            command_tools=list(raw_policy.get("command_tools", ToolPolicy().command_tools)),
            blocked_tools=list(raw_policy.get("blocked_tools", ToolPolicy().blocked_tools)),
            user_visible_tools=list(raw_policy.get("user_visible_tools", ToolPolicy().user_visible_tools)),
            require_admin_for=list(raw_policy.get("require_admin_for", ToolPolicy().require_admin_for)),
            require_owner_for=list(raw_policy.get("require_owner_for", ToolPolicy().require_owner_for)),
        )

    telegram_raw = raw.get("telegram", {}) or {}
    command_policy_raw = telegram_raw.get("command_policy", {}) or {}
    tool_policy_raw = telegram_raw.get("tool_policy", {}) or {}
    telegram_help_raw = telegram_raw.get("telegram_help", {}) or {}
    telegram = TelegramBotConfig(
        enabled=bool(telegram_raw.get("enabled", False)),
        bot_token=telegram_raw.get("bot_token"),
        allowed_chat_ids=[str(item) for item in (telegram_raw.get("allowed_chat_ids") or [])],
        owner_chat_ids=[str(item) for item in (telegram_raw.get("owner_chat_ids") or [])],
        admin_chat_ids=[str(item) for item in (telegram_raw.get("admin_chat_ids") or [])],
        allow_all_chats=bool(telegram_raw.get("allow_all_chats", False)),
        admin_pin_hash=telegram_raw.get("admin_pin_hash"),
        core_editing_enabled=bool(telegram_raw.get("core_editing_enabled", False)),
        require_owner_approval_for_core_changes=bool(telegram_raw.get("require_owner_approval_for_core_changes", True)),
        provider=str(telegram_raw.get("provider", codex.provider)),
        model=telegram_raw.get("model"),
        system_prompt=str(
            telegram_raw.get(
                "system_prompt",
                "You are a restricted Telegram bot powered by llama bridge. "
                "Answer helpfully, keep replies concise, use bridge tools only when clearly needed, "
                "and refuse unsafe or privileged actions.",
            )
        ),
        max_input_chars=max(200, int(telegram_raw.get("max_input_chars", 4000))),
        max_output_tokens=max(64, int(telegram_raw.get("max_output_tokens", 512))),
        poll_interval_seconds=max(0.5, float(telegram_raw.get("poll_interval_seconds", 2.0))),
        response_timeout_seconds=max(
            10.0, float(telegram_raw.get("response_timeout_seconds", 180.0))
        ),
        autonomous_enabled=bool(telegram_raw.get("autonomous_enabled", True)),
        autonomous_interval_seconds=max(
            60.0, float(telegram_raw.get("autonomous_interval_seconds", 1800.0))
        ),
        self_evolution_enabled=bool(telegram_raw.get("self_evolution_enabled", True)),
        self_evolution_min_events=max(1, int(telegram_raw.get("self_evolution_min_events", 3))),
        command_policy=_load_command_policy(command_policy_raw.get("commands", {}) if "commands" in command_policy_raw else command_policy_raw),
        tool_policy=_load_tool_policy(tool_policy_raw),
        telegram_help={
            "show_disabled_commands": bool(telegram_help_raw.get("show_disabled_commands", False)),
            "show_tool_list": bool(telegram_help_raw.get("show_tool_list", False)),
            "show_admin_commands_to_admins_only": bool(telegram_help_raw.get("show_admin_commands_to_admins_only", True)),
        },
    )
    if telegram.provider not in providers:
        raise ValueError(
            f"Unknown provider referenced by telegram.provider: {telegram.provider}"
        )

    vs_copilot_models = _load_vs_copilot_models(raw, providers, aliases, pi, codex)
    tools = _load_tool_config(raw)
    openwebui = _load_openwebui_config(raw, server, tools)

    bridge_config = BridgeConfig(
        server=server,
        ngrok=ngrok,
        providers=providers,
        anthropic_models=aliases,
        pi=pi,
        codex=codex,
        copilot_cli=copilot_cli,
        opencode=opencode,
        openclaw=openclaw,
        poolside=poolside,
        telegram=telegram,
        vs_copilot_models=vs_copilot_models,
        tools=tools,
        openwebui=openwebui,
        source_path=config_path,
    )
    write_claude_api_settings(
        config_path.parent / DEFAULT_API_SETTINGS_PATH.name,
        server=server,
        aliases=aliases,
        force=True,
    )
    return bridge_config


def _load_tool_config(raw: dict[str, Any]) -> ToolConfig:
    tools_raw = raw.get("tools", {}) or {}
    max_exposed = int(tools_raw.get("max_exposed", 8))
    if max_exposed < 1:
        raise ValueError("tools.max_exposed must be at least 1")
    confidence_threshold = float(tools_raw.get("confidence_threshold", 0.5))
    if not 0.0 <= confidence_threshold <= 10.0:
        raise ValueError("tools.confidence_threshold must be between 0 and 10")
    default_search_provider = str(tools_raw.get("default_search_provider", "tavily")).lower()
    if default_search_provider not in {"tavily", "serpapi"}:
        raise ValueError("tools.default_search_provider must be 'tavily' or 'serpapi'")
    cache_ttl_seconds = int(tools_raw.get("cache_ttl_seconds", 300))
    if cache_ttl_seconds < 0:
        raise ValueError("tools.cache_ttl_seconds must be non-negative")
    serpapi = _external_tool_provider(tools_raw.get("serpapi"), "https://serpapi.com/search")
    tavily = _external_tool_provider(tools_raw.get("tavily"), "https://api.tavily.com/search")
    if default_search_provider == "tavily" and not tavily.enabled and serpapi.enabled:
        default_search_provider = "serpapi"
    elif default_search_provider == "serpapi" and not serpapi.enabled and tavily.enabled:
        default_search_provider = "tavily"
    return ToolConfig(
        enabled=bool(tools_raw.get("enabled", True)),
        expose_http=bool(tools_raw.get("expose_http", True)),
        require_auth=bool(tools_raw.get("require_auth", True)),
        country=tools_raw.get("country"),
        include=list(tools_raw.get("include") or []),
        serpapi=serpapi,
        tavily=tavily,
        weather=_external_tool_provider(
            tools_raw.get("weather"),
            "https://api.open-meteo.com/v1/forecast",
            enabled=True,
        ),
        wikipedia=_external_tool_provider(
            tools_raw.get("wikipedia"),
            "https://en.wikipedia.org",
            enabled=True,
        ),
        max_exposed=max_exposed,
        relevance_filter=bool(tools_raw.get("relevance_filter", True)),
        force_for_keywords=bool(tools_raw.get("force_for_keywords", True)),
        confidence_threshold=confidence_threshold,
        log_outputs=bool(tools_raw.get("log_outputs", False)),
        default_search_provider=default_search_provider,
        cache_enabled=bool(tools_raw.get("cache_enabled", True)),
        cache_ttl_seconds=cache_ttl_seconds,
        tool_system_instructions=tools_raw.get("tool_system_instructions"),
        pi_system_instructions=tools_raw.get("pi_system_instructions"),
        # Tool management fields
        management_enabled=bool(tools_raw.get("management_enabled", True)),
        compact_manifest_enabled=bool(tools_raw.get("compact_manifest_enabled", True)),
        compact_manifest_max_tools=int(tools_raw.get("compact_manifest_max_tools", 20)),
        always_expose_management_tools=bool(tools_raw.get("always_expose_management_tools", True)),
        expose_full_schema_policy=str(tools_raw.get("expose_full_schema_policy", "relevant")),
        schema_memory_enabled=bool(tools_raw.get("schema_memory_enabled", True)),
        schema_memory_ttl_seconds=int(tools_raw.get("schema_memory_ttl_seconds", 86400)),
        schema_fetch_max_tools=int(tools_raw.get("schema_fetch_max_tools", 3)),
        compact_summary_max_chars=int(tools_raw.get("compact_summary_max_chars", 160)),
        full_schema_token_budget=int(tools_raw.get("full_schema_token_budget", 5000)),
        fallback_to_full_schemas_for_unsupported_clients=bool(
            tools_raw.get("fallback_to_full_schemas_for_unsupported_clients", True)
        ),
    )


def _external_tool_provider(
    value: Any,
    default_base_url: str,
    *,
    enabled: bool = False,
) -> ExternalToolProviderConfig:
    raw = value if isinstance(value, dict) else {}
    api_key = raw.get("api_key")
    auto_enabled = bool(
        api_key
        and isinstance(api_key, str)
        and api_key.strip()
        and not api_key.strip().startswith("${")
    )
    defaults = {
        key: item
        for key, item in raw.items()
        if key not in {"enabled", "api_key", "base_url", "defaults"}
    }
    defaults.update(raw.get("defaults", {}) or {})
    return ExternalToolProviderConfig(
        enabled=bool(raw.get("enabled", enabled or auto_enabled)),
        api_key=api_key,
        base_url=raw.get("base_url") or default_base_url,
        defaults=defaults,
    )


def _load_vs_copilot_models(
    raw: dict[str, Any],
    providers: dict[str, ProviderConfig],
    aliases: dict[str, ModelAlias],
    pi: PiConfig,
    codex: CodexConfig,
) -> list[VsCopilotModel]:
    raw_models = (raw.get("vs_copilot", {}) or {}).get("models")
    if raw_models is None:
        raw_models = _default_vs_copilot_models(providers, aliases, pi, codex)
    if not isinstance(raw_models, list):
        raise ValueError("vs_copilot.models must be a list")
    if not 1 <= len(raw_models) <= 3:
        raise ValueError("vs_copilot.models must contain between 1 and 3 models")

    models: list[VsCopilotModel] = []
    for index, value in enumerate(raw_models, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"vs_copilot.models[{index}] must be a mapping")
        name = str(value.get("name") or value.get("model") or "").strip()
        provider_name = str(value.get("provider") or "").strip()
        if not name:
            raise ValueError(f"vs_copilot.models[{index}] is missing name")
        if not provider_name:
            raise ValueError(f"vs_copilot.models[{index}] is missing provider")
        if provider_name not in providers:
            raise ValueError(
                f"Unknown provider referenced by vs_copilot.models[{index}].provider: "
                f"{provider_name}"
            )
        model = value.get("model")
        provider = providers[provider_name]
        if not model and not provider.default_model:
            raise ValueError(
                f"vs_copilot model '{name}' has no model and provider "
                f"'{provider_name}' has no default_model configured"
            )
        models.append(
            VsCopilotModel(
                name=name,
                provider=provider_name,
                model=model,
                context_size=int(value.get("context_size", 65536)),
                modified_at=value.get("modified_at"),
                size=int(value["size"]) if value.get("size") is not None else None,
                digest=value.get("digest"),
            )
        )
    return models


def _default_vs_copilot_models(
    providers: dict[str, ProviderConfig],
    aliases: dict[str, ModelAlias],
    pi: PiConfig,
    codex: CodexConfig,
) -> list[dict[str, Any]]:
    if codex.model:
        return [{"name": codex.model, "provider": codex.provider, "model": codex.model}]
    codex_provider = providers.get(codex.provider)
    if codex_provider and codex_provider.default_model:
        return [
            {
                "name": codex_provider.default_model,
                "provider": codex.provider,
                "model": codex_provider.default_model,
            }
        ]
    if pi.model:
        return [{"name": pi.model, "provider": pi.provider, "model": pi.model}]
    for alias_name in ("sonnet", "opus", "haiku"):
        alias = aliases.get(alias_name)
        if alias and alias.model:
            return [{"name": alias.model, "provider": alias.provider, "model": alias.model}]
        if alias and providers[alias.provider].default_model:
            model = providers[alias.provider].default_model
            return [{"name": model, "provider": alias.provider, "model": model}]
    alias = next(iter(aliases.values()))
    provider = providers[alias.provider]
    model = alias.model or provider.default_model or alias.alias
    return [{"name": model, "provider": alias.provider, "model": model}]


def _load_openwebui_config(raw: dict[str, Any], server: ServerConfig, tools: ToolConfig) -> OpenWebUIConfig:
    owui_raw = raw.get("openwebui", {}) or {}
    web_search_providers: dict[str, ExternalToolProviderConfig] = {}
    ow_providers_raw = owui_raw.get("web_search_providers", {}) or {}
    for name in ("ollama", "tavily", "serpapi", "searchapi"):
        p_raw = ow_providers_raw.get(name, {}) or {}
        web_search_providers[name] = ExternalToolProviderConfig(
            enabled=bool(p_raw.get("enabled", True)),
            api_key=p_raw.get("api_key"),
            base_url=p_raw.get("base_url"),
            defaults=p_raw.get("defaults", {}) or {},
        )
    return OpenWebUIConfig(
        enabled=bool(owui_raw.get("enabled", True)),
        host=str(owui_raw.get("host", "127.0.0.1")),
        port=int(owui_raw.get("port", 8080)),
        bridge_tools_port=int(owui_raw["bridge_tools_port"]) if owui_raw.get("bridge_tools_port") is not None else None,
        bridge_llm_only_port=int(owui_raw["bridge_llm_only_port"]) if owui_raw.get("bridge_llm_only_port") is not None else None,
        auth_enabled=bool(owui_raw.get("auth_enabled", False)),
        auto_login=bool(owui_raw.get("auto_login", True)),
        web_search_enabled=bool(owui_raw.get("web_search_enabled", tools.default_search_provider != "disabled" if hasattr(tools, 'default_search_provider') else True)),
        web_search_provider=str(owui_raw.get("web_search_provider", "ollama")),
        web_search_providers=web_search_providers,
        search_result_count=max(1, int(owui_raw.get("search_result_count", 3))),
        concurrent_requests=max(1, int(owui_raw.get("concurrent_requests", 1))),
        bypass_embedding_and_retrieval=bool(owui_raw.get("bypass_embedding_and_retrieval", False)),
        bypass_web_loader=bool(owui_raw.get("bypass_web_loader", False)),
        hf_token=owui_raw.get("hf_token"),
        openai_base_url_mode=str(owui_raw.get("openai_base_url_mode", "llm_only")),
        openwebui_data_dir=owui_raw.get("openwebui_data_dir"),
        extra_env=dict(owui_raw.get("extra_env", {}) or {}),
        preferred_env_name=str(owui_raw.get("preferred_env_name", "omx-open-webui")),
        preferred_python=owui_raw.get("preferred_python"),
        preferred_command=owui_raw.get("preferred_command"),
        auto_discover=bool(owui_raw.get("auto_discover", True)),
    )


def _default_config_template_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "config.example.yml",
        Path.cwd() / "config.example.yml",
    ]

    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.extend(
            [
                Path(bundle_dir) / "config.example.yml",
            ]
        )

    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            executable_dir / "config.example.yml",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find bundled config.example.yml template")
