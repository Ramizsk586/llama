from __future__ import annotations

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
    if (package_root / "pyproject.toml").exists() and (package_root / "env.yml").exists():
        return package_root

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

DEFAULT_CONFIG_TEMPLATE = """server:
  host: 127.0.0.1
  port: 8089
  auth_token: change-me

providers:
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

  ollama_cloud:
    type: ollama_cloud
    base_url: https://ollama.com
    api_key: ${OLLAMA_API_KEY}
    supports_tools: true
    default_model: qwen3.5:cloud

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

  openai_compatible:
    type: openai_compatible
    base_url: https://your-provider.example.com/v1
    api_key: ${CUSTOM_PROVIDER_API_KEY}
    supports_tools: true
    default_model: your-model

anthropic_models:
  haiku:
    provider: ollama_cloud
    model: qwen3.5:cloud
  sonnet:
    provider: ollama_cloud
    model: qwen3.5:cloud
  opus:
    provider: ollama_cloud
    model: qwen3.5:cloud
  small_fast:
    provider: ollama_cloud
    model: qwen3.5:cloud

pi:
  provider: ollama_cloud
  model: qwen3.5:cloud
  api: openai-completions
  config_dir: ~/.pi/agent
  install_package: "@mariozechner/pi-coding-agent"
  web_search: true

codex:
  provider: ollama_cloud
  model: qwen3.5:cloud
  config_path: ~/.codex/config.toml
  profile: llama_bridge
  install_package: "@openai/codex"

copilot_cli:
  provider: ollama_cloud
  model: qwen3.5:cloud
  wire_api: responses
  max_prompt_tokens: 65536
  max_output_tokens: 2048
  install_package: "@github/copilot"

opencode:
  provider: ollama_cloud
  model: qwen3.5:cloud
  install_package: "opencode-ai"

tools:
  enabled: true
  expose_http: true
  require_auth: true
  country: India
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
    - master_review
  # Tool selection and filtering (NEW)
  max_exposed: 8  # Limit tools in request (after filtering)
  relevance_filter: true  # Enable relevance-based filtering
  force_for_keywords: true  # Strongly prefer obvious tools for weather/time/latest/source queries
  confidence_threshold: 0.5  # Minimum relevance score (0-10)
  log_outputs: false  # Log tool arguments and results
  default_search_provider: tavily  # Prefer "tavily" or "serpapi"
  cache_enabled: true
  cache_ttl_seconds: 300
  tool_system_instructions: null
  # Pi CLI specific (NEW)
  pi_system_instructions: null  # Optional custom system instructions
  # External tool providers
  serpapi:
    enabled: false
    api_key: ${SERPAPI_API_KEY}
    base_url: https://serpapi.com/search
    defaults:
      engine: google
      num: 5
  tavily:
    enabled: false
    api_key: ${TAVILY_API_KEY}
    base_url: https://api.tavily.com/search
    defaults:
      search_depth: basic
      max_results: 5
  weather:
    enabled: true
    provider: open_meteo
    base_url: https://api.open-meteo.com/v1/forecast
    geocoding_url: https://geocoding-api.open-meteo.com/v1/search
  wikipedia:
    enabled: true
    language: en
    base_url: https://en.wikipedia.org

master_review:
  enabled: true
  run_after_deep_research: true
  mode: balanced
  groq:
    enabled: true
    base_url: https://api.groq.com/openai/v1
    model: llama-3.3-70b-versatile
    api_keys:
      - ${GROQ_API_KEY_1}
      - ${GROQ_API_KEY_2}
      - ${GROQ_API_KEY_3}
    timeout_seconds: 45
    max_retries_per_key: 2
    cooldown_seconds_after_429: 60
    cooldown_seconds_after_5xx: 20
    max_parallel_agents: 1
    requests_per_minute_per_key: 30
    tokens_per_minute_per_key: 12000
    requests_per_day_per_key: 1000
    tokens_per_day_per_key: 100000
    rate_limit_wait_seconds: 45
  sub_agents:
    spelling_grammar: true
    evidence_validity: true
    evidence_reliability: true
    citation_coverage: true
    neutrality_bias: true
    logic_consistency: true
    format_quality: true
    final_synthesis: true
  debate:
    enabled: true
    rounds: 2
    require_disagreement: true
    max_points_per_agent: 8
  output:
    return_review_trace: true
    return_revised_draft: true
    return_final_llm_instructions: true
    max_instruction_tokens: 1800

vs_copilot:
  models:
    - name: qwen3.5:cloud
      provider: ollama_cloud
      model: qwen3.5:cloud
      context_size: 65536
      modified_at: "2026-04-02T09:00:00-08:00"
      size: 1000000000
      digest: dummy-qwen35-cloud
    - name: local-llama
      provider: ollama_local
      model: llama3.1:8b
      context_size: 32768
      modified_at: "2026-04-02T09:00:00-08:00"
      size: 1000000000
      digest: dummy-local-llama
    - name: custom-openai-compatible
      provider: openai_compatible
      model: your-model
      context_size: 65536
      modified_at: "2026-04-02T09:00:00-08:00"
      size: 1000000000
      digest: dummy-custom-model
"""


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8089
    auth_token: str = "change-me"


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
class MasterGroqConfig:
    enabled: bool = True
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    api_keys: list[str] = field(default_factory=list)
    timeout_seconds: float = 45.0
    max_retries_per_key: int = 2
    cooldown_seconds_after_429: int = 60
    cooldown_seconds_after_5xx: int = 20
    max_parallel_agents: int = 1
    requests_per_minute_per_key: int = 30
    tokens_per_minute_per_key: int = 12000
    requests_per_day_per_key: int = 1000
    tokens_per_day_per_key: int = 100000
    rate_limit_wait_seconds: int = 45


@dataclass(slots=True)
class MasterSubAgentsConfig:
    spelling_grammar: bool = True
    evidence_validity: bool = True
    evidence_reliability: bool = True
    citation_coverage: bool = True
    neutrality_bias: bool = True
    logic_consistency: bool = True
    format_quality: bool = True
    final_synthesis: bool = True


@dataclass(slots=True)
class MasterDebateConfig:
    enabled: bool = True
    rounds: int = 2
    require_disagreement: bool = True
    max_points_per_agent: int = 8


@dataclass(slots=True)
class MasterOutputConfig:
    return_review_trace: bool = True
    return_revised_draft: bool = True
    return_final_llm_instructions: bool = True
    max_instruction_tokens: int = 1800


@dataclass(slots=True)
class MasterReviewConfig:
    enabled: bool = True
    run_after_deep_research: bool = True
    mode: str = "balanced"
    groq: MasterGroqConfig = field(default_factory=MasterGroqConfig)
    sub_agents: MasterSubAgentsConfig = field(default_factory=MasterSubAgentsConfig)
    debate: MasterDebateConfig = field(default_factory=MasterDebateConfig)
    output: MasterOutputConfig = field(default_factory=MasterOutputConfig)


@dataclass(slots=True)
class BridgeConfig:
    server: ServerConfig
    providers: dict[str, ProviderConfig]
    anthropic_models: dict[str, ModelAlias]
    pi: PiConfig
    codex: CodexConfig
    copilot_cli: CopilotCliConfig
    opencode: OpenCodeConfig
    vs_copilot_models: list[VsCopilotModel]
    tools: ToolConfig
    master_review: MasterReviewConfig
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

    preferred_aliases = ("sonnet", "opus", "haiku", "small_fast")
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

    preferred_aliases = ("sonnet", "opus", "haiku", "small_fast")
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

    preferred_aliases = ("sonnet", "opus", "haiku", "small_fast")
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

    preferred_aliases = ("sonnet", "opus", "haiku", "small_fast")
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

    preferred_aliases = ("sonnet", "opus", "haiku", "small_fast")
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
    small_fast_model = "small_fast" if "small_fast" in aliases else "haiku"
    haiku_model = "haiku" if "haiku" in aliases else small_fast_model
    sonnet_model = "sonnet" if "sonnet" in aliases else haiku_model
    opus_model = "opus" if "opus" in aliases else sonnet_model

    settings = _read_json_object(path) if path.exists() else {}
    env = dict(settings.get("env") or {})
    env.update(
        {
            "ANTHROPIC_BASE_URL": f"http://{server.host}:{server.port}",
            "ANTHROPIC_AUTH_TOKEN": server.auth_token,
            "API_TIMEOUT_MS": "3000000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_SMALL_FAST_MODEL": small_fast_model,
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
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=int(server_raw.get("port", 8089)),
        auth_token=server_raw.get("auth_token", "change-me"),
    )
    if server.auth_token == "change-me" and server.host not in {"127.0.0.1", "localhost", "::1"}:
        warnings.warn(
            "server.auth_token is still 'change-me' while server.host is not localhost.",
            RuntimeWarning,
            stacklevel=2,
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

    vs_copilot_models = _load_vs_copilot_models(raw, providers, aliases, pi, codex)
    tools = _load_tool_config(raw)
    master_review = _load_master_review_config(raw)

    bridge_config = BridgeConfig(
        server=server,
        providers=providers,
        anthropic_models=aliases,
        pi=pi,
        codex=codex,
        copilot_cli=copilot_cli,
        opencode=opencode,
        vs_copilot_models=vs_copilot_models,
        tools=tools,
        master_review=master_review,
        source_path=config_path,
    )
    write_claude_api_settings(
        config_path.parent / DEFAULT_API_SETTINGS_PATH.name,
        server=server,
        aliases=aliases,
        force=True,
    )
    return bridge_config


def _load_master_review_config(raw: dict[str, Any]) -> MasterReviewConfig:
    review_raw = raw.get("master_review", {}) or {}
    mode = str(review_raw.get("mode", "balanced")).lower()
    if mode not in {"fast", "balanced", "strict"}:
        warnings.warn(
            "master_review.mode must be one of: fast, balanced, strict; using balanced.",
            RuntimeWarning,
            stacklevel=2,
        )
        mode = "balanced"

    groq_raw = review_raw.get("groq", {}) or {}
    api_keys = [
        str(key).strip()
        for key in (groq_raw.get("api_keys") or [])
        if str(key or "").strip() and not str(key).strip().startswith("${")
    ]
    enabled = bool(review_raw.get("enabled", True))
    groq_enabled = bool(groq_raw.get("enabled", True))
    if enabled and groq_enabled and not api_keys:
        warnings.warn(
            "master_review.enabled is true but no real Groq keys are configured; local fallback review will be used.",
            RuntimeWarning,
            stacklevel=2,
        )
    groq = MasterGroqConfig(
        enabled=groq_enabled,
        base_url=str(groq_raw.get("base_url", "https://api.groq.com/openai/v1")).rstrip("/"),
        model=str(groq_raw.get("model", "llama-3.3-70b-versatile")),
        api_keys=api_keys,
        timeout_seconds=float(groq_raw.get("timeout_seconds", 45.0)),
        max_retries_per_key=max(0, int(groq_raw.get("max_retries_per_key", 2))),
        cooldown_seconds_after_429=max(0, int(groq_raw.get("cooldown_seconds_after_429", 60))),
        cooldown_seconds_after_5xx=max(0, int(groq_raw.get("cooldown_seconds_after_5xx", 20))),
        max_parallel_agents=max(1, int(groq_raw.get("max_parallel_agents", 1))),
        requests_per_minute_per_key=max(1, int(groq_raw.get("requests_per_minute_per_key", 30))),
        tokens_per_minute_per_key=max(1000, int(groq_raw.get("tokens_per_minute_per_key", 12000))),
        requests_per_day_per_key=max(1, int(groq_raw.get("requests_per_day_per_key", 1000))),
        tokens_per_day_per_key=max(1000, int(groq_raw.get("tokens_per_day_per_key", 100000))),
        rate_limit_wait_seconds=max(0, int(groq_raw.get("rate_limit_wait_seconds", 45))),
    )

    sub_raw = review_raw.get("sub_agents", {}) or {}
    sub_agents = MasterSubAgentsConfig(
        spelling_grammar=bool(sub_raw.get("spelling_grammar", True)),
        evidence_validity=bool(sub_raw.get("evidence_validity", True)),
        evidence_reliability=bool(sub_raw.get("evidence_reliability", True)),
        citation_coverage=bool(sub_raw.get("citation_coverage", True)),
        neutrality_bias=bool(sub_raw.get("neutrality_bias", True)),
        logic_consistency=bool(sub_raw.get("logic_consistency", True)),
        format_quality=bool(sub_raw.get("format_quality", True)),
        final_synthesis=bool(sub_raw.get("final_synthesis", True)),
    )

    debate_raw = review_raw.get("debate", {}) or {}
    debate = MasterDebateConfig(
        enabled=bool(debate_raw.get("enabled", True)),
        rounds=max(0, min(2, int(debate_raw.get("rounds", 2)))),
        require_disagreement=bool(debate_raw.get("require_disagreement", True)),
        max_points_per_agent=max(1, int(debate_raw.get("max_points_per_agent", 8))),
    )

    output_raw = review_raw.get("output", {}) or {}
    output = MasterOutputConfig(
        return_review_trace=bool(output_raw.get("return_review_trace", True)),
        return_revised_draft=bool(output_raw.get("return_revised_draft", True)),
        return_final_llm_instructions=bool(output_raw.get("return_final_llm_instructions", True)),
        max_instruction_tokens=max(200, int(output_raw.get("max_instruction_tokens", 1800))),
    )

    return MasterReviewConfig(
        enabled=enabled,
        run_after_deep_research=bool(review_raw.get("run_after_deep_research", True)),
        mode=mode,
        groq=groq,
        sub_agents=sub_agents,
        debate=debate,
        output=output,
    )


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
    for alias_name in ("sonnet", "opus", "haiku", "small_fast"):
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
