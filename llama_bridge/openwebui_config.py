from __future__ import annotations

import os
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import (
    BridgeConfig,
    OpenWebUIConfig,
    DEFAULT_CONFIG_PATH,
    load_config,
)


def load_openwebui_config(config_path: Path | None = None) -> OpenWebUIConfig:
    config = load_config(config_path)
    return config.openwebui


def save_openwebui_config(
    owui: OpenWebUIConfig,
    config_path: Path | None = None,
) -> Path:
    target = config_path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}

    raw.setdefault("openwebui", {})
    owui_raw = raw["openwebui"]

    owui_raw["enabled"] = owui.enabled
    owui_raw["host"] = owui.host
    owui_raw["port"] = owui.port
    if owui.bridge_tools_port is not None:
        owui_raw["bridge_tools_port"] = owui.bridge_tools_port
    else:
        owui_raw.pop("bridge_tools_port", None)
    if owui.bridge_llm_only_port is not None:
        owui_raw["bridge_llm_only_port"] = owui.bridge_llm_only_port
    else:
        owui_raw.pop("bridge_llm_only_port", None)
    owui_raw["auth_enabled"] = owui.auth_enabled
    owui_raw["auto_login"] = owui.auto_login
    owui_raw["web_search_enabled"] = owui.web_search_enabled
    owui_raw["web_search_provider"] = owui.web_search_provider
    owui_raw["search_result_count"] = owui.search_result_count
    owui_raw["concurrent_requests"] = owui.concurrent_requests
    owui_raw["bypass_embedding_and_retrieval"] = owui.bypass_embedding_and_retrieval
    owui_raw["bypass_web_loader"] = owui.bypass_web_loader
    if owui.hf_token:
        owui_raw["hf_token"] = owui.hf_token
    else:
        owui_raw.pop("hf_token", None)
    owui_raw["openai_base_url_mode"] = owui.openai_base_url_mode
    if owui.openwebui_data_dir:
        owui_raw["openwebui_data_dir"] = owui.openwebui_data_dir
    else:
        owui_raw.pop("openwebui_data_dir", None)

    providers_raw = owui_raw.setdefault("web_search_providers", {})
    for provider_name in ("ollama", "tavily", "serpapi", "searchapi"):
        pcfg = owui.web_search_providers.get(provider_name)
        if pcfg is None:
            continue
        p_raw = providers_raw.setdefault(provider_name, {})
        p_raw["enabled"] = pcfg.enabled
        if pcfg.api_key:
            p_raw["api_key"] = pcfg.api_key
        elif "api_key" in p_raw:
            del p_raw["api_key"]
        if pcfg.base_url:
            p_raw["base_url"] = pcfg.base_url
        if pcfg.defaults:
            p_raw["defaults"] = pcfg.defaults

    if owui.extra_env:
        owui_raw["extra_env"] = dict(owui.extra_env)
    else:
        owui_raw.pop("extra_env", None)

    owui_raw["preferred_env_name"] = owui.preferred_env_name
    if owui.preferred_python:
        owui_raw["preferred_python"] = owui.preferred_python
    else:
        owui_raw.pop("preferred_python", None)
    if owui.preferred_command:
        owui_raw["preferred_command"] = owui.preferred_command
    else:
        owui_raw.pop("preferred_command", None)
    owui_raw["auto_discover"] = owui.auto_discover

    target.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return target


OPENWEBUI_ENV_VARS = {
    "auth": {
        "WEBUI_AUTH": lambda c: "True" if c.auth_enabled else "False",
        "WEBUI_AUTO_LOGIN": lambda c: "False" if c.auth_enabled else "True",
    },
    "cors": {
        "CORS_ALLOW_ORIGIN": lambda c: f"http://{c.host}:{c.port}",
    },
    "hf": {
        "HF_TOKEN": lambda c: c.hf_token or "",
        "HUGGING_FACE_HUB_TOKEN": lambda c: c.hf_token or "",
    },
    "data": {
        "DATA_DIR": lambda c: str(c.openwebui_data_dir) if c.openwebui_data_dir else "",
    },
}


def generate_openwebui_env(
    owui: OpenWebUIConfig,
    bridge: BridgeConfig | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}

    if owui.auth_enabled:
        env["WEBUI_AUTH"] = "True"
        env["WEBUI_AUTO_LOGIN"] = "False"
    else:
        env["WEBUI_AUTH"] = "False"
        env["WEBUI_AUTO_LOGIN"] = "True"

    env["CORS_ALLOW_ORIGIN"] = f"http://{owui.host}:{owui.port}"

    if owui.hf_token:
        env["HF_TOKEN"] = owui.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = owui.hf_token

    if owui.openwebui_data_dir:
        env["DATA_DIR"] = owui.openwebui_data_dir

    llm_port = owui.bridge_llm_only_port
    if llm_port is None and bridge is not None:
        llm_port = bridge.server.openwebui_port
    if llm_port is None:
        llm_port = 11534

    bridge_host = "127.0.0.1"
    if bridge is not None:
        bridge_host = bridge.server.host

    mode = owui.openai_base_url_mode
    if mode == "llm_only":
        base_url = f"http://{bridge_host}:{llm_port}"
    elif mode == "tools":
        tools_port = owui.bridge_tools_port
        if tools_port is None and bridge is not None:
            tools_port = bridge.server.port
        if tools_port is None:
            tools_port = 11434
        base_url = f"http://{bridge_host}:{tools_port}"
    else:
        base_url = ""

    if base_url:
        env["OLLAMA_BASE_URL"] = base_url
        env["OPENAI_BASE_URL"] = f"{base_url}/v1"

    if owui.web_search_enabled:
        env["WEBUI_SEARCH_ENABLED"] = "True"
        provider = owui.web_search_provider
        env["WEBUI_SEARCH_PROVIDER"] = provider

        pcfg = owui.web_search_providers.get(provider)
        api_key = pcfg.api_key if pcfg else None

        if provider == "ollama" and api_key:
            env["OLLAMA_API_KEY"] = api_key
        elif provider == "tavily":
            env["TAVILY_API_KEY"] = api_key or ""
        elif provider == "serpapi":
            env["SERPAPI_API_KEY"] = api_key or ""
        elif provider == "searchapi":
            env["SEARCHAPI_API_KEY"] = api_key or ""
    else:
        env["WEBUI_SEARCH_ENABLED"] = "False"

    env["WEBUI_SECRET_KEY"] = "llama-bridge-webui-secret-key-change-in-production"

    env["BYPass_EMBEDDING_AND_RETRIEVAL"] = str(owui.bypass_embedding_and_retrieval)
    env["BYPass_WEB_LOADER"] = str(owui.bypass_web_loader)

    env["RAG_CONCURRENT_FILE_PROCESSING"] = str(owui.concurrent_requests)
    env["SEARCH_RESULT_COUNT"] = str(owui.search_result_count)

    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    env.update(owui.extra_env)

    return {k: v for k, v in env.items() if v}


VALID_SEARCH_PROVIDERS = {"ollama", "tavily", "serpapi", "searchapi", "disabled"}


def validate_web_search_provider(provider: str) -> str | None:
    if provider not in VALID_SEARCH_PROVIDERS:
        return f"Unknown search provider '{provider}'. Valid: {', '.join(sorted(VALID_SEARCH_PROVIDERS))}"

    api_key_var = {
        "tavily": "TAVILY_API_KEY",
        "serpapi": "SERPAPI_API_KEY",
        "searchapi": "SEARCHAPI_API_KEY",
    }.get(provider)

    if api_key_var and not os.environ.get(api_key_var):
        return f"{provider} requires {api_key_var} environment variable to be set"

    return None


def test_search_provider(
    provider: str,
    api_key: str | None = None,
    query: str = "Open WebUI test search",
) -> tuple[bool, str]:
    if provider == "disabled":
        return False, "Web search is disabled"

    if provider not in VALID_SEARCH_PROVIDERS:
        return False, f"Unknown search provider: {provider}"

    if provider == "ollama":
        if not api_key:
            return False, "Ollama API key is required for ollama cloud web search"
        try:
            import httpx
            resp = httpx.post(
                "https://ollama.com/api/search",
                json={"query": query, "max_results": 1},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            if resp.status_code == 401:
                return False, "Ollama API key is invalid (401 Unauthorized)"
            if resp.status_code == 429:
                return False, "Ollama API rate limited (429). Try again later."
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            return True, f"Success: {len(results)} results returned"
        except httpx.HTTPStatusError as exc:
            return False, f"Ollama API error: {exc.response.status_code} - {exc.response.text[:200]}"
        except httpx.RequestError as exc:
            return False, f"Ollama API unreachable: {exc}"
        except Exception as exc:
            return False, f"Ollama test failed: {exc}"

    provider_urls = {
        "tavily": "https://api.tavily.com/search",
        "serpapi": "https://serpapi.com/search",
        "searchapi": "https://www.searchapi.io/api/v1/search",
    }
    url = provider_urls.get(provider)
    if not url or not api_key:
        return False, f"Missing API key for {provider}"

    try:
        import httpx

        if provider == "tavily":
            payload = {"api_key": api_key, "query": query, "max_results": 1, "search_depth": "basic"}
        elif provider == "serpapi":
            payload = {"api_key": api_key, "q": query, "engine": "google", "num": 1}
        elif provider == "searchapi":
            payload = {"api_key": api_key, "q": query, "engine": "google", "num": 1}
        else:
            return False, f"Unsupported provider: {provider}"

        resp = httpx.post(url, json=payload, timeout=15.0)
        if resp.status_code == 401 or resp.status_code == 403:
            return False, f"{provider} API key is invalid ({resp.status_code})"
        if resp.status_code == 429:
            return False, f"{provider} rate limited (429)"
        resp.raise_for_status()
        data = resp.json()

        if provider == "tavily":
            results = data.get("results", []) if isinstance(data, dict) else []
        elif provider == "serpapi":
            results = data.get("organic_results", []) if isinstance(data, dict) else []
        elif provider == "searchapi":
            results = data.get("organic_results", []) if isinstance(data, dict) else []
        else:
            results = []

        count = len(results) if isinstance(results, list) else 0
        return True, f"Success: {count} results returned"
    except httpx.HTTPStatusError as exc:
        return False, f"{provider} API error: {exc.response.status_code} - {exc.response.text[:200]}"
    except httpx.RequestError as exc:
        return False, f"{provider} API unreachable: {exc}"
    except Exception as exc:
        return False, f"{provider} test failed: {exc}"


def get_effective_ports(owui: OpenWebUIConfig, bridge: BridgeConfig | None = None) -> dict[str, int]:
    tools_port = owui.bridge_tools_port
    if tools_port is None and bridge is not None:
        tools_port = bridge.server.port
    if tools_port is None:
        tools_port = 11434

    llm_port = owui.bridge_llm_only_port
    if llm_port is None and bridge is not None:
        llm_port = bridge.server.openwebui_port
    if llm_port is None:
        llm_port = 11534

    return {
        "openwebui": owui.port,
        "bridge_tools": tools_port,
        "bridge_llm_only": llm_port,
    }


def check_openwebui_installed() -> tuple[bool, str | None]:
    import shutil
    cmd = shutil.which("open-webui")
    if cmd:
        return True, cmd

    try:
        import open_webui
        return True, str(Path(open_webui.__file__).resolve().parent)
    except ImportError:
        pass

    try:
        import openwebui
        return True, str(Path(openwebui.__file__).resolve().parent)
    except ImportError:
        pass

    return False, None


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def write_pid(pid_path: Path, pid: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid), encoding="utf-8")


def clear_pid(pid_path: Path) -> None:
    pid_path.unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return bool(kernel32.GetLastError() == 5)
        except Exception:
            try:
                import psutil
                return psutil.pid_exists(pid)
            except ImportError:
                pass
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def port_in_use(port: int, host: str = "127.0.0.1") -> tuple[bool, str]:
    import socket
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True, _get_process_on_port(port, host)
    except (OSError, socket.timeout):
        return False, ""


def _get_process_on_port(port: int, host: str = "127.0.0.1") -> str:
    if os.name == "nt":
        try:
            result = subprocess_run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            for line in (result.stdout or "").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and f"{host}:{port}" in parts[1]:
                    pid = parts[4]
                    return f"PID {pid}"
            return f"port {port} is in use"
        except Exception:
            return f"port {port} is in use"
    return f"port {port} is in use"


def subprocess_run(*args, **kwargs):
    import subprocess as sp
    return sp.run(*args, **kwargs)


CONDA_ENV_NAME = "omx-open-webui"

_CONDA_PYTHON_CACHE: str | None = None
_CONDA_PYTHON_CACHED = False


def _find_conda_python() -> str | None:
    global _CONDA_PYTHON_CACHE, _CONDA_PYTHON_CACHED
    if _CONDA_PYTHON_CACHED:
        return _CONDA_PYTHON_CACHE

    user = os.environ.get("USERPROFILE", "")
    local = os.environ.get("LOCALAPPDATA", "")

    candidates: list[Path] = []
    for base in (user, local):
        if not base:
            continue
        for variant in ("miniconda3", "Miniconda3", "anaconda3", "Anaconda3"):
            candidates.append(Path(base) / variant / "envs" / CONDA_ENV_NAME / "python.exe")

    for candidate in candidates:
        if candidate.is_file():
            resolved = str(candidate.resolve())
            _CONDA_PYTHON_CACHE = resolved
            _CONDA_PYTHON_CACHED = True
            return resolved

    _CONDA_PYTHON_CACHE = None
    _CONDA_PYTHON_CACHED = True
    return None


def get_conda_python_path() -> str | None:
    return _find_conda_python()


# ── Open WebUI Discovery ─────────────────────────────────────────────────

@dataclass
class OpenWebUIDiscovery:
    installed: bool = False
    source: str = ""
    python_exe: Path | None = None
    command: Path | None = None
    package_path: Path | None = None
    version: str | None = None
    env_name: str | None = None
    env_path: Path | None = None
    details: list[str] = field(default_factory=list)

    def to_card_status(self) -> tuple[str, str, str]:
        """Returns (title, subtitle, status) for GUI card."""
        if not self.installed:
            return "Open WebUI", "Package missing", "Missing"
        return "Open WebUI", "Installed", "OK"

    def to_short_label(self) -> str:
        if not self.installed:
            return "Not installed"
        return self.source


_DISCOVERY_CACHE: OpenWebUIDiscovery | None = None


def _get_conda_roots() -> list[Path]:
    """Return common Conda installation root directories."""
    roots: list[Path] = []
    for env_var in ("USERPROFILE", "LOCALAPPDATA", "PROGRAMDATA"):
        val = os.environ.get(env_var)
        if not val:
            continue
        base = Path(val)
        for variant in ("miniconda3", "Miniconda3", "anaconda3", "Anaconda3"):
            roots.append(base / variant)
    return roots


def _get_conda_env_dirs(root: Path) -> list[Path]:
    """Return conda env directories under root/envs if it exists."""
    envs_dir = root / "envs"
    if envs_dir.is_dir():
        try:
            return sorted([d for d in envs_dir.iterdir() if d.is_dir()])
        except PermissionError:
            pass
    return []


def _run_probe_script(python_exe: Path, timeout: int = 8) -> dict | None:
    """Run import probe against a Python executable. Returns parsed JSON or None."""
    import subprocess
    probe = (
        'import json, sys, pathlib; '
        'try:\n'
        '    import open_webui\n'
        '    ver = getattr(open_webui, "__version__", None)\n'
        '    print(json.dumps({\n'
        '        "ok": True,\n'
        '        "python": sys.executable,\n'
        '        "package": str(pathlib.Path(open_webui.__file__).resolve().parent),\n'
        '        "version": ver\n'
        '    }))\n'
        'except Exception as e:\n'
        '    print(json.dumps({"ok": False, "error": str(e)}))\n'
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", probe],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if data.get("ok"):
                return data
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def probe_python_for_openwebui(python_exe: Path) -> OpenWebUIDiscovery | None:
    """Check if a specific Python executable has Open WebUI installed."""
    data = _run_probe_script(python_exe)
    if not data:
        return None
    return OpenWebUIDiscovery(
        installed=True,
        source="python",
        python_exe=python_exe.resolve(),
        package_path=Path(data["package"]) if data.get("package") else None,
        version=data.get("version"),
        details=[f"Import probe OK: {data.get('python', '')}"],
    )


def probe_openwebui_command(command: Path) -> OpenWebUIDiscovery | None:
    """Check if an open-webui executable exists and responds."""
    import subprocess
    if not command.is_file():
        return None
    name = command.name.lower()
    if "open-webui" not in name and "openwebui" not in name:
        return None
    try:
        result = subprocess.run(
            [str(command), "--help"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            result = subprocess.run(
                [str(command), "--version"],
                capture_output=True, text=True, timeout=5,
            )
        details = [f"Executable responds: {command.name}"]
        # Infer env python if inside Scripts folder
        python_exe = None
        env_path = None
        env_name = None
        if command.parent.name.lower() == "scripts":
            maybe_env = command.parent.parent
            maybe_python = maybe_env / "python.exe"
            if maybe_python.is_file():
                python_exe = maybe_python.resolve()
                env_path = maybe_env.resolve()
                env_name = maybe_env.parent.name
        return OpenWebUIDiscovery(
            installed=True,
            source="PATH",
            command=command.resolve(),
            python_exe=python_exe,
            env_path=env_path,
            env_name=env_name,
            details=details,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def _check_current_python() -> OpenWebUIDiscovery | None:
    """Check if the currently running Python has open_webui."""
    try:
        import open_webui
        return OpenWebUIDiscovery(
            installed=True,
            source="current Python",
            python_exe=Path(sys.executable).resolve(),
            package_path=Path(open_webui.__file__).resolve().parent,
            version=getattr(open_webui, "__version__", None),
            details=[f"Current Python import: {sys.executable}"],
        )
    except ImportError:
        pass
    try:
        import openwebui
        return OpenWebUIDiscovery(
            installed=True,
            source="current Python",
            python_exe=Path(sys.executable).resolve(),
            package_path=Path(openwebui.__file__).resolve().parent,
            details=[f"Current Python import (alt): {sys.executable}"],
        )
    except ImportError:
        pass
    return None


def _check_active_conda_env() -> OpenWebUIDiscovery | None:
    """Check CONDA_PREFIX env var."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    python_exe = Path(prefix) / "python.exe"
    if not python_exe.is_file():
        return None
    return probe_python_for_openwebui(python_exe)


def _check_preferred_conda_env(env_name: str) -> OpenWebUIDiscovery | None:
    """Search for a specific Conda env by name."""
    for root in _get_conda_roots():
        env_dir = root / "envs" / env_name
        python_exe = env_dir / "python.exe"
        if python_exe.is_file():
            result = probe_python_for_openwebui(python_exe)
            if result:
                result.source = "Conda env"
                result.env_name = env_name
                result.env_path = env_dir.resolve()
                result.details.append(f"Found in Conda env: {env_dir}")
                return result
            # Check Scripts too even if import failed
            cmd = env_dir / "Scripts" / "open-webui.exe"
            result = probe_openwebui_command(cmd)
            if result:
                result.source = "Conda env"
                result.env_name = env_name
                result.env_path = env_dir.resolve()
                result.details.append(f"Found via Scripts in: {env_dir}")
                return result
    return None


def _check_any_conda_env() -> OpenWebUIDiscovery | None:
    """Scan all Conda envs across all roots."""
    for root in _get_conda_roots():
        for env_dir in _get_conda_env_dirs(root):
            python_exe = env_dir / "python.exe"
            if python_exe.is_file():
                result = probe_python_for_openwebui(python_exe)
                if result:
                    result.source = "Conda env"
                    result.env_name = env_dir.name
                    result.env_path = env_dir.resolve()
                    result.details.append(f"Found in env: {env_dir}")
                    return result
                # Check Scripts
                cmd = env_dir / "Scripts" / "open-webui.exe"
                result = probe_openwebui_command(cmd)
                if result:
                    result.source = "Conda env"
                    result.env_name = env_dir.name
                    result.env_path = env_dir.resolve()
                    result.details.append(f"Found via Scripts: {env_dir}")
                    return result
    return None


def _check_conda_env_list() -> OpenWebUIDiscovery | None:
    """Run conda env list --json and check each env."""
    import subprocess
    conda = _find_conda_exe()
    if not conda:
        return None
    try:
        result = subprocess.run(
            [str(conda), "env", "list", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        envs = data.get("envs", [])
        for env_path_str in envs:
            env_dir = Path(env_path_str)
            python_exe = env_dir / "python.exe"
            if python_exe.is_file():
                disc = probe_python_for_openwebui(python_exe)
                if disc:
                    disc.source = "Conda env"
                    disc.env_name = env_dir.name
                    disc.env_path = env_dir.resolve()
                    disc.details.append(f"Found from conda env list: {env_dir}")
                    return disc
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return None


def _find_conda_exe() -> Path | None:
    """Locate conda executable."""
    import shutil
    cmd = shutil.which("conda")
    if cmd:
        return Path(cmd)
    for root in _get_conda_roots():
        for exe_name in ("conda.exe", "conda.bat"):
            candidate = root / "Scripts" / exe_name
            if candidate.is_file():
                return candidate
            candidate = root / "condabin" / exe_name
            if candidate.is_file():
                return candidate
    return None


def _check_path_commands() -> OpenWebUIDiscovery | None:
    """Check open-webui executables on PATH."""
    import shutil
    for name in ("open-webui", "open-webui.exe"):
        cmd = shutil.which(name)
        if cmd:
            result = probe_openwebui_command(Path(cmd))
            if result:
                return result
    return None


def _check_pipx_installs() -> OpenWebUIDiscovery | None:
    """Check pipx install locations."""
    candidates = [
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "open-webui.exe",
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "pipx" / "venvs" / "open-webui" / "Scripts" / "open-webui.exe",
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "pipx" / "venvs" / "open-webui" / "Scripts" / "python.exe",
        Path(os.environ.get("APPDATA", "")) / "pipx" / "venvs" / "open-webui" / "Scripts" / "open-webui.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "pipx" / "venvs" / "open-webui" / "Scripts" / "open-webui.exe",
    ]
    for cand in candidates:
        if cand.is_file():
            if cand.name == "python.exe":
                result = probe_python_for_openwebui(cand)
                if result:
                    result.source = "pipx"
                    result.details.append(f"Found pipx Python: {cand}")
                    return result
            else:
                result = probe_openwebui_command(cand)
                if result:
                    result.source = "pipx"
                    result.details.append(f"Found pipx exe: {cand}")
                    return result
    return None


def _check_user_site() -> OpenWebUIDiscovery | None:
    """Check user site-packages locations."""
    import subprocess
    # Check python -m site --user-site
    try:
        result = subprocess.run(
            [sys.executable, "-m", "site", "--user-site"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            user_site = result.stdout.strip()
            if user_site:
                # Check if open_webui is in user site
                test = subprocess.run(
                    [sys.executable, "-c", "import open_webui"],
                    capture_output=True, text=True, timeout=5,
                    env={**os.environ, "PYTHONPATH": user_site},
                )
                if test.returncode == 0:
                    return probe_python_for_openwebui(Path(sys.executable))
    except (subprocess.TimeoutExpired, OSError):
        pass
    # Check installed Python versions under LOCALAPPDATA
    local_python = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    if local_python.is_dir():
        try:
            for py_dir in sorted(local_python.iterdir(), reverse=True):
                python_exe = py_dir / "python.exe"
                if python_exe.is_file():
                    result = probe_python_for_openwebui(python_exe)
                    if result:
                        result.source = "Python install"
                        result.details.append(f"Found in: {py_dir}")
                        return result
        except PermissionError:
            pass
    # Check C:\Python* and C:\Program Files\Python*
    for root_dir in (Path("C:\\"), Path(os.environ.get("ProgramFiles", "C:\\Program Files"))):
        if not root_dir.is_dir():
            continue
        try:
            for item in root_dir.iterdir():
                name = item.name.lower()
                if name.startswith("python") and item.is_dir():
                    python_exe = item / "python.exe"
                    if python_exe.is_file():
                        result = probe_python_for_openwebui(python_exe)
                        if result:
                            result.source = "Python install"
                            result.details.append(f"Found in: {item}")
                            return result
        except (PermissionError, OSError):
            pass
    return None


def _check_manual_folders() -> OpenWebUIDiscovery | None:
    """Check common manual install folders (limited depth)."""
    from .config import DEFAULT_CONFIG_DIR
    user = os.environ.get("USERPROFILE", "")
    roots: list[Path] = [
        DEFAULT_CONFIG_DIR,
        DEFAULT_CONFIG_DIR.parent,
        Path(user) / "OpenWebUI",
        Path(user) / "open-webui",
        Path(user) / "Downloads" / "open-webui",
        Path(user) / "Desktop" / "open-webui",
        Path(user) / "Documents" / "open-webui",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        # Check for venv subfolders
        for venv_name in (".venv", "venv", "env"):
            python_exe = root / venv_name / "Scripts" / "python.exe"
            if python_exe.is_file():
                result = probe_python_for_openwebui(python_exe)
                if result:
                    result.source = "venv"
                    result.env_path = (root / venv_name).resolve()
                    result.env_name = venv_name
                    result.details.append(f"Found venv: {root / venv_name}")
                    return result
        # Check for Scripts/open-webui.exe directly
        cmd = root / "Scripts" / "open-webui.exe"
        result = probe_openwebui_command(cmd)
        if result:
            result.source = "manual folder"
            result.details.append(f"Found command in: {root}")
            return result
    return None


def discover_openwebui(
    preferred_env_name: str = "omx-open-webui",
    preferred_python: str | None = None,
    preferred_command: str | None = None,
) -> OpenWebUIDiscovery:
    """Full discovery of Open WebUI across all possible install locations.

    Search order:
      1. Saved preferred_python (if provided)
      2. Saved preferred_command (if provided)
      3. Active CONDA_PREFIX
      4. Preferred Conda env by name
      5. Any Conda env (by scanning)
      6. conda env list --json
      7. Current Python
      8. PATH commands
      9. pipx installs
      10. User site-packages / Python installs
      11. Manual folders
    """
    global _DISCOVERY_CACHE

    steps: list[tuple[str, str, callable]] = [
        ("preferred Python", "Checking saved preferred_python\u2026", lambda: _check_preferred_python(preferred_python) if preferred_python else None),
        ("preferred command", "Checking saved preferred_command\u2026", lambda: _check_preferred_command(preferred_command) if preferred_command else None),
        ("active CONDA_PREFIX", "Checking active CONDA_PREFIX\u2026", _check_active_conda_env),
        ("preferred Conda env", f"Checking preferred env {preferred_env_name}\u2026", lambda: _check_preferred_conda_env(preferred_env_name)),
        ("any Conda env", "Scanning Conda envs\u2026", _check_any_conda_env),
        ("conda env list", "Running conda env list\u2026", _check_conda_env_list),
        ("current Python", "Checking current Python\u2026", _check_current_python),
        ("PATH commands", "Checking PATH for open-webui\u2026", _check_path_commands),
        ("pipx installs", "Checking pipx installs\u2026", _check_pipx_installs),
        ("user site-packages", "Checking user Python installs\u2026", _check_user_site),
        ("manual folders", "Checking manual folders\u2026", _check_manual_folders),
    ]

    all_details: list[str] = []

    for source_name, log_msg, check_fn in steps:
        all_details.append(f"  [{source_name}] {log_msg}")
        try:
            result = check_fn()
            if result and result.installed:
                result.source = source_name
                result.details = all_details + result.details
                _DISCOVERY_CACHE = result
                return result
        except Exception:
            all_details.append(f"  [{source_name}] error during check")

    not_found = OpenWebUIDiscovery(
        installed=False,
        source="not found",
        details=all_details + ["Open WebUI not found after full search"],
    )
    _DISCOVERY_CACHE = not_found
    return not_found


def _check_preferred_python(python_path: str) -> OpenWebUIDiscovery | None:
    """Check a saved preferred Python path."""
    py = Path(python_path)
    if py.is_file():
        result = probe_python_for_openwebui(py)
        if result:
            result.source = "saved Python"
            result.details.append(f"Preferred Python: {py}")
            return result
    return None


def _check_preferred_command(cmd_path: str) -> OpenWebUIDiscovery | None:
    """Check a saved preferred command path."""
    cmd = Path(cmd_path)
    if cmd.is_file():
        result = probe_openwebui_command(cmd)
        if result:
            result.source = "saved command"
            result.details.append(f"Preferred command: {cmd}")
            return result
    return None


def clear_discovery_cache() -> None:
    """Clear the cached discovery result."""
    global _DISCOVERY_CACHE
    _DISCOVERY_CACHE = None


def get_cached_discovery() -> OpenWebUIDiscovery | None:
    """Return cached discovery result, or None if not yet discovered."""
    return _DISCOVERY_CACHE


def check_openwebui_installed_discovery(
    preferred_env_name: str = "omx-open-webui",
    preferred_python: str | None = None,
    preferred_command: str | None = None,
) -> tuple[bool, OpenWebUIDiscovery]:
    """Combined check: uses discovery to determine installation status."""
    disc = discover_openwebui(
        preferred_env_name=preferred_env_name,
        preferred_python=preferred_python,
        preferred_command=preferred_command,
    )
    return disc.installed, disc
