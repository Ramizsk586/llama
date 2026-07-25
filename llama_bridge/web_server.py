import os
import sys
import httpx
import yaml
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Llama Bridge Dashboard Server")

# Locate env.yml and web_ui directory
BRIDGE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BRIDGE_DIR / "env.yml"
WEB_UI_DIR = BRIDGE_DIR / "web_ui"

# Helper to import config utilities from llama_bridge
sys.path.insert(0, str(BRIDGE_DIR.parent))
from llama_bridge.config import write_config_data

@app.get("/api/config")
def get_config():
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Configuration file env.yml not found.")
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return raw
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")

@app.post("/api/config")
def save_config(config_data: Dict[str, Any]):
    try:
        write_config_data(CONFIG_PATH, config_data)
        return {"status": "ok", "message": "Configuration saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

@app.get("/api/status")
async def get_bridge_status():
    # Load backend server port from config
    port = 8089
    host = "127.0.0.1"
    auth_token = None
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        server_cfg = raw.get("server", {})
        port = int(server_cfg.get("port", 8089))
        host = server_cfg.get("host", "127.0.0.1")
        auth_token = server_cfg.get("auth_token")
    except Exception:
        pass

    backend_url = f"http://{host}:{port}/health"
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(backend_url, headers=headers)
            if resp.status_code == 200:
                return {
                    "running": True,
                    "url": f"http://{host}:{port}",
                    "details": resp.json()
                }
    except Exception:
        pass

    return {"running": False, "url": f"http://{host}:{port}"}

@app.post("/api/test-provider")
async def test_provider(provider_cfg: Dict[str, Any]):
    # Tests connection to a provider base_url
    url = provider_cfg.get("base_url")
    if not url:
        return {"success": False, "message": "Missing base_url"}
    
    headers = provider_cfg.get("headers", {}) or {}
    api_key = provider_cfg.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        # Check standard models endpoint first
        test_url = f"{url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(test_url, headers=headers)
            if resp.status_code < 400:
                return {"success": True, "message": "Connected successfully"}
            
            # Fallback to base url checking
            resp2 = await client.get(url, headers=headers)
            if resp2.status_code < 400:
                return {"success": True, "message": "Connected successfully"}
            return {"success": False, "message": f"Server returned HTTP {resp2.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/api/usage")
def get_usage_stats():
    import json
    usage_path = CONFIG_PATH.parent / "llama.usage.json"

    # Defaults
    stats = {
        "token_usage": "0",
        "sessions": 0,
        "messages": 0,
        "active_days": 0,
        "streak": 0,
        "favorite_model": "None",
        "favorite_model_share": 0,
        "heatmap": [],
        "tokens_per_day": {},
        "model_usage": []
    }

    if not usage_path.exists():
        return stats

    try:
        data = json.loads(usage_path.read_text(encoding="utf-8"))
        daily_history = data.get("daily_history", {})

        if not daily_history:
            return stats

        from datetime import datetime, timedelta

        # 1. Active days
        dates = list(daily_history.keys())
        stats["active_days"] = len(dates)

        # 2. Streak calculation
        if dates:
            parsed_dates = sorted([datetime.strptime(d, "%Y-%m-%d").date() for d in dates])
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)

            streak = 0
            if today in parsed_dates or yesterday in parsed_dates:
                current = today if today in parsed_dates else yesterday
                while current in parsed_dates:
                    streak += 1
                    current -= timedelta(days=1)
            stats["streak"] = streak

        # 3. Overall Totals and Model Breakdown
        total_tokens = 0
        total_sessions = 0
        total_messages = 0
        model_tokens = {}
        heatmap_data = []
        tokens_per_day = {}

        for date_str, day_entry in sorted(daily_history.items()):
            day_tokens = 0
            day_messages = 0
            day_sessions = day_entry.get("sessions", 1)
            total_sessions += day_sessions

            for model_name, model_data in day_entry.items():
                if model_name in ("timestamps", "sessions"):
                    continue
                in_tok = model_data.get("input_tokens", 0)
                out_tok = model_data.get("output_tokens", 0)
                reqs = model_data.get("requests", 0)

                tok_sum = in_tok + out_tok
                day_tokens += tok_sum
                day_messages += reqs

                model_tokens[model_name] = model_tokens.get(model_name, 0) + tok_sum

                # Stacked bar data: tokens_per_day[date][model] = tokens
                date_entry = tokens_per_day.setdefault(date_str, {})
                date_entry[model_name] = date_entry.get(model_name, 0) + tok_sum

            total_tokens += day_tokens
            total_messages += day_messages

            # Heatmap data: array of {date: str, count: requests}
            heatmap_data.append({
                "date": date_str,
                "count": day_messages
            })

        # Formatted total tokens
        if total_tokens >= 1_000_000:
            stats["token_usage"] = f"{total_tokens / 1_000_000:.1f}M"
        elif total_tokens >= 1_000:
            stats["token_usage"] = f"{total_tokens / 1_000:.1f}K"
        else:
            stats["token_usage"] = str(total_tokens)

        stats["messages"] = total_messages
        stats["sessions"] = total_sessions
        stats["heatmap"] = heatmap_data
        stats["tokens_per_day"] = tokens_per_day

        # 4. Favorite model & Donut chart data
        model_usage = []
        favorite_model = "None"
        max_tokens = 0

        for model_name, tokens in sorted(model_tokens.items(), key=lambda x: x[1], reverse=True):
            model_usage.append({
                "model": model_name,
                "tokens": tokens,
                "percentage": int((tokens / total_tokens * 100)) if total_tokens > 0 else 0
            })
            if tokens > max_tokens:
                max_tokens = tokens
                favorite_model = model_name

        stats["favorite_model"] = favorite_model
        if total_tokens > 0:
            stats["favorite_model_share"] = int((max_tokens / total_tokens * 100))
        stats["model_usage"] = model_usage

        return stats
    except Exception as e:
        return stats

@app.get("/api/provider/{provider_key}/models")
async def get_provider_models(provider_key: str):
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="env.yml not found")

    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        providers = raw.get("providers", {})
        if provider_key not in providers:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_key}' not found.")

        p = providers[provider_key]
        p_type = p.get("type", "openai_compatible")
        base_url = p.get("base_url", "")
        api_key = p.get("api_key")
        disabled_models = p.get("disabled_models", []) or []

        models = []

        if p_type == "antigravity":
            models = ["gemini-3.5-pro-agent", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        elif p_type == "ollama_local":
            url = f"{base_url.rstrip('/')}/api/tags"
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            except Exception:
                pass
        else:
            url = f"{base_url.rstrip('/')}/models"
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict) and "data" in data:
                            models = [m.get("id") for m in data["data"] if m.get("id")]
                        elif isinstance(data, list):
                            models = [m.get("id") if isinstance(m, dict) else str(m) for m in data]
            except Exception:
                pass

        default_model = p.get("default_model")
        all_unique = set(models)
        if default_model:
            all_unique.add(default_model)
        for m in disabled_models:
            all_unique.add(m)

        result = []
        for m in sorted(all_unique):
            result.append({
                "id": m,
                "visible": m not in disabled_models
            })

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/provider/{provider_key}/models")
async def save_provider_models(provider_key: str, disabled_models: list[str]):
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="env.yml not found")

    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        providers = raw.setdefault("providers", {})
        if provider_key not in providers:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_key}' not found.")

        providers[provider_key]["disabled_models"] = disabled_models
        write_config_data(CONFIG_PATH, raw)
        return {"status": "ok", "message": "Model visibility settings saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend static assets
if WEB_UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_UI_DIR), html=True), name="static")
else:
    @app.get("/")
    def no_ui():
        return {"error": "web_ui directory not found"}
