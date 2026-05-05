from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any


log = logging.getLogger(__name__)


def _text_from_content(content: Any, preserve_json: bool = True) -> str:
    """
    Convert content to text, optionally preserving JSON structures.
    
    Args:
        content: Content to convert
        preserve_json: If True, preserve JSON objects as formatted text
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(part for part in text_parts if part)
    return ""


def _preserve_tool_result_json(content: Any) -> str:
    """
    Convert tool result content to text while preserving JSON structure.
    Handles both structured tool results and plain text.
    """
    if isinstance(content, dict):
        # Tool result with ok/data/error structure
        if "ok" in content and "tool" in content:
            # Structured tool result - preserve the full JSON
            return json.dumps(content, indent=2, ensure_ascii=True)
        # Other dict - preserve as JSON
        return json.dumps(content, indent=2, ensure_ascii=True)
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Handle both old format (text blocks) and new format (mixed content)
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(part for part in text_parts if part)
    
    return str(content) if content else ""


def _system_to_text(system: Any) -> str | None:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            block.get("text", "") for block in system if block.get("type") == "text"
        )
    return None


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
        )
    return converted


def anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if not choice:
        return None
    if isinstance(choice, str):
        return choice
    if choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    if choice.get("type") == "any":
        return "required"
    if choice.get("type") == "auto":
        return "auto"
    if choice.get("type") == "none":
        return "none"
    return None


def anthropic_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        pending_tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        tool_call_map: dict[str, dict] = {}  # Map tool_use_id to tool_call
        
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_call = {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
                pending_tool_calls.append(tool_call)
                tool_call_map[block["id"]] = tool_call
            elif block_type == "tool_result":
                # Preserve structured JSON in tool results
                result_content = block.get("content", "")
                if isinstance(result_content, (dict, list)):
                    result_text = _preserve_tool_result_json(result_content)
                else:
                    result_text = _text_from_content(result_content)

                tool_result = {
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": result_text,
                }
                
                # If no matching tool_call exists, we need to create a synthetic one
                if block["tool_use_id"] not in tool_call_map:
                    # Will handle this after the loop
                    pass
                
                tool_results.append(tool_result)

        # Handle case where we have tool_results but no matching tool_calls
        if tool_results and not pending_tool_calls:
            # Create synthetic tool_calls for orphaned tool_results
            synthetic_calls = []
            for tr in tool_results:
                synthetic_calls.append({
                    "id": tr["tool_call_id"],
                    "type": "function",
                    "function": {"name": "unknown_tool", "arguments": "{}"},
                })
            converted.append({
                "role": "assistant",
                "content": "",
                "tool_calls": synthetic_calls,
            })
        elif pending_tool_calls:
            converted.append({
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part),
                "tool_calls": pending_tool_calls,
            })
        elif text_parts:
            converted.append({"role": role, "content": "\n".join(text_parts)})

        converted.extend(tool_results)
    return converted


def anthropic_request_to_openai(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": anthropic_messages_to_openai(body.get("messages", [])),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_tokens": body.get("max_tokens", 2048),
        "stop": body.get("stop_sequences"),
    }

    system_text = _system_to_text(body.get("system"))
    if system_text:
        payload["messages"].insert(0, {"role": "system", "content": system_text})

    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if choice is not None:
            payload["tool_choice"] = choice

    return {key: value for key, value in payload.items() if value is not None}


def openai_response_to_anthropic(
    data: dict[str, Any], alias: str, requested_model: str
) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content_blocks: list[dict[str, Any]] = []

    text = _openai_content_to_text(message.get("content"))
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        arguments = tool_call["function"].get("arguments") or "{}"
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": _json_object(arguments),
            }
        )

    finish_reason = choice.get("finish_reason")
    stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
    if finish_reason == "length":
        stop_reason = "max_tokens"

    usage = data.get("usage", {})
    response_model = requested_model or alias
    return {
        "id": data.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": response_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def estimate_input_tokens(body: dict[str, Any]) -> int:
    total_text = []
    system = _system_to_text(body.get("system"))
    if system:
        total_text.append(system)
    for message in body.get("messages", []):
        total_text.append(_text_from_content(message.get("content", "")))
    for tool in body.get("tools") or []:
        total_text.append(json.dumps(tool, ensure_ascii=True, sort_keys=True))
    if body.get("tool_choice") is not None:
        total_text.append(
            json.dumps(body["tool_choice"], ensure_ascii=True, sort_keys=True)
        )
    text = "\n".join(part for part in total_text if part)
    return max(1, len(text) // 4)


def _openai_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return ""


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        log.warning("Could not parse tool call arguments as JSON: %r", value)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def anthropic_stream_prefix(requested_model: str) -> list[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    return [
        sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": requested_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "created_at": created,
                },
            },
        )
    ]
