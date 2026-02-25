from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from backend.agents import AgentExecutor, MultiAgentManager

ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"
DB_JSON_PATH = Path(os.getenv("DB_JSON_PATH", str(ROOT_DIR / "DB.json"))).resolve()


def _default_state() -> Dict[str, Any]:
    return {
        "meta": {
            "next_llm_config_id": 1,
            "next_db_config_id": 1,
            "next_agent_id": 1,
            "next_thread_id": 1,
        },
        "llm_config": [],
        "db_config": [],
        "agents": [],
        "chat_threads": [],
    }


def _default_llm_config(record_id: int) -> Dict[str, Any]:
    return {
        "id": record_id,
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model_name": "llama3",
        "api_key": None,
    }


class JsonStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = RLock()
        self._initialize()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self.path.exists():
                self._write_unlocked(_default_state())

            state = self._normalize_unlocked(self._read_unlocked())
            self._write_unlocked(state)

    def _read_unlocked(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return _default_state()

    def _write_unlocked(self, data: Dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix=f"{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _normalize_unlocked(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized = _default_state()

        meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
        for key in normalized["meta"]:
            value = meta.get(key)
            normalized["meta"][key] = value if isinstance(value, int) and value > 0 else normalized["meta"][key]

        for key in ("llm_config", "db_config", "agents", "chat_threads"):
            value = state.get(key)
            normalized[key] = value if isinstance(value, list) else []

        # Ensure default LLM config exists.
        if len(normalized["llm_config"]) == 0:
            cfg_id = normalized["meta"]["next_llm_config_id"]
            normalized["llm_config"].append(_default_llm_config(cfg_id))
            normalized["meta"]["next_llm_config_id"] = cfg_id + 1

        # Keep ID counters coherent with existing records.
        normalized["meta"]["next_llm_config_id"] = max(
            normalized["meta"]["next_llm_config_id"],
            1 + max((int(item.get("id", 0)) for item in normalized["llm_config"]), default=0),
        )
        normalized["meta"]["next_db_config_id"] = max(
            normalized["meta"]["next_db_config_id"],
            1 + max((int(item.get("id", 0)) for item in normalized["db_config"]), default=0),
        )
        normalized["meta"]["next_agent_id"] = max(
            normalized["meta"]["next_agent_id"],
            1 + max((int(item.get("id", 0)) for item in normalized["agents"]), default=0),
        )
        normalized["meta"]["next_thread_id"] = max(
            normalized["meta"]["next_thread_id"],
            1 + max((int(item.get("id", 0)) for item in normalized["chat_threads"]), default=0),
        )

        return normalized

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return self._normalize_unlocked(self._read_unlocked())

    def mutate(self, fn: Callable[[Dict[str, Any]], Any]) -> Any:
        with self._lock:
            state = self._normalize_unlocked(self._read_unlocked())
            result = fn(state)
            self._write_unlocked(state)
            return result


def _next_id(state: Dict[str, Any], key: str) -> int:
    next_id = int(state["meta"][key])
    state["meta"][key] = next_id + 1
    return next_id


store = JsonStore(DB_JSON_PATH)
app = FastAPI(title="AI Data Agents Backend", version="2.2.0")


def parse_agent_config(agent_row: Dict[str, Any]) -> Dict[str, Any]:
    parsed_config: Dict[str, Any] = {}
    raw = agent_row.get("config")
    if isinstance(raw, str) and raw:
        try:
            parsed_config = json.loads(raw)
        except Exception:
            parsed_config = {}
    elif isinstance(raw, dict):
        parsed_config = raw

    return {
        **parsed_config,
        "role": agent_row.get("role"),
        "persona": agent_row.get("persona"),
        "objectives": agent_row.get("objectives"),
        "system_prompt": agent_row.get("system_prompt"),
    }


def build_agent_record(payload: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    db_config_id = payload.get("db_config_id")
    return {
        "id": agent_id,
        "name": payload.get("name"),
        "role": payload.get("role"),
        "objectives": payload.get("objectives"),
        "persona": payload.get("persona"),
        "tools": payload.get("tools"),
        "memory_settings": payload.get("memory_settings"),
        "system_prompt": payload.get("system_prompt") or "",
        "db_config_id": int(db_config_id) if db_config_id not in (None, "") else None,
        "agent_type": payload.get("agent_type") or "custom",
        "config": json.dumps(payload.get("config")) if payload.get("config") is not None else None,
    }


def build_db_config_record(payload: Dict[str, Any], config_id: int) -> Dict[str, Any]:
    return {
        "id": config_id,
        "type": payload.get("type") or "clickhouse",
        "host": payload.get("host") or "",
        "port": int(payload.get("port") or 0),
        "username": payload.get("username") or "",
        "password": payload.get("password") or "",
        "database_name": payload.get("database_name") or "",
    }


def _deserialize_agent_config(raw_config: Any) -> Any:
    if isinstance(raw_config, dict):
        return raw_config
    if isinstance(raw_config, str) and raw_config.strip():
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
                return parsed
        except Exception:
            return {}
    return {}


def serialize_agent_export(agent_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": agent_row.get("id"),
        "name": agent_row.get("name"),
        "role": agent_row.get("role"),
        "objectives": agent_row.get("objectives"),
        "persona": agent_row.get("persona"),
        "tools": agent_row.get("tools"),
        "memory_settings": agent_row.get("memory_settings"),
        "system_prompt": agent_row.get("system_prompt") or "",
        "db_config_id": agent_row.get("db_config_id"),
        "agent_type": agent_row.get("agent_type") or "custom",
        "config": _deserialize_agent_config(agent_row.get("config")),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_thread_record(payload: Dict[str, Any], thread_id: int) -> Dict[str, Any]:
    now = _now_iso()
    raw_title = str(payload.get("title") or "").strip()
    return {
        "id": thread_id,
        "agent_id": _safe_int(payload.get("agent_id")),
        "title": raw_title or f"Discussion {thread_id}",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def summarize_thread(thread: Dict[str, Any]) -> Dict[str, Any]:
    messages = thread.get("messages") if isinstance(thread.get("messages"), list) else []
    last_message = messages[-1] if messages else {}
    preview = str(last_message.get("content") or "").replace("\n", " ").strip()
    if len(preview) > 120:
        preview = f"{preview[:120]}..."
    return {
        "id": thread.get("id"),
        "agent_id": thread.get("agent_id"),
        "title": thread.get("title") or f"Discussion {thread.get('id')}",
        "created_at": thread.get("created_at"),
        "updated_at": thread.get("updated_at"),
        "message_count": len(messages),
        "last_message_preview": preview,
    }


def _format_memory_context(messages: List[Dict[str, Any]], max_messages: int = 16) -> str:
    if not messages:
        return ""

    selected = messages[-max_messages:]
    lines: List[str] = ["Conversation memory (most recent first):"]
    for message in selected:
        role = str(message.get("role") or "user").upper()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"- {role}: {content}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _find_thread(state: Dict[str, Any], thread_id: int) -> Optional[Dict[str, Any]]:
    return next((item for item in state["chat_threads"] if int(item.get("id", 0)) == thread_id), None)


def _append_message(thread: Dict[str, Any], role: str, content: str, details: Any = None) -> None:
    now = _now_iso()
    messages = thread.get("messages")
    if not isinstance(messages, list):
        messages = []
        thread["messages"] = messages

    message_record: Dict[str, Any] = {
        "role": role,
        "content": content,
        "created_at": now,
    }
    if details is not None:
        message_record["details"] = details
    messages.append(message_record)
    thread["updated_at"] = now


def _auto_title_thread_if_needed(thread: Dict[str, Any], user_message: str) -> None:
    current_title = str(thread.get("title") or "").strip()
    default_prefixes = ("discussion ", "new discussion", "nouvelle discussion")
    should_replace = (
        not current_title
        or any(current_title.lower().startswith(prefix) for prefix in default_prefixes)
    )

    if not should_replace:
        return

    normalized = " ".join(user_message.strip().split())
    if not normalized:
        return

    max_len = 72
    thread["title"] = normalized if len(normalized) <= max_len else f"{normalized[:max_len].rstrip()}..."


def build_llm(llm_config: Dict[str, Any]) -> Any:
    provider = llm_config.get("provider")
    if provider == "ollama":
        return ChatOllama(
            base_url=llm_config.get("base_url"),
            model=llm_config.get("model_name"),
        )

    if provider == "http":
        return ChatOpenAI(
            base_url=llm_config.get("base_url"),
            model=llm_config.get("model_name"),
            api_key=llm_config.get("api_key") or "dummy-key",
        )

    raise HTTPException(status_code=400, detail="Unsupported LLM provider")


@app.get("/api/config/llm")
async def get_llm_config() -> JSONResponse:
    state = store.read()
    return JSONResponse(content=state["llm_config"][-1])


@app.post("/api/config/llm")
async def save_llm_config(request: Request) -> JSONResponse:
    payload = await request.json()

    def mutate(state: Dict[str, Any]) -> None:
        config_id = _next_id(state, "next_llm_config_id")
        state["llm_config"].append(
            {
                "id": config_id,
                "provider": payload.get("provider"),
                "base_url": payload.get("base_url"),
                "model_name": payload.get("model_name"),
                "api_key": payload.get("api_key") or None,
            }
        )

    store.mutate(mutate)
    return JSONResponse(content={"success": True})


@app.post("/api/config/llm/test")
async def test_llm_config(request: Request) -> JSONResponse:
    payload = await request.json()
    provider = payload.get("provider")
    base_url = str(payload.get("base_url") or "").rstrip("/")
    api_key = payload.get("api_key")

    try:
        if provider == "ollama":
            response = await asyncio.to_thread(requests.get, f"{base_url}/api/tags", timeout=15)
            if not response.ok:
                raise RuntimeError("Failed to connect to Ollama")
            data = response.json()
            models = [model.get("name") for model in data.get("models", [])]
            return JSONResponse(content={"success": True, "models": models})

        if provider == "http":
            headers: Dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = await asyncio.to_thread(requests.get, f"{base_url}/v1/models", headers=headers, timeout=20)
            if not response.ok:
                raise RuntimeError("Failed to connect to HTTP LLM")
            data = response.json()
            models = [model.get("id") for model in data.get("data", [])]
            return JSONResponse(content={"success": True, "models": models})

        return JSONResponse(status_code=400, content={"success": False, "error": "Unsupported provider"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.get("/api/config/db")
async def get_db_configs() -> JSONResponse:
    state = store.read()
    return JSONResponse(content=state["db_config"])


@app.post("/api/config/db")
async def create_db_config(request: Request) -> JSONResponse:
    payload = await request.json()

    try:
        def mutate(state: Dict[str, Any]) -> None:
            config_id = _next_id(state, "next_db_config_id")
            state["db_config"].append(build_db_config_record(payload, config_id))

        store.mutate(mutate)
        return JSONResponse(content={"success": True})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.delete("/api/config/db/{config_id}")
async def delete_db_config(config_id: int) -> JSONResponse:
    def mutate(state: Dict[str, Any]) -> None:
        state["db_config"] = [item for item in state["db_config"] if int(item.get("id", 0)) != config_id]

    store.mutate(mutate)
    return JSONResponse(content={"success": True})


@app.post("/api/config/db/test")
async def test_db_config(request: Request) -> JSONResponse:
    payload = await request.json()
    host = payload.get("host")
    port = payload.get("port")
    db_type = payload.get("type")

    if not host or not port:
        return JSONResponse(status_code=400, content={"success": False, "error": "Host and port are required"})

    await asyncio.sleep(1)
    return JSONResponse(content={"success": True, "message": f"Successfully connected to {db_type} at {host}:{port}"})


@app.get("/api/agents")
async def get_agents() -> JSONResponse:
    state = store.read()
    return JSONResponse(content=state["agents"])


@app.get("/api/agents/export")
async def export_agents() -> JSONResponse:
    state = store.read()
    payload = {
        "format": "react_agents_export_v1",
        "exported_at": _now_iso(),
        "db_config": state.get("db_config", []),
        "agents": [serialize_agent_export(agent) for agent in state.get("agents", [])],
    }
    return JSONResponse(content=payload)


@app.post("/api/agents/import")
async def import_agents(request: Request) -> JSONResponse:
    payload = await request.json()
    imported_agents = payload.get("agents")
    imported_db_configs = payload.get("db_config")

    if not isinstance(imported_agents, list):
        return JSONResponse(status_code=400, content={"error": "Invalid payload: agents must be an array"})

    try:
        def mutate(state: Dict[str, Any]) -> Dict[str, int]:
            db_id_map: Dict[int, int] = {}

            if isinstance(imported_db_configs, list):
                state["db_config"] = []
                for db_item in imported_db_configs:
                    if not isinstance(db_item, dict):
                        continue
                    config_id = _next_id(state, "next_db_config_id")
                    old_id = _safe_int(db_item.get("id"))
                    if old_id is not None:
                        db_id_map[old_id] = config_id
                    state["db_config"].append(build_db_config_record(db_item, config_id))

            existing_db_ids = {int(item.get("id", 0)) for item in state.get("db_config", [])}

            state["agents"] = []
            state["chat_threads"] = []

            imported_count = 0
            for item in imported_agents:
                if not isinstance(item, dict):
                    continue

                raw_db_id = _safe_int(item.get("db_config_id"))
                if raw_db_id is None:
                    mapped_db_id = None
                elif raw_db_id in db_id_map:
                    mapped_db_id = db_id_map[raw_db_id]
                elif raw_db_id in existing_db_ids:
                    mapped_db_id = raw_db_id
                else:
                    mapped_db_id = None

                agent_payload = {
                    "name": item.get("name"),
                    "role": item.get("role"),
                    "objectives": item.get("objectives"),
                    "persona": item.get("persona"),
                    "tools": item.get("tools"),
                    "memory_settings": item.get("memory_settings"),
                    "system_prompt": item.get("system_prompt") or "",
                    "db_config_id": mapped_db_id,
                    "agent_type": item.get("agent_type") or "custom",
                    "config": _deserialize_agent_config(item.get("config")),
                }

                agent_id = _next_id(state, "next_agent_id")
                state["agents"].append(build_agent_record(agent_payload, agent_id))
                imported_count += 1

            return {
                "agents_imported": imported_count,
                "db_configs_imported": len(state.get("db_config", [])) if isinstance(imported_db_configs, list) else 0,
                "threads_reset": 1,
            }

        result = store.mutate(mutate)
        return JSONResponse(content={"success": True, **result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/agents")
async def create_agent(request: Request) -> JSONResponse:
    payload = await request.json()

    try:
        def mutate(state: Dict[str, Any]) -> None:
            agent_id = _next_id(state, "next_agent_id")
            state["agents"].append(build_agent_record(payload, agent_id))

        store.mutate(mutate)
        return JSONResponse(content={"success": True})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: int, request: Request) -> JSONResponse:
    payload = await request.json()

    try:
        def mutate(state: Dict[str, Any]) -> bool:
            for index, item in enumerate(state["agents"]):
                if int(item.get("id", 0)) == agent_id:
                    state["agents"][index] = build_agent_record(payload, agent_id)
                    return True
            return False

        updated = store.mutate(mutate)
        if not updated:
            return JSONResponse(status_code=404, content={"error": "Agent not found"})

        return JSONResponse(content={"success": True})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: int) -> JSONResponse:
    def mutate(state: Dict[str, Any]) -> None:
        state["agents"] = [item for item in state["agents"] if int(item.get("id", 0)) != agent_id]
        state["chat_threads"] = [item for item in state["chat_threads"] if int(item.get("agent_id") or -1) != agent_id]

    store.mutate(mutate)
    return JSONResponse(content={"success": True})


@app.get("/api/chat/threads")
async def get_chat_threads(agent_id: Optional[int] = None) -> JSONResponse:
    state = store.read()
    threads = state.get("chat_threads", [])

    if agent_id is not None:
        threads = [thread for thread in threads if int(thread.get("agent_id") or 0) == agent_id]

    summaries = [summarize_thread(thread) for thread in threads]
    summaries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return JSONResponse(content=summaries)


@app.post("/api/chat/threads")
async def create_chat_thread(request: Request) -> JSONResponse:
    payload = await request.json()

    try:
        def mutate(state: Dict[str, Any]) -> Dict[str, Any]:
            thread_id = _next_id(state, "next_thread_id")
            thread = build_thread_record(payload, thread_id)
            state["chat_threads"].append(thread)
            return thread

        created = store.mutate(mutate)
        return JSONResponse(content=created)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/chat/threads/{thread_id}")
async def get_chat_thread(thread_id: int) -> JSONResponse:
    state = store.read()
    thread = _find_thread(state, thread_id)
    if not thread:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    return JSONResponse(content=thread)


@app.post("/api/chat/threads/{thread_id}/clear")
async def clear_chat_thread(thread_id: int) -> JSONResponse:
    def mutate(state: Dict[str, Any]) -> bool:
        thread = _find_thread(state, thread_id)
        if not thread:
            return False
        thread["messages"] = []
        thread["updated_at"] = _now_iso()
        return True

    cleared = store.mutate(mutate)
    if not cleared:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    return JSONResponse(content={"success": True})


@app.delete("/api/chat/threads/{thread_id}")
async def delete_chat_thread(thread_id: int) -> JSONResponse:
    def mutate(state: Dict[str, Any]) -> None:
        state["chat_threads"] = [item for item in state["chat_threads"] if int(item.get("id", 0)) != thread_id]

    store.mutate(mutate)
    return JSONResponse(content={"success": True})


@app.post("/api/chat/{agent_id}")
async def chat_agent(agent_id: int, request: Request):
    payload = await request.json()
    message = str(payload.get("message") or "").strip()
    requested_thread_id = _safe_int(payload.get("threadId"))
    requested_thread_title = str(payload.get("threadTitle") or "").strip()
    is_streaming = request.query_params.get("stream") == "true"

    if not message:
        return JSONResponse(status_code=400, content={"error": "Message is required"})

    try:
        state = store.read()
        agent = next((a for a in state["agents"] if int(a.get("id", 0)) == agent_id), None)
        if not agent:
            return JSONResponse(status_code=404, content={"error": "Agent not found"})

        llm_config = state["llm_config"][-1] if state["llm_config"] else None
        if not llm_config:
            return JSONResponse(status_code=400, content={"error": "LLM configuration not found"})

        llm = build_llm(llm_config)
        full_config = parse_agent_config(agent)
        agent_name = agent.get("name") or agent.get("agent_type") or "agent"

        try:
            def mutate_user_message(state_mut: Dict[str, Any]) -> Dict[str, Any]:
                thread = _find_thread(state_mut, requested_thread_id) if requested_thread_id is not None else None

                if thread is None:
                    new_thread_id = _next_id(state_mut, "next_thread_id")
                    thread = build_thread_record(
                        {
                            "agent_id": agent_id,
                            "title": requested_thread_title or f"Discussion {new_thread_id}",
                        },
                        new_thread_id,
                    )
                    state_mut["chat_threads"].append(thread)

                thread_agent_id = _safe_int(thread.get("agent_id"))
                if thread_agent_id is None:
                    thread["agent_id"] = agent_id
                elif thread_agent_id != agent_id:
                    raise ValueError("Thread belongs to another agent")

                _auto_title_thread_if_needed(thread, message)
                _append_message(thread, "user", message)

                messages = thread.get("messages") if isinstance(thread.get("messages"), list) else []
                return {"thread_id": int(thread.get("id")), "messages": messages}

            thread_write_result = store.mutate(mutate_user_message)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        resolved_thread_id = int(thread_write_result["thread_id"])
        all_messages = thread_write_result.get("messages") if isinstance(thread_write_result.get("messages"), list) else []
        previous_messages = all_messages[:-1] if len(all_messages) > 0 else []
        memory_context = _format_memory_context(previous_messages)
        execution_input = f"{memory_context}\n\nCurrent user message:\n{message}" if memory_context else message

        def persist_assistant_message(content: str, details: Any = None) -> None:
            def mutate_assistant_message(state_mut: Dict[str, Any]) -> None:
                thread = _find_thread(state_mut, resolved_thread_id)
                if not thread:
                    return
                _append_message(thread, "assistant", content, details)

            store.mutate(mutate_assistant_message)

        if agent.get("agent_type") == "manager":
            all_agents = [a for a in state["agents"] if a.get("agent_type") != "manager"]

            if is_streaming:
                queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

                def on_step(step: Dict[str, Any]) -> None:
                    queue.put_nowait({"type": "step", "step": step})

                async def worker() -> None:
                    try:
                        result = await MultiAgentManager.run_stream(
                            execution_input,
                            all_agents,
                            llm,
                            full_config,
                            on_step=on_step,
                        )
                        response_text = str(result.get("answer") or "")
                        persist_assistant_message(
                            response_text,
                            {
                                "steps": result.get("steps"),
                                "agent_type": "manager",
                            },
                        )
                        await queue.put(
                            {
                                "type": "result",
                                "response": response_text,
                                "steps": result.get("steps"),
                                "threadId": resolved_thread_id,
                            }
                        )
                    except Exception as exc:
                        persist_assistant_message(f"Error: {exc}")
                        await queue.put({"type": "error", "error": str(exc)})
                    finally:
                        await queue.put({"_done": True})

                async def event_stream():
                    task = asyncio.create_task(worker())
                    try:
                        while True:
                            event = await queue.get()
                            if event.get("_done"):
                                break
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"
                    finally:
                        await task

                return StreamingResponse(
                    event_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            result = await MultiAgentManager.run_stream(execution_input, all_agents, llm, full_config)
            response_text = str(result.get("answer") or "")
            persist_assistant_message(
                response_text,
                {
                    "steps": result.get("steps"),
                    "agent_type": "manager",
                },
            )
            return JSONResponse(
                content={
                    "response": response_text,
                    "steps": result.get("steps"),
                    "threadId": resolved_thread_id,
                }
            )

        if is_streaming:
            queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

            def on_step(step: Dict[str, Any]) -> None:
                queue.put_nowait({"type": "step", "step": step})

            async def worker() -> None:
                try:
                    result = await AgentExecutor.execute(
                        agent.get("agent_type") or "custom",
                        full_config,
                        execution_input,
                        llm,
                        on_step=on_step,
                        agent_name=agent_name,
                    )
                    response_text = str(result.get("answer") or "")
                    persist_assistant_message(response_text, result)
                    await queue.put(
                        {
                            "type": "result",
                            "response": response_text,
                            "details": result,
                            "threadId": resolved_thread_id,
                        }
                    )
                except Exception as exc:
                    persist_assistant_message(f"Error: {exc}")
                    await queue.put({"type": "error", "error": str(exc)})
                finally:
                    await queue.put({"_done": True})

            async def event_stream():
                task = asyncio.create_task(worker())
                try:
                    while True:
                        event = await queue.get()
                        if event.get("_done"):
                            break
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"
                finally:
                    await task

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        result = await AgentExecutor.execute(
            agent.get("agent_type") or "custom",
            full_config,
            execution_input,
            llm,
            agent_name=agent_name,
        )
        response_text = str(result.get("answer") or "")
        persist_assistant_message(response_text, result)
        return JSONResponse(
            content={
                "response": response_text,
                "details": result,
                "threadId": resolved_thread_id,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        if is_streaming:
            async def error_stream():
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\\n\\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        return JSONResponse(status_code=500, content={"error": str(exc) or "Internal server error"})


if (DIST_DIR / "index.html").exists():
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(DIST_DIR / "index.html")


    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        target = DIST_DIR / full_path
        if target.exists() and target.is_file():
            return FileResponse(target)

        return FileResponse(DIST_DIR / "index.html")
else:

    @app.get("/", include_in_schema=False)
    async def no_frontend_built() -> PlainTextResponse:
        return PlainTextResponse("Backend running. Build the frontend to serve static files.")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=3000, reload=True)
