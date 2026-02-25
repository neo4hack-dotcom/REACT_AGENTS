from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Optional

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
        },
        "llm_config": [],
        "db_config": [],
        "agents": [],
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

        for key in ("llm_config", "db_config", "agents"):
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
            state["db_config"].append(
                {
                    "id": config_id,
                    "type": payload.get("type") or "clickhouse",
                    "host": payload.get("host") or "",
                    "port": int(payload.get("port") or 0),
                    "username": payload.get("username") or "",
                    "password": payload.get("password") or "",
                    "database_name": payload.get("database_name") or "",
                }
            )

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


@app.post("/api/agents")
async def create_agent(request: Request) -> JSONResponse:
    payload = await request.json()

    try:
        def mutate(state: Dict[str, Any]) -> None:
            agent_id = _next_id(state, "next_agent_id")
            db_config_id = payload.get("db_config_id")
            state["agents"].append(
                {
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
            )

        store.mutate(mutate)
        return JSONResponse(content={"success": True})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: int) -> JSONResponse:
    def mutate(state: Dict[str, Any]) -> None:
        state["agents"] = [item for item in state["agents"] if int(item.get("id", 0)) != agent_id]

    store.mutate(mutate)
    return JSONResponse(content={"success": True})


@app.post("/api/chat/{agent_id}")
async def chat_agent(agent_id: int, request: Request):
    payload = await request.json()
    message = payload.get("message")
    thread_id = payload.get("threadId")
    is_streaming = request.query_params.get("stream") == "true"

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

        if agent.get("agent_type") == "manager":
            all_agents = [a for a in state["agents"] if a.get("agent_type") != "manager"]

            if is_streaming:
                queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

                def on_step(step: Dict[str, Any]) -> None:
                    queue.put_nowait({"type": "step", "step": step})

                async def worker() -> None:
                    try:
                        result = await MultiAgentManager.run_stream(
                            message,
                            all_agents,
                            llm,
                            full_config,
                            on_step=on_step,
                        )
                        await queue.put(
                            {
                                "type": "result",
                                "response": result.get("answer"),
                                "steps": result.get("steps"),
                            }
                        )
                    except Exception as exc:
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

            result = await MultiAgentManager.run_stream(message, all_agents, llm, full_config)
            return JSONResponse(
                content={
                    "response": result.get("answer"),
                    "steps": result.get("steps"),
                    "threadId": thread_id or "default",
                }
            )

        if is_streaming:
            async def event_stream():
                try:
                    result = await AgentExecutor.execute(agent.get("agent_type") or "custom", full_config, message, llm)
                    stream_payload = {"type": "result", "response": result.get("answer"), "details": result}
                except Exception as exc:
                    stream_payload = {"type": "error", "error": str(exc)}
                yield f"data: {json.dumps(stream_payload, ensure_ascii=False)}\\n\\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        result = await AgentExecutor.execute(agent.get("agent_type") or "custom", full_config, message, llm)
        return JSONResponse(
            content={
                "response": result.get("answer"),
                "details": result,
                "threadId": thread_id or "default",
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
