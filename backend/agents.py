from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Callable, Dict, List, TypedDict

import feedparser
import requests
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph


class AgentResult(TypedDict, total=False):
    sql: str
    rows: List[Any]
    answer: str
    details: Any


StepCallback = Callable[[Dict[str, Any]], None]


def _emit_step(on_step: StepCallback | None, step: Dict[str, Any]) -> None:
    if not on_step:
        return
    try:
        on_step(step)
    except Exception:
        # Streaming callbacks should never break agent execution.
        return


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        return "\n".join(parts)

    return str(content)


def _extract_json_block(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1) if match else text


def build_agent_execution_config(agent: Dict[str, Any]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    raw_config = agent.get("config")

    if isinstance(raw_config, str):
        try:
            parsed = json.loads(raw_config)
        except Exception:
            parsed = {}
    elif isinstance(raw_config, dict):
        parsed = raw_config

    return {
        **parsed,
        "role": agent.get("role"),
        "persona": agent.get("persona"),
        "objectives": agent.get("objectives"),
        "system_prompt": agent.get("system_prompt"),
    }


def _is_generic_orchestration_text(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True

    generic_markers = (
        "starting multi-agent orchestration",
        "response ready",
        "analyzing request and preparing context",
        "generating response with the language model",
        "model response received",
    )
    return any(marker in lowered for marker in generic_markers)


def _extract_manager_answer_from_steps(steps: List[Dict[str, Any]]) -> str:
    for step in reversed(steps):
        status = str(step.get("status") or "")

        if status in {"manager_final", "manager_finalized_fallback"}:
            for key in ("answer", "manager_summary"):
                value = step.get(key)
                if isinstance(value, str) and value.strip() and not _is_generic_orchestration_text(value):
                    return value.strip()

        if status in {"agent_call_completed"}:
            result = step.get("result")
            if isinstance(result, dict):
                value = result.get("answer")
                if isinstance(value, str) and value.strip() and not _is_generic_orchestration_text(value):
                    return value.strip()

        if status in {"agent_call_failed", "manager_finalize_failed", "error"}:
            for key in ("error", "message"):
                value = step.get(key)
                if isinstance(value, str) and value.strip() and not _is_generic_orchestration_text(value):
                    return value.strip()
    return ""


def _summarize_steps_for_synthesis(steps: List[Dict[str, Any]], max_steps: int = 25) -> str:
    lines: List[str] = []
    for step in steps[-max_steps:]:
        status = str(step.get("status") or "unknown")
        agent = str(step.get("agent") or "")
        message = str(step.get("message") or "")
        rationale = str(step.get("rationale") or "")
        result = step.get("result")
        result_text = ""
        if isinstance(result, dict):
            result_text = str(result.get("answer") or "")
        sql = str(step.get("sql") or "")

        parts = [f"status={status}"]
        if agent:
            parts.append(f"agent={agent}")
        if message:
            parts.append(f"message={message}")
        if rationale:
            parts.append(f"rationale={rationale}")
        if result_text:
            parts.append(f"result={result_text}")
        if sql:
            parts.append(f"sql={sql[:400]}")
        lines.append("- " + " | ".join(parts))

    return "\n".join(lines)


async def _synthesize_manager_fallback_answer(
    llm: Any,
    user_input: str,
    conversation_history: str,
    scratchpad: Dict[str, Any],
    steps: List[Dict[str, Any]],
) -> str:
    prompt = f"""You are finalizing a manager-orchestrated task.
Generate a complete, user-facing final answer based on the execution trace.
Do not mention internal implementation details.
If information is missing, state exactly what is missing.

Return ONLY JSON:
{{ "final_answer": "..." }}

User request:
{user_input}

Conversation history:
{conversation_history}

Scratchpad:
{json.dumps(scratchpad, ensure_ascii=False)}

Execution steps:
{_summarize_steps_for_synthesis(steps)}
"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    content = _coerce_message_content(response.content).strip()
    if not content:
        return ""

    try:
        parsed = json.loads(_extract_json_block(content))
        if isinstance(parsed, dict):
            return str(parsed.get("final_answer") or parsed.get("answer") or "").strip()
    except Exception:
        pass

    return content


class AgentExecutor:
    @staticmethod
    async def execute(
        agent_type: str,
        config: Dict[str, Any],
        user_input: str,
        llm: Any,
        on_step: StepCallback | None = None,
        agent_name: str | None = None,
    ) -> AgentResult:
        system_prompt = "You are a helpful AI assistant."
        extra_context = ""
        agent_label = agent_name or agent_type or "agent"

        _emit_step(
            on_step,
            {
                "status": "agent_thinking",
                "agent": agent_label,
                "message": "Analyzing request and preparing context.",
            },
        )

        if agent_type == "sql_analyst":
            system_prompt = f"""You are a Senior ClickHouse Expert and SQL Analyst. Database: {config.get('database_name') or 'Unknown'}. 
Mode: {config.get('sql_use_case_mode') or 'llm_sql'}.
Template: {config.get('sql_query_template') or 'None'}.

CRITICAL CLICKHOUSE BEST PRACTICES:
- Never use PostgreSQL or MySQL syntax.
- For time-series gap analysis, use WITH FILL.
- For behavioral analysis (funnels), use windowFunnel().
- For fast descriptive stats on billions of rows, use quantileTiming, uniqHLL12, topK.
- For conditional aggregations, prefer sumIf, countIf over complex joins.
- For time-series joins, always consider ASOF JOIN.
- For distinct counts on large volumes, use uniq or uniqExact.
- Filters MUST use the table's sorting key (ORDER BY) for optimal performance.

CRITICAL FOR LARGE SCHEMAS & HUGE TABLES:
- Do not guess table or column names.
- If you lack schema context, write exploratory SQL first (e.g., SHOW TABLES, DESCRIBE table, or query system.columns/system.tables).
- Always LIMIT your queries when exploring data (e.g., SELECT * FROM table LIMIT 5).
- If parameters are missing, state what is missing in your answer.

Return your response in JSON format: {{ "sql": "SELECT ...", "answer": "Explanation..." }}"""
        elif agent_type == "clickhouse_table_manager":
            system_prompt = f"""You are a ClickHouse Table Manager. 
Protect existing tables: {config.get('protect_existing_tables') is not False}.
Allow inserts: {bool(config.get('allow_row_inserts'))}.
Allow updates: {bool(config.get('allow_row_updates'))}.
Allow deletes: {bool(config.get('allow_row_deletes'))}.

CRITICAL FOR LARGE SCHEMAS:
- Verify table existence and structure before applying changes.
- Do not drop or truncate tables unless explicitly permitted and verified.

Return your response in JSON format: {{ "sql": "CREATE TABLE ...", "answer": "Table created." }}"""
        elif agent_type == "clickhouse_writer":
            system_prompt = f"""You are a ClickHouse Data Writer.
Enforce agent prefix: {config.get('enforce_agent_prefix') is not False}.
Allow inserts: {config.get('allow_inserts') is not False}.

CRITICAL SECURITY RULE:
- You are ONLY allowed to create, modify, or insert into tables whose names start with "agent_".
- If the user asks to modify a table that does not start with "agent_", you MUST refuse and explain the security policy.
- Ensure all inserted data is properly escaped and formatted.

Return your response in JSON format: {{ "sql": "INSERT INTO agent_... ", "answer": "Data inserted." }}"""
        elif agent_type == "knowledge_base_assistant":
            system_prompt = f"""You are an internal Knowledge Base Assistant (Concierge Interne).
Vector DB: {config.get('vector_db') or 'qdrant'}
Embedding Model: {config.get('embedding_model') or 'text-embedding-004'}
Use Hybrid Search: {bool(config.get('use_hybrid_search'))}
Use Reranker: {bool(config.get('use_reranker'))}

CRITICAL RAG BEST PRACTICES:
- NEVER hallucinate. If the answer is not in the provided context, state clearly that you do not know.
- ALWAYS cite your sources (e.g., [Document Name, Page X]).
- Synthesize the retrieved chunks into a coherent, professional response.

Return your response in JSON format: {{ "answer": "Your synthesized answer with citations...", "sources": ["doc1", "doc2"] }}"""
        elif agent_type == "data_anomaly_hunter":
            system_prompt = f"""You are a Data Anomaly Hunter.
Detection Method: {config.get('detection_method') or 'statistical'}
Use Dynamic Thresholds: {bool(config.get('use_dynamic_thresholds'))}

CRITICAL ANOMALY DETECTION BEST PRACTICES:
- You receive statistical outliers (e.g., Z-score > 3, IQR anomalies) and KPI baselines.
- DO NOT read raw data rows. Rely on the statistical summaries provided.
- Explain WHY the anomaly might have happened in clear business terms.
- Format an alert message suitable for {config.get('alert_channel') or 'Slack'}.

Return your response in JSON format: {{ "answer": "Business explanation of the anomaly...", "alert_message": "🚨 *Anomaly Detected*..." }}"""
        elif agent_type == "text_to_sql_translator":
            system_prompt = f"""You are an expert ClickHouse Text-to-SQL Translator (Traducteur Métier).
Use Semantic Layer: {bool(config.get('use_semantic_layer'))}
Use Golden Queries: {bool(config.get('use_golden_queries'))}

CRITICAL TEXT-TO-SQL BEST PRACTICES:
- You will receive the DDL, column descriptions, and potentially "Golden Queries" as examples.
- Write precise ClickHouse SQL. Avoid complex joins if conditional aggregations (sumIf, countIf) suffice.
- If the query fails, you will receive the error and must auto-correct it (up to {config.get('max_retries') or 3} times).
- Generate a chart configuration (e.g., ECharts JSON) if requested and appropriate.

Return your response in JSON format: {{ "sql": "SELECT ...", "answer": "Explanation...", "chart_config": {{}} }}"""
        elif agent_type == "data_profiler_cleaner":
            system_prompt = f"""You are a Data Profiler & Cleaner (Data Steward).
Execution Engine: {config.get('execution_engine') or 'polars'}
Use Sampling: {bool(config.get('use_sampling'))} (Sample size: {config.get('sample_size') or 10000})

CRITICAL DATA PROFILING BEST PRACTICES:
- Review the provided statistical profile (nulls, cardinality, min/max, inferred types).
- Identify data quality issues (e.g., negative ages, inconsistent formats).
- Write a Python (Pandas/Polars) or SQL script to clean the data.
- NEVER execute destructive cleaning in production. Propose a script for human validation.

Return your response in JSON format: {{ "answer": "Audit summary...", "cleaning_script": "import polars as pl\\n..." }}"""
        elif agent_type == "unstructured_to_structured":
            schema = config.get("output_schema")
            schema_repr = schema if isinstance(schema, str) else json.dumps(schema or {})
            system_prompt = f"""You are a data extraction assistant. Convert the unstructured text into structured JSON.
Output Schema: {schema_repr}
Strict JSON: {config.get('strict_json') is not False}
Return ONLY valid JSON matching the schema."""
        elif agent_type == "email_cleaner":
            system_prompt = f"""You are an Email Cleaner. Condense the email into actionable sections.
Max bullets: {config.get('max_bullets') or 5}.
Include sections: {config.get('include_sections') or 'Action Items, Summary'}.
Return your response in JSON format: {{ "answer": "Cleaned email content..." }}"""
        elif agent_type == "file_assistant":
            system_prompt = f"You are a File Assistant. Answer questions based on the files in folder: {config.get('folder_path')}."
            folder_path = config.get("folder_path")
            if folder_path:
                try:
                    if os.path.exists(folder_path):
                        max_files = int(config.get("max_files") or 10)
                        files = os.listdir(folder_path)[:max_files]
                        extra_context = "\n\nFiles in directory:\n" + "\n".join(files)
                    else:
                        extra_context = f"\n\nError: Directory {folder_path} does not exist."
                except Exception as exc:
                    extra_context = f"\n\nError reading directory: {exc}"
        elif agent_type == "text_file_manager":
            system_prompt = (
                f"You are a Text File Manager. Folder: {config.get('folder_path')}. Allow overwrite: "
                f"{bool(config.get('allow_overwrite'))}.\n"
                "Return your response in JSON format: { \"answer\": \"File operations completed.\", "
                "\"details\": {\"action\": \"read|write|append\", \"file\": \"filename\"} }"
            )
        elif agent_type == "excel_manager":
            system_prompt = (
                f"You are an Excel Manager. Workbook: {config.get('workbook_path')}.\n"
                "Return your response in JSON format: { \"answer\": \"Excel operations completed.\", "
                "\"details\": {\"action\": \"read|write\", \"sheet\": \"sheetname\"} }"
            )
        elif agent_type == "word_manager":
            system_prompt = (
                f"You are a Word Manager. Document: {config.get('document_path')}.\n"
                "Return your response in JSON format: { \"answer\": \"Word operations completed.\", "
                "\"details\": {\"action\": \"read|write\"} }"
            )
        elif agent_type == "elasticsearch_retriever":
            system_prompt = (
                f"You are an Elasticsearch Retriever. Index: {config.get('index')}. "
                "Synthesize the retrieved documents.\n"
                "Return your response in JSON format: { \"answer\": \"Synthesized content...\" }"
            )
            if config.get("base_url") and config.get("index"):
                try:
                    es_kwargs: Dict[str, Any] = {
                        "hosts": [config["base_url"]],
                        "verify_certs": config.get("verify_ssl") is not False,
                    }
                    if config.get("api_key"):
                        es_kwargs["api_key"] = config.get("api_key")
                    elif config.get("username"):
                        es_kwargs["basic_auth"] = (
                            config.get("username"),
                            config.get("password") or "",
                        )

                    client = Elasticsearch(**es_kwargs)
                    fields = ["*"]
                    if config.get("fields"):
                        fields = [field.strip() for field in str(config["fields"]).split(",") if field.strip()]

                    result = await asyncio.to_thread(
                        client.search,
                        index=config["index"],
                        query={
                            "simple_query_string": {
                                "query": user_input,
                                "fields": fields,
                            }
                        },
                        size=int(config.get("top_k") or 5),
                    )
                    hits = [hit.get("_source") for hit in result.get("hits", {}).get("hits", [])]
                    extra_context = "\n\nElasticsearch Results:\n" + json.dumps(hits)[:5000]
                except Exception as exc:
                    extra_context = f"\n\nElasticsearch Error: {exc}"
        elif agent_type == "rag_context":
            system_prompt = (
                f"You are a RAG Assistant. Folder: {config.get('folder_path')}. "
                f"Top K chunks: {config.get('top_k_chunks') or 3}.\n"
                "Return your response in JSON format: { \"answer\": \"RAG response...\" }"
            )
        elif agent_type == "rss_news":
            system_prompt = (
                f"You are an RSS News summarizer. Feeds: {config.get('feed_urls')}. "
                f"Interests: {config.get('interests')}."
            )
            feed_urls = config.get("feed_urls")
            if feed_urls:
                try:
                    urls = [url.strip() for url in str(feed_urls).split(",") if url.strip()]
                    max_items = int(config.get("max_items_per_feed") or 5)
                    feed_items: List[Any] = []
                    for url in urls:
                        try:
                            response = await asyncio.to_thread(requests.get, url, timeout=20)
                            parsed = feedparser.parse(response.content)
                            feed_items.extend(parsed.entries[:max_items])
                        except Exception:
                            continue

                    lines: List[str] = []
                    for item in feed_items:
                        title = item.get("title", "")
                        summary = item.get("summary") or item.get("description") or ""
                        lines.append(f"- {title}: {summary}")

                    extra_context = "\n\nRecent News Items:\n" + "\n".join(lines)
                except Exception:
                    pass
        elif agent_type == "web_scraper":
            system_prompt = (
                f"You are a Web Scraper. Start URLs: {config.get('start_urls')}. "
                f"Allowed domains: {config.get('allowed_domains')}."
            )
            start_urls = config.get("start_urls")
            if start_urls:
                try:
                    urls = [url.strip() for url in str(start_urls).split(",") if url.strip()]
                    max_chars = int(config.get("max_chars_per_page") or 5000)
                    timeout = int(config.get("timeout_seconds") or 15)
                    scraped = ""

                    for url in urls:
                        try:
                            response = await asyncio.to_thread(requests.get, url, timeout=timeout)
                            soup = BeautifulSoup(response.text, "html.parser")
                            body_text = soup.get_text(" ", strip=True)
                            scraped += f"\n\nContent from {url}:\n{body_text[:max_chars]}"
                        except Exception:
                            continue

                    extra_context = "\n\nScraped Content:\n" + scraped
                except Exception:
                    pass
        elif agent_type == "web_navigator":
            system_prompt = (
                f"You are a Web Navigator. Start URL: {config.get('start_url')}. "
                f"Max steps: {config.get('max_steps') or 5}.\n"
                "Return your response in JSON format: { \"answer\": \"Navigation completed.\" }"
            )
            extra_context = "\n\nNote: Playwright is not installed in this environment. The agent is marked as unavailable."
        else:
            system_prompt = f"""Role: {config.get('role') or 'Assistant'}
Persona: {config.get('persona') or 'Helpful'}
Objectives: {config.get('objectives') or 'Assist the user'}
Instructions: {config.get('system_prompt') or ''}"""

        try:
            _emit_step(
                on_step,
                {
                    "status": "llm_call_started",
                    "agent": agent_label,
                    "message": "Generating response with the language model.",
                },
            )
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input + extra_context),
                ]
            )
            content = _coerce_message_content(response.content)

            _emit_step(
                on_step,
                {
                    "status": "llm_response_received",
                    "agent": agent_label,
                    "message": "Model response received. Preparing final answer.",
                },
            )

            if content.strip().startswith("{") or agent_type != "custom":
                try:
                    json_payload = json.loads(_extract_json_block(content))
                    result: AgentResult = {
                        "answer": json_payload.get("answer")
                        or (json.dumps(json_payload) if agent_type == "unstructured_to_structured" else "JSON parsed successfully."),
                        "sql": json_payload.get("sql"),
                        "rows": json_payload.get("rows") or [],
                        "details": json_payload,
                    }
                    _emit_step(
                        on_step,
                        {
                            "status": "agent_completed",
                            "agent": agent_label,
                            "message": "Response ready.",
                        },
                    )
                    return result
                except Exception:
                    if agent_type == "unstructured_to_structured":
                        fallback_result = {"answer": f"Fallback raw output:\n{content}"}
                        _emit_step(
                            on_step,
                            {
                                "status": "agent_completed",
                                "agent": agent_label,
                                "message": "Response ready.",
                            },
                        )
                        return fallback_result
                    plain_result = {"answer": content}
                    _emit_step(
                        on_step,
                        {
                            "status": "agent_completed",
                            "agent": agent_label,
                            "message": "Response ready.",
                        },
                    )
                    return plain_result

            plain_result = {"answer": content}
            _emit_step(
                on_step,
                {
                    "status": "agent_completed",
                    "agent": agent_label,
                    "message": "Response ready.",
                },
            )
            return plain_result
        except Exception as exc:
            _emit_step(
                on_step,
                {
                    "status": "agent_error",
                    "agent": agent_label,
                    "error": str(exc),
                    "message": "Agent execution failed.",
                },
            )
            return {"answer": f"Error executing agent: {exc}"}


class ManagerState(TypedDict):
    input: str
    active_agents: List[Dict[str, Any]]
    llm: Any
    manager_config: Dict[str, Any]
    steps: List[Dict[str, Any]]
    conversation_history: str
    scratchpad: Dict[str, Any]
    current_step: int
    max_steps: int
    done: bool
    final_answer: str


class MultiAgentManager:
    @staticmethod
    async def run_stream(
        user_input: str,
        agents: List[Dict[str, Any]],
        llm: Any,
        manager_config: Dict[str, Any] | None = None,
        on_step: Callable[[Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        cfg = manager_config or {}
        active_agents = [agent for agent in agents if agent.get("agent_type") != "web_navigator"]

        state: ManagerState = {
            "input": user_input,
            "active_agents": active_agents,
            "llm": llm,
            "manager_config": cfg,
            "steps": [],
            "conversation_history": f"User Request: {user_input}\n",
            "scratchpad": {},
            "current_step": 0,
            "max_steps": int(cfg.get("max_steps") or 10),
            "done": False,
            "final_answer": "",
        }

        def add_step(step: Dict[str, Any]) -> None:
            state["steps"].append(step)
            if on_step:
                on_step(step)

        add_step(
            {
                "status": "manager_start",
                "message": "Starting multi-agent orchestration",
                "active_agents": len(active_agents),
                "unavailable_agents": len(agents) - len(active_agents),
            }
        )

        async def iterate(current_state: ManagerState) -> ManagerState:
            if current_state["done"]:
                return current_state

            if current_state["current_step"] >= current_state["max_steps"]:
                current_state["final_answer"] = "Reached maximum steps without a final answer."
                add_step(
                    {
                        "status": "manager_final",
                        "answer": current_state["final_answer"],
                        "manager_summary": "Max steps reached.",
                    }
                )
                current_state["done"] = True
                return current_state

            manager_prompt = f"""You are an advanced Multi-Agent Orchestrator.
Available agents:
{chr(10).join([f"- {a.get('name')} ({a.get('agent_type')}): {a.get('role') or ''} - {a.get('objectives') or ''}" for a in current_state['active_agents']])}

Current Conversation History & Knowledge:
{current_state['conversation_history']}

Shared Memory (Scratchpad):
{json.dumps(current_state['scratchpad'], indent=2)}

CRITICAL INSTRUCTIONS FOR LARGE DATA/SCHEMAS:
1. Do not assume schema structures. Anticipate multiple phases:
   - Phase 1 (Discovery): Query table names, metadata, or file lists.
   - Phase 2 (Inspection): Query specific column definitions, data types, or file samples.
   - Phase 3 (Execution): Run the targeted SQL or data extraction.
   - Phase 4 (Synthesis): Analyze and format the final result.
2. At EVERY step, re-evaluate your plan based on the new knowledge acquired. Optimize the next calls.
3. You can call multiple agents in parallel if needed (Map-Reduce).
4. Do NOT send the entire conversation history to agents. Extract ONLY the relevant context (e.g., specific table names, specific metrics) and pass it in the "input" field for that agent.
5. You can update the "scratchpad" to store variables (like schema definitions) for future steps.

Decide the next step. Return ONLY a valid JSON object with the following structure:
{{
  "status": "thinking" | "calling_agent" | "final_answer",
  "current_plan": "Your updated step-by-step plan based on current knowledge",
  "rationale": "Why you are making this decision and how it optimizes the plan",
  "scratchpad_updates": {{ "key": "value" }},
  "calls": [{{"agent_name": "exact name of agent", "input": "highly detailed instruction, filtered to ONLY include necessary context"}}],
  "final_answer": "The comprehensive final answer to the user",
  "missing_information": "Any info you still need to proceed"
}}"""

            try:
                response = await current_state["llm"].ainvoke([HumanMessage(content=manager_prompt)])
                content = _coerce_message_content(response.content)
                decision = json.loads(_extract_json_block(content))

                updates = decision.get("scratchpad_updates")
                if isinstance(updates, dict):
                    current_state["scratchpad"] = {**current_state["scratchpad"], **updates}

                rationale = decision.get("rationale")
                plan = decision.get("current_plan")
                add_step({"status": "manager_decision", "rationale": rationale, "plan": plan})
                current_state["conversation_history"] += (
                    f"\n[Manager Plan Updated]: {plan}\n"
                    f"[Manager Rationale]: {rationale}\n"
                )

                if decision.get("status") == "final_answer":
                    current_state["final_answer"] = decision.get("final_answer") or ""
                    add_step(
                        {
                            "status": "manager_final",
                            "answer": current_state["final_answer"],
                            "manager_summary": rationale,
                            "judge_verdict": "Pass",
                            "missing_information": decision.get("missing_information"),
                        }
                    )
                    current_state["done"] = True
                elif (
                    decision.get("status") == "calling_agent"
                    and isinstance(decision.get("calls"), list)
                    and len(decision["calls"]) > 0
                ):
                    async def execute_call(call: Dict[str, Any]) -> str:
                        agent_name = call.get("agent_name")
                        call_input = call.get("input") or ""
                        selected = next(
                            (agent for agent in current_state["active_agents"] if agent.get("name") == agent_name),
                            None,
                        )

                        if not selected:
                            add_step({"status": "agent_not_found", "agent": agent_name})
                            return (
                                f"\nAttempted to call agent {agent_name} "
                                "but it was not found or unavailable.\n"
                            )

                        add_step(
                            {
                                "status": "agent_call_started",
                                "agent": selected.get("name"),
                                "input": call_input,
                            }
                        )

                        try:
                            execution_config = build_agent_execution_config(selected)
                            result = await AgentExecutor.execute(
                                selected.get("agent_type") or "custom",
                                execution_config,
                                call_input,
                                current_state["llm"],
                                on_step=add_step,
                                agent_name=selected.get("name") or selected.get("agent_type") or "agent",
                            )
                            add_step(
                                {
                                    "status": "agent_call_completed",
                                    "agent": selected.get("name"),
                                    "result": result,
                                }
                            )
                            if result.get("sql"):
                                add_step({"status": "sql_generated", "sql": result["sql"]})
                            return f"\nAgent {selected.get('name')} responded: {result.get('answer')}\n"
                        except Exception as exc:
                            add_step(
                                {
                                    "status": "agent_call_failed",
                                    "agent": selected.get("name"),
                                    "error": str(exc),
                                }
                            )
                            return f"\nAgent {selected.get('name')} failed: {exc}\n"

                    results = await asyncio.gather(*(execute_call(call) for call in decision["calls"]))
                    current_state["conversation_history"] += "".join(results)
                else:
                    current_state["conversation_history"] += f"\nManager thought: {rationale}\n"

            except Exception as exc:
                add_step({"status": "error", "message": f"Manager error: {exc}"})
                current_state["final_answer"] = f"Orchestration error: {exc}"
                current_state["done"] = True

            current_state["current_step"] += 1
            if current_state["current_step"] >= current_state["max_steps"] and not current_state["done"]:
                current_state["final_answer"] = "Reached maximum steps without a final answer."
                add_step(
                    {
                        "status": "manager_final",
                        "answer": current_state["final_answer"],
                        "manager_summary": "Max steps reached.",
                    }
                )
                current_state["done"] = True

            return current_state

        workflow = StateGraph(ManagerState)
        workflow.add_node("iterate", iterate)
        workflow.set_entry_point("iterate")
        workflow.add_conditional_edges("iterate", lambda s: END if s["done"] else "iterate")
        graph = workflow.compile()

        final_state = await graph.ainvoke(state)
        steps = final_state.get("steps")
        if not isinstance(steps, list):
            steps = []

        final_answer = str(final_state.get("final_answer") or "").strip()
        if not final_answer:
            final_answer = _extract_manager_answer_from_steps(steps)

        if not final_answer:
            try:
                synthesized = await _synthesize_manager_fallback_answer(
                    llm=final_state.get("llm"),
                    user_input=user_input,
                    conversation_history=str(final_state.get("conversation_history") or ""),
                    scratchpad=final_state.get("scratchpad") if isinstance(final_state.get("scratchpad"), dict) else {},
                    steps=steps,
                )
                if synthesized:
                    final_answer = synthesized
                    steps.append(
                        {
                            "status": "manager_finalized_fallback",
                            "answer": final_answer,
                            "manager_summary": "Fallback synthesis generated because manager did not provide a final answer.",
                        }
                    )
            except Exception as exc:
                steps.append(
                    {
                        "status": "manager_finalize_failed",
                        "error": str(exc),
                        "message": "Failed to synthesize fallback final answer.",
                    }
                )

        if not final_answer:
            final_answer = "Manager completed orchestration but did not produce a final answer."
            steps.append(
                {
                    "status": "manager_finalized_fallback",
                    "answer": final_answer,
                    "manager_summary": "Default fallback answer used.",
                }
            )

        return {
            "answer": final_answer,
            "steps": steps,
        }
