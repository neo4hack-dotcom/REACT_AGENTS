from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, TypedDict
from urllib.parse import urljoin, urlparse

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _agent_catalog_entry(agent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": agent.get("id"),
        "name": agent.get("name"),
        "agent_type": agent.get("agent_type"),
        "role": agent.get("role"),
        "objectives": agent.get("objectives"),
    }


def _agent_aliases(agent: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    name = str(agent.get("name") or "").strip()
    agent_type = str(agent.get("agent_type") or "").strip()
    role = str(agent.get("role") or "").strip()

    if name:
        aliases.append(name)
    if agent_type:
        aliases.append(agent_type)
        aliases.append(agent_type.replace("_", " "))
    if role:
        aliases.append(role)

    # Keep insertion order while removing duplicates.
    deduped: List[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(alias)
    return deduped


def _best_fuzzy_agent_match(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    norm_query = _normalize_identifier(query)
    if len(norm_query) < 4:
        return None

    scored: List[tuple[float, Dict[str, Any]]] = []
    for agent in candidates:
        best_for_agent = 0.0
        for alias in _agent_aliases(agent):
            norm_alias = _normalize_identifier(alias)
            if not norm_alias:
                continue
            score = SequenceMatcher(None, norm_query, norm_alias).ratio()
            if score > best_for_agent:
                best_for_agent = score
        if best_for_agent > 0:
            scored.append((best_for_agent, agent))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_agent = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    # Require a strong match. If two candidates are too close, avoid random routing.
    if best_score < 0.84:
        return None
    if (best_score - second_score) < 0.06 and best_score < 0.96:
        return None
    return best_agent


def _resolve_agent_from_call(call: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not candidates:
        return None

    requested_id = _safe_int(call.get("agent_id"))
    if requested_id is not None:
        direct = next((agent for agent in candidates if _safe_int(agent.get("id")) == requested_id), None)
        if direct:
            return direct

    requested_name = str(call.get("agent_name") or call.get("agent") or "").strip()
    requested_type = str(call.get("agent_type") or "").strip()

    if requested_name:
        normalized_name = _normalize_identifier(requested_name)

        # First, strong deterministic matches on aliases (name/type/role).
        for agent in candidates:
            aliases = _agent_aliases(agent)
            for alias in aliases:
                if alias.lower() == requested_name.lower():
                    return agent
                if normalized_name and _normalize_identifier(alias) == normalized_name:
                    return agent

        # Then, relaxed containment on normalized aliases.
        if normalized_name:
            for agent in candidates:
                for alias in _agent_aliases(agent):
                    normalized_alias = _normalize_identifier(alias)
                    if not normalized_alias:
                        continue
                    if normalized_name in normalized_alias or normalized_alias in normalized_name:
                        return agent

        fuzzy_name_match = _best_fuzzy_agent_match(requested_name, candidates)
        if fuzzy_name_match:
            return fuzzy_name_match

    if requested_type:
        normalized_type = _normalize_identifier(requested_type)
        for agent in candidates:
            agent_type = str(agent.get("agent_type") or "").strip()
            if agent_type.lower() == requested_type.lower():
                return agent
            if normalized_type and _normalize_identifier(agent_type) == normalized_type:
                return agent

        fuzzy_type_match = _best_fuzzy_agent_match(requested_type, candidates)
        if fuzzy_type_match:
            return fuzzy_type_match

    if len(candidates) == 1:
        return candidates[0]

    return None


def _extract_urls_from_text(text: str) -> List[str]:
    matches = re.findall(r"https?://[^\s<>\"]+", text or "")
    urls: List[str] = []
    seen: set[str] = set()
    for raw in matches:
        cleaned = raw.rstrip(").,;!?]")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        urls.append(cleaned)
    return urls


def _extract_requested_item_limit(text: str, default_limit: int = 10, hard_max: int = 50) -> int:
    lowered = (text or "").lower()
    patterns = [
        r"(?:latest|top|first)\s+(\d{1,3})",
        r"(\d{1,3})\s+(?:articles|results|links|titles|items)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        try:
            value = int(match.group(1))
            return max(1, min(value, hard_max))
        except Exception:
            continue
    return max(1, min(default_limit, hard_max))


def _is_probable_article_anchor(title: str, url: str) -> bool:
    title_norm = " ".join(title.split()).strip()
    if len(title_norm) < 8:
        return False

    lower_title = title_norm.lower()
    if lower_title in {"menu", "accueil", "home", "login", "connexion"}:
        return False

    lower_url = url.lower()
    blocked_tokens = [
        "/login",
        "/connexion",
        "/account",
        "/compte",
        "/abonnement",
        "/subscribe",
        "/newsletter",
        "/video",
        "/podcast",
        "/tag/",
        "#",
        "javascript:",
        "mailto:",
    ]
    if any(token in lower_url for token in blocked_tokens):
        return False

    return True


def _extract_article_links_from_html(
    page_url: str,
    html: str,
    max_links: int,
    same_domain_only: bool,
) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(page_url).netloc.lower()

    selectors = [
        "article a[href]",
        "main a[href]",
        "h1 a[href]",
        "h2 a[href]",
        "h3 a[href]",
        "[data-testid] a[href]",
        "a[href]",
    ]

    links: List[Dict[str, str]] = []
    seen_urls: set[str] = set()

    for selector in selectors:
        for anchor in soup.select(selector):
            href = str(anchor.get("href") or "").strip()
            if not href:
                continue
            if href.startswith("#") or href.lower().startswith("javascript:") or href.lower().startswith("mailto:"):
                continue

            absolute_url = urljoin(page_url, href)
            parsed = urlparse(absolute_url)
            if parsed.scheme not in {"http", "https"}:
                continue

            if same_domain_only and base_domain:
                target_domain = parsed.netloc.lower()
                if target_domain and target_domain != base_domain and not target_domain.endswith(f".{base_domain}"):
                    continue

            title = anchor.get_text(" ", strip=True)
            if not title:
                continue
            title = " ".join(title.split())
            if not _is_probable_article_anchor(title, absolute_url):
                continue

            normalized_url = absolute_url.rstrip("/")
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            links.append({"title": title, "url": absolute_url})

            if len(links) >= max_links:
                return links

    return links


def _extract_possible_json(text: str) -> Any:
    candidate = _extract_json_block(text or "").strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _is_virtual_data_path(raw_path: str) -> bool:
    normalized = str(raw_path or "").replace("\\", "/").strip()
    return normalized == "/data" or normalized.startswith("/data/")


def _resolve_workspace_storage_path(raw_path: str, fallback_dir: str | None = None) -> str:
    value = os.path.expandvars(os.path.expanduser(str(raw_path or "").strip()))
    if not value:
        base = fallback_dir or str(PROJECT_ROOT)
        return os.path.abspath(base)

    if _is_virtual_data_path(value):
        return os.path.abspath(os.path.join(str(PROJECT_ROOT), value.lstrip("/")))

    if os.path.isabs(value):
        return os.path.abspath(value)

    base = fallback_dir or str(PROJECT_ROOT)
    return os.path.abspath(os.path.join(base, value))


def _resolve_managed_root_path(
    config: Dict[str, Any],
    *,
    default_subdir: str,
) -> str:
    configured_folder = str(config.get("folder_path") or "").strip()
    default_folder = str(PROJECT_ROOT / default_subdir)
    if configured_folder:
        resolved_folder = _resolve_workspace_storage_path(configured_folder, fallback_dir=str(PROJECT_ROOT))
    else:
        resolved_folder = os.path.abspath(default_folder)
    auto_create_folder = bool(config.get("auto_create_folder", True))

    if os.path.exists(resolved_folder):
        if not os.path.isdir(resolved_folder):
            raise ValueError(f"Configured folder is not a directory: {resolved_folder}")
    elif auto_create_folder:
        os.makedirs(resolved_folder, exist_ok=True)
    else:
        raise ValueError(f"Configured folder does not exist: {resolved_folder}")

    return os.path.abspath(resolved_folder)


def _to_relative_path(root_path: str, target_path: str) -> str:
    root = Path(root_path).resolve()
    target = Path(target_path).resolve()
    try:
        relative = target.relative_to(root)
        text = str(relative)
        return text if text else "."
    except Exception:
        return str(target)


def _resolve_safe_managed_file_path(
    *,
    root_path: str,
    raw_path: str,
    required_suffix: str | None = None,
    allowed_suffixes: tuple[str, ...] | None = None,
    allow_outside_folder: bool = False,
) -> str:
    resolved_candidate = _resolve_workspace_storage_path(raw_path, fallback_dir=root_path)
    candidate = Path(resolved_candidate)

    if allowed_suffixes:
        normalized_allowed = tuple(str(item).lower() for item in allowed_suffixes if item)
        if not normalized_allowed:
            normalized_allowed = None
        if normalized_allowed:
            if not candidate.suffix:
                candidate = candidate.with_suffix(normalized_allowed[0])
            elif candidate.suffix.lower() not in normalized_allowed:
                joined = ", ".join(normalized_allowed)
                raise ValueError(f"Expected one of ({joined}) file extensions: {candidate}")
    elif required_suffix:
        normalized_suffix = required_suffix.lower()
        if not candidate.suffix:
            candidate = candidate.with_suffix(normalized_suffix)
        elif candidate.suffix.lower() != normalized_suffix:
            raise ValueError(f"Expected a '{normalized_suffix}' file path: {candidate}")

    resolved = candidate.resolve()
    root = Path(root_path).resolve()
    if not allow_outside_folder:
        try:
            resolved.relative_to(root)
        except Exception as exc:
            raise ValueError(f"Path '{resolved}' is outside the configured folder '{root}'.") from exc

    return str(resolved)


def _resolve_excel_workbook_path(config: Dict[str, Any], workbook_path: str | None = None) -> str:
    configured_workbook = str(config.get("workbook_path") or "").strip()
    requested_workbook = str(workbook_path or "").strip()
    root_path = _resolve_managed_root_path(config, default_subdir="data/spreadsheets")
    raw_path = requested_workbook or configured_workbook or "data.xlsx"
    resolved = _resolve_safe_managed_file_path(
        root_path=root_path,
        raw_path=raw_path,
        allowed_suffixes=(".xlsx", ".xlsm", ".xltx", ".xltm"),
        allow_outside_folder=bool(config.get("allow_outside_folder")),
    )
    parent = os.path.dirname(resolved) or root_path
    if bool(config.get("auto_create_folder", True)):
        os.makedirs(parent, exist_ok=True)
    return resolved


def _guess_sheet_name_from_text(text: str, default_name: str) -> str:
    patterns = [
        r"(?:sheet|onglet|feuille)\s+[\"']([^\"']+)[\"']",
        r"(?:sheet|onglet|feuille)\s+([A-Za-z0-9 _-]{1,50})",
    ]
    lowered = text or ""
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            value = " ".join(str(match.group(1)).split()).strip()
            if value:
                return value
    return default_name


def _guess_sheet_rename_from_text(text: str) -> tuple[str, str] | None:
    patterns = [
        r"(?:rename|renomme(?:r)?|renommer)\s+(?:the\s+|la\s+|le\s+|l')?(?:sheet|onglet|feuille)\s+[\"']([^\"']+)[\"']\s+(?:to|en|vers)\s+[\"']([^\"']+)[\"']",
        r"(?:rename|renomme(?:r)?|renommer)\s+(?:the\s+|la\s+|le\s+|l')?(?:sheet|onglet|feuille)\s+([A-Za-z0-9 _-]{1,50}?)\s+(?:to|en|vers)\s+([A-Za-z0-9 _-]{1,50})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not match:
            continue
        source = " ".join(str(match.group(1)).split()).strip()
        target = " ".join(str(match.group(2)).split()).strip()
        if source and target:
            return source, target
    return None


def _guess_workbook_path_from_text(text: str) -> str | None:
    quoted = re.search(
        r"[\"']([^\"']+\.(?:xlsx|xlsm|xltx|xltm))[\"']",
        text or "",
        flags=re.IGNORECASE,
    )
    if quoted:
        return str(quoted.group(1)).strip()

    bare = re.search(
        r"([A-Za-z]:[\\/][^\s\"']+\.(?:xlsx|xlsm|xltx|xltm)|\.{0,2}[\\/][^\s\"']+\.(?:xlsx|xlsm|xltx|xltm)|[A-Za-z0-9._/-]+\.(?:xlsx|xlsm|xltx|xltm))",
        text or "",
        flags=re.IGNORECASE,
    )
    if bare:
        return str(bare.group(1)).strip()
    return None


def _guess_cell_and_value_from_text(text: str) -> tuple[str, Any] | None:
    patterns = [
        r"(?:cellule|cell)\s+([A-Za-z]{1,3}\d{1,6})\s*(?:=|:|to|avec|with)\s*[\"']([^\"']*)[\"']",
        r"([A-Za-z]{1,3}\d{1,6})\s*(?:=|:)\s*[\"']([^\"']*)[\"']",
        r"(?:cellule|cell)\s+([A-Za-z]{1,3}\d{1,6})\s*(?:=|:|to|avec|with)\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not match:
            continue
        cell = str(match.group(1)).upper().strip()
        value = str(match.group(2)).strip()
        if cell:
            return (cell, value)
    return None


def _normalize_excel_action_payload(action: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(action)
    raw_action_name = str(normalized.get("action") or "").strip().lower()
    action_aliases = {
        "create_file": "create_workbook",
        "new_workbook": "create_workbook",
        "remove_workbook": "delete_workbook",
        "delete_file": "delete_workbook",
        "remove_sheet": "delete_sheet",
        "new_sheet": "create_sheet",
        "add_sheet": "create_sheet",
        "write_cell": "write_cells",
        "set_cell": "write_cells",
        "append_row": "append_rows",
        "read_rows": "read_sheet",
    }
    normalized["action"] = action_aliases.get(raw_action_name, raw_action_name)

    if normalized.get("workbook_path") in (None, ""):
        for key in ("file_path", "file", "path", "workbook", "workbook_file"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["workbook_path"] = value.strip()
                break

    if normalized.get("sheet") in (None, ""):
        for key in ("sheet_name", "worksheet", "tab", "onglet", "feuille"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["sheet"] = value.strip()
                break

    if normalized.get("new_name") in (None, ""):
        for key in ("target_sheet", "new_sheet", "target", "destination", "to"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["new_name"] = value.strip()
                break

    if normalized.get("action") == "write_cells" and not isinstance(normalized.get("cells"), dict):
        cell = normalized.get("cell")
        if cell is not None:
            normalized["cells"] = {str(cell): normalized.get("value")}

    if normalized.get("action") == "append_rows" and not isinstance(normalized.get("rows"), list):
        alt_rows = normalized.get("data") or normalized.get("values")
        if isinstance(alt_rows, list):
            normalized["rows"] = alt_rows

    return normalized


def _normalize_excel_actions(
    user_input: str,
    config: Dict[str, Any],
    allow_heuristics: bool = True,
) -> List[Dict[str, Any]] | None:
    parsed = _extract_possible_json(user_input)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            actions = [_normalize_excel_action_payload(action) for action in parsed["actions"] if isinstance(action, dict)]
            return actions if actions else None
        if isinstance(parsed.get("action"), str):
            return [_normalize_excel_action_payload(parsed)]
    if isinstance(parsed, list):
        actions = [_normalize_excel_action_payload(action) for action in parsed if isinstance(action, dict)]
        if actions:
            return actions

    if not allow_heuristics:
        return None

    text = (user_input or "").strip()
    lowered = text.lower()
    default_sheet = str(config.get("default_sheet") or "Sheet1")
    workbook_hint = _guess_workbook_path_from_text(text)
    heuristics: List[Dict[str, Any]] = []

    def add_action(action: Dict[str, Any]) -> None:
        if workbook_hint and not action.get("workbook_path"):
            action["workbook_path"] = workbook_hint
        heuristics.append(_normalize_excel_action_payload(action))

    if re.search(r"(list|show|affiche|liste).*(sheet|sheets|onglet|onglets|feuille|feuilles)", lowered):
        add_action({"action": "list_sheets"})
    elif re.search(r"(create|new|cr[eé]e|créer|ajoute|add).*(workbook|excel|classeur|fichier)", lowered):
        add_action({"action": "create_workbook"})
    elif re.search(r"(delete|remove|supprime|supprimer).*(workbook|excel|classeur|fichier)", lowered):
        add_action({"action": "delete_workbook"})
    elif re.search(r"(rename|renomme|renommer).*(sheet|onglet|feuille)", lowered):
        rename_pair = _guess_sheet_rename_from_text(text)
        if rename_pair:
            source, target = rename_pair
        else:
            source = _guess_sheet_name_from_text(text, default_sheet)
            target_match = re.search(
                r"(?:to|en|vers)\s+[\"']([^\"']+)[\"']|(?:to|en|vers)\s+([A-Za-z0-9 _-]{1,50})",
                text,
                flags=re.IGNORECASE,
            )
            target = source
            if target_match:
                target = (target_match.group(1) or target_match.group(2) or source).strip()
        add_action({"action": "rename_sheet", "sheet": source, "new_name": target})
    elif re.search(r"(create|new|add|cr[eé]e|créer|ajoute).*(sheet|onglet|feuille)", lowered):
        add_action({"action": "create_sheet", "sheet": _guess_sheet_name_from_text(text, default_sheet)})
    elif re.search(r"(delete|remove|supprime|supprimer).*(sheet|onglet|feuille)", lowered):
        add_action({"action": "delete_sheet", "sheet": _guess_sheet_name_from_text(text, default_sheet)})
    elif re.search(r"(read|lire|show|affiche).*(sheet|onglet|feuille)", lowered):
        add_action(
            {
                "action": "read_sheet",
                "sheet": _guess_sheet_name_from_text(text, default_sheet),
                "max_rows": int(config.get("max_rows_read") or 100),
            }
        )
    else:
        cell_value = _guess_cell_and_value_from_text(text)
        if cell_value:
            cell, value = cell_value
            add_action(
                {
                    "action": "write_cells",
                    "sheet": _guess_sheet_name_from_text(text, default_sheet),
                    "cells": {cell: value},
                }
            )

    return heuristics if heuristics else None


async def _execute_excel_actions(
    user_input: str,
    config: Dict[str, Any],
    on_step: StepCallback | None,
    agent_label: str,
    allow_heuristics: bool = True,
) -> AgentResult | None:
    actions = _normalize_excel_actions(user_input, config, allow_heuristics=allow_heuristics)
    if not actions:
        return None

    try:
        import openpyxl  # type: ignore
    except Exception as exc:
        return {
            "answer": (
                "Excel manager requires the `openpyxl` dependency. "
                f"Install it to enable workbook operations. ({exc})"
            )
        }

    default_sheet = str(config.get("default_sheet") or "Sheet1")
    auto_create_workbook = bool(config.get("auto_create_workbook"))
    allow_overwrite = bool(config.get("allow_overwrite"))
    max_rows_read = int(config.get("max_rows_read") or 100)

    operation_logs: List[Dict[str, Any]] = []
    last_read_rows: List[List[Any]] = []
    last_sheet_names: List[str] = []
    workbook_path_cache: str | None = None

    for action in actions:
        action_name = str(action.get("action") or "").strip().lower()
        try:
            workbook_path = _resolve_excel_workbook_path(config, action.get("workbook_path"))
        except Exception as exc:
            return {
                "answer": f"Unable to resolve workbook path: {exc}",
                "details": {"action": action_name, "requested_workbook_path": action.get("workbook_path")},
            }
        workbook_path_cache = workbook_path
        _emit_step(
            on_step,
            {
                "status": "excel_action_started",
                "agent": agent_label,
                "action": action_name,
                "workbook_path": workbook_path,
            },
        )

        if action_name == "create_workbook":
            if os.path.exists(workbook_path) and not allow_overwrite:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "Workbook already exists and overwrite is disabled.",
                        "workbook_path": workbook_path,
                    }
                )
                _emit_step(
                    on_step,
                    {
                        "status": "excel_action_completed",
                        "agent": agent_label,
                        "action": action_name,
                        "result": "skipped_existing",
                    },
                )
                continue

            workbook = openpyxl.Workbook()
            active = workbook.active
            active.title = str(action.get("sheet") or default_sheet)[:31] or "Sheet1"
            workbook.save(workbook_path)
            operation_logs.append({"action": action_name, "status": "ok", "workbook_path": workbook_path})
            _emit_step(
                on_step,
                {
                    "status": "excel_action_completed",
                    "agent": agent_label,
                    "action": action_name,
                    "result": "created",
                },
            )
            continue

        if action_name == "delete_workbook":
            if os.path.exists(workbook_path):
                os.remove(workbook_path)
                operation_logs.append({"action": action_name, "status": "ok", "workbook_path": workbook_path})
                _emit_step(
                    on_step,
                    {
                        "status": "excel_action_completed",
                        "agent": agent_label,
                        "action": action_name,
                        "result": "deleted",
                    },
                )
            else:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "Workbook does not exist.",
                        "workbook_path": workbook_path,
                    }
                )
            continue

        if not os.path.exists(workbook_path):
            if auto_create_workbook:
                workbook = openpyxl.Workbook()
                workbook.active.title = default_sheet[:31] or "Sheet1"
                workbook.save(workbook_path)
            else:
                return {
                    "answer": (
                        f"Workbook not found at {workbook_path}. "
                        "Enable auto_create_workbook or create it first."
                    ),
                    "details": {
                        "action": action_name,
                        "workbook_path": workbook_path,
                    },
                }

        workbook = openpyxl.load_workbook(workbook_path)
        changed = False

        try:
            if action_name == "list_sheets":
                last_sheet_names = list(workbook.sheetnames)
                operation_logs.append({"action": action_name, "status": "ok", "sheets": last_sheet_names})
            elif action_name == "create_sheet":
                sheet_name = str(action.get("sheet") or default_sheet).strip()[:31] or "Sheet1"
                if sheet_name in workbook.sheetnames:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Sheet already exists.", "sheet": sheet_name}
                    )
                else:
                    workbook.create_sheet(title=sheet_name)
                    changed = True
                    operation_logs.append({"action": action_name, "status": "ok", "sheet": sheet_name})
            elif action_name == "delete_sheet":
                sheet_name = str(action.get("sheet") or default_sheet).strip()
                if sheet_name not in workbook.sheetnames:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Sheet not found.", "sheet": sheet_name}
                    )
                elif len(workbook.sheetnames) <= 1:
                    operation_logs.append(
                        {
                            "action": action_name,
                            "status": "skipped",
                            "reason": "Cannot delete the last remaining sheet.",
                            "sheet": sheet_name,
                        }
                    )
                else:
                    del workbook[sheet_name]
                    changed = True
                    operation_logs.append({"action": action_name, "status": "ok", "sheet": sheet_name})
            elif action_name == "rename_sheet":
                source = str(action.get("sheet") or default_sheet).strip()
                target = str(action.get("new_name") or "").strip()[:31]
                if source not in workbook.sheetnames:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Source sheet not found.", "sheet": source}
                    )
                elif not target:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Target sheet name is empty.", "sheet": source}
                    )
                elif target in workbook.sheetnames:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Target sheet already exists.", "sheet": target}
                    )
                else:
                    workbook[source].title = target
                    changed = True
                    operation_logs.append({"action": action_name, "status": "ok", "sheet": source, "new_name": target})
            elif action_name == "write_cells":
                sheet_name = str(action.get("sheet") or default_sheet).strip() or default_sheet
                if sheet_name not in workbook.sheetnames:
                    workbook.create_sheet(title=sheet_name[:31])
                    changed = True
                sheet = workbook[sheet_name[:31]]
                cells = action.get("cells")
                if not isinstance(cells, dict) or len(cells) == 0:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "No cells payload provided.", "sheet": sheet_name}
                    )
                else:
                    written = 0
                    for cell_ref, value in cells.items():
                        ref = str(cell_ref).upper().strip()
                        if not re.fullmatch(r"[A-Z]{1,3}[1-9][0-9]{0,6}", ref):
                            continue
                        sheet[ref] = value
                        written += 1
                    changed = changed or (written > 0)
                    operation_logs.append(
                        {"action": action_name, "status": "ok", "sheet": sheet_name, "written_cells": written}
                    )
            elif action_name == "append_rows":
                sheet_name = str(action.get("sheet") or default_sheet).strip() or default_sheet
                if sheet_name not in workbook.sheetnames:
                    workbook.create_sheet(title=sheet_name[:31])
                    changed = True
                sheet = workbook[sheet_name[:31]]
                rows_payload = action.get("rows")
                if isinstance(rows_payload, list) and len(rows_payload) > 0:
                    appended = 0
                    for row in rows_payload:
                        if isinstance(row, list):
                            sheet.append(row)
                            appended += 1
                    changed = changed or (appended > 0)
                    operation_logs.append(
                        {"action": action_name, "status": "ok", "sheet": sheet_name, "appended_rows": appended}
                    )
                else:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "No rows payload provided.", "sheet": sheet_name}
                    )
            elif action_name == "read_sheet":
                sheet_name = str(action.get("sheet") or default_sheet).strip() or default_sheet
                if sheet_name not in workbook.sheetnames:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Sheet not found.", "sheet": sheet_name}
                    )
                else:
                    limit = int(action.get("max_rows") or max_rows_read)
                    limit = max(1, min(limit, 5000))
                    sheet = workbook[sheet_name]
                    last_read_rows = []
                    for row in sheet.iter_rows(min_row=1, max_row=limit, values_only=True):
                        last_read_rows.append(list(row))
                    operation_logs.append(
                        {"action": action_name, "status": "ok", "sheet": sheet_name, "rows_read": len(last_read_rows)}
                    )
            else:
                operation_logs.append({"action": action_name, "status": "skipped", "reason": "Unsupported action."})
        finally:
            if changed:
                workbook.save(workbook_path)
            workbook.close()

        _emit_step(
            on_step,
            {
                "status": "excel_action_completed",
                "agent": agent_label,
                "action": action_name,
            },
        )

    summary_parts = [f"{len(operation_logs)} action(s) processed"]
    if workbook_path_cache:
        summary_parts.append(f"workbook={workbook_path_cache}")
    if last_sheet_names:
        summary_parts.append(f"sheets={', '.join(last_sheet_names)}")
    if last_read_rows:
        summary_parts.append(f"rows_read={len(last_read_rows)}")

    return {
        "answer": "Excel operations completed: " + " | ".join(summary_parts),
        "details": {
            "workbook_path": workbook_path_cache,
            "operations": operation_logs,
            "sheet_names": last_sheet_names,
            "rows": last_read_rows[:200],
        },
    }


def _resolve_word_document_path(config: Dict[str, Any], document_path: str | None = None) -> str:
    configured_document = str(config.get("document_path") or "").strip()
    requested_document = str(document_path or "").strip()
    root_path = _resolve_managed_root_path(config, default_subdir="data/documents")
    raw_path = requested_document or configured_document or "report.docx"
    resolved = _resolve_safe_managed_file_path(
        root_path=root_path,
        raw_path=raw_path,
        allowed_suffixes=(".docx", ".docm", ".dotx", ".dotm"),
        allow_outside_folder=bool(config.get("allow_outside_folder")),
    )
    parent = os.path.dirname(resolved) or root_path
    if bool(config.get("auto_create_folder", True)):
        os.makedirs(parent, exist_ok=True)
    return resolved


def _guess_word_document_path_from_text(text: str) -> str | None:
    quoted = re.search(
        r"[\"']([^\"']+\.(?:docx|docm|dotx|dotm))[\"']",
        text or "",
        flags=re.IGNORECASE,
    )
    if quoted:
        return str(quoted.group(1)).strip()

    bare = re.search(
        r"([A-Za-z]:[\\/][^\s\"']+\.(?:docx|docm|dotx|dotm)|\.{0,2}[\\/][^\s\"']+\.(?:docx|docm|dotx|dotm)|[A-Za-z0-9._/-]+\.(?:docx|docm|dotx|dotm))",
        text or "",
        flags=re.IGNORECASE,
    )
    if bare:
        return str(bare.group(1)).strip()
    return None


def _coerce_word_paragraphs(value: Any) -> List[str]:
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines()]
        return [line for line in lines if line]
    if isinstance(value, list):
        paragraphs: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                paragraphs.append(text)
        return paragraphs
    return []


def _guess_replace_text_from_text(text: str) -> tuple[str, str] | None:
    patterns = [
        r"(?:replace|remplace(?:r)?)\s+[\"']([^\"']+)[\"']\s+(?:with|par|to)\s+[\"']([^\"']*)[\"']",
        r"(?:replace|remplace(?:r)?)(?:\s+the)?(?:\s+text)?\s+[\"']([^\"']+)[\"']\s+(?:with|par|to)\s+[\"']([^\"']*)[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            old_text = str(match.group(1)).strip()
            new_text = str(match.group(2))
            if old_text:
                return old_text, new_text
    return None


def _guess_word_payload_from_text(text: str) -> List[str]:
    quoted = [str(item).strip() for item in re.findall(r"[\"']([^\"']+)[\"']", text or "")]
    quoted = [item for item in quoted if item]
    if quoted:
        return quoted

    colon_match = re.search(r"[:：]\s*(.+)$", text or "", flags=re.IGNORECASE)
    if colon_match:
        payload = str(colon_match.group(1)).strip()
        if payload:
            return [payload]
    return []


def _normalize_word_action_payload(action: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(action)
    raw_action_name = str(normalized.get("action") or "").strip().lower()
    action_aliases = {
        "create_file": "create_document",
        "new_document": "create_document",
        "new_doc": "create_document",
        "delete_file": "delete_document",
        "remove_document": "delete_document",
        "remove_doc": "delete_document",
        "read": "read_document",
        "write": "write_paragraphs",
        "overwrite": "write_paragraphs",
        "append": "append_paragraphs",
        "append_text": "append_paragraphs",
        "replace": "replace_text",
    }
    normalized["action"] = action_aliases.get(raw_action_name, raw_action_name)

    if normalized.get("document_path") in (None, ""):
        for key in ("file_path", "file", "path", "document", "doc_path", "document_file"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["document_path"] = value.strip()
                break

    action_name = str(normalized.get("action") or "")

    if action_name in {"write_paragraphs", "append_paragraphs"}:
        raw_paragraphs = normalized.get("paragraphs")
        if raw_paragraphs in (None, ""):
            for key in ("text", "content", "body", "lines", "paragraph"):
                if normalized.get(key) not in (None, ""):
                    raw_paragraphs = normalized.get(key)
                    break
        normalized["paragraphs"] = _coerce_word_paragraphs(raw_paragraphs)

    if action_name == "replace_text":
        old_text = normalized.get("old_text")
        new_text = normalized.get("new_text")
        if old_text in (None, ""):
            for key in ("from", "find", "target", "search", "old"):
                if normalized.get(key) not in (None, ""):
                    old_text = normalized.get(key)
                    break
        if new_text in (None, ""):
            for key in ("to", "replace", "replacement", "new"):
                if normalized.get(key) is not None:
                    new_text = normalized.get(key)
                    break
        normalized["old_text"] = str(old_text or "")
        normalized["new_text"] = str(new_text or "")

    if action_name == "read_document" and normalized.get("max_paragraphs") in (None, ""):
        for key in ("limit", "max_rows", "max_lines"):
            value = normalized.get(key)
            if value not in (None, ""):
                normalized["max_paragraphs"] = value
                break

    return normalized


def _normalize_word_actions(
    user_input: str,
    config: Dict[str, Any],
    allow_heuristics: bool = True,
) -> List[Dict[str, Any]] | None:
    parsed = _extract_possible_json(user_input)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            actions = [_normalize_word_action_payload(action) for action in parsed["actions"] if isinstance(action, dict)]
            return actions if actions else None
        if isinstance(parsed.get("action"), str):
            return [_normalize_word_action_payload(parsed)]
    if isinstance(parsed, list):
        actions = [_normalize_word_action_payload(action) for action in parsed if isinstance(action, dict)]
        if actions:
            return actions

    if not allow_heuristics:
        return None

    text = (user_input or "").strip()
    lowered = text.lower()
    document_hint = _guess_word_document_path_from_text(text)
    max_paragraphs_read = int(config.get("max_paragraphs_read") or 100)
    heuristics: List[Dict[str, Any]] = []

    def add_action(action: Dict[str, Any]) -> None:
        if document_hint and not action.get("document_path"):
            action["document_path"] = document_hint
        heuristics.append(_normalize_word_action_payload(action))

    if re.search(r"(create|new|cr[eé]e|créer|add|ajoute).*(document|word|docx|fichier)", lowered):
        add_action({"action": "create_document"})
    elif re.search(r"(delete|remove|supprime|supprimer).*(document|word|docx|fichier)", lowered):
        add_action({"action": "delete_document"})
    elif re.search(r"(read|lire|show|affiche).*(document|word|docx|fichier)", lowered):
        add_action({"action": "read_document", "max_paragraphs": max_paragraphs_read})
    else:
        replace_payload = _guess_replace_text_from_text(text)
        if replace_payload:
            old_text, new_text = replace_payload
            add_action({"action": "replace_text", "old_text": old_text, "new_text": new_text})
        elif re.search(r"(append|add|ajoute).*(paragraph|paragraphe|texte)", lowered):
            paragraphs = _guess_word_payload_from_text(text)
            if paragraphs:
                add_action({"action": "append_paragraphs", "paragraphs": paragraphs})
        elif re.search(r"(write|ecris|écris|r[eé]dige|overwrite|replace all|remplace le contenu|modifie|modifier|edit)", lowered):
            paragraphs = _guess_word_payload_from_text(text)
            if paragraphs:
                add_action({"action": "write_paragraphs", "paragraphs": paragraphs})

    return heuristics if heuristics else None


def _clear_word_document_paragraphs(document: Any) -> None:
    for paragraph in list(document.paragraphs):
        element = paragraph._element
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)


async def _execute_word_actions(
    user_input: str,
    config: Dict[str, Any],
    on_step: StepCallback | None,
    agent_label: str,
    allow_heuristics: bool = True,
) -> AgentResult | None:
    actions = _normalize_word_actions(user_input, config, allow_heuristics=allow_heuristics)
    if not actions:
        return None

    try:
        from docx import Document  # type: ignore
    except Exception as exc:
        return {
            "answer": (
                "Word manager requires the `python-docx` dependency. "
                f"Install it to enable document operations. ({exc})"
            )
        }

    max_paragraphs_read = int(config.get("max_paragraphs_read") or 100)
    auto_create_document = bool(config.get("auto_create_document"))
    allow_overwrite = bool(config.get("allow_overwrite"))

    operation_logs: List[Dict[str, Any]] = []
    last_read_paragraphs: List[str] = []
    document_path_cache: str | None = None

    for action in actions:
        action_name = str(action.get("action") or "").strip().lower()
        try:
            document_path = _resolve_word_document_path(config, action.get("document_path"))
        except Exception as exc:
            return {
                "answer": f"Unable to resolve document path: {exc}",
                "details": {"action": action_name, "requested_document_path": action.get("document_path")},
            }
        document_path_cache = document_path

        _emit_step(
            on_step,
            {
                "status": "word_action_started",
                "agent": agent_label,
                "action": action_name,
                "document_path": document_path,
            },
        )

        if action_name == "create_document":
            if os.path.exists(document_path) and not allow_overwrite:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "Document already exists and overwrite is disabled.",
                        "document_path": document_path,
                    }
                )
                _emit_step(
                    on_step,
                    {
                        "status": "word_action_completed",
                        "agent": agent_label,
                        "action": action_name,
                        "result": "skipped_existing",
                    },
                )
                continue

            document = Document()
            paragraphs = _coerce_word_paragraphs(action.get("paragraphs"))
            for paragraph in paragraphs:
                document.add_paragraph(paragraph)
            document.save(document_path)
            operation_logs.append(
                {
                    "action": action_name,
                    "status": "ok",
                    "document_path": document_path,
                    "paragraphs_written": len(paragraphs),
                }
            )
            _emit_step(
                on_step,
                {
                    "status": "word_action_completed",
                    "agent": agent_label,
                    "action": action_name,
                    "result": "created",
                },
            )
            continue

        if action_name == "delete_document":
            if os.path.exists(document_path):
                os.remove(document_path)
                operation_logs.append({"action": action_name, "status": "ok", "document_path": document_path})
                _emit_step(
                    on_step,
                    {
                        "status": "word_action_completed",
                        "agent": agent_label,
                        "action": action_name,
                        "result": "deleted",
                    },
                )
            else:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "Document does not exist.",
                        "document_path": document_path,
                    }
                )
            continue

        if not os.path.exists(document_path):
            if auto_create_document:
                document = Document()
                document.save(document_path)
            else:
                return {
                    "answer": (
                        f"Document not found at {document_path}. "
                        "Enable auto_create_document or create it first."
                    ),
                    "details": {
                        "action": action_name,
                        "document_path": document_path,
                    },
                }

        try:
            document = Document(document_path)
        except Exception as exc:
            return {
                "answer": f"Unable to open document at {document_path}: {exc}",
                "details": {"action": action_name, "document_path": document_path},
            }
        changed = False

        if action_name == "read_document":
            limit = int(action.get("max_paragraphs") or max_paragraphs_read)
            limit = max(1, min(limit, 5000))
            last_read_paragraphs = []
            for paragraph in document.paragraphs:
                text = str(paragraph.text or "").strip()
                if not text:
                    continue
                last_read_paragraphs.append(text)
                if len(last_read_paragraphs) >= limit:
                    break
            operation_logs.append(
                {
                    "action": action_name,
                    "status": "ok",
                    "paragraphs_read": len(last_read_paragraphs),
                }
            )
        elif action_name == "write_paragraphs":
            paragraphs = _coerce_word_paragraphs(action.get("paragraphs"))
            if len(paragraphs) == 0:
                operation_logs.append(
                    {"action": action_name, "status": "skipped", "reason": "No paragraph payload provided."}
                )
            else:
                _clear_word_document_paragraphs(document)
                for paragraph in paragraphs:
                    document.add_paragraph(paragraph)
                changed = True
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "ok",
                        "paragraphs_written": len(paragraphs),
                    }
                )
        elif action_name == "append_paragraphs":
            paragraphs = _coerce_word_paragraphs(action.get("paragraphs"))
            if len(paragraphs) == 0:
                operation_logs.append(
                    {"action": action_name, "status": "skipped", "reason": "No paragraph payload provided."}
                )
            else:
                for paragraph in paragraphs:
                    document.add_paragraph(paragraph)
                changed = True
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "ok",
                        "paragraphs_appended": len(paragraphs),
                    }
                )
        elif action_name == "replace_text":
            old_text = str(action.get("old_text") or "")
            new_text = str(action.get("new_text") or "")
            if not old_text:
                operation_logs.append(
                    {"action": action_name, "status": "skipped", "reason": "No source text provided."}
                )
            else:
                replaced_count = 0
                for paragraph in document.paragraphs:
                    content = str(paragraph.text or "")
                    if old_text in content:
                        paragraph.text = content.replace(old_text, new_text)
                        replaced_count += 1
                if replaced_count > 0:
                    changed = True
                    operation_logs.append(
                        {
                            "action": action_name,
                            "status": "ok",
                            "paragraphs_replaced": replaced_count,
                        }
                    )
                else:
                    operation_logs.append(
                        {"action": action_name, "status": "skipped", "reason": "Text not found in document."}
                    )
        else:
            operation_logs.append({"action": action_name, "status": "skipped", "reason": "Unsupported action."})

        if changed:
            document.save(document_path)

        _emit_step(
            on_step,
            {
                "status": "word_action_completed",
                "agent": agent_label,
                "action": action_name,
            },
        )

    summary_parts = [f"{len(operation_logs)} action(s) processed"]
    if document_path_cache:
        summary_parts.append(f"document={document_path_cache}")
    if last_read_paragraphs:
        summary_parts.append(f"paragraphs_read={len(last_read_paragraphs)}")

    return {
        "answer": "Word operations completed: " + " | ".join(summary_parts),
        "details": {
            "document_path": document_path_cache,
            "operations": operation_logs,
            "paragraphs": last_read_paragraphs[:500],
        },
    }


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _looks_like_file_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "/" in text or "\\" in text:
        return True
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", text))


def _resolve_text_file_path(config: Dict[str, Any], file_path: str | None = None) -> tuple[str, str]:
    root_path = _resolve_managed_root_path(config, default_subdir="data/workspace")
    configured_path = str(config.get("default_file_path") or "").strip()
    requested_path = str(file_path or "").strip()
    raw_path = requested_path or configured_path or "output.txt"
    resolved = _resolve_safe_managed_file_path(
        root_path=root_path,
        raw_path=raw_path,
        allow_outside_folder=_coerce_bool(config.get("allow_outside_folder"), default=False),
    )
    return root_path, resolved


def _guess_text_file_path_from_text(text: str) -> str | None:
    quoted = re.search(
        r"[\"']([^\"']+\.(?:txt|md|csv|json|log|yaml|yml|ini|cfg|conf|xml|html|py|js|ts))[\"']",
        text or "",
        flags=re.IGNORECASE,
    )
    if quoted:
        return str(quoted.group(1)).strip()

    bare = re.search(
        r"([A-Za-z]:[\\/][^\s\"']+\.(?:txt|md|csv|json|log|yaml|yml|ini|cfg|conf|xml|html|py|js|ts)|\.{0,2}[\\/][^\s\"']+\.(?:txt|md|csv|json|log|yaml|yml|ini|cfg|conf|xml|html|py|js|ts)|[A-Za-z0-9._/-]+\.(?:txt|md|csv|json|log|yaml|yml|ini|cfg|conf|xml|html|py|js|ts))",
        text or "",
        flags=re.IGNORECASE,
    )
    if bare:
        return str(bare.group(1)).strip()
    return None


def _guess_text_content_from_text(text: str) -> str:
    quoted_values = [str(item).strip() for item in re.findall(r"[\"']([^\"']+)[\"']", text or "")]
    filtered = [item for item in quoted_values if item and not _looks_like_file_path(item)]
    if filtered:
        return filtered[-1]

    colon_match = re.search(r"[:：]\s*(.+)$", text or "", flags=re.IGNORECASE)
    if colon_match:
        return str(colon_match.group(1)).strip()
    return ""


def _normalize_text_action_payload(action: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(action)
    raw_action_name = str(normalized.get("action") or normalized.get("operation") or "").strip().lower()
    action_aliases = {
        "create": "create_file",
        "new_file": "create_file",
        "delete": "delete_file",
        "remove": "delete_file",
        "read": "read_file",
        "write": "write_file",
        "append": "append_file",
        "list": "list_files",
    }
    normalized["action"] = action_aliases.get(raw_action_name, raw_action_name)

    if normalized.get("file_path") in (None, ""):
        for key in ("file", "path", "target_path", "filename"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["file_path"] = value.strip()
                break

    if normalized.get("content") is None:
        for key in ("text", "value", "body", "data"):
            if key in normalized and normalized.get(key) is not None:
                normalized["content"] = normalized.get(key)
                break

    if normalized.get("pattern") in (None, "") and normalized.get("filter") not in (None, ""):
        normalized["pattern"] = normalized.get("filter")

    return normalized


def _normalize_text_file_actions(
    user_input: str,
    config: Dict[str, Any],
    *,
    allow_heuristics: bool = True,
) -> List[Dict[str, Any]] | None:
    parsed = _extract_possible_json(user_input)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            actions = [_normalize_text_action_payload(item) for item in parsed["actions"] if isinstance(item, dict)]
            return actions if actions else None
        if isinstance(parsed.get("action"), str) or isinstance(parsed.get("operation"), str):
            return [_normalize_text_action_payload(parsed)]
    if isinstance(parsed, list):
        actions = [_normalize_text_action_payload(item) for item in parsed if isinstance(item, dict)]
        if actions:
            return actions

    if not allow_heuristics:
        return None

    text = (user_input or "").strip()
    lowered = text.lower()
    guessed_path = _guess_text_file_path_from_text(text)
    guessed_content = _guess_text_content_from_text(text)

    heuristics: List[Dict[str, Any]] = []

    def add_action(action: Dict[str, Any]) -> None:
        if guessed_path and not action.get("file_path"):
            action["file_path"] = guessed_path
        heuristics.append(_normalize_text_action_payload(action))

    if re.search(r"(list|liste|show|affiche).*(files?|fichiers?)", lowered):
        add_action({"action": "list_files"})
    elif re.search(r"(delete|remove|supprime|supprimer).*(files?|fichiers?)", lowered):
        add_action({"action": "delete_file"})
    elif re.search(r"(create|new|cr[eé]e|créer).*(files?|fichiers?)", lowered):
        payload: Dict[str, Any] = {"action": "create_file"}
        if guessed_content:
            payload["content"] = guessed_content
        add_action(payload)
    elif re.search(r"(read|lire|open|ouvre|affiche).*(files?|fichiers?)", lowered):
        add_action({"action": "read_file", "max_chars": int(config.get("max_chars_read") or 10000)})
    elif re.search(r"(append|ajoute|ajouter).*(text|texte|files?|fichiers?)", lowered):
        if guessed_content:
            add_action({"action": "append_file", "content": guessed_content})
    elif re.search(r"(write|ecris|écris|modifie|modifier|replace).*(files?|fichiers?|text|texte)", lowered):
        if guessed_content:
            add_action({"action": "write_file", "content": guessed_content})

    return heuristics if heuristics else None


async def _execute_text_file_actions(
    user_input: str,
    config: Dict[str, Any],
    on_step: StepCallback | None,
    agent_label: str,
    *,
    allow_heuristics: bool = True,
) -> AgentResult | None:
    actions = _normalize_text_file_actions(user_input, config, allow_heuristics=allow_heuristics)
    if not actions:
        return None

    default_encoding = str(config.get("default_encoding") or "utf-8").strip() or "utf-8"
    allow_overwrite = _coerce_bool(config.get("allow_overwrite"), default=True)
    max_chars_read = _coerce_int(config.get("max_chars_read"), default=10000, minimum=200, maximum=500000)

    operation_logs: List[Dict[str, Any]] = []
    listed_files: List[Dict[str, Any]] = []
    preview_by_file: Dict[str, str] = {}
    root_path_cache = ""
    file_path_cache = ""

    for action in actions:
        action_name = str(action.get("action") or "").strip().lower()
        raw_file_path = str(action.get("file_path") or "").strip()
        resolved_file_path = ""

        try:
            if action_name == "list_files":
                root_path = _resolve_managed_root_path(config, default_subdir="data/workspace")
                root_path_cache = root_path
                target_path = root_path
                if raw_file_path:
                    target_path = _resolve_safe_managed_file_path(
                        root_path=root_path,
                        raw_path=raw_file_path,
                        allow_outside_folder=_coerce_bool(config.get("allow_outside_folder"), default=False),
                    )
                resolved_file_path = target_path
            else:
                root_path, resolved_file_path = _resolve_text_file_path(config, raw_file_path or None)
                root_path_cache = root_path
                file_path_cache = resolved_file_path
        except Exception as exc:
            return {
                "answer": f"Unable to resolve text-file path: {exc}",
                "details": {"action": action_name, "requested_file_path": raw_file_path},
            }

        _emit_step(
            on_step,
            {
                "status": "text_action_started",
                "agent": agent_label,
                "action": action_name,
                "file_path": resolved_file_path,
            },
        )

        if action_name == "list_files":
            recursive = _coerce_bool(action.get("recursive"), default=False)
            pattern = str(action.get("pattern") or "").strip().lower()
            entries: List[Dict[str, Any]] = []

            if os.path.isfile(resolved_file_path):
                entries = [
                    {
                        "path": _to_relative_path(root_path_cache, resolved_file_path),
                        "size_bytes": os.path.getsize(resolved_file_path),
                    }
                ]
            elif os.path.isdir(resolved_file_path):
                iterator = Path(resolved_file_path).rglob("*") if recursive else Path(resolved_file_path).glob("*")
                for item in iterator:
                    if not item.is_file():
                        continue
                    relative = _to_relative_path(root_path_cache, str(item))
                    if pattern and pattern not in relative.lower():
                        continue
                    entries.append({"path": relative, "size_bytes": item.stat().st_size})
                    if len(entries) >= 500:
                        break
            else:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "Target path does not exist.",
                        "file_path": resolved_file_path,
                    }
                )
                continue

            listed_files = entries
            operation_logs.append(
                {
                    "action": action_name,
                    "status": "ok",
                    "listed_files": len(entries),
                    "target": _to_relative_path(root_path_cache, resolved_file_path),
                }
            )
            _emit_step(
                on_step,
                {
                    "status": "text_action_completed",
                    "agent": agent_label,
                    "action": action_name,
                    "listed_files": len(entries),
                },
            )
            continue

        if action_name == "read_file":
            if not os.path.exists(resolved_file_path) or not os.path.isfile(resolved_file_path):
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "File does not exist.",
                        "file_path": resolved_file_path,
                    }
                )
                continue

            max_chars = _coerce_int(action.get("max_chars"), default=max_chars_read, minimum=200, maximum=500000)
            content = Path(resolved_file_path).read_text(encoding=default_encoding, errors="ignore")
            preview = content[:max_chars]
            if len(content) > max_chars:
                preview += "\n...[truncated]"
            relative = _to_relative_path(root_path_cache, resolved_file_path)
            preview_by_file[relative] = preview
            operation_logs.append(
                {
                    "action": action_name,
                    "status": "ok",
                    "file_path": relative,
                    "chars_read": len(content),
                }
            )
            _emit_step(
                on_step,
                {
                    "status": "text_action_completed",
                    "agent": agent_label,
                    "action": action_name,
                    "file_path": relative,
                },
            )
            continue

        content = str(action.get("content") or "")
        target = Path(resolved_file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        relative_path = _to_relative_path(root_path_cache, resolved_file_path)

        if action_name == "create_file":
            if target.exists() and not allow_overwrite:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "File already exists and overwrite is disabled.",
                        "file_path": relative_path,
                    }
                )
            else:
                if content:
                    target.write_text(content, encoding=default_encoding)
                else:
                    target.touch(exist_ok=True)
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "ok",
                        "file_path": relative_path,
                        "chars_written": len(content),
                    }
                )
        elif action_name == "write_file":
            if target.exists() and not allow_overwrite:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "Overwrite disabled.",
                        "file_path": relative_path,
                    }
                )
            else:
                target.write_text(content, encoding=default_encoding)
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "ok",
                        "file_path": relative_path,
                        "chars_written": len(content),
                    }
                )
        elif action_name == "append_file":
            prefix = ""
            if _coerce_bool(action.get("prepend_newline"), default=False) and target.exists() and target.stat().st_size > 0:
                prefix = "\n"
            with target.open("a", encoding=default_encoding) as handle:
                handle.write(prefix + content)
            operation_logs.append(
                {
                    "action": action_name,
                    "status": "ok",
                    "file_path": relative_path,
                    "chars_appended": len(prefix + content),
                }
            )
        elif action_name == "delete_file":
            if target.exists():
                target.unlink()
                operation_logs.append({"action": action_name, "status": "ok", "file_path": relative_path})
            else:
                operation_logs.append(
                    {
                        "action": action_name,
                        "status": "skipped",
                        "reason": "File does not exist.",
                        "file_path": relative_path,
                    }
                )
        else:
            operation_logs.append({"action": action_name, "status": "skipped", "reason": "Unsupported action."})

        _emit_step(
            on_step,
            {
                "status": "text_action_completed",
                "agent": agent_label,
                "action": action_name,
                "file_path": relative_path,
            },
        )

    summary_parts = [f"{len(operation_logs)} action(s) processed"]
    if root_path_cache:
        summary_parts.append(f"folder={root_path_cache}")
    if file_path_cache:
        summary_parts.append(f"file={file_path_cache}")
    if listed_files:
        summary_parts.append(f"listed_files={len(listed_files)}")
    if preview_by_file:
        summary_parts.append(f"files_read={len(preview_by_file)}")

    return {
        "answer": "Text file operations completed: " + " | ".join(summary_parts),
        "details": {
            "folder_path": root_path_cache,
            "file_path": file_path_cache,
            "operations": operation_logs,
            "listed_files": listed_files[:500],
            "previews": preview_by_file,
        },
    }


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
            text_result = await _execute_text_file_actions(user_input, config, on_step, agent_label)
            if text_result is not None:
                _emit_step(
                    on_step,
                    {
                        "status": "agent_completed",
                        "agent": agent_label,
                        "message": "Text file operations executed.",
                    },
                )
                return text_result

            system_prompt = (
                f"You are a Text File Manager. Folder: {config.get('folder_path')}. Allow overwrite: "
                f"{bool(config.get('allow_overwrite'))}.\n"
                "If the user asks to list/create/read/write/append/delete files, return JSON actions using this schema:\n"
                "{ \"actions\": [ {\"action\": \"list_files|create_file|read_file|write_file|append_file|delete_file\", "
                "\"file_path\": \"optional\", \"content\": \"optional\", \"max_chars\": 10000, \"recursive\": false, "
                "\"pattern\": \"optional\", \"prepend_newline\": false } ] }\n"
                "Otherwise return your response in JSON format: { \"answer\": \"File operations completed.\" }"
            )
        elif agent_type == "excel_manager":
            excel_result = await _execute_excel_actions(user_input, config, on_step, agent_label)
            if excel_result is not None:
                _emit_step(
                    on_step,
                    {
                        "status": "agent_completed",
                        "agent": agent_label,
                        "message": "Excel operations executed.",
                    },
                )
                return excel_result

            system_prompt = (
                f"You are an Excel Manager. Workbook: {config.get('workbook_path')}.\n"
                "If the user asks to create/modify/delete workbook/sheets/cells, return JSON actions using this schema:\n"
                "{ \"actions\": [ {\"action\": \"create_workbook|delete_workbook|list_sheets|create_sheet|delete_sheet|rename_sheet|write_cells|append_rows|read_sheet\", "
                "\"workbook_path\": \"optional\", \"sheet\": \"optional\", \"new_name\": \"optional\", \"cells\": {\"A1\": \"value\"}, "
                "\"rows\": [[\"v1\", \"v2\"]], \"max_rows\": 100 } ] }\n"
                "Otherwise return your response in JSON format: { \"answer\": \"Excel operations completed.\" }"
            )
        elif agent_type == "word_manager":
            word_result = await _execute_word_actions(user_input, config, on_step, agent_label)
            if word_result is not None:
                _emit_step(
                    on_step,
                    {
                        "status": "agent_completed",
                        "agent": agent_label,
                        "message": "Word operations executed.",
                    },
                )
                return word_result

            system_prompt = (
                f"You are a Word Manager. Document: {config.get('document_path')}.\n"
                "If the user asks to create/modify/delete/read document content, return JSON actions using this schema:\n"
                "{ \"actions\": [ {\"action\": \"create_document|delete_document|read_document|write_paragraphs|append_paragraphs|replace_text\", "
                "\"document_path\": \"optional\", \"paragraphs\": [\"line 1\", \"line 2\"], "
                "\"old_text\": \"optional\", \"new_text\": \"optional\", \"max_paragraphs\": 100 } ] }\n"
                "Otherwise return your response in JSON format: { \"answer\": \"Word operations completed.\" }"
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
            navigation_urls = _extract_urls_from_text(user_input)
            if not navigation_urls:
                start_url = str(config.get("start_url") or "").strip()
                if start_url:
                    navigation_urls = [start_url]

            max_nav_steps = max(1, min(int(config.get("max_steps") or 5), 10))
            default_item_limit = int(config.get("max_links") or config.get("max_items") or 10)
            item_limit = _extract_requested_item_limit(user_input, default_limit=default_item_limit, hard_max=50)
            timeout_seconds = int(config.get("timeout_seconds") or 20)
            same_domain_only = config.get("same_domain_only")
            if same_domain_only is None:
                same_domain_only = True
            same_domain_only = bool(same_domain_only)

            if len(navigation_urls) == 0:
                _emit_step(
                    on_step,
                    {
                        "status": "agent_completed",
                        "agent": agent_label,
                        "message": "No URL provided to web navigator.",
                    },
                )
                return {"answer": "No URL provided. Please include a target URL or configure `start_url`."}

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
            }

            collected_links: List[Dict[str, str]] = []
            failures: List[str] = []
            requested_urls = navigation_urls[:max_nav_steps]

            for target_url in requested_urls:
                _emit_step(
                    on_step,
                    {
                        "status": "navigator_fetch_started",
                        "agent": agent_label,
                        "url": target_url,
                    },
                )
                try:
                    response = await asyncio.to_thread(
                        requests.get,
                        target_url,
                        headers=headers,
                        timeout=timeout_seconds,
                        allow_redirects=True,
                    )
                    if not response.ok:
                        failures.append(f"{target_url} -> HTTP {response.status_code}")
                        _emit_step(
                            on_step,
                            {
                                "status": "navigator_fetch_failed",
                                "agent": agent_label,
                                "url": target_url,
                                "error": f"HTTP {response.status_code}",
                            },
                        )
                        continue

                    resolved_url = response.url or target_url
                    remaining = max(item_limit - len(collected_links), 0)
                    if remaining <= 0:
                        break

                    links = _extract_article_links_from_html(
                        resolved_url,
                        response.text,
                        max_links=remaining,
                        same_domain_only=same_domain_only,
                    )
                    collected_links.extend(links)
                    _emit_step(
                        on_step,
                        {
                            "status": "navigator_fetch_completed",
                            "agent": agent_label,
                            "url": resolved_url,
                            "links_found": len(links),
                        },
                    )

                    if len(collected_links) >= item_limit:
                        break
                except Exception as exc:
                    failures.append(f"{target_url} -> {exc}")
                    _emit_step(
                        on_step,
                        {
                            "status": "navigator_fetch_failed",
                            "agent": agent_label,
                            "url": target_url,
                            "error": str(exc),
                        },
                    )

            if len(collected_links) > 0:
                payload = collected_links[:item_limit]
                answer = json.dumps(payload, ensure_ascii=False, indent=2)
                _emit_step(
                    on_step,
                    {
                        "status": "agent_completed",
                        "agent": agent_label,
                        "message": f"Navigation completed with {len(payload)} links.",
                    },
                )
                return {
                    "answer": answer,
                    "details": {
                        "links": payload,
                        "requested_urls": requested_urls,
                        "failures": failures,
                    },
                }

            fallback_message = (
                "Web navigation failed to extract article links from the target page(s). "
                "The site may block automated requests or require JavaScript rendering."
            )
            if failures:
                fallback_message += f" Details: {' | '.join(failures[:3])}"

            _emit_step(
                on_step,
                {
                    "status": "agent_completed",
                    "agent": agent_label,
                    "message": "Navigation finished with no extractable links.",
                },
            )
            return {"answer": fallback_message, "details": {"requested_urls": requested_urls, "failures": failures}}
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

            if agent_type == "excel_manager":
                excel_llm_result = await _execute_excel_actions(
                    content,
                    config,
                    on_step,
                    agent_label,
                    allow_heuristics=False,
                )
                if excel_llm_result is not None:
                    _emit_step(
                        on_step,
                        {
                            "status": "agent_completed",
                            "agent": agent_label,
                            "message": "Excel operations executed.",
                        },
                    )
                    return excel_llm_result
            if agent_type == "text_file_manager":
                text_llm_result = await _execute_text_file_actions(
                    content,
                    config,
                    on_step,
                    agent_label,
                    allow_heuristics=False,
                )
                if text_llm_result is not None:
                    _emit_step(
                        on_step,
                        {
                            "status": "agent_completed",
                            "agent": agent_label,
                            "message": "Text file operations executed.",
                        },
                    )
                    return text_llm_result
            if agent_type == "word_manager":
                word_llm_result = await _execute_word_actions(
                    content,
                    config,
                    on_step,
                    agent_label,
                    allow_heuristics=False,
                )
                if word_llm_result is not None:
                    _emit_step(
                        on_step,
                        {
                            "status": "agent_completed",
                            "agent": agent_label,
                            "message": "Word operations executed.",
                        },
                    )
                    return word_llm_result

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


def _runtime_unavailable_reason_for_agent(agent: Dict[str, Any]) -> str | None:
    agent_type = str(agent.get("agent_type") or "").strip().lower()
    config = build_agent_execution_config(agent)

    if agent_type == "excel_manager":
        try:
            import openpyxl  # type: ignore  # noqa: F401
        except Exception:
            return "Dependency missing: openpyxl is required."

    if agent_type == "word_manager":
        try:
            from docx import Document  # type: ignore  # noqa: F401
        except Exception:
            return "Dependency missing: python-docx is required."

    if agent_type == "elasticsearch_retriever":
        if not str(config.get("base_url") or "").strip() or not str(config.get("index") or "").strip():
            return "Missing Elasticsearch configuration (base_url/index)."

    return None


def _collect_runtime_unavailable_agents(agents: List[Dict[str, Any]]) -> Dict[int, str]:
    unavailable: Dict[int, str] = {}
    for agent in agents:
        reason = _runtime_unavailable_reason_for_agent(agent)
        if not reason:
            continue
        agent_id = _safe_int(agent.get("id"))
        if agent_id is None:
            continue
        unavailable[agent_id] = reason
    return unavailable


def _normalize_manager_call_input(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    return normalized[:600]


def _manager_call_signature(agent_id: int | None, call_input: str) -> str:
    return f"{agent_id or 0}|{_normalize_manager_call_input(call_input)}"


def _should_mark_agent_unavailable(error_text: str) -> bool:
    normalized = str(error_text or "").lower()
    permanent_markers = (
        "requires the `openpyxl` dependency",
        "requires the `python-docx` dependency",
        "no module named",
        "dependency missing",
        "unsupported llm provider",
    )
    return any(marker in normalized for marker in permanent_markers)


class ManagerState(TypedDict):
    input: str
    active_agents: List[Dict[str, Any]]
    llm: Any
    manager_config: Dict[str, Any]
    steps: List[Dict[str, Any]]
    conversation_history: str
    scratchpad: Dict[str, Any]
    current_step: int
    agent_calls_count: int
    max_agent_calls: int
    max_steps: int
    done: bool
    final_answer: str
    unavailable_agents: Dict[int, str]
    successful_call_signatures: set[str]


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
        active_agents = [agent for agent in agents if agent.get("agent_type") != "manager"]
        unavailable_agents = _collect_runtime_unavailable_agents(active_agents)
        plannable_agents = [
            agent for agent in active_agents if _safe_int(agent.get("id")) not in unavailable_agents
        ]

        state: ManagerState = {
            "input": user_input,
            "active_agents": active_agents,
            "llm": llm,
            "manager_config": cfg,
            "steps": [],
            "conversation_history": f"User Request: {user_input}\n",
            "scratchpad": {},
            "current_step": 0,
            "agent_calls_count": 0,
            "max_agent_calls": int(cfg.get("max_agent_calls") or 20),
            "max_steps": int(cfg.get("max_steps") or 10),
            "done": False,
            "final_answer": "",
            "unavailable_agents": unavailable_agents,
            "successful_call_signatures": set(),
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
                "plannable_agents": len(plannable_agents),
                "unavailable_agents": [
                    {
                        "id": agent.get("id"),
                        "name": agent.get("name"),
                        "agent_type": agent.get("agent_type"),
                        "reason": unavailable_agents.get(_safe_int(agent.get("id")) or -1),
                    }
                    for agent in active_agents
                    if (_safe_int(agent.get("id")) or -1) in unavailable_agents
                ],
                "max_steps": state["max_steps"],
                "max_agent_calls": state["max_agent_calls"],
            }
        )

        async def iterate(current_state: ManagerState) -> ManagerState:
            if current_state["done"]:
                return current_state

            if len(current_state["active_agents"]) == 0:
                current_state["final_answer"] = "No specialized agents are configured. Create at least one non-manager agent."
                add_step(
                    {
                        "status": "manager_final",
                        "answer": current_state["final_answer"],
                        "manager_summary": "No available worker agents.",
                    }
                )
                current_state["done"] = True
                return current_state

            current_plannable_agents = [
                agent
                for agent in current_state["active_agents"]
                if _safe_int(agent.get("id")) not in current_state["unavailable_agents"]
            ]
            if len(current_plannable_agents) == 0:
                current_state["final_answer"] = (
                    "No runnable agents are currently available. "
                    "Fix dependencies/configuration and retry."
                )
                add_step(
                    {
                        "status": "manager_final",
                        "answer": current_state["final_answer"],
                        "manager_summary": "All worker agents are unavailable at runtime.",
                        "unavailable_agents": [
                            {
                                "id": agent.get("id"),
                                "name": agent.get("name"),
                                "agent_type": agent.get("agent_type"),
                                "reason": current_state["unavailable_agents"].get(_safe_int(agent.get("id")) or -1),
                            }
                            for agent in current_state["active_agents"]
                            if (_safe_int(agent.get("id")) or -1) in current_state["unavailable_agents"]
                        ],
                    }
                )
                current_state["done"] = True
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

            agent_catalog = [_agent_catalog_entry(agent) for agent in current_plannable_agents]
            unavailable_summary = [
                {
                    "id": agent.get("id"),
                    "name": agent.get("name"),
                    "agent_type": agent.get("agent_type"),
                    "reason": current_state["unavailable_agents"].get(_safe_int(agent.get("id")) or -1),
                }
                for agent in current_state["active_agents"]
                if (_safe_int(agent.get("id")) or -1) in current_state["unavailable_agents"]
            ]
            manager_prompt = f"""You are an advanced Multi-Agent Orchestrator.
Available agents catalog (always use agent_id when possible and only choose from this list):
{json.dumps(agent_catalog, ensure_ascii=False, indent=2)}

Unavailable agents (DO NOT CALL):
{json.dumps(unavailable_summary, ensure_ascii=False, indent=2)}

Execution budget:
- max_steps: {current_state["max_steps"]}
- current_step: {current_state["current_step"]}
- max_agent_calls: {current_state["max_agent_calls"]}
- agent_calls_used: {current_state["agent_calls_count"]}
- agent_calls_remaining: {max(current_state["max_agent_calls"] - current_state["agent_calls_count"], 0)}

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
2. At EVERY step, re-evaluate your plan based on the new knowledge acquired. Optimize the next calls and avoid redundant repeated calls.
3. You can call multiple agents in parallel if needed (Map-Reduce), but stay within budget.
4. Do NOT send the entire conversation history to agents. Extract ONLY the relevant context and pass it in the "input" field.
5. You can update the "scratchpad" to store variables (schema, metrics, intermediate findings).
6. If you have enough information, return "final_answer" immediately.

Decide the next step. Return ONLY a valid JSON object with the following structure:
{{
  "status": "thinking" | "calling_agent" | "final_answer",
  "current_plan": "Your updated step-by-step plan based on current knowledge",
  "rationale": "Why you are making this decision and how it optimizes the plan",
  "scratchpad_updates": {{ "key": "value" }},
  "calls": [{{"agent_id": 1, "agent_name": "optional fallback", "agent_type": "optional", "input": "highly detailed instruction with only required context"}}],
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
                    remaining_agent_calls = current_state["max_agent_calls"] - current_state["agent_calls_count"]
                    if remaining_agent_calls <= 0:
                        current_state["final_answer"] = "Reached max agent-call budget. Provide a final answer from gathered evidence."
                        add_step(
                            {
                                "status": "manager_final",
                                "answer": current_state["final_answer"],
                                "manager_summary": "Max agent-call budget reached.",
                            }
                        )
                        current_state["done"] = True
                    else:
                        raw_calls = [call for call in decision["calls"] if isinstance(call, dict)]
                        bounded_calls = raw_calls[:remaining_agent_calls]

                        unique_calls: List[Dict[str, Any]] = []
                        seen_signatures: set[str] = set()
                        for call in bounded_calls:
                            signature = "|".join(
                                [
                                    str(call.get("agent_id") or ""),
                                    _normalize_identifier(call.get("agent_name") or call.get("agent") or call.get("agent_type")),
                                    str(call.get("input") or "").strip(),
                                ]
                            )
                            if signature in seen_signatures:
                                continue
                            seen_signatures.add(signature)
                            unique_calls.append(call)

                        resolved_calls: List[tuple[Dict[str, Any], Dict[str, Any], str, str]] = []
                        for call in unique_calls:
                            selected = _resolve_agent_from_call(call, current_plannable_agents)
                            if selected is None:
                                add_step(
                                    {
                                        "status": "agent_not_found",
                                        "requested_call": call,
                                        "available_agents": [
                                            {
                                                "id": agent.get("id"),
                                                "name": agent.get("name"),
                                                "agent_type": agent.get("agent_type"),
                                            }
                                            for agent in current_plannable_agents
                                        ],
                                    }
                                )
                                continue
                            call_input = str(call.get("input") or "").strip() or str(current_state["input"])
                            selected_id = _safe_int(selected.get("id"))
                            signature = _manager_call_signature(selected_id, call_input)
                            if signature in current_state["successful_call_signatures"]:
                                add_step(
                                    {
                                        "status": "manager_warning",
                                        "message": (
                                            "Skipping redundant call because an equivalent successful "
                                            f"call already exists for agent '{selected.get('name')}'."
                                        ),
                                        "agent": selected.get("name"),
                                        "agent_id": selected_id,
                                        "input": call_input,
                                    }
                                )
                                continue
                            resolved_calls.append((call, selected, call_input, signature))

                        if len(resolved_calls) == 0:
                            add_step(
                                {
                                    "status": "manager_no_valid_calls",
                                    "message": "Manager requested calls, but none matched existing agents.",
                                }
                            )
                        else:
                            current_state["agent_calls_count"] += len(resolved_calls)
                            add_step(
                                {
                                    "status": "manager_dispatch",
                                    "calls_dispatched": len(resolved_calls),
                                    "agent_calls_used": current_state["agent_calls_count"],
                                    "agent_calls_remaining": max(
                                        current_state["max_agent_calls"] - current_state["agent_calls_count"],
                                        0,
                                    ),
                                }
                            )

                            async def execute_call(
                                call: Dict[str, Any],
                                selected: Dict[str, Any],
                                call_input: str,
                                call_signature: str,
                            ) -> str:
                                selected_id = _safe_int(selected.get("id"))
                                add_step(
                                    {
                                        "status": "agent_call_started",
                                        "agent": selected.get("name"),
                                        "agent_id": selected_id,
                                        "agent_type": selected.get("agent_type"),
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
                                            "agent_id": selected_id,
                                            "result": result,
                                        }
                                    )
                                    if result.get("sql"):
                                        add_step({"status": "sql_generated", "sql": result["sql"]})
                                    answer_text = str(result.get("answer") or "").strip()
                                    if answer_text:
                                        current_state["successful_call_signatures"].add(call_signature)
                                    if _should_mark_agent_unavailable(answer_text) and selected_id is not None:
                                        current_state["unavailable_agents"][selected_id] = answer_text[:500]
                                        add_step(
                                            {
                                                "status": "agent_marked_unavailable",
                                                "agent": selected.get("name"),
                                                "agent_id": selected_id,
                                                "reason": answer_text,
                                            }
                                        )
                                    return f"\nAgent {selected.get('name')} responded: {result.get('answer')}\n"
                                except Exception as exc:
                                    error_text = str(exc)
                                    add_step(
                                        {
                                            "status": "agent_call_failed",
                                            "agent": selected.get("name"),
                                            "agent_id": selected_id,
                                            "error": error_text,
                                        }
                                    )
                                    if _should_mark_agent_unavailable(error_text) and selected_id is not None:
                                        current_state["unavailable_agents"][selected_id] = error_text[:500]
                                        add_step(
                                            {
                                                "status": "agent_marked_unavailable",
                                                "agent": selected.get("name"),
                                                "agent_id": selected_id,
                                                "reason": error_text,
                                            }
                                        )
                                    return f"\nAgent {selected.get('name')} failed: {exc}\n"

                            results = await asyncio.gather(
                                *(
                                    execute_call(call, selected, call_input, signature)
                                    for call, selected, call_input, signature in resolved_calls
                                )
                            )
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
