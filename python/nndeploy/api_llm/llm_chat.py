# llm_chat_node.py
# -*- coding: utf-8 -*-

import os
import json
import time
import logging
import random
import typing as T
from dataclasses import dataclass

import requests  # Used for general HTTP requests (OpenAI-compatible /v1/chat/completions)

import nndeploy.dag
from nndeploy.base import Status


# ──────────────────────────────────────────────────────────────────────────────
# Parameter structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMParams:
    api_base: str = "https://api.openai.com"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: str = "You are a helpful assistant."
    timeout_s: float = 60.0
    retry: int = 2
    retry_backoff_s: float = 1.2
    stream: bool = False
    response_json_mode: bool = False
    json_schema: T.Optional[dict] = None


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def _read_api_key(env_name: str) -> str:
    key = os.getenv(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"[LLMChatNode] Missing API key. Please set environment variable: {env_name}"
        )
    return key


def _normalize_messages(
    user_input: T.Union[str, dict, list],
    system_prompt: str
) -> list:
    """
    Normalize user input into an OpenAI-compatible messages array:
      - str: treated as one user message
      - list[dict]: if already in messages format (with role/content), use directly; otherwise convert to string
      - dict: if contains "messages" list, use directly; otherwise convert whole dict to string
    Ensure that there is always one system message (add if missing).
    """
    sys_msg = {"role": "system", "content": system_prompt}

    if isinstance(user_input, str):
        return [sys_msg, {"role": "user", "content": user_input}]

    if isinstance(user_input, list):
        ok = all(isinstance(m, dict) and "role" in m and "content" in m for m in user_input)
        if ok:
            has_system = any(m.get("role") == "system" for m in user_input)
            return ([sys_msg] + user_input) if not has_system else user_input
        return [sys_msg, {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}]

    if isinstance(user_input, dict):
        if "messages" in user_input and isinstance(user_input["messages"], list):
            msgs = user_input["messages"]
            has_system = any(isinstance(m, dict) and m.get("role") == "system" for m in msgs)
            return ([sys_msg] + msgs) if not has_system else msgs
        return [sys_msg, {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}]

    # Fallback for other types
    return [sys_msg, {"role": "user", "content": str(user_input)}]


def _calc_backoff(attempt: int, base: float = 1.2, jitter: float = 0.3) -> float:
    """Exponential backoff with jitter."""
    return (base ** attempt) * (1.0 + random.uniform(-jitter, jitter))


def _should_stop(node) -> bool:
    """Respect NNDeploy interruption if available."""
    try:
        return hasattr(node, "checkInterruptStatus") and node.checkInterruptStatus()
    except Exception:
        return False


def _mask_key(key: str) -> str:
    """Return a masked preview like 'sk-****abcd' (no real key leakage)."""
    if not key:
        return ""
    tail = key[-4:]
    prefix = "sk-" if key.startswith("sk-") else ""
    return f"{prefix}****{tail}"


# ──────────────────────────────────────────────────────────────────────────────
# Core HTTP caller
# ──────────────────────────────────────────────────────────────────────────────

def call_llm(
    params: LLMParams,
    messages: list,
    *,
    api_key_override: str = None,
    node_for_interrupt: T.Optional[nndeploy.dag.Node] = None,
    emit_usage_to_log: bool = True,
    stream_print: bool = False,            # print streaming pieces to stdout
    proxies: T.Optional[dict] = None,      # e.g. {"http": "...", "https": "..."}
    verify: T.Union[bool, str, None] = None # True/False or CA bundle path
) -> str:
    """
    Call an OpenAI-compatible Chat Completions endpoint.
    - Supports non-stream and stream modes.
    - Respects 429 Retry-After; uses exponential backoff with jitter.
    - Logs usage if available.
    Returns the final concatenated content as a string.
    """
    api_key = (api_key_override or _read_api_key(params.api_key_env)).strip()
    url = params.api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": params.model,
        "messages": messages,
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "stream": bool(params.stream),
    }

    # JSON strict mode (if provider supports OpenAI response_format)
    if params.response_json_mode:
        payload["response_format"] = {"type": "json_object"}
        if isinstance(params.json_schema, dict):
            # Some providers may ignore 'json_schema'; keep best-effort
            payload["response_format"]["json_schema"] = params.json_schema

    # Proxies & TLS verification (when not provided by caller)
    if proxies is None:
        proxies = {
            "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
            "https": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
        }
    if verify is None:
        verify = os.getenv("REQUESTS_CA_BUNDLE", True)  # custom CA bundle path or bool

    # Helper to check interrupt between retries/stream chunks
    def _interrupted() -> bool:
        return _should_stop(node_for_interrupt)

    last_err = None
    for attempt in range(params.retry + 1):
        if _interrupted():
            return ""  # Return empty on interrupt

        try:
            if not payload["stream"]:
                # Non-streaming
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=params.timeout_s,
                    proxies=proxies, verify=verify
                )
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else _calc_backoff(attempt + 1, base=params.retry_backoff_s)
                    logging.warning(f"[LLMChatNode] 429 Too Many Requests; sleeping {sleep_s:.2f}s")
                    time.sleep(sleep_s)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"[LLMChatNode] HTTP {resp.status_code}: {resp.text[:800]}")

                data = resp.json()
                # optional usage logging
                usage = data.get("usage", {})
                if emit_usage_to_log and usage:
                    logging.info(f"[LLMChatNode] usage={usage}")

                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"[LLMChatNode] Empty choices: {data}")
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                return content

            else:
                # Streaming
                chunks: list[str] = []
                with requests.post(
                    url, headers=headers, json=payload, timeout=params.timeout_s,
                    stream=True, proxies=proxies, verify=verify
                ) as r:
                    if r.status_code == 429:
                        retry_after = r.headers.get("Retry-After")
                        sleep_s = float(retry_after) if retry_after else _calc_backoff(attempt + 1, base=params.retry_backoff_s)
                        logging.warning(f"[LLMChatNode] 429 (stream); sleeping {sleep_s:.2f}s")
                        time.sleep(sleep_s)
                        continue
                    if r.status_code >= 400:
                        raise RuntimeError(f"[LLMChatNode] HTTP {r.status_code}: {r.text[:800]}")

                    for raw in r.iter_lines(decode_unicode=True):
                        if _interrupted():
                            return "".join(chunks)
                        if not raw:
                            continue
                        # SSE line format: "data: {...}" or "data: [DONE]"
                        if not raw.startswith("data:"):
                            continue
                        data_line = raw[5:].strip()
                        if data_line == "[DONE]":
                            break
                        try:
                            delta = json.loads(data_line)
                            choice0 = (delta.get("choices") or [{}])[0]
                            piece = (choice0.get("delta") or {}).get("content")
                            # Some providers may stream full 'message' frames (rare)
                            if piece is None:
                                piece = (choice0.get("message") or {}).get("content")
                            if piece:
                                chunks.append(piece)
                                if stream_print:
                                    print(piece, end="", flush=True)
                        except Exception:
                            # Ignore malformed lines
                            continue

                # No usage is guaranteed in streaming mode
                return "".join(chunks)

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt < params.retry:
                sleep_s = _calc_backoff(attempt + 1, base=params.retry_backoff_s)
                logging.warning(f"[LLMChatNode] network issue; retry in {sleep_s:.2f}s: {e}")
                time.sleep(sleep_s)
                continue
            break
        except Exception as e:
            last_err = e
            if attempt < params.retry:
                sleep_s = _calc_backoff(attempt + 1, base=params.retry_backoff_s)
                logging.warning(f"[LLMChatNode] call failed; retry in {sleep_s:.2f}s: {e}")
                time.sleep(sleep_s)
                continue
            break

    raise RuntimeError(f"[LLMChatNode] call failed after retries: {last_err}")


# ──────────────────────────────────────────────────────────────────────────────
# Node implementation
# ──────────────────────────────────────────────────────────────────────────────

class LLMChatNode(nndeploy.dag.Node):
    """
    LLM chat node (OpenAI-compatible)
    Input: str / list[messages] / dict({"messages":[...]})
    Output: str (model generated text)
    """
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.api_llm.LLMChatNode")
        super().set_desc("LLM chat node (OpenAI-compatible)")
        self.set_input_type(str)
        self.set_output_type(str)
        self.set_node_type(nndeploy.dag.NodeType.Intermediate)

        # Frontend editable parameters (exposed via serialize/deserialize)
        self.api_base = "https://api.openai.com"
        self.api_key_env = "OPENAI_API_KEY"  # environment variable to read Key (optional)
        self.api_key = ""                    # allow frontend to provide Key inline (priority if allowed)
        self.allow_inline_api_key = True     # security switch for inline key usage

        self.model = "gpt-4o-mini"
        self.temperature = 0.2
        self.max_tokens = 512
        self.system_prompt = "You are a helpful assistant."
        self.timeout_s = 60.0
        self.retry = 2
        self.retry_backoff_s = 1.2

        # New features
        self.stream = False                  # streaming mode
        self.stream_print = False            # print streaming pieces to stdout
        self.response_json_mode = False      # enforce JSON object responses (if supported)
        self.json_schema: T.Optional[dict] = None
        self.emit_usage_to_log = True        # log token usage if present
        self.verify_tls: T.Union[bool, str] = True  # True/False or CA bundle path

        # NEW: Proxies as frontend parameters
        self.http_proxy: str = ""            # e.g. "http://user:pass@host:port"
        self.https_proxy: str = ""           # e.g. "https://user:pass@host:port"

        # Aggregate into a dict for frontend rendering/editing
        self.frontend_params = {
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "api_key": self.api_key,                     # UI should render this field; value won't be persisted
            "allow_inline_api_key": self.allow_inline_api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "timeout_s": self.timeout_s,
            "retry": self.retry,
            "retry_backoff_s": self.retry_backoff_s,
            "stream": self.stream,
            "stream_print": self.stream_print,
            "response_json_mode": self.response_json_mode,
            "json_schema": self.json_schema,
            "emit_usage_to_log": self.emit_usage_to_log,
            "verify_tls": self.verify_tls,
            # NEW: proxies exposed to UI
            "http_proxy": self.http_proxy,
            "https_proxy": self.https_proxy,
        }

    # Sync frontend parameters back to instance variables
    def _sync_params_from_frontend(self):
        fp = self.frontend_params or {}
        self.api_base = str(fp.get("api_base", self.api_base))
        self.api_key_env = str(fp.get("api_key_env", self.api_key_env))
        self.api_key = str(fp.get("api_key", self.api_key))  # UI can set it each time
        self.allow_inline_api_key = bool(fp.get("allow_inline_api_key", self.allow_inline_api_key))

        self.model = str(fp.get("model", self.model))
        self.temperature = float(fp.get("temperature", self.temperature))
        self.max_tokens = int(fp.get("max_tokens", self.max_tokens))
        self.system_prompt = str(fp.get("system_prompt", self.system_prompt))
        self.timeout_s = float(fp.get("timeout_s", self.timeout_s))
        self.retry = int(fp.get("retry", self.retry))
        self.retry_backoff_s = float(fp.get("retry_backoff_s", self.retry_backoff_s))

        self.stream = bool(fp.get("stream", self.stream))
        self.stream_print = bool(fp.get("stream_print", self.stream_print))
        self.response_json_mode = bool(fp.get("response_json_mode", self.response_json_mode))
        self.json_schema = fp.get("json_schema", self.json_schema)
        self.emit_usage_to_log = bool(fp.get("emit_usage_to_log", self.emit_usage_to_log))
        self.verify_tls = fp.get("verify_tls", self.verify_tls)

        # NEW: proxies
        self.http_proxy = str(fp.get("http_proxy", self.http_proxy))
        self.https_proxy = str(fp.get("https_proxy", self.https_proxy))

    def run(self):
        # 0) Interrupt guard
        if _should_stop(self):
            return Status.ok()

        # 1) Get input
        input_edge = self.get_input(0)
        user_input = input_edge.get(self)  # May be str / list / dict

        # 2) Sync parameters
        self._sync_params_from_frontend()

        # 3) API key selection (priority: inline key if allowed > environment variable)
        api_key = ""
        if self.allow_inline_api_key:
            api_key = (self.api_key or "").strip()
        if not api_key:
            api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            err = (f"[LLMChatNode] Missing API key. "
                   f"Please set 'frontend_params.api_key' (if allowed) or environment variable '{self.api_key_env}'.")
            logging.error(err)
            output_edge = self.get_output(0)
            output_edge.set(err)
            return Status.ok()

        # 4) Build params object and messages
        params = LLMParams(
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            timeout_s=self.timeout_s,
            retry=self.retry,
            retry_backoff_s=self.retry_backoff_s,
            stream=self.stream,
            response_json_mode=self.response_json_mode,
            json_schema=self.json_schema if isinstance(self.json_schema, dict) else None,
        )
        messages = _normalize_messages(user_input, params.system_prompt)

        # 5) Log target (mask scheme)
        safe_base = (self.api_base or "").replace("https://", "").replace("http://", "")
        logging.info(f"[LLMChatNode] base={safe_base} model={self.model} stream={self.stream}")

        # 6) Build proxies from frontend or environment
        proxies = {
            "http": self.http_proxy or os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
            "https": self.https_proxy or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
        }

        # 7) Call LLM
        try:
            text = call_llm(
                params,
                messages,
                api_key_override=api_key,
                node_for_interrupt=self,
                emit_usage_to_log=self.emit_usage_to_log,
                stream_print=self.stream_print,
                proxies=proxies,
                verify=self.verify_tls
            )
        except Exception as e:
            logging.exception("[LLMChatNode] LLM call failed")
            text = f"[LLMChatNode ERROR] {e}"

        # 8) Write output to edge 0
        output_edge = self.get_output(0)
        output_edge.set(text)
        return Status.ok()

    # Serialization / Deserialization
    def serialize(self):
        base = json.loads(super().serialize())
        fp = dict(self.frontend_params)

        # Always include the api_key field for UI rendering, but NEVER return the real value
        fp["api_key"] = ""

        # Provide status & masked preview so UI can show "configured" state
        has_key = bool(self.api_key)
        fp["has_inline_api_key"] = has_key
        if has_key:
            fp["api_key_preview"] = _mask_key(self.api_key)

        # Proxies are returned as-is so UI can render & persist them
        # (If you don't want to persist proxies with credentials, mask or drop them here.)

        base["frontend_params"] = fp
        return json.dumps(base, ensure_ascii=False)

    def deserialize(self, target: str):
        obj = json.loads(target)
        if "frontend_params" in obj:
            self.frontend_params = obj["frontend_params"] or self.frontend_params
        self._sync_params_from_frontend()
        return super().deserialize(target)


# ──────────────────────────────────────────────────────────────────────────────
# Node creator & registration
# ──────────────────────────────────────────────────────────────────────────────

class LLMChatNodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        self.node: T.Optional[LLMChatNode] = None

    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = LLMChatNode(name, inputs, outputs)
        return self.node


# Register into NNDeploy
llm_chat_node_creator = LLMChatNodeCreator()
nndeploy.dag.register_node("nndeploy.api_llm.LLMChatNode", llm_chat_node_creator)
