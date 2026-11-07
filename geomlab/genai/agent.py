"""Single tool-calling agent (doc §7.3: "a single tool-calling agent … is enough and easier to
defend than a complex multi-agent system"). Provider-switchable (doc §7.2 "local for dev, API
for demo"):

  LLM_PROVIDER=anthropic  Claude via the Anthropic SDK (default model claude-opus-5-5)
  LLM_PROVIDER=ollama     a local open model via Ollama's /api/chat tool calling

The loop is written by hand (not the SDK's beta tool runner) because the same loop must
drive both providers and collect citations from tool outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from geomlab.genai.prompts import system_prompt
from geomlab.logging import get_logger
from geomlab.settings import get_settings
from geomlab.tools import registry

log = get_logger(__name__)
MAX_STEPS = 8


@dataclass
class ToolTrace:
    name: str
    args: dict
    is_error: bool
    output: dict


@dataclass
class AgentAnswer:
    answer: str
    tool_calls: list[ToolTrace] = field(default_factory=list)
    model: str = ""
    stop_reason: str = ""

    @property
    def citations(self) -> dict[str, list]:
        """Model vs document evidence, kept separate (doc §7.4.2 'combine, do not blend')."""
        model_ev, docs, versions = [], [], set()
        for t in self.tool_calls:
            if t.is_error:
                continue
            out = t.output
            if out.get("model_version"):
                versions.add((out["model_version"], out.get("data_date")))
            if t.name == "recommend_locations":
                model_ev += [{"h3_id": r["h3_id"], "business_type": out["business_type"], "rank": r["rank"]}
                             for r in out.get("results", [])]
            elif t.name in ("explain_location", "simulate_location"):
                model_ev.append({"h3_id": out["h3_id"], "business_type": out["business_type"]})
            elif t.name == "search_documents":
                docs += [{"document": r["document"], "page": r["page"], "section": r["section"],
                          "similarity": r["similarity"]} for r in out.get("results", [])]
        return {"model": model_ev, "documents": docs,
                "model_versions": [{"model_version": v, "data_date": d} for v, d in sorted(versions)]}

    def to_dict(self) -> dict:
        return {"answer": self.answer, "citations": self.citations, "model": self.model,
                "stop_reason": self.stop_reason,
                "tool_calls": [{"name": t.name, "args": t.args, "is_error": t.is_error} for t in self.tool_calls]}


def _run_tool(name: str, args: dict, trace: list[ToolTrace]) -> tuple[str, bool]:
    payload, is_error = registry.call(name, args)
    trace.append(ToolTrace(name, args, is_error, payload))
    log.info("tool_call name=%s error=%s", name, is_error)
    return registry.to_json(payload), is_error


# ----------------------------------------------------------------------------- Claude
class AnthropicProvider:
    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or get_settings().llm_model

    def run(self, history: list[dict], tools: list[dict]) -> AgentAnswer:
        trace: list[ToolTrace] = []
        messages: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in history]
        tool_defs = tools  # inputs are validated by the tool functions themselves
        stop = ""
        for _ in range(MAX_STEPS):
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system_prompt(),
                tools=tool_defs,
                messages=messages,
                output_config={"effort": "medium"},
                # On a safety-classifier decline, re-run server-side on Anthropic's recommended fallback.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            stop = resp.stop_reason or ""
            if stop == "refusal":
                return AgentAnswer("I can't help with this request.", trace, resp.model, stop)
            # Append the full content (tool_use + thinking blocks) — required for the next turn.
            messages.append({"role": "assistant", "content": resp.content})
            if stop == "pause_turn":
                continue
            calls = [b for b in resp.content if b.type == "tool_use"]
            if stop != "tool_use" or not calls:
                text = "".join(b.text for b in resp.content if b.type == "text")
                return AgentAnswer(text, trace, resp.model, stop)
            results = []
            for b in calls:  # all results go back in ONE user message
                out, err = _run_tool(b.name, dict(b.input), trace)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
            messages.append({"role": "user", "content": results})
        return AgentAnswer("I could not finish within the allowed number of steps.", trace, self.model, stop)


# ----------------------------------------------------------------------------- Ollama
class OllamaProvider:
    def __init__(self, model: str | None = None):
        s = get_settings()
        self.url, self.model = s.ollama_url.rstrip("/"), model or s.ollama_model

    def run(self, history: list[dict], tools: list[dict]) -> AgentAnswer:
        trace: list[ToolTrace] = []
        fn_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                      "parameters": t["input_schema"]}} for t in tools]
        messages = [{"role": "system", "content": system_prompt()}] + [
            {"role": m["role"], "content": m["content"]} for m in history]
        for _ in range(MAX_STEPS):
            r = httpx.post(f"{self.url}/api/chat", json={"model": self.model, "messages": messages, "tools": fn_tools,
                                                         "stream": False, "options": {"temperature": 0.1}}, timeout=600)
            r.raise_for_status()
            msg = r.json()["message"]
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return AgentAnswer(msg.get("content", ""), trace, self.model, "end_turn")
            for c in calls:
                fn = c["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                out, _ = _run_tool(fn["name"], args, trace)
                messages.append({"role": "tool", "content": out, "tool_name": fn["name"]})
        return AgentAnswer("I could not finish within the allowed number of steps.", trace, self.model, "max_steps")


def get_provider(name: str | None = None):
    name = (name or get_settings().llm_provider).lower()
    return OllamaProvider() if name == "ollama" else AnthropicProvider()


def ask(question: str, history: list[dict] | None = None, provider: str | None = None,
        enabled_tools: list[str] | None = None) -> AgentAnswer:
    """``history`` is a list of {role: user|assistant, content: str} from previous turns."""
    msgs = list(history or []) + [{"role": "user", "content": question}]
    return get_provider(provider).run(msgs, registry.tool_schemas(enabled_tools))
