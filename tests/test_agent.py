"""Agent loop with a fake Anthropic client: tool calls are executed, results go back in a
single user message, and citations separate model and document evidence."""

from types import SimpleNamespace

from geomlab.genai import agent


def _block(**kw):
    return SimpleNamespace(**kw)


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        if len(self.calls) == 1:
            return SimpleNamespace(stop_reason="tool_use", model="claude-opus-5-5", content=[
                _block(type="tool_use", id="t1", name="recommend_locations", input={"business_type": "cafe", "city": "rabat"}),
                _block(type="tool_use", id="t2", name="search_documents", input={"query": "licence café"}),
            ])
        return SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                               content=[_block(type="text", text="Top: 89abc… (modèle v1)")])


def test_anthropic_loop(monkeypatch, env):
    fake = FakeMessages()
    prov = agent.AnthropicProvider.__new__(agent.AnthropicProvider)
    prov.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    prov.model = "claude-opus-5-5"

    def fake_call(name, args):
        if name == "recommend_locations":
            return {"business_type": "cafe", "model_version": "m:v1", "data_date": "2026-11-01",
                    "results": [{"h3_id": "89abc", "rank": 1}]}, False
        return {"results": [{"document": "Guide licences", "page": 3, "section": "Café", "similarity": 0.7}]}, False

    monkeypatch.setattr(agent.registry, "call", fake_call)
    ans = prov.run([{"role": "user", "content": "Où ouvrir un café à Rabat ?"}], agent.registry.tool_schemas())
    assert ans.answer.startswith("Top")
    second = fake.calls[1]["messages"]
    assert second[-1]["role"] == "user" and len(second[-1]["content"]) == 2  # both results in one message
    assert fake.calls[0]["model"] == "claude-opus-5-5"
    cites = ans.citations
    assert cites["model"] == [{"h3_id": "89abc", "business_type": "cafe", "rank": 1}]
    assert cites["documents"][0]["page"] == 3
    assert cites["model_versions"] == [{"model_version": "m:v1", "data_date": "2026-11-01"}]


def test_registry_returns_errors_to_the_llm(env):
    from geomlab.tools import registry

    out, is_error = registry.call("recommend_locations", {"business_type": "casino", "city": "rabat"})
    assert is_error and "pharmacy" in out["valid_values"]
    out, is_error = registry.call("explain_location", {"h3_id": "nope", "business_type": "cafe"})
    assert is_error and out["field"] == "h3_id"


def test_api_validation_without_db(env):
    from fastapi.testclient import TestClient

    from geomlab.api.main import app

    c = TestClient(app)
    assert {x["slug"] for x in c.get("/meta/cities").json()} >= {"casablanca", "rabat", "marrakech"}
    r = c.get("/recommendations", params={"business_type": "cafe", "city": "paris"})
    assert r.status_code == 422 and r.json()["field"] == "city"


def test_chat_without_llm_credentials_is_a_clear_503(env, monkeypatch):
    from fastapi.testclient import TestClient

    from geomlab.api.main import app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = TestClient(app).post("/chat", json={"message": "bonjour", "provider": "anthropic"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


class FakeStream:
    """Stands in for ``httpx.stream``: replays NDJSON chunks, one scripted response per call."""

    def __init__(self, responses):
        self.responses, self.bodies = list(responses), []

    def __call__(self, method, url, json, timeout):  # noqa: A002 - mirrors httpx's signature
        self.bodies.append({**json, "messages": list(json.get("messages", []))})
        lines = self.responses.pop(0)
        stream = SimpleNamespace(status_code=200, iter_lines=lambda: iter(lines))

        class _Ctx:
            def __enter__(self_inner):
                return stream

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


def _nd(*chunks):
    import json

    return [json.dumps(c) for c in chunks]


def test_ollama_loop_streams_and_compacts(monkeypatch, env):
    fake = FakeStream([
        _nd({"message": {"content": "", "tool_calls": [{"function": {
            "name": "recommend_locations", "arguments": {"business_type": "cafe", "city": "rabat"}}}]}, "done": True}),
        _nd({"message": {"content": "Top "}}, {"message": {"content": "89abc"}}, {"done": True, "message": {}}),
    ])
    monkeypatch.setattr(agent.httpx, "stream", fake)
    monkeypatch.setattr(agent.registry, "call", lambda name, args: ({
        "business_type": "cafe", "model_version": "m:v1", "data_date": "2026-11-01", "evidence": "model",
        "results": [{"h3_id": "89abc", "rank": 1, "lon": -6.8, "lat": 34.0, "final_score": 0.123456}]}, False))
    events = []
    ans = agent.OllamaProvider().run([{"role": "user", "content": "café Rabat ?"}],
                                     agent.registry.tool_schemas(), events.append)
    assert ans.answer == "Top 89abc"
    first = fake.bodies[0]
    assert first["think"] is False and first["options"]["num_ctx"] == 8192
    tool_msg = fake.bodies[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and '"lon"' not in tool_msg["content"] and "0.123" in tool_msg["content"]
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["tool", "tool_result"] and kinds.count("delta") == 2
    assert ans.citations["model"] == [{"h3_id": "89abc", "business_type": "cafe", "rank": 1}]


def test_ollama_error_is_raised_with_its_message(monkeypatch, env):
    import pytest

    monkeypatch.setattr(agent.httpx, "stream", FakeStream([_nd({"error": "model 'x' not found"})]))
    with pytest.raises(agent.OllamaError, match="not found"):
        agent.OllamaProvider().run([{"role": "user", "content": "hi"}], [])


def test_compact_drops_noise_and_rounds():
    from geomlab.tools.registry import compact

    out = compact({"lon": 1.0, "x": 1234.567, "y": 0.123456, "z": None, "e": [], "l": list(range(30)),
                   "nested": {"source_path": "/a", "keep": "ok"}})
    assert out == {"x": 1235, "y": 0.123, "l": list(range(10)), "nested": {"keep": "ok"}}


def test_ui_context_is_prefixed():
    q = agent.with_context("et ici ?", {"city": "rabat", "business_type": "cafe", "h3_id": None})
    assert q.startswith("[UI context: city = rabat, business type = cafe]") and q.endswith("et ici ?")
    assert agent.with_context("bonjour", None) == "bonjour"


def test_chat_stream_emits_events_then_done(env, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from geomlab.api import main

    def fake_ask(question, history, provider=None, context=None, on_event=None, **_):
        on_event({"type": "tool", "name": "recommend_locations", "args": {}})
        on_event({"type": "delta", "text": "ok"})
        return agent.AgentAnswer("ok", [], "fake", "end_turn")

    monkeypatch.setattr(agent, "ask", fake_ask)
    r = TestClient(main.app).post("/chat/stream", json={"message": "x", "provider": "ollama",
                                                        "context": {"city": "rabat"}})
    events = [json.loads(line) for line in r.text.splitlines() if line]
    assert [e["type"] for e in events] == ["tool", "delta", "done"] and events[-1]["answer"] == "ok"


def test_chat_stream_reports_unreachable_ollama(env, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from geomlab.api import main

    def boom(*a, **k):
        raise agent.httpx.ConnectError("Name or service not known")

    monkeypatch.setattr(agent, "ask", boom)
    r = TestClient(main.app).post("/chat/stream", json={"message": "x", "provider": "ollama"})
    last = json.loads(r.text.strip().splitlines()[-1])
    assert last["type"] == "error" and "Ollama is not reachable" in last["detail"]


def test_pois_route_is_not_captured_by_the_scores_route(env, monkeypatch):
    from fastapi.testclient import TestClient

    from geomlab.api import main

    monkeypatch.setattr(main.core, "list_pois", lambda city: [{"name": "x", "role": "attractor", "category": "school",
                                                                "lon": -6.8, "lat": 34.0}])
    r = TestClient(main.app).get("/map/rabat/pois")
    assert r.status_code == 200 and r.json()["pois"][0]["category"] == "school"
