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
