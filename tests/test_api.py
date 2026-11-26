"""End-to-end tests of POST /ask with the real router, retriever and service,
but fake LLM and Superhero API (dependency override)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.router import QuestionRouter
from app.service import NO_INFO, OFF_TOPIC, ChatService
from tests.conftest import FakeLLM, FakeSuperheroes


@pytest.fixture
def make_client(retriever):
    def _make(llm, heroes=None):
        service = ChatService(QuestionRouter(llm), retriever, heroes or FakeSuperheroes(), llm)
        app.dependency_overrides[get_service] = lambda: service
        return TestClient(app, raise_server_exceptions=False)
    yield _make
    app.dependency_overrides.clear()


def test_dataset_question_cites_dataset(make_client):
    client = make_client(FakeLLM('{"route": "dataset", "heroes": []}'))
    body = client.post("/ask", json={"question": "Who created Spider-Man?"}).json()
    assert body["route"] == "dataset"
    assert body["sources"][0]["type"] == "dataset"
    assert body["sources"][0]["reference"] == "spiderman.md"


def test_both_question_cites_both_sources(make_client):
    llm = FakeLLM('{"route": "both", "heroes": ["Batman"]}')
    body = make_client(llm).post("/ask", json={"question": "Who created Batman and how smart is he?"}).json()
    assert body["route"] == "both"
    assert {s["type"] for s in body["sources"]} == {"dataset", "superhero_api"}
    assert "Bruce Wayne" in llm.answer_prompts[0]        # API data reached the LLM context


def test_unknown_hero_returns_no_info_without_calling_llm(make_client):
    llm = FakeLLM('{"route": "superhero", "heroes": ["Nobody Man"]}')
    body = make_client(llm).post("/ask", json={"question": "What are Nobody Man's powers?"}).json()
    assert body["answer"].startswith(NO_INFO)
    assert body["sources"] == []
    assert llm.answer_prompts == []                       # no context, so no generation


def test_off_topic(make_client):
    body = make_client(FakeLLM('{"route": "none", "heroes": []}')).post(
        "/ask", json={"question": "What is the capital of France?"}).json()
    assert (body["answer"], body["route"], body["sources"]) == (OFF_TOPIC, "none", [])


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 501}])
def test_invalid_input_is_rejected(make_client, payload):
    assert make_client(FakeLLM()).post("/ask", json=payload).status_code == 422


def test_llm_outage_returns_502(make_client):
    response = make_client(FakeLLM(fail=True)).post("/ask", json={"question": "Who created Superman?"})
    assert response.status_code == 502
    assert "unavailable" in response.json()["detail"]


def test_follow_up_is_rewritten_using_history(make_client):
    llm = FakeLLM('{"route": "superhero", "heroes": ["Batman"]}', rewrite="What are Batman's powers?")
    client = make_client(llm)
    first = client.post("/ask", json={"question": "Tell me about Batman"}).json()
    assert first["standalone_question"] is None             # first question: no rewrite
    assert llm.rewrite_prompts == []                        # ... and no extra LLM call

    second = client.post("/ask", json={"question": "What are his powers?",
                                       "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]
    assert second["standalone_question"] == "What are Batman's powers?"
    assert "Tell me about Batman" in llm.rewrite_prompts[0]   # history was used
    assert llm.routed_questions[-1] == "What are Batman's powers?"


def test_new_session_has_no_history(make_client):
    llm = FakeLLM('{"route": "superhero", "heroes": ["Batman"]}')
    client = make_client(llm)
    a = client.post("/ask", json={"question": "Tell me about Batman"}).json()
    b = client.post("/ask", json={"question": "Tell me about Batman"}).json()
    assert a["session_id"] != b["session_id"]
    assert llm.rewrite_prompts == []


def test_invalid_session_id_is_rejected(make_client):
    response = make_client(FakeLLM()).post("/ask", json={"question": "Who is Batman?", "session_id": "bad id!"})
    assert response.status_code == 422


def test_characters_sharing_a_name_are_all_cited(make_client):
    heroes = FakeSuperheroes({"batman": [
        {"id": "69", "name": "Batman", "full_name": "Terry McGinnis", "powerstats": {}},
        {"id": "70", "name": "Batman", "full_name": "Bruce Wayne", "powerstats": {}},
    ]})
    llm = FakeLLM('{"route": "superhero", "heroes": ["Batman"]}')
    body = make_client(llm, heroes).post("/ask", json={"question": "How smart is Batman?"}).json()
    assert [s["title"] for s in body["sources"]] == ["Batman (Terry McGinnis)", "Batman (Bruce Wayne)"]
    assert "2 different characters" in llm.answer_prompts[0]


def test_router_hero_names_expand_the_dataset_search(make_client):
    # The question uses a nickname that appears nowhere in the dataset.
    llm = FakeLLM('{"route": "dataset", "heroes": ["Spider-Man"]}')
    body = make_client(llm).post("/ask", json={"question": "Who drew the web-slinger?"}).json()
    assert body["sources"] and body["sources"][0]["reference"] == "spiderman.md"


def test_query_expansion_does_not_repeat_names():
    from app.service import ChatService
    assert ChatService._search_query("Who created Batman?", ["Batman"]) == "Who created Batman?"
    assert ChatService._search_query("Who drew the web-slinger?", ["Spider-Man"]) == "Who drew the web-slinger? Spider-Man"


def test_app_starts_even_if_service_cannot_be_built(monkeypatch):
    import app.main as main

    def broken():
        raise ValueError("GROQ_API_KEY is not set")
    monkeypatch.setattr(main, "get_service", broken)
    with TestClient(main.app) as client:          # runs the startup (lifespan) code
        assert client.get("/health").json() == {"status": "ok"}


def test_router_names_are_canonicalised_before_the_api_call(make_client):
    heroes = FakeSuperheroes({"spider-man": {"id": "620", "name": "Spider-Man", "full_name": "Peter Parker",
                                             "powerstats": {}}})
    llm = FakeLLM('{"route": "superhero", "heroes": ["spiderman"]}')
    body = make_client(llm, heroes).post("/ask", json={"question": "How strong is spiderman?"}).json()
    assert heroes.calls == ["Spider-Man"]
    assert body["sources"][0]["title"] == "Spider-Man (Peter Parker)"


def test_answer_step_is_told_how_a_typo_was_resolved(make_client):
    heroes = FakeSuperheroes({"spider-man": {"id": "620", "name": "Spider-Man", "powerstats": {}}})
    llm = FakeLLM('{"route": "superhero", "heroes": ["Spider-Man"]}')
    make_client(llm, heroes).post("/ask", json={"question": "How strong is Spidremen?"})
    assert "The question refers to: Spider-Man." in llm.answer_prompts[0]


def test_no_interpretation_note_when_the_name_is_in_the_question(make_client):
    llm = FakeLLM('{"route": "superhero", "heroes": ["Batman"]}')
    make_client(llm).post("/ask", json={"question": "How strong is Batman?"})
    assert "The question refers to" not in llm.answer_prompts[0]


def test_superhero_miss_falls_back_to_dataset(make_client):
    # "X-Men" is a team, not an API character: answer from Wikipedia instead of "no info".
    llm = FakeLLM('{"route": "superhero", "heroes": ["X-Men"]}')
    body = make_client(llm).post("/ask", json={"question": "How strong are the X-Men?"}).json()
    assert body["sources"] and {s["type"] for s in body["sources"]} == {"dataset"}
    assert all("X-Men" in s["title"] or "xmen" in s["reference"] for s in body["sources"])
    assert len(llm.answer_prompts) == 1


def test_answer_prompt_never_exposes_internal_wording():
    from app.service import ANSWER_PROMPT
    assert "my sources" in ANSWER_PROMPT and 'never mention "context"' in ANSWER_PROMPT


class BuggyService:
    def ask(self, question, session_id=None):
        raise ValueError("a bug, not a configuration problem")


def test_unexpected_error_returns_json_500():
    app.dependency_overrides[get_service] = lambda: BuggyService()
    try:
        response = TestClient(app, raise_server_exceptions=False).post("/ask", json={"question": "Who is Batman?"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500                      # not a misleading 503
    assert response.json() == {"detail": "Internal server error."}


def test_missing_configuration_returns_503(monkeypatch):
    from app import main
    def broken():
        raise ValueError("GROQ_API_KEY is not set")
    monkeypatch.setattr(main, "_build_service", broken)
    main.get_service.cache_clear()
    try:
        response = TestClient(app, raise_server_exceptions=False).post("/ask", json={"question": "Who is Batman?"})
    finally:
        main.get_service.cache_clear()
    assert response.status_code == 503
    assert response.json() == {"detail": "Service is not configured correctly."}


@pytest.mark.parametrize("question", ["Hi", "hello!", "How are you?", "Hey, how are you doing?",
                                      "Thanks!", "bye", "What can you do?"])
def test_small_talk_gets_a_fixed_reply_without_llm(make_client, question):
    llm = FakeLLM()
    body = make_client(llm).post("/ask", json={"question": question}).json()
    assert body["route"] == "greeting"
    assert body["sources"] == []
    assert llm.routed_questions == [] and llm.answer_prompts == []    # no LLM call at all


@pytest.mark.parametrize("question", ["Hi, who created Batman?", "Thanks, and how strong is he?"])
def test_greeting_with_a_real_question_is_answered(make_client, question):
    body = make_client(FakeLLM()).post("/ask", json={"question": question}).json()
    assert body["route"] != "greeting"
