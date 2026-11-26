import pytest

from app.router import QuestionRouter, keyword_route
from tests.conftest import FakeLLM


def test_llm_route_is_used_when_valid():
    router = QuestionRouter(FakeLLM('{"route": "both", "heroes": ["Batman"]}'))
    decision = router.route("Who created the Dark Knight and how strong is he?")
    assert decision.route == "both"
    assert decision.heroes == ["Batman"]
    assert decision.method == "llm"


@pytest.mark.parametrize("bad_json", [
    "not json",
    '{"route": "weather", "heroes": []}',          # unknown route
    '{"route": "superhero", "heroes": []}',        # superhero route but no hero
])
def test_invalid_llm_output_falls_back_to_keywords(bad_json):
    decision = QuestionRouter(FakeLLM(bad_json)).route("What are Superman's powers?")
    assert decision.method == "fallback"
    assert decision.route == "superhero"
    assert decision.heroes == ["superman"]


def test_llm_outage_falls_back_to_keywords():
    decision = QuestionRouter(FakeLLM(fail=True)).route("When was the Comics Code created?")
    assert (decision.route, decision.method) == ("dataset", "fallback")


@pytest.mark.parametrize("question, expected", [
    ("What is Wolverine's strength?", "superhero"),
    ("Who created Spider-Man?", "dataset"),
    ("Who created Batman and what are his powers?", "both"),
    ("What is the capital of France?", "none"),
])
def test_keyword_routes(question, expected):
    assert keyword_route(question).route == expected


def test_heroes_are_capped():
    router = QuestionRouter(FakeLLM('{"route": "superhero", "heroes": ["A", "B", "C", "D"]}'), max_heroes=3)
    assert router.route("compare A B C D").heroes == ["A", "B", "C"]


@pytest.mark.parametrize("spelling", ["spiderman", "spider man", "Spider-Man"])
def test_fallback_finds_hero_however_the_name_is_spelled(spelling):
    assert keyword_route(f"What are {spelling}'s powers?").heroes == ["spider-man"]


@pytest.mark.parametrize("spelling", ["ironman", "iron man", "Iron-Man"])
def test_fallback_finds_spaced_names_typed_as_one_word(spelling):
    assert keyword_route(f"How strong is {spelling}?").heroes == ["iron man"]


def test_fallback_handles_special_dashes():
    assert keyword_route("What are Spider\u2011Man's powers?").heroes == ["spider-man"]


@pytest.mark.parametrize("raw", [
    '```json\n{"route": "dataset", "heroes": []}\n```',      # markdown fence
    'Here is the JSON: {"route": "dataset", "heroes": []}',  # text before it
])
def test_router_accepts_wrapped_json(raw):
    decision = QuestionRouter(FakeLLM(raw)).route("Who created Superman?")
    assert (decision.route, decision.method) == ("dataset", "llm")


def test_known_hero_overrides_off_topic_route():
    decision = QuestionRouter(FakeLLM('{"route": "none", "heroes": []}')).route("What is Hulk's favourite food?")
    assert (decision.route, decision.heroes, decision.method) == ("superhero", ["hulk"], "llm+override")


def test_real_off_topic_stays_off_topic():
    decision = QuestionRouter(FakeLLM('{"route": "none", "heroes": []}')).route("What is the capital of France?")
    assert decision.route == "none"
