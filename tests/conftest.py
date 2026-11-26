"""Shared fakes: tests never call the real LLM or Superhero API."""
from pathlib import Path

import pytest

from app.llm import LLMError
from app.retriever import Retriever, load_passages
from app.service import REWRITE_PROMPT
from app.superhero import SuperheroNotFound


class FakeLLM:
    """Returns a canned routing JSON for router calls, a canned rewrite for follow-ups,
    and records the prompts it receives."""

    def __init__(self, route_json='{"route": "dataset", "heroes": []}', answer="Fake answer.",
                 rewrite=None, fail=False):
        self.route_json = route_json
        self.answer = answer
        self.rewrite = rewrite
        self.fail = fail
        self.answer_prompts = []
        self.rewrite_prompts = []
        self.routed_questions = []

    def chat(self, system, user, json_mode=False, temperature=0.2):
        if self.fail:
            raise LLMError("simulated outage")
        if json_mode:
            self.routed_questions.append(user)
            return self.route_json
        if system == REWRITE_PROMPT:
            self.rewrite_prompts.append(user)
            return self.rewrite or user.rsplit("LATEST QUESTION: ", 1)[-1]
        self.answer_prompts.append(user)
        return self.answer


class FakeSuperheroes:
    def __init__(self, known=None):
        self.known = known or {
            "batman": {"id": "70", "name": "Batman", "full_name": "Bruce Wayne", "publisher": "DC Comics",
                       "first_appearance": "Detective Comics #27", "alignment": "good",
                       "powerstats": {"intelligence": "100", "strength": "26"}, "occupation": "Businessman"},
        }
        self.calls = []

    def search(self, name):
        self.calls.append(name)
        heroes = self.known.get(name.lower())
        if not heroes:
            raise SuperheroNotFound(name)
        return heroes if isinstance(heroes, list) else [heroes]


# Tests use a small fixed dataset, so they don't depend on the downloaded Wikipedia files.
FIXTURE_DATA = Path(__file__).parent / "fixtures" / "comics"


@pytest.fixture(scope="session")
def retriever():
    return Retriever(load_passages(FIXTURE_DATA))
