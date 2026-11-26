"""Decide which source(s) a question needs: superhero, dataset, both, or none.

Primary path: the LLM classifies the question and extracts hero names (it copes with
paraphrases such as "how strong is the Dark Knight").
Fallback: simple keyword rules, used only if the LLM call fails or returns invalid JSON,
so the bot keeps working when the classifier is unavailable.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from app.llm import LLMClient, LLMError
from app.names import CANONICAL_HEROES, normalize_dashes

log = logging.getLogger(__name__)

ROUTES = {"superhero", "dataset", "both", "none"}

ROUTER_PROMPT = """You route questions for a chatbot that has two sources:
1. "superhero": the Superhero API - profile data for a named fictional character: powerstats
   (intelligence, strength, speed, durability, power, combat), real/full name, publisher,
   alignment (good/bad), occupation, aliases.
2. "dataset": Wikipedia articles about characters (Superman, Batman, Wonder Woman, Spider-Man,
   X-Men, Captain America, Iron Man, Fantastic Four, Wolverine, Hulk), creators (Stan Lee,
   Jack Kirby), publishers (Marvel, DC, Timely), the Golden/Silver/Bronze Ages, the Comics Code
   Authority and superhero films - who created what, when it first appeared, history, media.
Choose "both" if the question needs facts from both sources, "none" only if it is not about comics
or superheroes. Any question about a superhero or comics character is on-topic, even if the
sources may not hold the answer (route it to "superhero" with the character's name).
"heroes" lists the fictional characters mentioned, by canonical name ("Batman" for "the Dark
Knight", "Spider-Man" for "the web-slinger"). Never list real people such as writers or artists.

Examples:
Q: How tough is the Dark Knight? -> {"route": "superhero", "heroes": ["Batman"]}
Q: Who created Spider-Man and how strong is he? -> {"route": "both", "heroes": ["Spider-Man"]}
Q: Who is Jack Kirby? -> {"route": "dataset", "heroes": []}
Q: What is Hulk's favourite food? -> {"route": "superhero", "heroes": ["Hulk"]}
Q: What is the capital of France? -> {"route": "none", "heroes": []}

Reply ONLY with JSON: {"route": "superhero|dataset|both|none", "heroes": ["name", ...]}"""

KNOWN_HEROES = [h.lower() for h in CANONICAL_HEROES]
HISTORY_WORDS = re.compile(
    r"\b(creat\w*|invent\w*|first appear\w*|histor\w*|publish\w*|debut\w*|golden age|silver age|"
    r"bronze age|comics code|film|movie|origin)\b", re.I)
PROFILE_WORDS = re.compile(
    r"\b(power\w*|strong\w*|strength|stats?|intelligen\w*|speed|durab\w*|combat|real name|"
    r"full name|alignment|occupation|abilit\w*)\b", re.I)
COMICS_WORDS = re.compile(r"\b(comic\w*|marvel|dc|superhero\w*|mutant\w*)\b", re.I)


@dataclass
class RouteDecision:
    route: str
    heroes: list[str] = field(default_factory=list)
    method: str = "llm"   # "llm", "llm+override" or "fallback" - useful when debugging


def extract_json(raw: str) -> dict:
    """Parse the JSON object in an LLM reply, tolerating ```json fences or text around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in router reply")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("router reply is not a JSON object")
    return data


def find_heroes(question: str) -> list[str]:
    q = normalize_dashes(question).lower()
    found = []
    for hero in sorted(KNOWN_HEROES, key=len, reverse=True):  # longest first: "spider-man" before "man"
        # "spider-man" also matches "spiderman" and "spider man"; "iron man" also "ironman"
        pattern = re.escape(hero).replace(r"\-", "[- ]?").replace(r"\ ", "[- ]?")
        if re.search(rf"\b{pattern}\b", q) and not any(hero in f for f in found):
            found.append(hero)
    return found


def keyword_route(question: str) -> RouteDecision:
    heroes = find_heroes(question)
    wants_history = bool(HISTORY_WORDS.search(question))
    wants_profile = bool(PROFILE_WORDS.search(question))
    if heroes and wants_history and wants_profile:
        route = "both"
    elif heroes and wants_profile:
        route = "superhero"
    elif heroes or wants_history or COMICS_WORDS.search(question):
        route = "dataset"
    else:
        route = "none"
    return RouteDecision(route, heroes, method="fallback")


class QuestionRouter:
    def __init__(self, llm: LLMClient, max_heroes: int = 3):
        self.llm = llm
        self.max_heroes = max_heroes

    def route(self, question: str) -> RouteDecision:
        try:
            raw = self.llm.chat(ROUTER_PROMPT, question, json_mode=True, temperature=0)
            data = extract_json(raw)
            route = str(data.get("route", "")).lower()
            heroes = [str(h).strip() for h in data.get("heroes", []) if str(h).strip()]
            if route not in ROUTES:
                raise ValueError(f"unknown route {route!r}")
            if route in ("superhero", "both") and not heroes:
                raise ValueError("superhero route without a hero name")
            if route == "none":
                # Safety net: a known hero in the question means it is on-topic, even if the
                # LLM judged it off-topic ("What is Hulk's favourite food?").
                known = find_heroes(question)
                if known:
                    log.info("router said 'none' but found heroes %s; using 'superhero'", known)
                    return RouteDecision("superhero", known[: self.max_heroes], method="llm+override")
            return RouteDecision(route, heroes[: self.max_heroes])
        except (LLMError, ValueError, TypeError, AttributeError) as exc:   # JSONDecodeError is a ValueError
            log.warning("LLM routing failed (%s); using keyword fallback", exc)
            decision = keyword_route(question)
            decision.heroes = decision.heroes[: self.max_heroes]
            return decision
