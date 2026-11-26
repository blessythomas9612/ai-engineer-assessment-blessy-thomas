"""Core flow: (rewrite a follow-up using the conversation history), route the question,
gather context from the chosen sources, ask the LLM to answer ONLY from that context,
and return the answer with its sources.

Sources are built in code from what was actually retrieved, not written by the LLM,
so the attribution can't be hallucinated.
"""
import json
import logging
import re
import time
import uuid

from app.llm import LLMClient, LLMError
from app.memory import SessionStore, Turn
from app.names import canonical_name, name_key, normalize_dashes
from app.retriever import Retriever
from app.router import QuestionRouter
from app.schemas import AskResponse, Source
from app.superhero import SuperheroClient, SuperheroError, SuperheroNotFound

log = logging.getLogger(__name__)

OFF_TOPIC = "I can only answer questions about superheroes and comics history."
NO_INFO = ("I couldn't find information about that in my sources "
           "(a comics history dataset and the Superhero API).")

HELP = ("I can answer questions about superheroes and comics history. "
        "Try \"Who created Batman?\" or \"How strong is Wolverine?\"")

# Small talk gets a fixed reply: no LLM call, so nothing ungrounded, and it costs nothing.
# Each pattern must match the WHOLE message, so "Hi, who created Batman?" is still answered.
SMALL_TALK = [
    (re.compile(r"(hi|hello|hey|hiya|good (morning|afternoon|evening))( there)?"), "Hi! " + HELP),
    (re.compile(r"(hi|hello|hey)?,? ?how are you( doing)?( today)?"), "I'm doing well, thanks! " + HELP),
    (re.compile(r"(thanks|thank you|thank you so much|thx|ok thanks)"), "You're welcome! Ask me anything else about superheroes or comics."),
    (re.compile(r"(bye|goodbye|see you|see ya)"), "Goodbye! Come back any time with more superhero questions."),
    (re.compile(r"(who are you|what are you|what can you do|help)"), HELP),
]


def small_talk_reply(question: str) -> str | None:
    text = re.sub(r"[^a-z ,]", "", question.lower()).strip(" ,")   # drop "?", "!", emojis
    for pattern, reply in SMALL_TALK:
        if pattern.fullmatch(text):
            return reply
    return None


ANSWER_PROMPT = """You answer questions about superheroes and comics history.
Use ONLY the facts in the CONTEXT. Do not invent facts or use outside knowledge.
If the context answers only part of the question, answer that part and say which part is not
covered. If it answers none of it, say plainly that your sources do not cover it.
For questions about strength or powers, use the powerstats and any abilities described in
the articles.
Stay faithful to how the context states things: do not add your own judgements or qualifiers
(such as "limited" or "minor") that the context does not state. If the context gives differing
accounts (for example, disputed credit), present them neutrally and say they differ.
Superhero API powerstats are scores from 0 to 100. If several characters share a name, say
which is which. Refer to sources only as "Wikipedia", "the Superhero API" or "my sources";
never mention "context", "notes" or "provided information" - the user cannot see them.
Write plain text with no markdown, in at most 5 sentences. Do not list sources; they are
attached separately."""

REWRITE_PROMPT = """Rewrite the user's latest question so it can be understood without the conversation.
Replace pronouns and references ("he", "his", "they", "that team") with the names they refer to,
and complete short follow-ups using the earlier question
(after "How strong is Batman?", "And Superman?" becomes "How strong is Superman?").
Keep the user's meaning; do not answer the question or add new details.
If the question is already standalone, return it unchanged. Reply with the question only."""


class ChatService:
    def __init__(self, router: QuestionRouter, retriever: Retriever,
                 superheroes: SuperheroClient, llm: LLMClient, top_k: int = 3,
                 memory: SessionStore | None = None):
        self.router = router
        self.retriever = retriever
        self.superheroes = superheroes
        self.llm = llm
        self.top_k = top_k
        self.memory = memory or SessionStore()

    def ask(self, question: str, session_id: str | None = None) -> AskResponse:
        session_id = session_id or uuid.uuid4().hex
        question = normalize_dashes(question)
        reply = small_talk_reply(question)
        if reply:
            # Not stored in memory, so "thanks" doesn't become the context for a follow-up.
            return AskResponse(answer=reply, route="greeting", sources=[], session_id=session_id)
        history = self.memory.history(session_id)
        standalone = self._rewrite(question, history) if history else question

        answer, route, sources = self._answer(standalone)
        self.memory.add(session_id, standalone, answer)   # only reached if answering succeeded
        return AskResponse(answer=answer, route=route, sources=sources, session_id=session_id,
                           standalone_question=standalone if standalone != question else None)

    def _rewrite(self, question: str, history: list[Turn]) -> str:
        """Turn a follow-up ("what are his powers?") into a standalone question."""
        transcript = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in history)
        try:
            rewritten = self.llm.chat(REWRITE_PROMPT, f"CONVERSATION:\n{transcript}\n\nLATEST QUESTION: {question}",
                                      temperature=0)
        except LLMError as exc:
            log.warning("Rewrite failed (%s); using the question as asked", exc)
            return question
        rewritten = rewritten.strip().strip('"')
        # Guard against a rambling reply: fall back to the original question.
        if not rewritten or len(rewritten) > 500 or "\n" in rewritten:
            return question
        log.info("rewrote %r -> %r", question, rewritten)
        return rewritten

    @staticmethod
    def _search_query(question: str, heroes: list[str]) -> str:
        """Query expansion: add the canonical names the router found, so nicknames
        ("the web-slinger", "the Dark Knight") still match the right articles."""
        extra = [h for h in heroes if name_key(h) not in name_key(question)]
        return f"{question} {' '.join(extra)}".strip()

    def _add_passages(self, question: str, heroes: list[str], context: list[str],
                      sources: list[Source], keep=lambda p: True) -> None:
        for p in self.retriever.search(self._search_query(question, heroes), k=self.top_k):
            if not keep(p):
                continue
            context.append(f"[Article: {p.doc_title}]\n{p.text}")
            sources.append(Source(type="dataset", title=p.doc_title,
                                  reference=p.reference, snippet=p.text[:160]))

    def _answer(self, question: str) -> tuple[str, str, list[Source]]:
        timings, start = {}, time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal start
            timings[stage] = round((time.perf_counter() - start) * 1000)
            start = time.perf_counter()

        try:
            return self._answer_steps(question, lap)
        finally:
            log.info("timings_ms %s", timings)       # where each request spends its time

    def _answer_steps(self, question: str, lap) -> tuple[str, str, list[Source]]:
        decision = self.router.route(question)
        # One spelling per hero, as the API uses it ("spiderman" -> "Spider-Man"), no duplicates.
        decision.heroes = list(dict.fromkeys(canonical_name(h) for h in decision.heroes))
        lap("route")
        log.info("route=%s heroes=%s method=%s", decision.route, decision.heroes, decision.method)
        if decision.route == "none":
            return OFF_TOPIC, "none", []

        context, sources, notes = [], [], []

        if decision.route in ("dataset", "both"):
            self._add_passages(question, decision.heroes, context, sources)

        lap("retrieval")

        if decision.route in ("superhero", "both"):
            for name in decision.heroes:
                try:
                    matches = self.superheroes.search(name)
                except SuperheroNotFound:
                    notes.append(f"The Superhero API has no character named '{name}'.")
                    continue
                except SuperheroError as exc:
                    # Partial failure: keep going with whatever else we have.
                    log.warning("Superhero API failed for %s: %s", name, exc)
                    notes.append(f"The Superhero API was unavailable for '{name}'.")
                    continue
                if len(matches) > 1:
                    notes.append(f"The Superhero API has {len(matches)} different characters named "
                                 f"'{matches[0]['name']}'; mention which is which.")
                for hero in matches:
                    label = hero["name"] + (f" ({hero['full_name']})" if hero.get("full_name") else "")
                    context.append(f"[Superhero API: {label}]\n{json.dumps(hero, ensure_ascii=False)}")
                    sources.append(Source(type="superhero_api", title=label,
                                          reference=f"superheroapi.com /search/{name} (id {hero['id']})"))

        lap("superhero_api")

        if decision.route == "superhero" and not context:
            # The API had nothing (a team such as "X-Men", a missing character, or an outage):
            # try Wikipedia before giving up. Keep only passages that name the character, so a
            # character the dataset doesn't cover ("Nobody Man") still gets "no info", not
            # loosely related text.
            log.info("no Superhero API data; falling back to the dataset")
            keys = [name_key(h) for h in decision.heroes]
            self._add_passages(question, decision.heroes, context, sources,
                               keep=lambda p: any(k in name_key(p.doc_title + " " + p.text) for k in keys))
            lap("dataset_fallback")

        if not context:
            return NO_INFO + (" " + " ".join(notes) if notes else ""), decision.route, []

        # Tell the answer step how the router read names that aren't in the question as typed
        # (typos, nicknames): otherwise it may refuse, seeing "Spidremen" but context about Spider-Man.
        resolved = [h for h in decision.heroes if name_key(h) not in name_key(question)]
        if resolved:
            notes.insert(0, f"The question refers to: {', '.join(resolved)}.")

        prompt = "CONTEXT:\n" + "\n\n".join(context)
        if notes:
            prompt += "\n\nNOTES:\n" + "\n".join(notes)
        prompt += f"\n\nQUESTION: {question}"
        answer = self.llm.chat(ANSWER_PROMPT, prompt)
        lap("answer")
        return answer, decision.route, sources
