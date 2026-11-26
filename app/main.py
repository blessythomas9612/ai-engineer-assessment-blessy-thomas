"""FastAPI app exposing POST /ask."""
import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from typing import Annotated

from fastapi import Body, Depends, FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.llm import LLMClient, LLMError
from app.memory import SessionStore
from app.retriever import Retriever, load_passages
from app.router import QuestionRouter
from app.schemas import AskRequest, AskResponse
from app.service import ChatService
from app.superhero import SuperheroClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs every request URL at INFO, and the Superhero API puts the token in the URL.
# Keep HTTP client logs to warnings only, so secrets never reach the logs.
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the service (load the dataset, build the TF-IDF index) at startup, so configuration
    problems show up immediately. If keys are missing the app still starts; /ask then returns 503."""
    try:
        get_service()
    except Exception as exc:
        log.warning("Service not ready at startup (%s); /ask will return 503 until fixed", exc)
    yield


app = FastAPI(title="Superhero & Comics History Chatbot", version="1.0.0", lifespan=lifespan)


class ServiceNotConfigured(Exception):
    """Missing API keys or an empty dataset: the service cannot be built."""


@lru_cache
def get_service() -> ChatService:
    """Build the service once (the TF-IDF index is built at first use) and reuse it."""
    try:
        return _build_service()
    except ValueError as exc:   # only construction errors land here, not errors while answering
        raise ServiceNotConfigured(str(exc)) from exc


def _build_service() -> ChatService:
    llm = LLMClient(settings.groq_api_key, settings.groq_model, settings.groq_base_url, settings.http_timeout_s,
                    reasoning_effort=settings.groq_reasoning_effort)
    return ChatService(
        router=QuestionRouter(llm, settings.max_heroes_per_question),
        retriever=Retriever(load_passages(settings.data_dir), settings.retrieval_min_score),
        superheroes=SuperheroClient(settings.superhero_api_token, settings.superhero_base_url,
                                    settings.http_timeout_s),
        llm=llm,
        top_k=settings.top_k_passages,
        memory=SessionStore(max_turns=settings.memory_turns, ttl_s=settings.memory_ttl_s),
    )


@app.exception_handler(LLMError)
async def llm_error_handler(_, exc: LLMError):
    log.error("LLM failure: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "The language model is unavailable. Try again later."})


@app.exception_handler(ServiceNotConfigured)
async def config_error_handler(_, exc: ServiceNotConfigured):
    log.error("Configuration error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Service is not configured correctly."})


@app.exception_handler(Exception)
async def unexpected_error_handler(_, exc: Exception):
    # Last resort: a bug should still give a JSON error, with the details only in the logs.
    log.exception("Unexpected error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
def health():
    return {"status": "ok"}


ASK_EXAMPLES = {
    "both": {"summary": "Both sources", "value": {"question": "Who created Batman and what are his powers?"}},
    "dataset": {"summary": "Dataset only", "value": {"question": "What was the Comics Code Authority?"}},
    "superhero": {"summary": "Superhero API only", "value": {"question": "How strong is Wolverine?"}},
    "follow_up": {"summary": "Follow-up (paste a session_id from an earlier answer)",
                  "value": {"question": "What are his powers?", "session_id": "paste-session-id-here"}},
    "off_topic": {"summary": "Off-topic", "value": {"question": "What is the weather in Doha?"}},
}


@app.post("/ask", response_model=AskResponse)
def ask(req: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)],
        service: ChatService = Depends(get_service)) -> AskResponse:
    return service.ask(req.question, req.session_id)
