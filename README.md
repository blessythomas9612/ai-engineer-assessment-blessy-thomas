# Superhero & Comics History Chatbot

A FastAPI service with one endpoint, `POST /ask`. It answers questions about superheroes (live data from the [Superhero API](https://superheroapi.com)) and about comics history (Wikipedia articles stored in `data/comics/`). It decides which source a question needs, or uses both, and every answer lists its sources.

## Dataset

20 English Wikipedia articles on comics history: major characters (Superman, Batman, Spider-Man, X-Men ...), publishers and creators (Marvel, DC, Stan Lee, Jack Kirby), the Golden/Silver/Bronze Ages, the Comics Code Authority and superhero film. It complements the Superhero API: the API gives a character's current profile and powers, the articles give origins and history, so some questions need one source and some need both.

`scripts/fetch_wikipedia.py` downloads them through the Wikipedia API, drops reference sections and caps each article at about 12,000 characters. Each file records its source URL, which is returned as the source of every answer built from it. Text is from Wikipedia under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## How it works

0. **Small talk:** greetings, thanks and "what can you do?" get a fixed friendly reply (route `greeting`) with no LLM call, so the bot never chats without a source. A greeting with a real question ("Hi, who created Batman?") is answered normally.
1. **Route:** an LLM call (Groq) classifies the question as `superhero`, `dataset`, `both` or `none`, and extracts hero names. If that call fails or its reply can't be parsed, keyword rules take over. If the LLM calls a question off-topic but it names a known hero ("What is Hulk's favourite food?"), it is routed to the Superhero API anyway.
2. **Retrieve:** dataset questions use TF-IDF search over the article paragraphs (top 3 above a 0.1 similarity threshold), plus the lead paragraph of the best-matching article. The search query is expanded with the hero names the router found, so nicknames like "the web-slinger" still match. Names are matched however they are typed ("spiderman", "Spider-Man", "ironman", "Iron Man"). Superhero questions call `/api/{token}/search/{name}`. If the API has nothing (for example a team such as "X-Men"), the articles are searched for passages that name the character.
3. **Answer:** the LLM answers using only the retrieved context. Sources are attached by the code, not written by the LLM.

**Follow-up questions:** every response includes a `session_id`. Send it back with the next question and the bot remembers the last 5 turns. A follow-up such as "What are his powers?" is first rewritten into a standalone question ("What are Batman's powers?"), then goes through the same steps. Memory is held in process and forgotten after 30 minutes idle.


## Screenshots

**Swagger UI (`/docs`):** a single `POST /ask` endpoint:

<img src="docs/screenshots/docs-page.png" alt="Swagger UI with POST /ask and GET /health" width="700">

**A question that needs both sources:** the answer combines Wikipedia (creators) and the Superhero API (powerstats), and lists both as sources:

<img src="docs/screenshots/answer-both-sources.png" alt="Answer with route both and sources from Wikipedia and the Superhero API" width="700">

**A follow-up question:** "What are his powers?" is rewritten using the session's memory into "What are Batman's powers?":

<img src="docs/screenshots/follow-up.png" alt="Follow-up question with standalone_question" width="700">

**Validation:** a blank question is rejected with 422 before any LLM or API call:

<img src="docs/screenshots/validation-error.png" alt="422 validation error for a blank question" width="700">

**Tests:** 99 tests pass on a fresh clone, with no API keys needed:

<img src="docs/screenshots/tests-passing.png" alt="pytest: 99 passed" width="500">

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env
```

Then put your keys in `.env`:
- `GROQ_API_KEY`: from [console.groq.com](https://console.groq.com)
- `SUPERHERO_API_TOKEN`: from [superheroapi.com](https://superheroapi.com) (sign in with GitHub)

The dataset is already in `data/comics/`. To rebuild it: `python scripts/fetch_wikipedia.py`.

## Run

```bash
uvicorn app.main:app --reload --reload-dir app
```

Open http://127.0.0.1:8000/docs to try it, or:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Who created Batman and what are his powers?\"}"
```

Example response:

```json
{
  "answer": "Batman was created by Bob Kane with Bill Finger and first appeared in Detective Comics #27 in 1939 ...",
  "route": "both",
  "sources": [
    {"type": "dataset", "title": "Batman", "reference": "https://en.wikipedia.org/wiki/Batman", "snippet": "..."},
    {"type": "superhero_api", "title": "Batman", "reference": "superheroapi.com /search/Batman (id 70)", "snippet": null}
  ],
  "session_id": "3f9c...",
  "standalone_question": null
}
```

Each request logs how long every stage took, e.g. `timings_ms {'route': 900, 'retrieval': 15, 'superhero_api': 600, 'answer': 1100}`.

## Tests

```bash
pytest
```

The tests use fakes for the LLM and the Superhero API and a small fixed dataset in `tests/fixtures/`, so they need no keys or network access.

## Errors

| Case | Response |
|---|---|
| Empty, blank or over-500-character question, or invalid `session_id` | `422` |
| Greeting or thanks ("Hi", "How are you?") | `200`, route `greeting`, a fixed reply with no sources |
| Off-topic question | `200` with a polite refusal, no sources |
| Hero not found, or no relevant passage | `200`, says no information was found |
| Superhero API down or returns an unexpected format | `200`, answers from the other source and says the API was unavailable |
| LLM unavailable | `502` |
| Missing API keys or empty dataset | `503` |
| Any other unexpected error | `500` with `{"detail": "Internal server error."}`; details only in the server log |
