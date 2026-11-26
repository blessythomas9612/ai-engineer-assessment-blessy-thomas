"""Client for the Superhero API: GET /api/{token}/search/{name}."""
from urllib.parse import quote

import httpx

from app.names import name_key


class SuperheroError(Exception):
    """The API could not be reached or returned an unexpected response."""


class SuperheroNotFound(SuperheroError):
    """The API answered, but no character matches the name."""


class SuperheroClient:
    def __init__(self, token: str, base_url: str, timeout: float = 15, transport: httpx.BaseTransport | None = None):
        if not token:
            raise ValueError("SUPERHERO_API_TOKEN is not set")
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout, transport=transport, follow_redirects=True,
            headers={"User-Agent": "ai-engineer-assessment/1.0"},   # some hosts block default client UAs
        )

    def search(self, name: str, max_matches: int = 3) -> list[dict]:
        """Return compact profiles for `name`.

        The API does partial matching ("Batman" also returns "Batgirl") and several
        characters can share a name exactly (two "Batman" entries: Terry McGinnis and
        Bruce Wayne). So: all exact-name matches (up to `max_matches`), or the first
        result if nothing matches exactly. The LLM then sees every candidate.
        """
        url = f"{self._base_url}/{self._token}/search/{quote(name.strip(), safe='')}"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise SuperheroError("Superhero API timed out") from exc
        except httpx.HTTPStatusError as exc:
            # Never include `url` in messages: it contains the token.
            raise SuperheroError(f"Superhero API returned HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SuperheroError(f"Superhero API request failed ({type(exc).__name__})") from exc

        # Never trust the shape of an outside response: a list or an HTML error page parsed
        # as JSON would otherwise crash with AttributeError and lose the whole answer.
        if not isinstance(data, dict):
            raise SuperheroError("Superhero API returned an unexpected response")
        if data.get("response") != "success" or not data.get("results"):
            error = str(data.get("error", "")).lower()
            if "not found" in error or data.get("response") == "success":
                raise SuperheroNotFound(f"No superhero found for '{name}'")
            raise SuperheroError(f"Superhero API error: {data.get('error', 'unknown')}")

        results = data["results"]
        if not isinstance(results, list):
            raise SuperheroError("Superhero API returned an unexpected response")
        results = [r for r in results if isinstance(r, dict)]
        if not results:
            raise SuperheroError("Superhero API returned an unexpected response")
        exact = [r for r in results if name_key(r.get("name", "")) == name_key(name)]
        return [self._compact(r) for r in (exact or results[:1])[:max_matches]]

    @staticmethod
    def _compact(hero: dict) -> dict:
        bio, work = hero.get("biography", {}), hero.get("work", {})
        return {
            "id": hero.get("id"),
            "name": hero.get("name"),
            "full_name": bio.get("full-name"),
            "publisher": bio.get("publisher"),
            "first_appearance": bio.get("first-appearance"),
            "alignment": bio.get("alignment"),
            "powerstats": hero.get("powerstats", {}),
            "occupation": work.get("occupation"),
        }
