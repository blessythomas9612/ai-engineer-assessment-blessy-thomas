import httpx
import pytest

from app.superhero import SuperheroClient, SuperheroError, SuperheroNotFound

BATMAN = {
    "id": "70", "name": "Batman",
    "powerstats": {"intelligence": "100", "strength": "26"},
    "biography": {"full-name": "Bruce Wayne", "publisher": "DC Comics",
                  "first-appearance": "Detective Comics #27", "alignment": "good"},
    "work": {"occupation": "Businessman"},
}


def client_returning(handler):
    return SuperheroClient("secret-token", "https://superheroapi.com/api", transport=httpx.MockTransport(handler))


def test_success_keeps_only_exact_name_matches():
    batgirl = {**BATMAN, "id": "63", "name": "Batgirl"}
    client = client_returning(lambda req: httpx.Response(200, json={"response": "success",
                                                                   "results": [batgirl, BATMAN]}))
    heroes = client.search("Batman")
    assert [h["name"] for h in heroes] == ["Batman"]
    assert heroes[0]["full_name"] == "Bruce Wayne"
    assert heroes[0]["powerstats"]["intelligence"] == "100"


def test_characters_sharing_a_name_are_all_returned():
    beyond = {**BATMAN, "id": "69", "biography": {**BATMAN["biography"], "full-name": "Terry McGinnis"}}
    client = client_returning(lambda req: httpx.Response(200, json={"response": "success",
                                                                   "results": [beyond, BATMAN]}))
    assert [h["full_name"] for h in client.search("batman")] == ["Terry McGinnis", "Bruce Wayne"]


def test_no_exact_match_falls_back_to_first_result():
    client = client_returning(lambda req: httpx.Response(200, json={"response": "success", "results": [BATMAN]}))
    assert [h["name"] for h in client.search("bat")] == ["Batman"]


def test_not_found():
    client = client_returning(lambda req: httpx.Response(
        200, json={"response": "error", "error": "character with given name not found"}))
    with pytest.raises(SuperheroNotFound):
        client.search("Nobody")


def test_bad_token_is_an_error_not_not_found():
    client = client_returning(lambda req: httpx.Response(200, json={"response": "error", "error": "access denied"}))
    with pytest.raises(SuperheroError) as exc:
        client.search("Batman")
    assert not isinstance(exc.value, SuperheroNotFound)


def test_http_failure_does_not_leak_token():
    client = client_returning(lambda req: httpx.Response(500))
    with pytest.raises(SuperheroError) as exc:
        client.search("Batman")
    assert "secret-token" not in str(exc.value)


def test_missing_token_is_rejected():
    with pytest.raises(ValueError):
        SuperheroClient("", "https://superheroapi.com/api")


def test_redirect_is_followed():
    def handler(req):
        if req.url.host == "superheroapi.com":
            return httpx.Response(301, headers={"Location": str(req.url.copy_with(host="www.superheroapi.com"))})
        return httpx.Response(200, json={"response": "success", "results": [BATMAN]})
    assert client_returning(handler).search("Batman")[0]["name"] == "Batman"


def test_token_never_appears_in_logs(caplog):
    import logging
    import app.main  # noqa: F401  (applies the logging configuration)

    client = client_returning(lambda req: httpx.Response(200, json={"response": "success", "results": [BATMAN]}))
    with caplog.at_level(logging.DEBUG):
        client.search("Batman")
    assert "secret-token" not in caplog.text


def test_exact_match_ignores_hyphens_spaces_and_case():
    spidey = {**BATMAN, "id": "620", "name": "Spider-Man"}
    spider_woman = {**BATMAN, "id": "623", "name": "Spider-Woman"}
    client = client_returning(lambda req: httpx.Response(200, json={"response": "success",
                                                                   "results": [spider_woman, spidey]}))
    assert [h["id"] for h in client.search("spiderman")] == ["620"]


def test_name_with_slash_stays_in_one_path_segment():
    seen = []

    def handler(req):
        seen.append(req.url.raw_path.decode())
        return httpx.Response(200, json={"response": "success", "results": [BATMAN]})

    client_returning(handler).search("AC/DC")
    assert seen[0].endswith("/search/AC%2FDC")


@pytest.mark.parametrize("payload", [
    ["unexpected", "list"],                               # not an object
    {"response": "success", "results": "oops"},           # results not a list
    {"response": "success", "results": ["not-a-dict"]},   # items not objects
])
def test_unexpected_response_shape_is_a_clean_error(payload):
    client = client_returning(lambda req: httpx.Response(200, json=payload))
    with pytest.raises(SuperheroError, match="unexpected response"):
        client.search("Batman")
