from app.retriever import Retriever, load_passages


def test_relevant_document_ranks_first(retriever):
    results = retriever.search("Who created Spider-Man?")
    assert results, "expected at least one passage"
    assert results[0].file_name == "spiderman.md"


def test_scores_are_sorted(retriever):
    scores = [p.score for p in retriever.search("Comics Code Authority 1954")]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_question_returns_nothing(retriever):
    assert retriever.search("How do I bake sourdough bread?") == []


def test_source_header_is_parsed(tmp_path):
    (tmp_path / "batman.md").write_text(
        "# Batman\nSource: https://en.wikipedia.org/wiki/Batman\nLicense: CC BY-SA 4.0\n\n"
        "Batman is a superhero who first appeared in Detective Comics in 1939.\n", encoding="utf-8")
    passage = Retriever(load_passages(tmp_path)).search("When did Batman first appear?")[0]
    assert passage.source_url == "https://en.wikipedia.org/wiki/Batman"
    assert passage.reference == "https://en.wikipedia.org/wiki/Batman"
    assert "Source:" not in passage.text          # header lines are not indexed as content


def test_reference_falls_back_to_file_name(retriever):
    assert retriever.search("Who created Spider-Man?")[0].reference == "spiderman.md"


def test_lead_paragraph_of_top_article_is_included(tmp_path):
    (tmp_path / "hero.md").write_text(
        "# Spider-Man\n\n"
        "Spider-Man is a superhero created by writer Stan Lee and artist Steve Ditko.\n\n"
        "Spider-Man appeared in Spider-Man films, Spider-Man games and Spider-Man toys for many years.\n\n"
        "Peter Parker lives in Queens, New York, with his aunt May and works as a photographer.\n",
        encoding="utf-8")
    results = Retriever(load_passages(tmp_path)).search("Spider-Man films games toys", k=1)
    assert [p.position for p in results] == [1, 0]          # top match, then the lead paragraph
    assert "Stan Lee" in results[1].text


def test_lead_is_not_duplicated_when_already_ranked(retriever):
    results = retriever.search("Who created Spider-Man?")
    leads = [p for p in results if p.file_name == "spiderman.md" and p.position == 0]
    assert len(leads) == 1


def test_hyphenated_names_match_when_typed_without_hyphen(retriever):
    results = retriever.search("Who is spiderman?")      # the dataset writes "Spider-Man"
    assert results and results[0].file_name == "spiderman.md"


def test_spaced_names_match_when_typed_as_one_word(tmp_path):
    (tmp_path / "iron.md").write_text("# Iron Man\n\nIron Man is a superhero created by Stan Lee, Larry Lieber, Don Heck and Jack Kirby.\n", encoding="utf-8")
    (tmp_path / "bat.md").write_text("# Batman\n\nBatman is a superhero created by Bill Finger and Bob Kane in 1939.\n", encoding="utf-8")
    results = Retriever(load_passages(tmp_path)).search("Who created ironman?")
    assert results[0].file_name == "iron.md"


def test_special_dash_in_query_still_matches(retriever):
    assert retriever.search("Who created Spider\u2011Man?")[0].file_name == "spiderman.md"
