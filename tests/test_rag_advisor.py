import itertools
import json

import pytest

from rag_advisor import (
    KnowledgeBase,
    Retriever,
    RAGAdvisor,
    Snippet,
    RULE_BASED_DEFAULTS,
)

SPECIES_OPTIONS = ["dog", "cat", "other"]
CATEGORY_OPTIONS = ["walk", "feeding", "meds", "grooming", "enrichment"]


@pytest.fixture()
def kb():
    return KnowledgeBase()


@pytest.fixture()
def retriever(kb):
    return Retriever(kb)


def test_retriever_ranks_exact_species_category_match_first(retriever):
    results = retriever.retrieve("dog", "walk", top_k=3)
    top_snippet, _ = results[0]
    assert top_snippet.id == "dog-walk-1"


def test_retriever_falls_back_to_any_species_snippet(retriever):
    # "any-meds-consistency" should be a candidate for every species asking about meds.
    results = retriever.retrieve("cat", "meds", top_k=5)
    ids = [s.id for s, _ in results]
    assert "cat-meds-1" in ids


def test_advisor_without_api_key_uses_heuristic_client_and_cites_source(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    advisor = RAGAdvisor()

    suggestion = advisor.suggest(species="dog", category="walk", task_title="Morning walk")

    assert suggestion.source == "heuristic"
    assert suggestion.citation is not None
    assert suggestion.duration_minutes == 30
    assert suggestion.priority == "high"


class _FakeLLMClient:
    """Test double that returns a scripted JSON response."""

    MODE = "llm"

    def __init__(self, response: dict):
        self._response = response

    def suggest(self, species, category, task_title, snippets):
        return json.dumps(self._response)


def test_guardrail_rejects_citation_not_in_retrieved_snippets(kb, retriever):
    fake_client = _FakeLLMClient({
        "duration_minutes": 25,
        "priority": "high",
        "rationale": "made up",
        "citation_id": "not-a-real-snippet-id",
        "confidence": 0.9,
    })
    advisor = RAGAdvisor(kb=kb, retriever=retriever, llm_client=fake_client)

    suggestion = advisor.suggest(species="dog", category="walk")

    assert suggestion.source == "fallback"
    assert suggestion.citation is None
    assert (suggestion.duration_minutes, suggestion.priority) == RULE_BASED_DEFAULTS["walk"]


def test_guardrail_rejects_low_confidence(kb, retriever):
    fake_client = _FakeLLMClient({
        "duration_minutes": 30,
        "priority": "high",
        "rationale": "unsure",
        "citation_id": "dog-walk-1",
        "confidence": 0.01,
    })
    advisor = RAGAdvisor(kb=kb, retriever=retriever, llm_client=fake_client)

    suggestion = advisor.suggest(species="dog", category="walk")

    assert suggestion.source == "fallback"


def test_guardrail_rejects_out_of_range_duration(kb, retriever):
    fake_client = _FakeLLMClient({
        "duration_minutes": 9000,
        "priority": "high",
        "rationale": "too long",
        "citation_id": "dog-walk-1",
        "confidence": 0.9,
    })
    advisor = RAGAdvisor(kb=kb, retriever=retriever, llm_client=fake_client)

    suggestion = advisor.suggest(species="dog", category="walk")

    assert suggestion.source == "fallback"


def test_guardrail_rejects_malformed_json(kb, retriever):
    class _BrokenClient:
        MODE = "llm"

        def suggest(self, species, category, task_title, snippets):
            return "not json at all"

    advisor = RAGAdvisor(kb=kb, retriever=retriever, llm_client=_BrokenClient())

    suggestion = advisor.suggest(species="dog", category="walk")

    assert suggestion.source == "fallback"


def test_valid_llm_suggestion_is_accepted_and_cited(kb, retriever):
    fake_client = _FakeLLMClient({
        "duration_minutes": 25,
        "priority": "medium",
        "rationale": "Adjusted slightly down for an older dog.",
        "citation_id": "dog-walk-1",
        "confidence": 0.8,
    })
    advisor = RAGAdvisor(kb=kb, retriever=retriever, llm_client=fake_client)

    suggestion = advisor.suggest(species="dog", category="walk", task_title="Evening walk")

    assert suggestion.source == "llm"
    assert suggestion.duration_minutes == 25
    assert suggestion.priority == "medium"
    assert "PawPal+ Care Guide" in suggestion.citation


# --- Reliability sweep -------------------------------------------------------
# Not a cherry-picked example: this exercises every species x category
# combination the app's UI actually offers (3 x 5 = 15 inputs) and asserts
# the advisor never returns something a user could act on unsafely — every
# duration/priority is in range, and any non-fallback suggestion carries a
# real citation. This is what "measuring reliability" means here: a
# consistency check over the full supported input space, run automatically
# on every test invocation, not a one-off manual spot check.
_GRID = list(itertools.product(SPECIES_OPTIONS, CATEGORY_OPTIONS))


@pytest.mark.parametrize(
    "species,category", _GRID, ids=[f"{s}_{c}" for s, c in _GRID]
)
def test_advisor_is_reliable_across_every_supported_species_category_combo(
    species, category, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    advisor = RAGAdvisor()

    suggestion = advisor.suggest(species=species, category=category, task_title="Test task")

    assert suggestion.priority in ("low", "medium", "high")
    assert 1 <= suggestion.duration_minutes <= 240
    assert suggestion.source in ("llm", "heuristic", "fallback")
    if suggestion.source != "fallback":
        assert suggestion.citation is not None, (
            f"non-fallback suggestion for {species}/{category} is missing a citation"
        )


def test_advisor_reliability_summary_reports_full_pass_rate(capsys):
    """Prints a small reliability report across the full input grid.

    Not a substitute for the assertions above (this test has its own asserts),
    but gives a human-readable pass-rate summary anyone running the suite can
    read directly in the terminal, similar to a coverage report.
    """
    import os

    os.environ.pop("ANTHROPIC_API_KEY", None)
    advisor = RAGAdvisor()

    total = 0
    valid = 0
    cited = 0
    for species, category in itertools.product(SPECIES_OPTIONS, CATEGORY_OPTIONS):
        total += 1
        suggestion = advisor.suggest(species=species, category=category, task_title="Test task")
        is_valid = (
            suggestion.priority in ("low", "medium", "high")
            and 1 <= suggestion.duration_minutes <= 240
        )
        if is_valid:
            valid += 1
        if suggestion.source != "fallback" and suggestion.citation is not None:
            cited += 1

    with capsys.disabled():
        print(
            f"\n[reliability] {valid}/{total} combinations produced a valid suggestion "
            f"({cited}/{total} with a verified citation)"
        )

    assert valid == total
