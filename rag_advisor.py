"""RAG-based care-task suggestion advisor for PawPal+.

Pipeline: retrieve relevant snippets from a curated knowledge base -> ask an
LLM to propose a duration/priority default (citing a retrieved snippet) ->
validate the suggestion against guardrails -> fall back to safe rule-based
defaults if the suggestion can't be trusted. The existing Task/Scheduler
classes in pawpal_system.py are untouched by this module; callers decide
whether to apply a Suggestion to a Task.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger("pawpal.rag")
if not logger.handlers:
    _handler = logging.FileHandler(Path(__file__).parent / "logs" / "rag_advisor.log")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

KB_PATH = Path(__file__).parent / "knowledge" / "pet_care_guidelines.json"

# Guardrail fallback: same safe defaults regardless of species, keyed by category.
RULE_BASED_DEFAULTS = {
    "walk": (20, "medium"),
    "feeding": (10, "high"),
    "meds": (5, "high"),
    "grooming": (15, "medium"),
    "enrichment": (15, "low"),
}

DISCLAIMER = "General educational guidance only, not a substitute for individualized veterinary advice."


@dataclass
class Snippet:
    id: str
    species: str
    category: str
    text: str
    suggested_duration_minutes: int
    suggested_priority: str
    source: str


@dataclass
class Suggestion:
    duration_minutes: int
    priority: str
    rationale: str
    citation: Optional[str]
    confidence: float
    source: str  # "llm" (live Claude call), "heuristic" (offline stand-in), or "fallback" (guardrail rejected)


class SuggestionValidationError(Exception):
    """Raised when an LLM suggestion fails a guardrail check."""


class KnowledgeBase:
    """Loads the curated pet-care snippet collection from knowledge/pet_care_guidelines.json."""

    def __init__(self, path: Path = KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.disclaimer: str = raw.get("disclaimer", DISCLAIMER)
        self.snippets: list[Snippet] = [Snippet(**entry) for entry in raw["snippets"]]

    def all(self) -> list[Snippet]:
        return list(self.snippets)


class Retriever:
    """TF-IDF retrieval over the knowledge base.

    Deterministic and offline (no network or API key required) so retrieval
    quality can be tested without depending on an LLM.
    """

    def __init__(self, kb: KnowledgeBase):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.kb = kb
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [f"{s.species} {s.category} {s.text}" for s in kb.all()]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, species: str, category: str, top_k: int = 3) -> list[tuple[Snippet, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([f"{species} {category}"])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        # Exact species/category matches rank first; text similarity breaks ties
        # and covers the "other" species / cross-cutting snippets.
        ranked = sorted(
            zip(self.kb.all(), scores),
            key=lambda pair: (
                pair[0].species not in (species, "any"),
                pair[0].category != category,
                -pair[1],
            ),
        )
        results = [(snip, float(score)) for snip, score in ranked[:top_k]]
        logger.info(
            "retrieve species=%s category=%s -> %s",
            species, category, [(s.id, round(sc, 3)) for s, sc in results],
        )
        return results


class LLMClient(Protocol):
    MODE: str  # "llm" (live model call) or "heuristic" (offline stand-in) — surfaced in Suggestion.source

    def suggest(self, species: str, category: str, task_title: str, snippets: list[Snippet]) -> str:
        """Return raw JSON text describing a suggestion (see RAGAdvisor._validate)."""
        ...


class AnthropicLLMClient:
    """Live suggestion generator backed by the Anthropic API. Requires ANTHROPIC_API_KEY."""

    MODE = "llm"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def suggest(self, species: str, category: str, task_title: str, snippets: list[Snippet]) -> str:
        snippet_block = "\n".join(
            f'- id="{s.id}": {s.text} (default: {s.suggested_duration_minutes} min, '
            f'{s.suggested_priority} priority)'
            for s in snippets
        )
        prompt = (
            "You are a care-planning assistant for a pet-care scheduling app called PawPal+. "
            "Using ONLY the retrieved guideline snippets below, propose a duration and priority "
            f'default for a task titled "{task_title}" (species: {species}, category: {category}).\n\n'
            f"Retrieved snippets:\n{snippet_block}\n\n"
            "Respond with ONLY a JSON object (no prose, no markdown fences) with these exact keys: "
            '"duration_minutes" (integer minutes), "priority" (one of "low", "medium", "high"), '
            '"rationale" (one short sentence), "citation_id" (the id of the single snippet you relied on '
            'most, exactly as given above), "confidence" (float between 0 and 1). '
            "If none of the snippets are relevant, set confidence to 0."
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class HeuristicLLMClient:
    """Deterministic, offline stand-in used when ANTHROPIC_API_KEY is not set.

    It does not call any network service. Rather than generating novel text,
    it composes a suggestion directly from the top-ranked retrieved snippet,
    scaled by that snippet's retrieval score as a stand-in for confidence.
    This keeps the app fully runnable (and tests deterministic) without an
    API key, at the cost of not producing genuinely LLM-authored rationale.
    """

    MODE = "heuristic"

    def suggest(self, species: str, category: str, task_title: str, snippets: list[Snippet]) -> str:
        if not snippets:
            return json.dumps({
                "duration_minutes": 0, "priority": "low", "rationale": "No guidance retrieved.",
                "citation_id": "", "confidence": 0.0,
            })
        top = snippets[0]
        exact_match = top.species in (species, "any") and top.category == category
        confidence = 0.85 if exact_match else 0.3
        return json.dumps({
            "duration_minutes": top.suggested_duration_minutes,
            "priority": top.suggested_priority,
            "rationale": f"Matches retrieved guidance for {top.species}/{top.category} tasks.",
            "citation_id": top.id,
            "confidence": confidence,
        })


class RAGAdvisor:
    """Orchestrates retrieval -> generation -> guardrail validation -> fallback."""

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[LLMClient] = None,
        confidence_threshold: float = 0.15,
    ):
        self.kb = kb or KnowledgeBase()
        self.retriever = retriever or Retriever(self.kb)
        self.llm_client = llm_client or self._default_llm_client()
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _default_llm_client() -> LLMClient:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicLLMClient()
        logger.info("ANTHROPIC_API_KEY not set - using offline HeuristicLLMClient")
        return HeuristicLLMClient()

    def suggest(self, species: str, category: str, task_title: str = "") -> Suggestion:
        """Return a validated Suggestion, or a safe rule-based fallback if validation fails."""
        retrieved = self.retriever.retrieve(species, category, top_k=3)
        snippets = [s for s, _ in retrieved]

        try:
            raw = self.llm_client.suggest(species, category, task_title, snippets)
            parsed = json.loads(raw)
            suggestion = self._validate(parsed, snippets, mode=getattr(self.llm_client, "MODE", "llm"))
            logger.info(
                "suggestion accepted: species=%s category=%s duration=%s priority=%s confidence=%s",
                species, category, suggestion.duration_minutes, suggestion.priority, suggestion.confidence,
            )
            return suggestion
        except (SuggestionValidationError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "suggestion rejected for species=%s category=%s (%s), falling back to a rule based default",
                species, category, exc,
            )
            return self._fallback(category)

    def _validate(self, parsed: dict, snippets: list[Snippet], mode: str = "llm") -> Suggestion:
        citation_id = parsed.get("citation_id")
        matching = next((s for s in snippets if s.id == citation_id), None)
        confidence = float(parsed.get("confidence", 0))
        duration = int(parsed["duration_minutes"])
        priority = parsed["priority"]

        if matching is None:
            raise SuggestionValidationError("citation_id does not match a retrieved snippet")
        if confidence < self.confidence_threshold:
            raise SuggestionValidationError(f"confidence {confidence} below threshold {self.confidence_threshold}")
        if priority not in ("low", "medium", "high"):
            raise SuggestionValidationError(f"invalid priority {priority!r}")
        if not (1 <= duration <= 240):
            raise SuggestionValidationError(f"duration {duration} out of sane range (1-240 minutes)")

        return Suggestion(
            duration_minutes=duration,
            priority=priority,
            rationale=str(parsed.get("rationale", "")),
            citation=f'{matching.source}: "{matching.text}"',
            confidence=confidence,
            source=mode,
        )

    def _fallback(self, category: str) -> Suggestion:
        duration, priority = RULE_BASED_DEFAULTS.get(category, (15, "medium"))
        return Suggestion(
            duration_minutes=duration,
            priority=priority,
            rationale="Rule based default (no verified AI suggestion available for this input).",
            citation=None,
            confidence=0.0,
            source="fallback",
        )
