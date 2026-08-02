# PawPal+ RAG Advisor: Model Card & Responsible AI Reflection

This document covers the RAG based care suggestion feature added to PawPal+ (`rag_advisor.py`, `knowledge/pet_care_guidelines.json`, and the "Suggest duration & priority" section of `app.py`). It's the graded reflection on limitations, misuse, reliability, and AI collaboration for this extension. See `README.md` for the system's design and setup, and `reflection.md` for the original (Modules 1 to 3) scheduler's reflection.

## Limitations and Biases

The knowledge base (`knowledge/pet_care_guidelines.json`) is small and hand authored, with only 15 snippets covering exactly 3 species buckets (dog, cat, other) times 5 categories. It has no breed specific, age specific, or health condition specific guidance, so a suggestion for "other/grooming" is the same generic default whether the pet is a rabbit, a bearded lizard, or a parrot, even though their actual grooming needs differ enormously. This is a real bias toward the average case baked into the knowledge base, not something the retriever or guardrail can correct.

The offline heuristic path isn't really AI generated. When `ANTHROPIC_API_KEY` isn't set, `HeuristicLLMClient` deterministically echoes the top retrieved snippet's stored defaults, which is useful for making the whole app runnable and testable without a key. It is templated retrieval, not generation, so every reliability number reported for offline mode reflects that limitation.

## Misuse Potential and Mitigations

The biggest realistic risk is an owner mistaking a scheduling default, such as duration or priority, for actual medical guidance, especially for the `meds` category. Every suggestion shown in the UI carries the disclaimer "General educational guidance only, not a substitute for individualized veterinary advice," and the guardrail restricts durations to a sane range of 1 to 240 minutes so it can't produce something like "give medication for 0 minutes" or "300 minutes," even if the generator proposed it.


## Surprises in Reliability Testing
The TFIDF machinery barely mattered. I expected cosine similarity scores to meaningfully influence ranking, but in practice the sort key of species mismatch, category mismatch, and score means an exact species and category match always wins regardless of the TFIDF score. For this 15 snippet, tightly templated knowledge base, a plain species and category to snippet dictionary lookup would have produced nearly identical retrieval behavior, and TFIDF only earns its place if the knowledge base grows past what a lookup table can cover cleanly.

## AI Collaboration

I worked with Claude (Claude Code) across this whole extension: from brainstorming three project directions, to the system diagram, to implementing `rag_advisor.py` and wiring it into `app.py`, to writing the reliability test sweep.

A helpful suggestion came when I asked for tests proving the guardrail actually works. Claude's approach was to make `RAGAdvisor` accept an injectable `llm_client`, any object with a `.suggest()` method, instead of hard wiring it to the Anthropic API. That let `tests/test_rag_advisor.py` use small scripted fake clients (`_FakeLLMClient`, `_BrokenClient`) to deterministically force every bad output scenario actually needed, such as a hallucinated citation id, a too low confidence score, an out of range duration, and malformed JSON, without ever needing a real, nondeterministic API call in the test suite. I wouldn't have thought to structure the dependency that way up front, and it's the reason the guardrail could be tested at all without an API key.

A flawed suggestion came in an early pass, when `RAGAdvisor._validate()` labeled every guardrail accepted suggestion with `source="llm"`, whether it came from a real Claude call or from the offline `HeuristicLLMClient`. That's technically accurate to the code path, since both go through the same validation, but misleading in the UI, since it would have shown "AI suggestion" for a templated, nongenerative fallback exactly the same way it would for a genuine model response, conflating two very different levels of trust. This was caught while reviewing the design, before it reached the UI, and split into three labels, `"llm"`, `"heuristic"`, and `"fallback"`, so `app.py` can honestly tell a user which one produced their suggestion. It's a good example of an AI written first pass being internally consistent but wrong about what it was communicating to a human.
