# PawPal+: AI Assisted Pet Care Scheduler

## Original Project (Modules 1 to 3)

PawPal+ is the original project this repo is built on. It's a rule based pet care planning assistant built with Python and Streamlit: an owner enters their pets, available time, and a list of care tasks (walks, feeding, meds, grooming, enrichment), and a `Scheduler` class produces a daily plan. It places fixed time tasks, fills remaining time with flexible tasks by priority, flags time conflicts as warnings instead of crashing, and explains why each task landed where it did. That original system (`pawpal_system.py`, `app.py`, `tests/test_pawpal.py`) has no AI in it at all; it's pure deterministic scheduling logic, and it remains completely unchanged by this extension.

## Title and Summary

This extension adds a Retrieval Augmented Generation (RAG) care advisor to PawPal+. When an owner is adding a task, they can ask the app to suggest a duration and priority default for it. The system retrieves relevant snippets from a small curated pet care knowledge base, asks an LLM to propose a suggestion grounded in those snippets, validates that suggestion against a guardrail before it's ever shown, and, if accepted, surfaces it with a citation the owner can check, inside the plan's existing reasoning explanation.

This matters because it's a small, honest demonstration of what makes RAG trustworthy in a real product: the model isn't free to say anything it wants, it's constrained to cite something a human curated, every suggestion is machine checked before a user ever sees it, and a human still has the final say before it touches the schedule.

## Architecture Overview

Full system diagram: [`diagrams/rag_system.mmd`](diagrams/rag_system.mmd) (class diagram for the original scheduler: [`diagrams/uml_final.mmd`](diagrams/uml_final.mmd)).

The pipeline, left to right:

1. The owner picks a species, category, and task title in the Streamlit UI (`app.py`).
2. A TFIDF index (`rag_advisor.Retriever`) over `knowledge/pet_care_guidelines.json` returns the top ranked snippets for that species and category.
3. An LLM (`rag_advisor.LLMClient`), or without an API key a deterministic offline stand in, proposes a `duration_minutes` and `priority` default, citing exactly one retrieved snippet by id.
4. The guardrail (`rag_advisor.RAGAdvisor._validate`) rejects the suggestion unless the cited snippet id is one that was actually retrieved, confidence clears a threshold, priority is a valid value, and duration is in a sane range. A rejection never crashes the app; it falls back to fixed rule based defaults.
5. The owner sees the suggestion, or the fallback clearly labeled, along with a disclaimer, and must click "Use this suggestion" before it's applied; they can just as easily ignore it and type their own numbers.
6. The accepted values become an ordinary `Task`, and `Scheduler.schedule_day()` places it exactly as it always has. The only change to `pawpal_system.py` is one new optional field, `Task.ai_citation`, which, if present, gets appended to `DailySchedule.explain()`'s reasoning text.
7. A separate, parallel path, `tests/test_rag_advisor.py`, runs the retriever and guardrail against fixed, scripted inputs, including deliberately bad LLM outputs, checking the whole "what if the model misbehaves" surface independently of the live app.

## Setup Instructions

```bash
# 1. Clone and enter the repo
git clone <repository_url>
cd applied-ai-project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) enable live Claude generated suggestions
export ANTHROPIC_API_KEY=your_api_key_here     # Windows: set ANTHROPIC_API_KEY=your_api_key_here
# Without this, the app still runs fully: it uses a deterministic offline
# suggestion generator instead of a live model call (see Design Decisions).

# 5. Run the app
streamlit run app.py

# 6. Run the tests
python3 -m pytest -v
```

No API key is required to run or test the app. Every `RAGAdvisor` decision (retrieval results, acceptance, or guardrail rejection) is logged to `logs/rag_advisor.log` for auditing.

## Sample Interactions

These are real outputs captured by running the advisor directly (offline mode, no API key set), not hand written examples.

Example 1: a clean match for dog and walk.
```
>>> RAGAdvisor().suggest(species="dog", category="walk", task_title="Morning walk")
source=heuristic duration=30 priority=high confidence=0.85
rationale: Matches retrieved guidance for dog/walk tasks.
citation: PawPal+ Care Guide: Canine Exercise Basics: "Adult dogs generally do well with
at least 30 minutes of walking activity a day, often split across one or two walks
depending on breed energy level and age."
```

Example 2: a different species and category, cat and enrichment.
```
>>> RAGAdvisor().suggest(species="cat", category="enrichment", task_title="Wand toy play")
source=heuristic duration=15 priority=low confidence=0.85
rationale: Matches retrieved guidance for cat/enrichment tasks.
citation: PawPal+ Care Guide: Enrichment Basics: "Interactive toys or short clicker training
sessions of 10 to 15 minutes help satisfy a cat's natural hunting instincts and reduce
boredom."
```

Example 3: the guardrail catching a bad suggestion. A scripted hallucinating LLM client cites a snippet id that was never retrieved.
```
>>> advisor.suggest(species="dog", category="walk", task_title="Long weekend walk")
source=fallback duration=20 priority=medium citation=None
rationale: Rule based default (no verified AI suggestion available for this input).
```
The corresponding log line shows why it was rejected:
```
WARNING suggestion rejected for species=dog category=walk
  (citation_id does not match a retrieved snippet), falling back to a rule based default
```

Example 4: the accepted suggestion flowing all the way into a generated schedule (`Task.ai_citation` feeds into `DailySchedule.explain()`).
```
Daily plan for Biscuit (Jordan) — 2026-08-02:
  08:00–08:30  Morning walk (30 min) [high]
Total: 30 min

Reasoning for Biscuit's schedule on 2026-08-02:
  • Morning walk: #1, priority: high, flexible — fit into available time, category: walk,
    AI suggestion source: PawPal+ Care Guide: Canine Exercise Basics: "Adult dogs generally
    do well with at least 30 minutes of walking activity a day, often split across one or
    two walks depending on breed energy level and age."
```
Note: the em dashes and the time range separator in that last block ("Jordan) — 2026-08-02", "08:00–08:30", "flexible — fit into available time") come from `pawpal_system.py`'s original, unmodified formatting, part of the Modules 1 to 3 project and not this extension's writing, so they're left exactly as the original author wrote them.

## Design Decisions

* TFIDF retrieval was used instead of embeddings because the knowledge base is small (15 hand authored snippets) and keyword overlap between species and category queries and snippet text is already a strong signal. It is deterministic, free, and needs no network call, which made it easy to test in isolation. The trade off is that it won't generalize as well as a real embedding model to oddly worded task titles, which is acceptable here since retrieval is keyed primarily on species and category, not free text.
* A pluggable LLM client with an offline fallback avoids being just an offline mock. `RAGAdvisor` depends on an `LLMClient` protocol, where `AnthropicLLMClient` calls Claude when `ANTHROPIC_API_KEY` is set and `HeuristicLLMClient` is used otherwise, deterministically composing a suggestion from the top retrieved snippet. This means the whole pipeline, retrieval, generation, guardrail, and UI, is runnable and testable by anyone who clones the repo, whether or not they have an API key. The UI and logs always label which mode produced a suggestion, so this never gets confused with a genuine model call.
* The guardrail rejects and falls back rather than raising to the UI. This mirrors a decision already made in the original PawPal+ project (see `reflection.md`): warn, don't crash. A rejected suggestion silently becomes a safe rule based default (`RULE_BASED_DEFAULTS` in `rag_advisor.py`) rather than an error page.
* `Task.ai_citation` is additive, not structural. Adding one new optional field to `Task` (default `None`) was enough to let `Scheduler._build_reason` surface a citation, without touching the scheduling algorithm itself. All 6 original scheduler tests pass unmodified, which was the intent: the RAG feature changes what defaults get proposed, not how scheduling works.
* A citation is only kept if the accepted values are still on screen. When a user clicks "Use this suggestion," the app remembers which exact `(duration, priority)` pair was applied, and if they later edit those fields the citation is dropped rather than misattributed to a hand typed number. A known limitation is that a user who edits back to the exact same numbers would keep the citation, an accepted simplification for this project's scope.

## Testing Summary

There are 30 automated tests, all passing. The original 6 scheduler tests (`tests/test_pawpal.py`, unchanged) plus 24 new tests (`tests/test_rag_advisor.py`) cover the RAG advisor. This is real, runnable proof, not a claim: clone the repo and run `python3 -m pytest -v -s` and you'll see the same output below, no API key needed, since everything here runs against the deterministic offline path.

```
tests/test_pawpal.py::test_mark_complete_changes_status PASSED
tests/test_pawpal.py::test_add_task_increases_pet_task_count PASSED
tests/test_pawpal.py::test_sort_by_time_returns_chronological_order PASSED
tests/test_pawpal.py::test_recurring_daily_task_creates_task_for_next_day PASSED
tests/test_pawpal.py::test_scheduler_flags_duplicate_fixed_times_as_conflict PASSED
tests/test_pawpal.py::test_check_conflicts_flags_overlapping_times_across_pets PASSED
tests/test_rag_advisor.py::test_retriever_ranks_exact_species_category_match_first PASSED
tests/test_rag_advisor.py::test_retriever_falls_back_to_any_species_snippet PASSED
tests/test_rag_advisor.py::test_advisor_without_api_key_uses_heuristic_client_and_cites_source PASSED
tests/test_rag_advisor.py::test_guardrail_rejects_citation_not_in_retrieved_snippets PASSED
tests/test_rag_advisor.py::test_guardrail_rejects_low_confidence PASSED
tests/test_rag_advisor.py::test_guardrail_rejects_out_of_range_duration PASSED
tests/test_rag_advisor.py::test_guardrail_rejects_malformed_json PASSED
tests/test_rag_advisor.py::test_valid_llm_suggestion_is_accepted_and_cited PASSED
tests/test_rag_advisor.py::test_advisor_is_reliable_across_every_supported_species_category_combo[dog_walk] PASSED
tests/test_rag_advisor.py::test_advisor_is_reliable_across_every_supported_species_category_combo[dog_feeding] PASSED
... (15 combinations total: dog/cat/other x walk/feeding/meds/grooming/enrichment)
tests/test_rag_advisor.py::test_advisor_reliability_summary_reports_full_pass_rate
[reliability] 15/15 combinations produced a valid suggestion (15/15 with a verified citation)
PASSED

============================== 30 passed in 1.38s ==============================
```

How reliability is actually measured, not just asserted:
* `test_advisor_is_reliable_across_every_supported_species_category_combo` is parametrized over all 15 species and category pairs the UI exposes, not a hand picked sample, and checks every one produces an in range duration, a valid priority, and, whenever the guardrail didn't fall back, a real citation. This is the "does the system give consistent, safe answers everywhere it's allowed to be used" check.
* `test_advisor_reliability_summary_reports_full_pass_rate` runs that same sweep once more and prints a human readable 15/15 pass rate line, so anyone running the suite sees a plain language reliability score in the terminal, not just green dots.
* The 4 `test_guardrail_rejects_*` tests each inject a specific way an LLM could misbehave, such as a hallucinated citation, low confidence, out of range duration, or malformed JSON, via a scripted fake client, and assert the guardrail catches it every time and falls back safely instead of propagating bad data.

What worked:
* Retrieval ranking is correct and easy to verify: exact species and category matches always outrank partial matches, and the species agnostic "any" snippet surfaces for every species.
* The guardrail catches every bad output case I could think of, proven with scripted fake LLM clients rather than a real, nondeterministic model call, so the tests are fast and repeatable.
* The offline `HeuristicLLMClient` makes the entire pipeline, including the 15 combination reliability sweep, deterministic and CI friendly, with no dependency on network access or a real API key.

What didn't work, and the limitations:
* There's no automated test against the real Anthropic API. A live call is nondeterministic and would require a secret in CI, so the JSON contract was tested via stubs instead, meaning a real prompt formatting bug in `AnthropicLLMClient.suggest()` wouldn't be caught by the test suite; it was only verified manually against the offline path.
* The knowledge base is intentionally small (3 species x 5 categories, 15 snippets). Coverage outside that grid, such as breed specific advice or unusual task titles, is untested and would likely fall through to the rule based fallback.
* The reliability sweep measures validity (in range values, real citations) across all supported inputs, not quality (whether 30 minutes is actually the right answer for a given dog). Judging suggestion quality would need either a labeled test set or human review, neither of which is in scope here.

What I learned: the most useful tests here weren't about the LLM's output quality; they were about the guardrail's behavior when the LLM misbehaves, and about whether the system holds up across its entire supported input space rather than one or two examples. Designing `LLMClient` as a swappable interface meant those tests could be written before ever calling a real model, which forced the validation rules (citation match, confidence floor, sane ranges) to be explicit and checkable rather than implicit trust in whatever came back.

## Reflection

Building the guardrail before the real model call changed how I thought about adding AI to an app. The interesting design work wasn't the prompt, it was deciding exactly what a suggestion has to prove about itself before a user ever sees it. It also reinforced a lesson from the original PawPal+ build: constraints that turn failure into a warning instead of a crash, there conflict detection, here the guardrail fallback, make a system feel trustworthy even when the underlying component, a human's schedule or an LLM's guess, is inherently unreliable.

The graded responsible AI reflection, how I collaborated with AI on this extension, one helpful and one flawed AI suggestion, and this system's limitations, is in [`model_card.md`](model_card.md), not here.
