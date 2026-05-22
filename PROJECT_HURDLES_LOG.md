# Project Hurdles & Solutions Log

Purpose: Keep a running record of implementation issues, root causes, fixes, and follow-up actions for `RAG_Arena`.

How to maintain:
- Add a new entry for every blocker or unexpected behavior.
- Keep entries factual and reproducible.
- Include commands/files changed where relevant.
- Do not store secrets in this file.

---

## Entry Template

### [YYYY-MM-DD] Short Hurdle Title
- **Area:** (e.g., Neo4j, RAGAS, UI, Env, Performance)
- **Symptom/Error:** exact error or observed behavior
- **Root Cause:** why it happened
- **Solution Applied:** what changed
- **Files/Commands:** files touched or commands used
- **Result:** how we verified fix
- **Follow-up:** optional improvements

---

## Hurdles Faced So Far

### [2026-05-20] Neo4j APOC startup failure
- **Area:** GraphRAG / Neo4j integration
- **Symptom/Error:** `ValueError: Could not use APOC procedures... apoc.meta.data()`
- **Root Cause:** `Neo4jGraphStore` schema refresh requires APOC procedures that may be restricted in Aura configurations.
- **Solution Applied:** Switched graph store implementation to `Neo4jPropertyGraphStore` and disabled startup schema refresh (`refresh_schema=False`).
- **Files/Commands:** `modules/graph_pipeline.py`
- **Result:** Graph engine initialization proceeded without APOC schema introspection failure.
- **Follow-up:** Keep Aura compatibility checks in startup validation.

### [2026-05-20] Neo4j database routing errors (database not found)
- **Area:** Neo4j Aura configuration
- **Symptom/Error:** `Database.DatabaseNotFound ... database 'neo4j' does not exist`
- **Root Cause:** Aura instance used non-default DB name (`c288b90b`) rather than `neo4j`.
- **Solution Applied:** Added database input in sidebar and propagated it to graph store build path. Added URI/username-based default inference and `NEO4J_DATABASE` support.
- **Files/Commands:** `app.py`, `modules/graph_pipeline.py`, `.env.example`
- **Result:** Successful connection when DB name matched Aura instance database.
- **Follow-up:** Add a quick connection test button to validate URI/user/db before indexing.

### [2026-05-20] Inaccurate UI interpretation during benchmark mode
- **Area:** Streamlit UX / Results display
- **Symptom/Error:** Top answer cards showed benchmark last question answer (e.g., `99.95%`) instead of user-typed query.
- **Root Cause:** Benchmark branch overwrote display variables with `eval_result["vector"][-1]` and `eval_result["graph"][-1]`.
- **Solution Applied:** Kept top cards tied to user query; benchmark output shown separately below.
- **Files/Commands:** `app.py`
- **Result:** Query cards now consistently reflect typed query; benchmark panel is independent.
- **Follow-up:** Add explicit labels ("Live Query Result" vs "Benchmark Result").

### [2026-05-20] RAGAS returned no numeric metrics (NaN/empty)
- **Area:** Evaluation pipeline
- **Symptom/Error:** Benchmark chart empty; warning indicated no numeric metrics.
- **Root Cause:** RAGAS judge metrics returned NaN/empty under current runtime/model conditions.
- **Solution Applied:** Implemented fallback scoring metrics and safe numeric filtering:
  - `token_f1`, `exact_match` initially
  - later expanded with `fact_recall`, `keyword_recall`
- **Files/Commands:** `modules/evaluator.py`, `app.py`
- **Result:** Benchmark chart and winner panel continue to render even when RAGAS is unavailable.
- **Follow-up:** Add per-question diagnostic panel for RAGAS raw outputs.

### [2026-05-20] Winner panel biased by strict string matching
- **Area:** Evaluation fairness
- **Symptom/Error:** Graph or Vector appeared to "win quality" despite semantically equivalent answers.
- **Root Cause:** `exact_match` favored terse answers matching ground-truth string format; semantically correct paraphrases were penalized.
- **Solution Applied:** Added fact/keyword coverage metrics, suite-specific expected terms, and tie thresholds for winner labels.
- **Files/Commands:** `modules/evaluator.py`, `app.py`
- **Result:** Quality winner now better reflects factual/semantic coverage, not just phrasing.
- **Follow-up:** Add per-question winner reason and confidence breakdown.

### [2026-05-20] Chart title mismatch with fallback metrics
- **Area:** UI clarity
- **Symptom/Error:** Chart title always said `Faithfulness / Answer Relevance / Context Recall` even when plotting fallback metrics.
- **Root Cause:** Static title string not tied to actual metric keys.
- **Solution Applied:** Updated chart title to display dynamic joined metric names.
- **Files/Commands:** `app.py`
- **Result:** Chart now communicates exactly what metrics are plotted.
- **Follow-up:** Group metrics into "RAGAS" vs "Fallback" sections visually.

### [2026-05-20] Secret handling risk
- **Area:** Security hygiene
- **Symptom/Error:** Credentials were stored in a plain text project file.
- **Root Cause:** Temporary local storage approach during setup/debugging.
- **Solution Applied:** Added ignores for `.env` and credential files; recommended key rotation and `.env` usage.
- **Files/Commands:** `.gitignore`, `.env.example`
- **Result:** Reduced risk of accidental commit/exposure.
- **Follow-up:** Add startup warning when credential text files are detected.

### [2026-05-20] Groq TPD rate-limit interruption during benchmark
- **Area:** LLM API quota / evaluation robustness
- **Symptom/Error:** `RateLimitError 429 ... tokens per day (TPD) limit reached`
- **Root Cause:** Benchmark sends multiple query+synthesis calls across both pipelines, exceeding Groq daily token budget.
- **Solution Applied:** Added per-question exception handling in evaluator, partial-result continuation, and early stop on rate-limit detection. Excluded failed rows from metric aggregation and surfaced warnings in UI.
- **Files/Commands:** `modules/evaluator.py`, `app.py`
- **Result:** App no longer crashes on quota hit; shows partial benchmark with clear warning.
- **Follow-up:** Add "quick benchmark" mode (2 questions) and token budget estimator.

### [2026-05-21] RAGAS evaluate returning NaN & timeouts with Groq LLM
- **Area:** Evaluation pipeline / RAGAS / Groq Integration
- **Symptom/Error:** Streamlit benchmark page failing with empty charts, NaN metrics in dashboard, and execution timeouts of over 3 minutes.
- **Root Cause:** 
  1. **LlamaIndex LLM Integration**: The LlamaIndex `Groq` LLM object was passed directly to Ragas. Ragas expects standard LangChain/OpenAI interfaces and calls `.generate()`, which fails with `AttributeError: 'Groq' object has no attribute 'generate'` because LlamaIndex LLMs do not expose this interface.
  2. **Sequential Execution Limit**: The legacy `LlamaIndexLLMWrapper` sequential execution mode does not support parallel completion requests (`n > 1` concurrency, required by metrics like `answer_relevancy`), causing timeouts of 3+ minutes per benchmark run.
  3. **Low Temperature Validation Loop**: When using low temperatures (e.g. `0.01`), Ragas retries on Groq JSON validation errors (such as extra trailing braces `}}`) were deterministic, leading to repeating failures and eventual timeouts.
- **Solution Applied:** 
  - Intercepted LlamaIndex LLM and Embedding parameters inside `run_comparative_evaluation` in `modules/evaluator.py`.
  - For LLMs, instantiated an OpenAI client pointing directly to the Groq endpoints (`https://api.groq.com/openai/v1`), wrapping it natively with Ragas's `llm_factory` at `temperature=0.4` to ensure retry diversity.
  - For Embedding models, wrapped them securely with `LlamaIndexEmbeddingsWrapper`.
  - Configured `max_retries=1` on both the main Groq LLM (in `app.py`) and OpenAI/Groq evaluator client wrapper to prevent 120s+ interface freezes when Groq quota limits are encountered.
- **Files/Commands:** `modules/evaluator.py`, `app.py`
- **Result:** Benchmark metrics evaluate successfully in under 3 seconds per benchmark suite under normal operations. Under API rate-limiting, the application fails fast and gracefully switches to fallback metrics (`exact_match`, `fact_recall`, etc.) without freeze or timeout delay.
- **Follow-up:** Keep monitoring Groq daily token budgets and consider implementing a quick-run/fewer-questions suite option.

---

## Maintenance Note

This log should be updated on every future blocker/fix cycle.  
During future development requests, append new entries at the bottom with date/time and verification notes.

