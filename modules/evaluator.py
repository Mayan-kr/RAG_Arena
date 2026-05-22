"""RAGAS comparative benchmarking across vector and graph engines."""

from __future__ import annotations

import time
import re
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

import os
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from llama_index.core.llms import LLM as LlamaIndexLLM
from llama_index.core.embeddings import BaseEmbedding as LlamaIndexEmbedding


def _extract_answer(response: Any) -> str:
    return str(getattr(response, "response", response))


def _extract_contexts(response: Any) -> list[str]:
    """Pull retrieved chunk text for RAGAS context metrics."""
    contexts: list[str] = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            text = node.node.get_content() if hasattr(node, "node") else str(node)
            if text:
                contexts.append(text)
    if not contexts:
        contexts = [_extract_answer(response)]
    return contexts


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_fact_tokens(text: str) -> list[str]:
    """
    Extract fact-bearing tokens:
    - percentages and decimals (e.g., 99.95%, 2%)
    - durations/numbers (e.g., 90, 15)
    - meaningful words (len >= 4)
    """
    normalized = _normalize_text(text)
    numeric_tokens = re.findall(r"\d+(?:\.\d+)?%?", normalized)
    word_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{3,}", normalized)
    # Preserve order while de-duplicating.
    seen: set[str] = set()
    tokens: list[str] = []
    for token in [*numeric_tokens, *word_tokens]:
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    common = 0
    for token, count in pred_counts.items():
        if token in ref_counts:
            common += min(count, ref_counts[token])
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _contains_token(answer: str, token: str) -> bool:
    return token in _normalize_text(answer)


def _keyword_recall(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 0.0
    hits = sum(1 for term in expected_terms if _contains_token(answer, _normalize_text(term)))
    return hits / len(expected_terms)


def _fallback_scores(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"token_f1": 0.0, "exact_match": 0.0, "fact_recall": 0.0, "keyword_recall": 0.0}

    f1_scores = [_token_f1(r["answer"], r["ground_truth"]) for r in rows]
    exact_match_scores = [
        1.0 if _normalize_text(r["answer"]) == _normalize_text(r["ground_truth"]) else 0.0
        for r in rows
    ]
    fact_recall_scores = []
    keyword_recall_scores = []
    for row in rows:
        fact_tokens = _extract_fact_tokens(row["ground_truth"])
        if fact_tokens:
            fact_recall_scores.append(_keyword_recall(row["answer"], fact_tokens))
        else:
            fact_recall_scores.append(0.0)
        keyword_recall_scores.append(
            _keyword_recall(row["answer"], row.get("expected_terms", []))
        )
    return {
        "token_f1": float(sum(f1_scores) / len(f1_scores)),
        "exact_match": float(sum(exact_match_scores) / len(exact_match_scores)),
        "fact_recall": float(sum(fact_recall_scores) / len(fact_recall_scores)),
        "keyword_recall": float(sum(keyword_recall_scores) / len(keyword_recall_scores)),
    }


def run_comparative_evaluation(
    vector_engine,
    graph_engine,
    evaluation_questions: list[str],
    ground_truths: list[str],
    expected_terms: list[list[str]] | None = None,
    llm=None,
    embed_model=None,
) -> dict[str, Any]:
    """
    Run identical question sweeps on both engines and score with RAGAS.

    Returns per-pipeline rows (answers, latencies) plus aggregate metric dicts.
    """
    vector_rows: list[dict] = []
    graph_rows: list[dict] = []
    warnings: list[str] = []

    for i, (question, ground_truth) in enumerate(zip(evaluation_questions, ground_truths)):
        row_expected_terms = expected_terms[i] if expected_terms and i < len(expected_terms) else []
        v_answer = ""
        v_contexts: list[str] = []
        g_answer = ""
        g_contexts: list[str] = []
        v_error = ""
        g_error = ""

        t0 = time.perf_counter()
        try:
            v_response = vector_engine.query(question)
            v_answer = _extract_answer(v_response)
            v_contexts = _extract_contexts(v_response)
        except Exception as exc:  # noqa: BLE001
            v_error = str(exc)
            v_answer = f"[query_failed] {v_error}"
            warnings.append(f"Vector query failed for '{question}': {v_error}")
        v_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            g_response = graph_engine.query(question)
            g_answer = _extract_answer(g_response)
            g_contexts = _extract_contexts(g_response)
        except Exception as exc:  # noqa: BLE001
            g_error = str(exc)
            g_answer = f"[query_failed] {g_error}"
            warnings.append(f"Graph query failed for '{question}': {g_error}")
        g_time = time.perf_counter() - t0

        vector_rows.append(
            {
                "question": question,
                "answer": v_answer,
                "contexts": v_contexts,
                "ground_truth": ground_truth,
                "expected_terms": row_expected_terms,
                "latency_s": v_time,
                "error": v_error,
            }
        )
        graph_rows.append(
            {
                "question": question,
                "answer": g_answer,
                "contexts": g_contexts,
                "ground_truth": ground_truth,
                "expected_terms": row_expected_terms,
                "latency_s": g_time,
                "error": g_error,
            }
        )

        if "rate limit" in f"{v_error} {g_error}".lower():
            warnings.append("Rate limit reached during benchmark; results are partial.")
            break

    if llm is not None and isinstance(llm, LlamaIndexLLM):
        api_key = getattr(llm, "api_key", os.environ.get("GROQ_API_KEY", ""))
        model_name = getattr(llm, "model", "llama-3.3-70b-versatile")
        groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, max_retries=1)
        llm = llm_factory(model_name, client=groq_client, temperature=0.4)

    if embed_model is not None and isinstance(embed_model, LlamaIndexEmbedding):
        embed_model = LlamaIndexEmbeddingsWrapper(embeddings=embed_model)

    metrics = [faithfulness, answer_relevancy, context_recall]
    eval_kwargs: dict[str, Any] = {}
    if llm is not None:
        eval_kwargs["llm"] = llm
    if embed_model is not None:
        eval_kwargs["embeddings"] = embed_model

    vector_scores = _run_ragas(vector_rows, metrics, eval_kwargs)
    graph_scores = _run_ragas(graph_rows, metrics, eval_kwargs)

    return {
        "vector": vector_rows,
        "graph": graph_rows,
        "vector_scores": vector_scores,
        "graph_scores": graph_scores,
        "warnings": warnings,
    }


def _run_ragas(rows: list[dict], metrics, eval_kwargs: dict) -> dict[str, float]:
    valid_rows = [r for r in rows if not r.get("error")]
    if not valid_rows:
        return {}

    fallback = _fallback_scores(valid_rows)

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in valid_rows],
            "answer": [r["answer"] for r in valid_rows],
            "contexts": [r["contexts"] for r in valid_rows],
            "ground_truth": [r["ground_truth"] for r in valid_rows],
        }
    )

    try:
        result = evaluate(dataset, metrics=metrics, **eval_kwargs)
        df = result.to_pandas()
        numeric_df = df.select_dtypes("number")
        clean_scores = {
            col: float(numeric_df[col].dropna().mean())
            for col in numeric_df.columns
            if not numeric_df[col].dropna().empty
        }
        if not clean_scores:
            fallback["ragas_warning"] = "RAGAS returned no numeric metric values (all NaN/empty)."
            return fallback
        
        fallback.update(clean_scores)
        return fallback
    except Exception as exc:
        fallback["ragas_warning"] = f"RAGAS failed: {exc}"
        return fallback



BENCHMARK_SUITES: dict[str, dict[str, list[str]]] = {
    "Fact Retrieval (Vector-friendly)": {
        "questions": [
            "How long before a nexus-agent is marked DEGRADED after a missed heartbeat?",
            "What is the AuthForge access token TTL?",
            "What is the Analytics Lake p95 ingestion SLA to Athena?",
            "What monthly uptime SLA does nexus-api target?",
        ],
        "ground_truths": [
            "90 seconds",
            "15 minutes",
            "5 minutes",
            "99.95%",
        ],
        "expected_terms": [
            ["90", "seconds", "degraded"],
            ["15", "minutes", "access", "ttl"],
            ["5", "minutes", "athena", "ingestion"],
            ["99.95%"],
        ],
    },
    "Relationship & Multi-hop (Graph-friendly)": {
        "questions": [
            "Which service validates JWT tokens using the AuthForge JWKS endpoint?",
            "Which component consumes EventBridge metrics.workload.v1 events for analytics ingestion?",
            "When heartbeat events are stale, which service marks nodes DEGRADED and what alert is triggered?",
            "Name the full deployment flow from release creation to analytics correlation.",
        ],
        "ground_truths": [
            "nexus-api validates JWT using AuthForge JWKS endpoint.",
            "Analytics Lake Flink job lake-ingest-primary consumes EventBridge events.",
            "nexus-api marks nodes DEGRADED and triggers ALERT-NODE-STALE.",
            "CI Runner deploy-v3 -> POST /v1/releases -> Postgres releases table -> EventBridge ReleaseCreated -> nexus-agent SSE /stream/releases -> artifact pull -> Analytics Lake correlation.",
        ],
        "expected_terms": [
            ["nexus-api", "authforge", "jwks"],
            ["analytics lake", "lake-ingest-primary", "eventbridge"],
            ["nexus-api", "degraded", "alert-node-stale"],
            ["deploy-v3", "/v1/releases", "eventbridge", "/stream/releases", "analytics lake"],
        ],
    },
    "Mixed Operations": {
        "questions": [
            "What error-rate threshold and duration triggers automatic rollback?",
            "Which systems provide logs, traces, and metrics in the observability stack?",
            "What are the RPO and RTO targets in disaster recovery?",
            "Which role requires MFA plus INC-* ticket for break-glass access?",
        ],
        "ground_truths": [
            "Error rate above 2% for 10 minutes.",
            "Logs via Vector->Loki, traces via OpenTelemetry->Tempo, metrics via Prometheus.",
            "RPO 15 minutes and RTO 60 minutes.",
            "ROLE_BREAK_GLASS.",
        ],
        "expected_terms": [
            ["2%", "10", "minutes", "rollback"],
            ["loki", "tempo", "prometheus"],
            ["rpo", "15", "rto", "60"],
            ["role_break_glass", "mfa", "inc-"],
        ],
    },
}

DEFAULT_EVAL_QUESTIONS = BENCHMARK_SUITES["Fact Retrieval (Vector-friendly)"]["questions"]
DEFAULT_GROUND_TRUTHS = BENCHMARK_SUITES["Fact Retrieval (Vector-friendly)"]["ground_truths"]
