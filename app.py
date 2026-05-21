"""
RAG Arena — Streamlit dashboard comparing Vector RAG vs GraphRAG.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from modules.data_ingestion import load_source_documents
from modules.evaluator import (
    BENCHMARK_SUITES,
    DEFAULT_EVAL_QUESTIONS,
    DEFAULT_GROUND_TRUTHS,
    run_comparative_evaluation,
)
from modules.graph_pipeline import build_graph_rag
from modules.vector_pipeline import build_vector_rag

load_dotenv()


def infer_aura_database(uri: str, username: str) -> str:
    """
    Aura Free/Professional often names the database after the instance id
    (same as username), not the legacy default 'neo4j'.
    """
    if uri and ".databases.neo4j.io" in uri:
        try:
            host = uri.split("://", 1)[1].split("/")[0].split(":")[0]
            instance_id = host.split(".")[0]
            if instance_id:
                return instance_id
        except IndexError:
            pass
    if username:
        return username
    return "neo4j"


_default_uri = os.getenv("NEO4J_URI", "")
_default_user = os.getenv("NEO4J_USERNAME", "")
_default_db = os.getenv("NEO4J_DATABASE") or infer_aura_database(_default_uri, _default_user)

st.set_page_config(page_title="RAG Arena", layout="wide")
st.title("RAG Arena: Vector RAG vs GraphRAG")
st.caption("Zero-local-weight benchmark — Groq + Chroma (memory) + Neo4j Aura")

# --- Sidebar: secrets & connection (never hardcoded) ---
with st.sidebar:
    st.header("Credentials & Connections")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    st.subheader("Neo4j Aura (GraphRAG)")
    neo4j_uri = st.text_input("Bolt URI", value=os.getenv("NEO4J_URI", ""))
    neo4j_user = st.text_input(
        "Username",
        value=os.getenv("NEO4J_USERNAME", ""),
        help="Aura often uses your instance id (e.g. c288b90b), not the literal word neo4j.",
    )
    neo4j_password = st.text_input(
        "Password",
        type="password",
        value=os.getenv("NEO4J_PASSWORD", ""),
    )
    neo4j_database = st.text_input(
        "Database name",
        value=_default_db,
        help="Use NEO4J_DATABASE from Aura. For many Aura instances this matches "
        "your instance id (same as username), e.g. c288b90b — not 'neo4j'.",
    )

    st.subheader("Query")
    user_question = st.text_area(
        "Question",
        value="What triggers automatic deploy rollback on NexusGrid?",
        height=100,
    )
    run_benchmark = st.checkbox("Run full RAGAS benchmark (4 questions)", value=False)
    benchmark_suite_name = st.selectbox(
        "Benchmark suite",
        options=list(BENCHMARK_SUITES.keys()),
        index=0,
    )

# --- Guardrails: halt until required inputs exist ---
if not groq_key:
    st.warning("Enter your Groq API key in the sidebar.")
    st.stop()

neo4j_ok = bool(neo4j_uri and neo4j_user and neo4j_password and neo4j_database)

if not neo4j_ok:
    st.warning("Enter Neo4j Aura Bolt URI, username, and password to build GraphRAG.")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model (BGE-small)...")
def get_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Configuring Groq LLM...")
def get_llm(api_key: str):
    from llama_index.llms.groq import Groq

    return Groq(model="llama-3.3-70b-versatile", api_key=api_key)


@st.cache_resource(show_spinner="Indexing Vector RAG (Chroma in-memory)...")
def get_vector_engine(_llm_id: str, _embed_id: str):
    documents = load_source_documents(str(Path(__file__).parent / "dataset"))
    return build_vector_rag(documents, get_embed_model())


@st.cache_resource(show_spinner="Building GraphRAG (Neo4j + LLM extraction)...")
def get_graph_engine(_llm_id: str, _embed_id: str, uri: str, user: str, pwd: str, database: str):
    documents = load_source_documents(str(Path(__file__).parent / "dataset"))
    llm = get_llm(groq_key)
    return build_graph_rag(
        documents, llm, get_embed_model(), uri, user, pwd, database=database
    )


embed_model = get_embed_model()
llm = get_llm(groq_key)
Settings.llm = llm
Settings.embed_model = embed_model

llm_cache_key = f"groq-{bool(groq_key)}"
vector_engine = get_vector_engine(llm_cache_key, "bge-small")
graph_engine = get_graph_engine(
    llm_cache_key,
    "bge-small",
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    neo4j_database,
)

col_v, col_g = st.columns(2)

display_q = user_question.strip()
if not display_q:
    st.error("Enter a question in the sidebar.")
    st.stop()

t0 = time.perf_counter()
v_resp = vector_engine.query(display_q)
v_latency = time.perf_counter() - t0

t0 = time.perf_counter()
g_resp = graph_engine.query(display_q)
g_latency = time.perf_counter() - t0

last_v = {
    "answer": str(getattr(v_resp, "response", v_resp)),
    "latency_s": v_latency,
}
last_g = {
    "answer": str(getattr(g_resp, "response", g_resp)),
    "latency_s": g_latency,
}

if run_benchmark:
    selected_suite = BENCHMARK_SUITES[benchmark_suite_name]
    with st.spinner(f"Running comparative RAGAS evaluation: {benchmark_suite_name}"):
        eval_result = run_comparative_evaluation(
            vector_engine,
            graph_engine,
            selected_suite["questions"],
            selected_suite["ground_truths"],
            expected_terms=selected_suite.get("expected_terms"),
            llm=llm,
            embed_model=embed_model,
        )
    eval_result["suite_name"] = benchmark_suite_name
    st.session_state["eval_result"] = eval_result

with col_v:
    st.subheader("Vector RAG Result")
    st.markdown(f"**Latency:** `{last_v['latency_s']:.3f}` s")
    st.write(last_v["answer"])

with col_g:
    st.subheader("GraphRAG Result")
    st.markdown(f"**Latency:** `{last_g['latency_s']:.3f}` s")
    st.write(last_g["answer"])

if "eval_result" in st.session_state:
    st.divider()
    st.subheader("RAGAS Benchmark Comparison")

    ev = st.session_state["eval_result"]
    v_raw_scores = ev.get("vector_scores", {})
    g_raw_scores = ev.get("graph_scores", {})
    v_scores = {
        k: float(v)
        for k, v in v_raw_scores.items()
        if isinstance(v, (int, float))
    }
    g_scores = {
        k: float(v)
        for k, v in g_raw_scores.items()
        if isinstance(v, (int, float))
    }
    st.caption(f"Suite: {ev.get('suite_name', 'Unknown')}")
    if ev.get("warnings"):
        st.warning(" | ".join(ev["warnings"][:2]))
    vector_warn = v_raw_scores.get("error") or v_raw_scores.get("ragas_warning")
    graph_warn = g_raw_scores.get("error") or g_raw_scores.get("ragas_warning")
    if vector_warn or graph_warn:
        st.warning(
            "RAGAS returned incomplete metric values. Showing available data. "
            f"Vector issue: {vector_warn or 'none'} | "
            f"Graph issue: {graph_warn or 'none'}"
        )

    if v_scores and g_scores:
        metrics = sorted(set(v_scores) & set(g_scores))
        vector_avg_quality = sum(v_scores[m] for m in metrics) / len(metrics)
        graph_avg_quality = sum(g_scores[m] for m in metrics) / len(metrics)
        vector_avg_latency = sum(r["latency_s"] for r in ev["vector"]) / len(ev["vector"])
        graph_avg_latency = sum(r["latency_s"] for r in ev["graph"]) / len(ev["graph"])
        quality_delta = vector_avg_quality - graph_avg_quality
        latency_delta = graph_avg_latency - vector_avg_latency
        quality_winner = "Tie" if abs(quality_delta) < 0.03 else (
            "Vector RAG" if quality_delta > 0 else "GraphRAG"
        )
        latency_winner = "Tie" if abs(latency_delta) < 0.1 else (
            "Vector RAG" if latency_delta > 0 else "GraphRAG"
        )
        # Weighted score: prioritize quality, then latency.
        vector_overall = (0.7 * vector_avg_quality) + (
            0.3 * (1.0 / (1.0 + vector_avg_latency))
        )
        graph_overall = (0.7 * graph_avg_quality) + (
            0.3 * (1.0 / (1.0 + graph_avg_latency))
        )
        overall_delta = vector_overall - graph_overall
        overall_winner = "Tie" if abs(overall_delta) < 0.03 else (
            "Vector RAG" if overall_delta > 0 else "GraphRAG"
        )

        w1, w2, w3 = st.columns(3)
        with w1:
            st.metric(
                "Quality Winner",
                quality_winner,
                delta=f"V:{vector_avg_quality:.3f} | G:{graph_avg_quality:.3f}",
            )
        with w2:
            st.metric(
                "Latency Winner",
                latency_winner,
                delta=f"V:{vector_avg_latency:.3f}s | G:{graph_avg_latency:.3f}s",
            )
        with w3:
            st.metric(
                "Overall Winner",
                overall_winner,
                delta=f"V:{vector_overall:.3f} | G:{graph_overall:.3f}",
            )

        x = range(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - width / 2 for i in x], [v_scores[m] for m in metrics], width, label="Vector RAG")
        ax.bar([i + width / 2 for i in x], [g_scores[m] for m in metrics], width, label="GraphRAG")
        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(" / ".join(metrics))
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        st.pyplot(fig)

        latency_df = pd.DataFrame(
            {
                "pipeline": ["vector"] * len(ev["vector"]) + ["graph"] * len(ev["graph"]),
                "latency_s": [r["latency_s"] for r in ev["vector"]]
                + [r["latency_s"] for r in ev["graph"]],
                "question": [r["question"] for r in ev["vector"]]
                + [r["question"] for r in ev["graph"]],
            }
        )
        st.dataframe(latency_df, use_container_width=True)
        detail_df = pd.DataFrame(
            {
                "question": [r["question"] for r in ev["vector"]],
                "vector_answer": [r["answer"] for r in ev["vector"]],
                "graph_answer": [r["answer"] for r in ev["graph"]],
                "ground_truth": [r["ground_truth"] for r in ev["vector"]],
                "vector_latency_s": [r["latency_s"] for r in ev["vector"]],
                "graph_latency_s": [r["latency_s"] for r in ev["graph"]],
            }
        )
        st.dataframe(detail_df, use_container_width=True)
    else:
        st.error(
            "RAGAS evaluation failed. Check API quotas and RAGAS version compatibility.\n\n"
            f"Vector: {ev.get('vector_scores')}\n\nGraph: {ev.get('graph_scores')}"
        )
