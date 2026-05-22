<div align="center">

# ⚔️ RAG Arena

### Vector RAG vs GraphRAG — Side-by-Side Benchmark

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-8B5CF6?logo=data:image/svg+xml;base64,&logoColor=white)](https://llamaindex.ai)
[![Groq](https://img.shields.io/badge/Groq-LLama_3.3_70B-F55036?logo=groq&logoColor=white)](https://groq.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-Aura-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/cloud/aura/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A zero-local-weight benchmark app that indexes the same document using two different RAG architectures and lets you compare their answers — powered by Groq, ChromaDB, and Neo4j Aura.*

---

[Getting Started](#-getting-started) · [How It Works](#-how-it-works) · [Benchmark Suites](#-benchmark-suites) · [Tech Stack](#-tech-stack) · [Troubleshooting](#-troubleshooting)

</div>

---

## 📖 What is RAG Arena?

**RAG Arena** is an interactive Streamlit application that puts two RAG (Retrieval-Augmented Generation) architectures head-to-head on the same source document:

| | Pipeline A — **Vector RAG** | Pipeline B — **GraphRAG** |
|---|---|---|
| **Indexing** | Chunks text → embeds with BGE-small → stores in ChromaDB | LLM extracts entities & relationships → stores in Neo4j graph |
| **Retrieval** | Top-3 cosine similarity search | Graph traversal + vector similarity |
| **Storage** | In-memory (ephemeral) | Neo4j Aura (persistent, cloud) |
| **Strengths** | Fast fact lookup, keyword-style queries | Multi-hop reasoning, relationship queries |

Ask any question, and see both pipelines answer **side-by-side** with latency metrics. Then run built-in benchmark suites to quantitatively compare them using RAGAS evaluation metrics.

---

## ✨ Key Features

- **🔄 Side-by-Side Comparison** — Ask a question once, get two answers instantly with response times
- **📊 RAGAS Benchmarking** — Automated evaluation with faithfulness, answer relevancy, and context recall
- **📈 Visual Analytics** — Bar charts and DataFrames showing metric breakdowns per pipeline
- **🏆 Winner Determination** — Weighted scoring (70% quality, 30% latency) with tie detection
- **🛡️ Robust Error Handling** — Graceful rate-limit recovery, partial results, and fallback scoring when RAGAS fails
- **⚡ Zero Local Weight** — All heavy inference runs on Groq Cloud; only embeddings are local
- **🔒 Flexible Credentials** — Use `.env` file or enter credentials directly in the sidebar

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.12 or higher |
| **Groq API Key** | Free at [console.groq.com](https://console.groq.com) |
| **Neo4j Aura Instance** | Free tier at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) |

### 1. Clone the Repository

```bash
git clone https://github.com/Mayan-kr/RAG_Arena.git
cd RAG_Arena
```

### 2. Create & Activate Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
GROQ_API_KEY=gsk_your_key_here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j
```

> **💡 Tip:** You can also enter these directly in the app's sidebar — no `.env` file required!

### 5. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌─────────────┐                                                 │
│  │   Sidebar    │    ┌─────────────────────────────────────────┐ │
│  │ ─────────── │    │         Side-by-Side Results            │ │
│  │ Groq Key    │    │                                         │ │
│  │ Neo4j Creds │    │  ┌──────────┐     ┌──────────────────┐  │ │
│  │ Query Input │───▶│  │Vector RAG│     │    GraphRAG       │  │ │
│  │ Benchmark ☐ │    │  │  Answer  │     │     Answer        │  │ │
│  └─────────────┘    │  │  0.82s   │     │     1.24s         │  │ │
│                      │  └──────────┘     └──────────────────┘  │ │
│                      └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   ┌─────────────────────┐       ┌──────────────────────┐
   │   Pipeline A:       │       │   Pipeline B:         │
   │   Vector RAG        │       │   GraphRAG            │
   │                     │       │                       │
   │ Documents           │       │ Documents             │
   │   ↓ SentenceSplitter│       │   ↓ LLM Extraction    │
   │   ↓ BGE-small embed │       │   ↓ Entities/Relations│
   │   ↓ ChromaDB (RAM)  │       │   ↓ Neo4j Aura        │
   │   ↓ Top-3 similarity│       │   ↓ Graph traversal   │
   │   ↓ Groq LLM        │       │   ↓ Groq LLM          │
   │   → Answer          │       │   → Answer             │
   └─────────────────────┘       └──────────────────────┘
```

### Step-by-Step Flow

1. **Launch** — Open the app and enter your API credentials in the sidebar
2. **Ingest** — The app loads the source document (`dataset/system_specs.md`) using LlamaIndex
3. **Index** — Both pipelines index the same document simultaneously (first run takes longer for GraphRAG due to LLM entity extraction)
4. **Query** — Type a question in the sidebar and hit enter
5. **Retrieve & Generate** — Each pipeline retrieves relevant context and generates an answer via Groq
6. **Display** — Answers appear side-by-side with response latency
7. **Benchmark** *(optional)* — Enable the benchmark checkbox to run RAGAS evaluation suites

---

## 📋 Benchmark Suites

The app includes **3 curated benchmark suites** (4 questions each) specifically designed to test different retrieval strengths:

| Suite | Purpose | Example Question |
|---|---|---|
| **Fact Retrieval** | Direct fact lookup (vector-friendly) | *"What is the heartbeat timeout for nexus-agent?"* |
| **Relationship & Multi-hop** | Entity traversal (graph-friendly) | *"Trace the JWT validation chain from login to API call"* |
| **Mixed Operations** | Combined reasoning | *"What are the rollback thresholds for canary deployments?"* |

### Evaluation Metrics

| Metric | Source | What It Measures |
|---|---|---|
| Faithfulness | RAGAS | Are claims supported by the retrieved context? |
| Answer Relevancy | RAGAS | Does the answer address the question? |
| Context Recall | RAGAS | Was the relevant information retrieved? |
| Token F1 | Fallback | Token-level overlap with ground truth |
| Fact Recall | Fallback | Coverage of key facts (numbers, percentages) |
| Keyword Recall | Fallback | Coverage of expected technical terms |

> **📌 Note:** If RAGAS metrics return NaN (common with some LLM configurations), the app automatically falls back to custom scoring metrics for reliable results.

### Winner Determination

The overall winner is calculated using a **weighted score**:

```
Score = (Quality × 0.70) + (Latency Factor × 0.30)
```

A tie is declared when scores are within a close threshold, ensuring fair comparison.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|---|---|---|
| **UI** | [Streamlit](https://streamlit.io) | Interactive web interface |
| **RAG Framework** | [LlamaIndex](https://llamaindex.ai) | Document indexing, retrieval, and query orchestration |
| **LLM** | [Groq](https://groq.com) — Llama 3.3 70B | Fast cloud-based inference |
| **Embeddings** | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Local text embeddings via HuggingFace |
| **Vector Store** | [ChromaDB](https://www.trychroma.com) | In-memory vector similarity search |
| **Graph Store** | [Neo4j Aura](https://neo4j.com/cloud/aura/) | Cloud property graph database |
| **Evaluation** | [RAGAS](https://docs.ragas.io) | RAG quality metrics with fallback scoring |
| **Visualization** | Matplotlib + Pandas | Benchmark charts and data tables |

---

## 📁 Project Structure

```
RAG_Arena/
│
├── app.py                        # 🎯 Main Streamlit application
├── requirements.txt              # 📦 Python dependencies
├── .env.example                  # 🔑 Environment variable template
├── .gitignore                    # 🚫 Git exclusions
│
├── modules/
│   ├── __init__.py               # Package marker
│   ├── data_ingestion.py         # 📄 Document loader (SimpleDirectoryReader)
│   ├── vector_pipeline.py        # 🔷 ChromaDB vector RAG pipeline
│   ├── graph_pipeline.py         # 🔶 Neo4j GraphRAG pipeline
│   └── evaluator.py              # 📊 RAGAS benchmark runner + fallback scoring
│
├── dataset/
│   └── system_specs.md           # 📋 Sample document (NexusGrid Platform spec)
│
├── guide.txt                     # 📘 Detailed operational guide
└── PROJECT_HURDLES_LOG.md        # 📝 Development log & solutions
```

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ Neo4j connection fails or shows routing errors</b></summary>

- Ensure your Neo4j Aura instance is **running** (check the Aura console)
- Verify the URI format: `neo4j+s://xxxxx.databases.neo4j.io`
- Double-check the database name — Aura uses instance-specific names, not always `neo4j`
- The app tries to auto-infer the database name from the URI; verify it in the sidebar

</details>

<details>
<summary><b>❌ Groq API returns 429 (Rate Limit) errors</b></summary>

- The free Groq tier has request limits — wait a moment and retry
- During benchmarks, the app handles rate limits gracefully and returns partial results
- Consider spacing out benchmark runs

</details>

<details>
<summary><b>❌ RAGAS metrics show NaN</b></summary>

- This is a known issue with certain LLM/metric combinations
- The app automatically falls back to custom metrics (token F1, fact recall, keyword recall)
- Results are still meaningful and comparable

</details>

<details>
<summary><b>❌ GraphRAG indexing is very slow on first run</b></summary>

- This is expected — the LLM must extract entities and relationships from the entire document
- Subsequent runs use Streamlit's cache (`@st.cache_resource`) and load instantly
- The cache persists until the app is restarted

</details>

<details>
<summary><b>❌ Import errors or missing modules</b></summary>

- Make sure your virtual environment is activated
- Re-run `pip install -r requirements.txt`
- Verify you're using Python 3.12+: `python --version`

</details>

---

## 💡 Usage Tips

- **Start simple** — Try a direct factual question first (e.g., *"What is the SLA for nexus-api?"*) to verify both pipelines are working
- **Test relationships** — Ask multi-hop questions (e.g., *"How does a deployment flow from commit to production?"*) to see where GraphRAG shines
- **Run all suites** — Use the benchmark dropdown to compare across different question types
- **Watch the latency** — Vector RAG is typically faster; GraphRAG trades speed for deeper reasoning
- **Swap the document** — Replace `dataset/system_specs.md` with your own document to benchmark on your data

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📝 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Mayan-kr](https://github.com/Mayan-kr)**

*If you found this useful, give it a ⭐!*

</div>

