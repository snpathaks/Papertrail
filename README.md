# PaperTrail: Research Discovery Agent

PaperTrail is an autonomous, agentic research discovery and literature review system. Built with **LangGraph**, **Streamlit**, **ChromaDB**, and local/cloud LLMs via **Ollama**, PaperTrail discovers academic papers, extracts structured findings with verbatim evidence, clusters methodologies, highlights research gaps, and drafts evidence-backed literature reviews with automated citation checking.

---

## 📸 Demo & Sample Outputs (`demo/`)

Sample outputs from verified end-to-end runs are stored directly in the [`demo/`](demo/) directory for reference:

| File | Description | Link |
| :--- | :--- | :--- |
| **`demo_review.md`** | Complete generated academic literature review with inline citations (`[P1]`, `[P2]`) and formatted references. | [View Review](demo/demo_review.md) |
| **`demo_comparison_table.json`** | Cross-paper methodology comparison matrix, mapped datasets, and approach family clusters. | [View Table](demo/demo_comparison_table.json) |
| **`demo_gaps.json`** | Identified research gaps, statistical evidence, confidence scores, and proposed research questions. | [View Gaps](demo/demo_gaps.json) |

All 3 complete smoke test runs are preserved under [`demo/run_1_...`](demo/run_1_graph_neural_networks_for_drug/), [`demo/run_2_...`](demo/run_2_retrieval_augmented_generation/), and [`demo/run_3_...`](demo/run_3_diffusion_models_for_medical_i/).

---

## 🧪 Smoke Test & Frozen Release (`v1-local`)

The system was smoke-tested end-to-end across **3 diverse academic topics** with **5 papers each** and frozen under git tag `v1-local`:

| Run | Research Topic | Discovered / Selected | Gaps Identified | Review Length | Citation Verification |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | `graph neural networks for drug discovery` | 10 / 5 papers | 5 gaps | 2,817 chars | **1.0 (10/10 verified)** |
| **2** | `retrieval augmented generation hallucination mitigation` | 10 / 5 papers | 2 gaps | 2,362 chars | **1.0 (10/10 verified)** |
| **3** | `diffusion models for medical image synthesis` | 10 / 5 papers | 3 gaps | 2,507 chars | **1.0 (10/10 verified)** |

To check out the frozen working tag:
```bash
git checkout v1-local
```

---

## 🚀 Completed Architecture & Features

### 1. Core Architecture & State Management
- **Centralized Configuration (`config.py`)**: Configurable LLM models (`qwen2.5:3b`), embedding models (`nomic-embed-text`), Ollama endpoints, context window settings (`NUM_CTX = 4096`), and paper search thresholds.
- **Typed State Graph (`state.py`)**: `PaperTrailState` manages end-to-end execution across search queries, retrieved papers, user-approved selections, structured extractions, methodology comparisons, identified research gaps, drafted reviews, and verification metrics.
- **Strict Pydantic Schemas (`schemas.py`)**:
  - `Queries`: Structured query expansion for multi-angle academic search.
  - `RelevanceDecision`: Relevance classification with 1-sentence rationales.
  - `MethodInfo`: Problem statements, proposed approaches, and exact quote evidence.
  - `ResultsInfo` & `Metric`: Dataset names, metric values, and verbatim results quotes.
  - `LimitInfo`: Stated limitations, future work, and grounding quotes.
  - `ClusterLabel` & `GapExplanation`: Categorical grouping and grounded research question generation.

### 2. Multi-Agent Workflow (LangGraph)
- **Phase 1: Discovery Graph (`graph.py`)**:
  - **Expand Node (`agents/search_agent.py`)**: Synthesizes alternate academic queries, synonyms, and broader concepts.
  - **Search Node (`agents/search_agent.py`)**: Queries academic APIs concurrently, deduplicating papers by ID and normalized titles.
  - **Filter Node (`agents/search_agent.py`)**: Validates relevance against the target research topic with fallback keyword matching.
  - **Conditional Routing**: Automatically loops back to broaden queries if fewer than `MIN_PAPERS` are found within max retry limits.
- **Phase 2: Analysis Graph (`graph.py`)**:
  - **Extract Node (`agents/extraction_agent.py`)**: Deconstructs each paper into structured methodology, results, and limitations.
  - **Compare Node (`agents/comparison_agent.py`)**: Clusters papers into approach families, maps shared datasets, and produces comparative insights.
  - **Gaps Node (`agents/gap_agent.py`)**: Identifies unexplored methodology-dataset intersections and recurring limitations.
  - **Write Node (`agents/writer_agent.py`)**: Generates a 5-section literature review with inline citations (`[P1]`, `[P2]`).
  - **Verify Node (`verify.py`)**: Audits drafted claims and numbers against original paper text.

### 3. Data Ingestion & Storage Pipeline
- **Dual-Source Academic Search (`tools/search_tools.py`)**:
  - **OpenAlex API**: Full metadata, abstract reconstruction from inverted index, open-access PDF links, and citation counts.
  - **arXiv API**: Direct preprint retrieval via the official `arxiv` client.
- **PDF Extraction (`tools/pdf_parser.py`)**:
  - Automatic PDF downloader with User-Agent headers, 15-second timeouts, and PDF magic-byte validation.
  - Section-aware markdown conversion via `pymupdf4llm` preserving page numbers and section headings (`Front matter`, `Methodology`, `Results`, etc.).
  - Automatic fallback to abstracts if open-access PDFs are unavailable.
- **Vector Store & Caching (`storage/`)**:
  - **ChromaDB (`storage/vectorstore.py`)**: Persistent cosine-distance vector collection for paper chunk indexing and semantic retrieval.
  - **SQLite Database (`storage/db.py`)**: SHA-256 hashed key-value cache to avoid re-extracting previously analyzed papers.

### 4. Analysis, Verification & Export
- **Evidence Verification (`agents/extraction_agent.py`)**: Employs `rapidfuzz` partial-ratio fuzzy matching to verify that extracted quotes exist in the source document chunks (>= 85% match).
- **Thematic Clustering (`tools/cluster.py`)**: KMeans clustering on paper methodology embeddings with an automatic TF-IDF fallback when offline.
- **Algorithmic Gap Detection (`agents/gap_agent.py`)**:
  - Dataset concentration analysis (over-reliance on benchmark datasets).
  - Cross-tabulation coverage matrix (identifying unstudied methodology × dataset pairs).
  - Aspect coverage analysis (robustness, efficiency, fairness, interpretability, generalization).
  - Recurring limitation clustering.
- **Citation Hallucination Checker (`verify.py`)**:
  - Verifies cited papers exist in the candidate pool.
  - Checks that numerical claims, percentages, and decimals match source evidence.
  - Computes citation support scores.
- **Export Formats (`export.py`)**:
  - Markdown literature review with human-readable references (`Author et al., Year`).
  - Standard BibTeX generation for LaTeX/Overleaf integration.

### 5. Interactive Streamlit App (`app.py`)
- Two-phase interactive workflow:
  1. **Discovery & Human-in-the-Loop Curation**: Search papers by topic, filter by publication year, and review relevance reasons with an interactive editable table.
  2. **Automated Deep Analysis**: Visual status tracker detailing agent execution across extraction, clustering, gap analysis, drafting, and verification.
- Five dedicated result tabs:
  - **Papers**: Summarized paper metadata, problem statements, and abstracts.
  - **Comparison**: Interactive methodology matrix, approach clusters, and shared benchmark datasets.
  - **Gaps**: Confidence-tagged research gaps with proposed research questions and supporting paper references.
  - **Literature Review**: Citation-verified formatted review with download buttons for Markdown (`.md`) and BibTeX (`.bib`).
  - **Evidence**: Interactive evidence inspector showing extracted quotes, source page numbers, and verification status.

---

## 📂 Project Structure

```
Papertrail/
├── app.py                      # Streamlit interactive frontend
├── config.py                   # Central settings, models, paths & thresholds
├── export.py                   # Markdown review & BibTeX export formatters
├── graph.py                    # LangGraph Discovery & Analysis StateGraphs
├── llm.py                      # Ollama client initialization & resilient fallbacks
├── schemas.py                  # Pydantic data validation schemas
├── state.py                    # TypedDict graph state definition
├── verify.py                   # Citation support & hallucination verification
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules for caches, DBs, and runtime data
├── agents/                     # Autonomous pipeline agents
│   ├── search_agent.py         # Query expansion, search, & relevance filtering
│   ├── extraction_agent.py     # Structured paper extraction & quote verification
│   ├── comparison_agent.py     # Methodology clustering & comparison tables
│   ├── gap_agent.py            # Algorithmic research gap detection
│   └── writer_agent.py         # 5-section academic review drafting
├── tools/                      # Core search, parsing, & ML tools
│   ├── search_tools.py         # OpenAlex & arXiv search integrations
│   ├── pdf_parser.py           # Downloader & pymupdf4llm section chunker
│   └── cluster.py              # KMeans & TF-IDF clustering
├── storage/                    # Persistence layers
│   ├── db.py                   # SQLite key-value hash cache
│   └── vectorstore.py          # ChromaDB chunk indexing & retrieval
├── eval/                       # Benchmark & evaluation suite
│   ├── eval_extraction.py      # Extraction F1 & quote verification benchmark
│   └── gold_extraction.json    # Gold evaluation dataset
└── demo/                       # Frozen smoke-test outputs & demo artifacts
    ├── demo_review.md          # Sample generated literature review
    ├── demo_comparison_table.json
    ├── demo_gaps.json
    ├── run_1_graph_neural_networks_for_drug/
    ├── run_2_retrieval_augmented_generation/
    └── run_3_diffusion_models_for_medical_i/
```

---

## 🛠️ Tech Stack

- **Workflow Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain)
- **LLM & Embeddings**: [langchain-ollama](https://github.com/langchain-ai/langchain-ollama), [Ollama](https://ollama.com/) (`qwen2.5:3b`, `nomic-embed-text`)
- **Web Interface**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **PDF & Document Parsing**: [PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF)
- **Academic APIs**: OpenAlex, [arXiv](https://pypi.org/project/arxiv/)
- **Machine Learning & Verification**: [scikit-learn](https://scikit-learn.org/), [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## 📦 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. (Optional) Run Local LLM
If using Ollama for local offline processing:
```bash
ollama run qwen2.5:3b
ollama pull nomic-embed-text
```

### 3. Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```

### 4. Run Evaluation Benchmark
Run the extraction evaluation benchmark:
```bash
python eval/eval_extraction.py
```
