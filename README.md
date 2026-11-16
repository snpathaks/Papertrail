# PaperTrail: Research Discovery Agent

PaperTrail is an autonomous, agentic research discovery and literature review system. Built with **LangGraph**, **Streamlit**, and flexible LLM/vector backends (**Ollama**, **Groq**, **Gemini**, **FastEmbed**, **NumPy**, **ChromaDB**), PaperTrail discovers academic papers, extracts structured findings with verbatim evidence, clusters methodologies, highlights research gaps, and drafts evidence-backed literature reviews with automated citation checking.

---

## 🌐 Deployable Multi-Provider Architecture (Step 2)

To ensure zero-barrier deployment on free cloud hosting (such as **Streamlit Cloud** and **Hugging Face Spaces**) where Ollama and custom C-extensions/SQLite cannot run:

1. **LLM Provider Switch (`LLM_PROVIDER`)**:
   - `ollama` *(default)*: Local offline inference (`qwen2.5:3b`, `llama3`, etc.).
   - `groq`: Fast cloud inference with Llama 3 models (`llama-3.3-70b-versatile`).
   - `gemini`: Cloud inference using Google Gemini (`gemini-2.5-flash`).
2. **Deployable Embeddings (`fastembed`)**:
   - Local ONNX CPU embeddings via `BAAI/bge-small-en-v1.5` used automatically whenever the provider is not Ollama. Eliminates the need for any background model server.
3. **In-Memory NumPy Vector Store (`VECTOR_BACKEND=memory`)**:
   - Lightweight, zero-dependency in-memory vector store computing normalized cosine similarities in pure NumPy. Avoids SQLite version mismatches common with ChromaDB on free cloud containers.
4. **Resilient Rate-Limit Backoff**:
   - Automated detection and exponential backoff retry for HTTP 429 errors across cloud provider API calls (`ask_structured` & `ask_text`).
5. **Automatic Fallback Clustering**:
   - Seamless TF-IDF fallback in `tools/cluster.py` when vector embeddings are unavailable or offline.

---

## 📊 Provider Comparison: Ollama vs. Groq

Both backends were evaluated on the same benchmark topic (`transformer models for time-series forecasting`):

| Feature / Metric | `LLM_PROVIDER=ollama` (Local) | `LLM_PROVIDER=groq` (Cloud Deployable) |
| :--- | :--- | :--- |
| **Embeddings Backend** | `OllamaEmbeddings` | `FastEmbed` (`BAAI/bge-small-en-v1.5`) |
| **Vector Store** | In-Memory NumPy (`memory`) | In-Memory NumPy (`memory`) |
| **Papers Analyzed** | 5 | 5 |
| **Gaps Discovered** | 4 | 4 |
| **Citation Verification Score** | **1.0 (10/10 verified)** | **1.0 (10/10 verified)** |
| **Cloud Deployability** | Requires self-hosted Ollama server | **100% Serverless / Cloud-Ready** |
| **Streamlit Cloud Compatibility** | Low (Server daemon required) | **Production-Ready** (Zero daemon, free tier API) |

*Full evaluation outputs are saved under [`eval/provider_comparison/`](eval/provider_comparison/).*

---

## 📸 Demo & Sample Outputs (`demo/`)

Sample outputs from verified end-to-end runs are stored directly in the [`demo/`](demo/) directory:

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

---

## 🚀 Completed Architecture & Features

### 1. Core Architecture & State Management
- **Centralized Configuration (`config.py`)**: Configurable provider switches (`LLM_PROVIDER`, `VECTOR_BACKEND`), models, endpoints, and search thresholds.
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
  - Automatic PDF downloader with User-Agent headers, 30-second timeouts, and a 15 MB streaming size cap.
  - Section-aware markdown conversion via `pymupdf4llm` preserving page numbers and section headings (`Front matter`, `Methodology`, `Results`, etc.).
  - Automatic fallback to abstracts if open-access PDFs are unavailable, oversized, or unparseable.
- **Vector Store & Caching (`storage/`)**:
  - **In-Memory NumPy (`storage/vectorstore.py`)**: Zero-dependency cosine vector store for serverless deployment.
  - **ChromaDB (`storage/vectorstore.py`)**: Persistent cosine-distance vector collection for local indexing.
  - **SQLite Database (`storage/db.py`)**: SHA-256 hashed key-value cache to avoid re-extracting previously analyzed papers.

### 4. Analysis, Verification & Export
- **Evidence Verification (`agents/extraction_agent.py`)**: Employs `rapidfuzz` partial-ratio fuzzy matching to verify that extracted quotes exist in the source document chunks (>= 85% match).
- **Thematic Clustering (`tools/cluster.py`)**: KMeans clustering on paper methodology embeddings with an automatic TF-IDF fallback when offline.
- **Algorithmic Gap & Contradiction Detection (`agents/gap_agent.py`)**:
  - **Contradiction Detection**: Flags empirical performance divergences (>= 10 points on normalized metrics like accuracy, F1, precision, recall, AUC) on identical datasets between papers (e.g. `P1` vs `P2`), epistemically framed to account for differing experimental settings or baselines.
  - **Dataset Concentration**: Identifies over-reliance on a single benchmark dataset.
  - **Coverage Matrix**: Cross-tabulation identifying unstudied methodology × dataset pairs.
  - **Aspect Coverage**: Evaluates coverage of robustness, efficiency, fairness, interpretability, and generalization.
  - **Recurring Limitations**: Clusters stated limitations and proposed future work across papers.
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
  2. **Automated Deep Analysis**:
     - **Per-Paper Live Progress**: Dynamic UI indicators (`Extracting 3/8: Title...`) during extraction.
     - **Auto-Save Runs**: Saves completed analysis states to `runs/<timestamp>.json`.
     - **Load Previous Runs**: Sidebar selector to reload and inspect previous runs instantly without re-running models.
     - **Issues Panel**: Dedicated expander and tab reporting papers that fell back to abstract-only due to PDF download/parsing constraints.
- Dedicated result tabs:
  - **Papers**: Summarized paper metadata, problem statements, abstracts, and full-text/fallback badges.
  - **Comparison**: Interactive methodology matrix, approach clusters, and shared benchmark datasets.
  - **Gaps**: Priority-ordered research gaps and empirical contradictions with proposed research questions.
  - **Literature Review**: Citation-verified formatted review with download buttons for Markdown (`.md`) and BibTeX (`.bib`).
  - **Evidence**: Interactive evidence inspector showing extracted quotes, source page numbers, and verification status.
  - **Issues**: Complete audit table of all papers with extraction warnings or PDF fallbacks.

---

## 📂 Project Structure

```
Papertrail/
├── app.py                      # Streamlit interactive frontend
├── config.py                   # Central settings, models, paths & thresholds
├── export.py                   # Markdown review & BibTeX export formatters
├── graph.py                    # LangGraph Discovery & Analysis StateGraphs
├── llm.py                      # Multi-provider LLM & FastEmbed initialization
├── schemas.py                  # Pydantic data validation schemas
├── state.py                    # TypedDict graph state definition
├── verify.py                   # Citation support & hallucination verification
├── requirements.txt            # Pinned project dependencies
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
│   └── vectorstore.py          # In-memory NumPy & ChromaDB vector stores
├── runs/                       # Persisted JSON analysis run snapshots
├── eval/                       # Evaluation & verification benchmarks
│   ├── eval_extraction.py      # Dataset extraction & quote verification benchmark
│   ├── test_step3.py           # Unit tests for contradictions, timeouts & fallbacks
│   └── provider_comparison/    # Ollama vs. Groq benchmark artifacts
├── eval/                       # Benchmark & evaluation suite
│   ├── eval_extraction.py      # Extraction F1 & quote verification benchmark
│   ├── gold_extraction.json    # Gold evaluation dataset
│   └── provider_comparison/   # Ollama vs Groq comparative runs & report
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
- **LLM Providers**: [langchain-ollama](https://github.com/langchain-ai/langchain-ollama) (Local), [langchain-groq](https://github.com/langchain-ai/langchain-groq) (Cloud), [langchain-google-genai](https://github.com/langchain-ai/langchain-google) (Cloud)
- **Embeddings**: [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`), Ollama (`nomic-embed-text`)
- **Vector Storage**: In-Memory NumPy Vector Store, [ChromaDB](https://www.trychroma.com/)
- **Web Interface**: [Streamlit](https://streamlit.io/)
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

### 2. Configure Execution Provider
Choose your deployment mode via environment variables:

**Option A: Deployable Serverless Mode (Groq + FastEmbed + Memory Store)**
```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_groq_api_key
export VECTOR_BACKEND=memory
```

**Option B: Local Mode (Ollama)**
```bash
export LLM_PROVIDER=ollama
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
