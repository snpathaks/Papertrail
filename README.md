# PaperTrail: Research Discovery Agent

PaperTrail is an autonomous, agentic research discovery and literature review system. Built with **LangGraph**, **Streamlit**, **ChromaDB**, and local/cloud LLMs via **Ollama**, PaperTrail discovers academic papers, extracts structured findings with verbatim evidence, clusters methodologies, highlights research gaps, and drafts evidence-backed literature reviews with automated citation checking.

---

## 🚀 Completed Work & Features

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
  - Automatic PDF downloader with User-Agent headers and PDF magic-byte validation.
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

### 6. Testing, Evaluation & Git Hygiene
- **Evaluation Suite (`eval/`)**: `eval_extraction.py` benchmarking dataset extraction F1 and quote verification rates against `gold_extraction.json`.
- **Repository Cleanliness (`.gitignore`)**: Configured and committed `.gitignore` ignoring Python bytecode caches, virtual environments, SQLite databases, ChromaDB stores, and downloaded PDFs.

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

### 4. Run Evaluation
Run the extraction evaluation benchmark:
```bash
python eval/eval_extraction.py
```
