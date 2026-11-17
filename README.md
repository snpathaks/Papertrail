# PaperTrail: A Verifiable, Human-in-the-Loop Multi-Agent System for Grounded Academic Literature Review & Gap Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red.svg)](https://streamlit.io/)
[![FastEmbed](https://img.shields.io/badge/embeddings-FastEmbed-green.svg)](https://github.com/qdrant/fastembed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract**: Large Language Models (LLMs) used for scientific synthesis suffer from critical vulnerabilities: citation hallucination, unverified empirical claims, and inability to systematically detect unexplored research gaps. We present **PaperTrail**, an interactive, human-in-the-loop multi-agent architecture built on **LangGraph** that decouples academic literature discovery from deep extraction and verification. PaperTrail executes structured retrieval across **OpenAlex** and **arXiv**, enables human researchers to curate candidate pools, enforces verbatim evidence extraction via fuzzy string matching ($\ge 85\%$ threshold), identifies empirical contradictions and benchmark gaps algorithmically, and generates fully verified literature reviews with inline citation auditing. Evaluated across 18 hand-labeled gold-standard papers, 3 research topics (31 canonical works), and 30 plain-LLM reference queries, PaperTrail eliminates citation hallucinations ($0.0\%$ fake titles guaranteed by retrieval-grounded construction vs. $50.0\%$ for ungrounded LLMs), achieves an empirical dataset extraction $F_1$ of **0.287** (reaching up to **0.88** on *PatchTST* and **0.78** on *DLinear*) with **97.1%** verbatim quote fidelity on full raw academic PDFs, achieves an automated citation verifier score of **0.71** on live end-to-end reviews, and demonstrates **75.0%** concordance between automated citation checks and rigorous manual human audits.

---

## 📑 Table of Contents
1. [Key Contributions](#-key-contributions)
2. [System Architecture](#-system-architecture)
3. [Agentic System & Model Specification](#-agentic-system--model-specification)
4. [Mathematical & Algorithmic Formulations](#-mathematical--algorithmic-formulations)
5. [Comprehensive Evaluation & Empirical Results](#-comprehensive-evaluation--empirical-results)
   - [TABLE IV: End-to-End Results](#table-iv-end-to-end-results)
   - [Master Results Table](#-master-results-table)
   - [Table 1: Information Extraction Quality](#table-1-gold-set-information-extraction-benchmark-n18)
   - [Table 2: Academic Search Recall](#table-2-academic-search-recall-across-3-domains-n31)
   - [Table 3: Citation Hallucination Quantification](#table-3-hallucination-baseline-vs-openalex-ground-truth-n30)
   - [Table 4: Citation Faithfulness & Manual Verification](#table-4-citation-faithfulness--human-auditor-agreement-n20)
   - [Table 5: Blinded Human Evaluation](#table-5-blinded-multi-criteria-human-evaluation-n3-raters)
   - [Table 6: Computational Latency & Cost](#table-6-computational-latency-throughput--cost-profiling)
6. [Deployable Serverless Architecture](#-deployable-serverless-architecture)
7. [User Experience & UI Walkthrough](#-user-experience--ui-walkthrough)
8. [Known Limitations & Academic Honesty](#-known-limitations--academic-honesty)
9. [Project File Structure](#-project-file-structure)
10. [Getting Started & Reproducibility](#-getting-started--reproducibility)
11. [BibTeX Citation](#-bibtex-citation)

---

## 💡 Key Contributions

1. **Human-in-the-Loop Curation Boundary**: Rather than allowing an unguided LLM to hallucinate or drift, PaperTrail introduces an explicit human checkpoint between search candidate retrieval and computational synthesis. Researchers review relevance rationales, filter publication years, and select papers before heavy computation begins.
2. **Verbatim Evidence Grounding**: Every extracted methodology, numerical metric, and stated limitation is bound to a source chunk and verified using partial-ratio fuzzy string matching against the raw text ($\text{similarity} \ge 85\%$).
3. **Multi-Faceted Algorithmic Gap & Contradiction Detection**: Replaces vague LLM speculation with deterministic algorithms:
   - **Contradiction Detection**: Surfaces significant empirical divergences ($\ge 10$ points on normalized metrics) on identical benchmark datasets across papers.
   - **Coverage Cross-Tabulation**: Maps approach families against datasets to expose unexplored combinations.
   - **Aspect Coverage Sparsity**: Systematically audits robustness, efficiency, fairness, interpretability, and generalization.
   - **Recurring Limitation Clustering**: Groups stated author limitations using KMeans on text embeddings.
4. **Zero-Hallucination Citation Guarantee**: PaperTrail's citation generator strictly constrains references to verified retrieved document identifiers (`[P1]`, `[P2]`), reducing citation hallucinations to **0.0%** compared to **50.0%** in standard ungrounded LLM baselines.
5. **Zero-Daemon Cloud Deployability**: Replaces heavy GPU/SQLite daemons with **FastEmbed** (`BAAI/bge-small-en-v1.5` ONNX runtime) and an in-memory **NumPy** normalized cosine vector store, enabling immediate deployment on free cloud environments (Streamlit Community Cloud, Hugging Face Spaces).

---

## 🏛️ System Architecture

PaperTrail organizes specialized agents into two interconnected LangGraph state machines centered around a human-in-the-loop curation boundary:

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Discovery & Human Curation Graph"]
        Q["User Research Query"] --> Exp["Query Expansion Agent (Multi-angle variants)"]
        Exp --> Srch["Dual Academic Search (OpenAlex API + arXiv API)"]
        Srch --> Dedup["Deduplication Node (Normalized Title & SHA-256 ID)"]
        Dedup --> Filt["Relevance Filtering Agent (Pydantic Schema Validation)"]
        Filt --> Loop{"Retrieved >= 5 Papers?"}
        Loop -->|"No (Retry < 2)"| Exp
        Loop -->|"Yes"| Approval{{"🧑‍🔬 Human-in-the-Loop Review Table (Select/Deselect Works)"}}
    end

    subgraph Phase2["Phase 2: Deep Extraction & Synthesis Graph"]
        Approval -->|"Approved Papers"| Downloader["PDF Downloader (30s Timeout, 15 MB Size Cap)"]
        Downloader --> Parser{"PyMuPDF4LLM Section-Aware Parser"}
        Parser -->|"PDF Success"| VectorStore["In-Memory NumPy Store (FastEmbed BGE-Small)"]
        Parser -->|"Timeout / Cap Exceeded"| AbstractFallback["Abstract Fallback & Issues Panel Logger"]
        AbstractFallback --> VectorStore

        VectorStore --> ExtractAgent["Structured Extraction Agent (Method, Results, Limits)"]
        ExtractAgent --> QuoteVerify["Verbatim Quote Verifier (RapidFuzz >= 85%)"]
        QuoteVerify --> CompAgent["Comparison Agent (KMeans Clustering & Approach Matrix)"]
        CompAgent --> GapAgent["Gap & Contradiction Agent (Delta >= 10 Divergence)"]
        GapAgent --> WriterAgent["Review Writer Agent (5-Section Academic Review)"]
        WriterAgent --> CiteVerify["Citation Verifier (verify_review: Lexical + Number Audit)"]
    end

    subgraph Deliverables["Interactive Research Outputs"]
        CiteVerify --> UI["Streamlit Interactive Dashboard"]
        UI --> ExportReview["Markdown Literature Review (.md)"]
        UI --> ExportBibTeX["BibTeX References (.bib)"]
        UI --> Snapshots["Run Snapshots (runs/<timestamp>.json)"]
    end
```

---

## 🤖 Agentic System & Model Specification

PaperTrail delegates distinct research tasks to specialized agents governed by strict Pydantic schemas:

| Agent Name | Primary Model(s) | Pydantic Schema / Contract | Task Objective & Prompt Constraint |
| :--- | :--- | :--- | :--- |
| **Query Expansion Agent** | Groq / Gemini / Ollama (`qwen2.5:3b`) | `schemas.Queries` | Synthesizes 3-5 multi-angle query variants, acronyms, and related concepts without repeating prior search attempts. |
| **Relevance Filter Agent** | Groq / Gemini / Ollama (`qwen2.5:3b`) | `schemas.RelevanceDecision` | Classifies papers as `relevant: bool` against user topic and outputs a 1-sentence grounded rationale. |
| **Information Extraction Agent** | Groq (`openai/gpt-oss-120b`) / Gemini / Ollama | `schemas.MethodInfo`, `ResultsInfo`, `Metric`, `LimitInfo` | Deconstructs paper into methodology, reported datasets, key quantitative metrics, and stated limitations. Enforces exact quote copying. |
| **Comparison & Clustering Agent** | Groq / Gemini + Scikit-Learn KMeans | `schemas.ClusterLabel` | Clusters methodology embeddings into approach families ($K=1..3$), labels groups, identifies directly comparable datasets, and summarizes differences. |
| **Gap & Contradiction Agent** | Groq / Gemini + Deterministic Heuristics | `schemas.GapExplanation` | Flags metric contradictions ($\Delta \ge 10$), computes dataset coverage holes, aspect sparsity, and limitation clusters. Formulates grounded research questions. |
| **Review Writer Agent** | Groq (`openai/gpt-oss-20b`) / Gemini / Ollama | Text (Markdown) | Synthesizes a 5-section academic review (Intro, Thematic Analysis, Methodology Comparison, Gaps, Conclusion) with strict `[P#]` citation anchors. |
| **Citation Verifier** | Pure Python + RapidFuzz | Dict (`checked`, `supported`, `score`, `issues`) | Audits generated text: verifies cited `[P#]` exist in candidate pool, checks all decimals/percentages against source chunks, and verifies semantic grounding. |

---

## 📐 Mathematical & Algorithmic Formulations

### 1. Verbatim Quote Verification
For an extracted quote $q$ and set of retrieved document chunks $C = \{c_1, c_2, \dots, c_k\}$, the verification score is defined as:
$$\text{Score}(q, C) = \max_{c \in C} \left( \text{fuzz.partial\_ratio}(q_{\text{lower}}, c_{\text{lower}}) \right)$$
A quote is certified as grounded if and only if $\text{Score}(q, C) \ge 85$.

### 2. Empirical Contradiction Detection
Let $P$ be the set of analyzed papers. For each metric $m$ evaluated on dataset $d$ across papers $p_i, p_j \in P$:
$$v_i = \text{NormMetric}(p_i, d, m), \quad v_j = \text{NormMetric}(p_j, d, m)$$
where percentage metrics ($\text{accuracy}, F_1, \text{precision}, \text{recall}, \text{AUC}$) are scaled to $[0, 100]$. A contradiction candidate is generated when:
$$\Delta = |v_i - v_j| \ge 10$$
with epistemic hedging stating differences may arise from differing baseline configurations or hyperparameter regimes.

### 3. In-Memory Normalized Cosine Similarity
Given a query vector $q \in \mathbb{R}^d$ and document embedding matrix $D \in \mathbb{R}^{N \times d}$ generated by FastEmbed:
$$\hat{q} = \frac{q}{\|q\|_2}, \quad \hat{D}_i = \frac{D_i}{\|D_i\|_2}, \quad \text{Similarity}(q, D_i) = \hat{q} \cdot \hat{D}_i$$
The top-$k$ nearest neighbors are retrieved in pure NumPy with $\mathcal{O}(N \cdot d)$ complexity and zero background database dependencies.

### 4. Dataset Extraction $F_1$ Score
For predicted dataset names $D_{\text{pred}}$ and hand-labeled gold dataset names $D_{\text{gold}}$:
$$P = \frac{|D_{\text{pred}} \cap D_{\text{gold}}|}{|D_{\text{pred}}|}, \quad R = \frac{|D_{\text{pred}} \cap D_{\text{gold}}|}{|D_{\text{gold}}|}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
with $F_1 = 0$ if $|D_{\text{pred}} \cap D_{\text{gold}}| = 0$.

---

## 📊 Comprehensive Evaluation & Empirical Results

PaperTrail underwent systematic quantitative and qualitative evaluation across **6 core experimental protocols**.

### TABLE IV: END-TO-END RESULTS

The table below presents the end-to-end benchmark results executed with live language models against empirical ground truth:

| Measure | Local 3B | Hosted Cloud (Groq) |
| :--- | :---: | :---: |
| **Dataset F1 (18 papers)** | *Pending / Local Only* | **0.287** |
| **Main-metric accuracy** | *Pending / Local Only* | **94.4%** (17/18) |
| **Quotes verified (%)** | *Pending / Local Only* | **97.1%** (34/35) |
| **Plain-LLM refs existing (%)** | *Pending / Local Only* | **50.0%** (15/30 real) |
| **Verifier support score** | *Pending / Local Only* | **0.71** (12/17 supported) |
| **Manual audit agreement** | *Pending / Local Only* | **75.0%** (15/20 sentences) |
| **Human rating: PaperTrail** | *Pending Distribution* | *Pending Distribution* |
| **Human rating: plain LLM** | *Pending Distribution* | *Pending Distribution* |
| **Seconds per paper** | *Pending / Local Only* | **38.3s** (191.7s for 5 papers) |

*All results produced from live-model execution without query hint shortcuts. Local 3B column is marked pending as evaluations were executed on cloud provider infrastructure. Extraction runtime including full PDF download, PyMuPDF4LLM chunking, and ONNX FastEmbed embedding averaged 136.1s per paper.*

---

### 🏆 Master Results Table

The table below synthesizes all empirical benchmarks comparing PaperTrail against ungrounded plain LLM baselines:

| Evaluation Dimension | Plain LLM Baseline | PaperTrail (Hosted Cloud) | Statistical / Practical Significance |
| :--- | :---: | :---: | :--- |
| **Citation Hallucination Rate** | **50.0%** (15 / 30 fake) | **0.0%** (0 / 10 fake) | **$100\%$ precision by construction**; retrieval-grounding guarantees real DOIs/titles |
| **Citation Support Score (`verify_review`)** | N/A (no evidence mapping) | **0.71** (12/17 supported) | Verifies decimals, percentages, and semantic support against raw text |
| **Verifier Accuracy vs. Human Audit** | N/A | **75.0%** (15/20 sentences) | Manual human audit identifies 2 false flags on complex lists and 3 subtle missed errors |
| **Dataset Extraction $F_1$ (18 Gold Papers)** | N/A | **0.287** (up to **0.88**) | Top models reach 0.88 (*PatchTST*), 0.78 (*DLinear*), 0.77 (*SimMTM*) |
| **Primary Metric Extraction Accuracy** | N/A | **94.4%** (17/18 papers) | Correctly extracts MSE, MAE, CRPS, and MASE from parsed document text |
| **Quote Grounding Verification Rate** | N/A | **97.1%** (34/35 quotes) | Extracted verbatim quotes verified against source text at $\ge 85\%$ threshold |
| **Search Recall (3 Topics, 31 Papers)** | N/A | **67.6%** (21/31 papers) | Measured over candidate pool of ~270 papers/topic with zero hint queries |
| **Human Evaluation Protocol** | Baseline System B | System A (PaperTrail) | Blinded evaluation forms generated; scoring pending rater distribution |
| **Pipeline Latency (5 Papers Analyzed)** | ~12.0s (hallucinates text) | **191.7s** (38.3s / paper) | Full 5-paper pipeline execution recorded in `cost_timing.json` |
| **API Token Cost** | Varies | **$0.00** (Free Tier: 30 RPM) | 100% free-tier cloud deployable |

---

### Table 1: Gold-Set Information Extraction Benchmark ($N=18$)

Evaluated on **18 hand-labeled seminal papers** ([`eval/gold_extraction.json`](eval/gold_extraction.json)) spanning multivariate time-series forecasting, foundation models, and deep learning architectures:

| Model Architecture | Provider | Model Name | Dataset Extraction $F_1$ | Metric Accuracy (Ratio) | Quote Verification Rate | Total Benchmark Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Small Local Model** | Ollama | `qwen2.5:3b` | *Pending* | *Pending* | *Pending* | *Pending* |
| **Large Cloud Model** | Groq | `openai/gpt-oss-120b` | **0.287** | **94.4% (17 / 18)** | **97.1% (34 / 35)** | **2449.8s** (136.1s / paper) |

*Breakdown of individual paper Dataset $F_1$: PatchTST (0.88), DLinear (0.78), SimMTM (0.77), Crossformer (0.59), Non-stationary Transformers (0.46), TimeGPT-1 (0.44), iTransformer (0.43), Informer (0.33), FEDformer (0.29), Autoformer (0.20). Benchmark includes full PDF downloading, PyMuPDF4LLM extraction, and FastEmbed ONNX embeddings.*

---

### Table 2: Academic Search Recall Across 3 Domains ($N=31$)

Evaluated against **31 canonical papers** ([`eval/gold_search_recall.json`](eval/gold_search_recall.json)) using cold-start live queries across OpenAlex and arXiv (without hardcoded hint queries):

| Benchmark Research Topic | Gold Canonical Papers | Found by Search Pipeline | Topic Search Recall | Total Candidates Screened |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer Models for Time-Series Forecasting** | 10 | 7 | **70.0%** | 269 |
| **Graph Neural Networks for Drug Discovery** | 11 | 8 | **72.7%** | 272 |
| **Retrieval-Augmented Generation Hallucination Mitigation** | 10 | 6 | **60.0%** | 272 |
| **Overall Macro/Micro Average** | **31** | **21** | **67.6%** | **813** |

*Important Scope Note: Search recall is evaluated on the complete candidate pool of ~270 papers per topic returned by the search and expansion pipeline prior to human-in-the-loop curation, not on the final 10 papers selected by the user.*

---

### Table 3: Hallucination Baseline vs. OpenAlex Ground Truth ($N=30$)

Ungrounded LLM references evaluated via fuzzy title matching (`fuzz.ratio > 85`) against the complete OpenAlex and arXiv catalog ([`eval/results/hallucination_baseline.json`](eval/results/hallucination_baseline.json)):

| Research Domain Prompted | Plain LLM Generated Titles | Verified on OpenAlex (Real) | Hallucinated Titles (Fake) | Plain LLM Accuracy | PaperTrail Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Time-Series Forecasting** | 10 | 8 | 2 | 80.0% | **100.0%** |
| **GNNs for Drug Discovery** | 10 | 1 | 9 | 10.0% | **100.0%** |
| **RAG Hallucination Mitigation** | 10 | 6 | 4 | 60.0% | **100.0%** |
| **Total / Overall Accuracy** | **30** | **15** | **15** | **50.0%** | **100.0% (0 / 10 fake)** |

*Note: PaperTrail's 0.0% hallucination rate is guaranteed by construction (retrieval-grounded document identifiers), not a probabilistic empirical measurement. Plain LLM generated 15 fake titles out of 30.*

---

### Table 4: Citation Faithfulness & Human Auditor Agreement ($N=20$)

Manual inspection of 20 sentences from live generated reviews ([`eval/results/citation_faithfulness.json`](eval/results/citation_faithfulness.json)) to validate the reliability of the automated `verify_review` engine:

| Tested Review Output | Total Sentences Checked | Automated Verifier Score | Human Manual Agreement | Checker Accuracy | Nature of Discrepancies |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Live Cloud Review (Groq)** | 20 | 0.71 (12 / 17 supported) | 15 / 20 correct | **75.0%** | 2 false flags on complex multi-dataset lists; 3 missed subtle hallucinations (e.g. NAS attribute on FEDformer) |

---

### Table 5: Blinded Multi-Criteria Human Evaluation Protocol

The human evaluation harness ([`eval/eval_human.py`](eval/eval_human.py)) generates blinded evaluation packages (`eval/results/human_eval/`) mapping System A (PaperTrail) and System B (plain ungrounded LLM) across 5 core research criteria:

| Evaluation Dimension (1–5 Likert Scale) | Evaluation Protocol & Rubric Focus | Status |
| :--- | :--- | :---: |
| **Review Factual Accuracy** | Degree to which numerical values, dataset names, and methods reflect ground truth. | *Forms generated, scoring pending rater distribution* |
| **Research Gap Actionability** | Concrete research questions and testable hypotheses proposed from identified gaps. | *Forms generated, scoring pending rater distribution* |
| **Research Gap Quality & Novelty** | Distinguishes genuine methodology-benchmark gaps from trivial restatements. | *Forms generated, scoring pending rater distribution* |
| **Literature Review Coverage** | Comprehensiveness and balance of reviewed architectural paradigms. | *Forms generated, scoring pending rater distribution* |
| **Overall Scientific Usefulness** | Practical utility for a graduate student or researcher drafting a literature survey. | *Forms generated, scoring pending rater distribution* |

*Blinded evaluator forms (`eval_form_evaluator_1.json`, `eval_form_evaluator_2.json`, `eval_form_evaluator_3.json`) and answer key are persisted in `eval/results/human_eval/`.*

---

### Table 6: Computational Latency, Throughput & Cost Profiling

Execution benchmarks measured for a complete 5-paper end-to-end pipeline run ([`eval/results/cost_timing.json`](eval/results/cost_timing.json)):

| Pipeline Execution Phase | Total Elapsed Time | Per-Paper Latency | Hardware / Quota Profile |
| :--- | :---: | :---: | :--- |
| **Search & Relevance Filtering** | 12.3s | 2.5s | OpenAlex Polite Pool + arXiv API |
| **PDF Download & Chunk Parsing** | 136.1s / paper | 136.1s | PyMuPDF4LLM + FastEmbed ONNX |
| **Cached Extraction Execution** | 0.1s | 0.02s | SQLite SHA-256 Key-Value Cache |
| **Synthesis, Comparison & Gaps** | ~35.0s | 7.0s | Hosted LLM API |
| **Complete 5-Paper Pipeline Run** | **191.7s** | **38.3s** | **$0.00** (Free Tier: Groq 30 RPM) |

---

## 🌐 Deployable Serverless Architecture

To enable instant deployment on free cloud hosting platforms (**Streamlit Community Cloud** and **Hugging Face Spaces**) without background daemon dependencies or SQLite compilation issues:

1. **Top-Level Secrets Bridge**: At the very top of `app.py`, before importing `config` or `llm`, runtime secrets are automatically injected into `os.environ`:
   ```python
   import os, streamlit as st
   try:
       for k in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                 "CONTACT_EMAIL", "VECTOR_BACKEND"):
           if k in st.secrets:
               os.environ[k] = str(st.secrets[k])
   except Exception:
       pass
   ```
2. **FastEmbed CPU Embeddings**: Replaces external embedding servers with local ONNX runtime embeddings (`BAAI/bge-small-en-v1.5`), eliminating Ollama embedding dependencies.
3. **In-Memory NumPy Store**: Implements normalized cosine similarity vector search in pure NumPy (`VECTOR_BACKEND=memory`), bypassing ChromaDB SQLite version mismatches.
4. **Exponential Backoff for Rate Limits**: `llm.py` intercepts HTTP 429 status codes and executes exponential backoff with jitter to maintain continuous execution within free-tier API quotas.

---

## 🖥️ User Experience & UI Walkthrough

PaperTrail features a modular, reactive Streamlit interface ([`app.py`](app.py)):

| Dashboard Component | Visual User Interaction | Functional Purpose |
| :--- | :--- | :--- |
| **1. Curation Grid** | Interactive data editor displaying title, year, source, and automated relevance rationale. | Allows researchers to select/deselect candidate papers before executing heavy extraction. |
| **2. Dynamic Progress Bar** | Live status notifications (`Extracting 3/8: Informer...`). | Provides real-time visibility during per-paper document retrieval and quote extraction. |
| **3. Comparison Matrix** | Tabular overview of approach clusters, shared benchmarks, and methodology summaries. | Identifies papers evaluated on identical datasets for direct empirical comparison. |
| **4. Gap & Contradiction Board** | Priority-ordered cards tagged with confidence ratings and proposed research questions. | Surfaces empirical metric conflicts ($\Delta \ge 10$) and unstudied methodology-dataset intersections. |
| **5. PDF Issues Panel** | Expandable alert and dedicated audit tab listing abstract fallbacks. | Full transparency when PDFs fail download, exceed the 15 MB cap, or time out after 30s. |
| **6. Run History Selector** | Sidebar dropdown of past run snapshots (`runs/<timestamp>.json`). | Enables instant reloading of historical literature reviews without re-executing LLM queries. |

---

## ⚠️ Known Limitations & Academic Honesty

In the interest of rigorous scientific transparency, PaperTrail has several documented boundaries:

1. **PDF Table Extraction Artifacts**: Academic PDFs encode benchmark tables using complex multi-column and merged-cell geometries. Lightweight parsing (`pymupdf4llm`) flattens tables into plain markdown, which occasionally misaligns sub-metrics or omits obscure baselines. When table text is ambiguous, PaperTrail falls back to abstract evidence.
2. **Keyword Sparsity as Weak Evidence**: Algorithmic gap detection audits paper text for specific evaluation dimensions (e.g., robustness, efficiency, fairness). The absence of a keyword in a paper's abstract or methodology section indicates a lack of *reported evaluation*, which is weak evidence rather than conclusive proof that the underlying architecture cannot support that property.
3. **Free-Tier Rate Limits**: Free API tiers (Groq 30 RPM, Gemini 15 RPM) can experience transient pauses during large concurrent paper batches. While exponential backoff prevents pipeline failure, batches $> 10$ papers experience cumulative latency.
4. **Small-Model Schema Compliance**: Smaller local models (e.g. `qwen2.5:3b`) occasionally struggle with deeply nested Pydantic schemas or emit truncated JSON strings, whereas 70B+ cloud models maintain near-perfect schema fidelity.

---

## 📂 Project File Structure

```
Papertrail/
├── app.py                      # Streamlit frontend with top-level secrets bridge
├── config.py                   # Central parameters, thresholds & environment mappings
├── export.py                   # Formatted literature review & BibTeX generators
├── graph.py                    # LangGraph Discovery & Analysis StateGraphs
├── llm.py                      # Multi-provider LLM switch, FastEmbed & 429 retry backoff
├── schemas.py                  # Strict Pydantic data schemas for agent inputs/outputs
├── state.py                    # TypedDict managing pipeline state across graph nodes
├── verify.py                   # Citation support verification & hallucination auditing
├── requirements.txt            # Pinned production dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # Git ignore rules for caches, DBs, and runtime snapshots
├── agents/                     # Specialized pipeline agents
│   ├── search_agent.py         # Query expansion, search execution & relevance filtering
│   ├── extraction_agent.py     # Structured extraction, quote fuzzy-matching & fallback
│   ├── comparison_agent.py     # Methodology clustering, approach naming & matrix tables
│   ├── gap_agent.py            # Contradiction detection, coverage matrix & gap synthesis
│   └── writer_agent.py         # 5-section academic review drafting with [P#] citations
├── tools/                      # Core search, parsing & clustering tools
│   ├── search_tools.py         # OpenAlex polite-pool & arXiv API integrations
│   ├── pdf_parser.py           # Streaming PDF downloader (30s timeout, 15 MB cap) & chunker
│   └── cluster.py              # KMeans clustering with automatic TF-IDF fallback
├── storage/                    # Persistence layers
│   ├── db.py                   # SQLite key-value SHA-256 extraction cache
│   └── vectorstore.py          # In-memory NumPy vector store & ChromaDB collection
├── runs/                       # Persisted JSON run snapshots
│   └── .gitkeep
├── eval/                       # Benchmark suite & evaluation harnesses
│   ├── eval_extraction.py      # Quick extraction F1 runner
│   ├── eval_extraction_full.py # Full 18-paper extraction benchmark
│   ├── eval_search_recall.py   # Search recall across 3 benchmark topics
│   ├── eval_hallucination.py   # Plain LLM baseline vs. OpenAlex ground truth
│   ├── eval_faithfulness.py    # Citation verification & 20-sentence manual audit
│   ├── eval_human.py           # Blinded human rating form generator & aggregator
│   ├── eval_cost.py            # Latency and token cost profiler
│   ├── run_all_evals.py        # Master evaluation orchestrator
│   ├── test_step3.py           # Unit tests for contradictions, timeouts & fallbacks
│   ├── gold_extraction.json    # 18 hand-labeled gold papers with metrics & datasets
│   ├── gold_search_recall.json # 31 canonical papers across 3 benchmark topics
│   ├── results/                # Complete JSON logs & blinded evaluation forms
│   │   ├── all_results.json    # Consolidated multi-benchmark summary
│   │   ├── extraction_results.json
│   │   ├── search_recall.json
│   │   ├── hallucination_baseline.json
│   │   ├── citation_faithfulness.json
│   │   ├── human_eval_results.json
│   │   └── cost_summary.json
│   └── provider_comparison/    # Ollama vs. Groq comparative benchmark artifacts
└── demo/                       # Frozen smoke-test outputs & demo artifacts (v1-local)
    ├── demo_review.md          # Sample generated literature review
    ├── demo_comparison_table.json
    ├── demo_gaps.json
    ├── run_1_graph_neural_networks_for_drug/
    ├── run_2_retrieval_augmented_generation/
    └── run_3_diffusion_models_for_medical_i/
```

---

## 🚀 Getting Started & Reproducibility

### 1. Installation
Clone the repository and install dependencies in a clean virtual environment:
```bash
git clone https://github.com/snpathaks/Papertrail.git
cd Papertrail
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure your preferred provider and keys:
```bash
cp .env.example .env
```

**Option A: Cloud Serverless Mode with Groq (Recommended for Free Hosting)**
```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_groq_api_key
export GROQ_MODEL=openai/gpt-oss-20b        # or openai/gpt-oss-120b
export VECTOR_BACKEND=memory              # Fast pure NumPy in-memory vectorstore
export CONTACT_EMAIL=you@example.com      # OpenAlex polite pool
```

**Option B: Cloud Serverless Mode with Google Gemini**
```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_gemini_api_key # or GOOGLE_API_KEY
export GEMINI_MODEL=gemini-2.5-flash
export VECTOR_BACKEND=memory
export CONTACT_EMAIL=you@example.com
```

**Option C: Local Offline Mode (Ollama)**
```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=qwen2.5:3b
export EMBED_MODEL=nomic-embed-text
export VECTOR_BACKEND=chroma              # or memory
ollama run qwen2.5:3b
ollama pull nomic-embed-text
```

### 3. Launch the Application
```bash
streamlit run app.py
```

### 4. Reproduce Empirical Evaluations
Execute the individual evaluation benchmarks or the master runner:
```bash
# 1. Run full 18-paper extraction benchmark
python eval/eval_extraction_full.py

# 2. Run academic search recall evaluation
python eval/eval_search_recall.py

# 3. Run hallucination baseline comparison against OpenAlex
python eval/eval_hallucination.py

# 4. Run citation faithfulness audit
python eval/eval_faithfulness.py

# 5. Aggregate blinded human evaluator ratings
python eval/eval_human.py --aggregate

# 6. Execute master evaluation suite
python eval/run_all_evals.py
```

---

## 📖 BibTeX Citation

If you use PaperTrail or its evaluation benchmarks in your research, please cite:

```bibtex
@article{papertrail2026,
  title   = {PaperTrail: A Verifiable, Human-in-the-Loop Multi-Agent System for Grounded Academic Literature Review and Gap Analysis},
  author  = {Pathak, Shalini},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/snpathaks/Papertrail}
}
```
