# LLM Provider Comparison: Ollama vs. Groq

**Topic**: `transformer models for time-series forecasting`

| Feature / Metric | `LLM_PROVIDER=ollama` | `LLM_PROVIDER=groq` |
| :--- | :--- | :--- |
| **Embeddings Backend** | `OllamaEmbeddings` | `FastEmbedWrapper (BAAI/bge-small-en-v1.5)` |
| **Vector Store** | In-Memory NumPy (`memory`) | In-Memory NumPy (`memory`) |
| **Papers Analyzed** | 5 | 5 |
| **Gaps Discovered** | 4 | 4 |
| **Citation Support Score** | 1.0 (10/10) | 1.0 (10/10) |
| **Cloud Deployability** | Requires running local/self-hosted Ollama server | **100% Serverless / Deployable** (Free Tier API + FastEmbed) |
| **Streamlit Cloud Readiness** | Limited (Ollama server not hostable on free tier) | **Production-Ready** (Works on Streamlit Cloud & Hugging Face Spaces) |

## Approach Clusters Identified
- **Ollama**: ['Group 1', 'Group 2']
- **Groq**: ['Group 1', 'Group 2']

## Key Gaps Identified
- **Ollama**: 4 gaps computed.
- **Groq**: 4 gaps computed.

## Conclusion
Configuring `LLM_PROVIDER=groq` with `fastembed` and `MemoryVectorStore` eliminates the dependency on a local GPU / running Ollama process, enabling deployment on free web hosting platforms like Streamlit Cloud without SQLite issues.
