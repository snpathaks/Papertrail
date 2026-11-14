import config
from llm import llm, ask_structured
from schemas import Queries, RelevanceDecision
from tools.search_tools import search_openalex, search_arxiv

def expand_queries(state):
    attempt = state.get("attempts", 0)
    prev = state.get("expanded_queries", [])
    extra = f" Do NOT repeat these: {prev}. Use broader terms." if attempt else ""
    prompt = (f"Research topic: {state['query']}\n"
              "Write 3 short, different academic search queries (synonyms, related terms)." + extra)
    try:
        structured = llm.with_structured_output(Queries)
        qs = structured.invoke(prompt).queries[:3]
    except Exception:
        qs = []
    queries = list(dict.fromkeys(prev + [state["query"]] + qs))
    return {"expanded_queries": queries, "attempts": attempt + 1}

def search_papers(state):
    queries = state.get("expanded_queries", [state.get("query", "")])
    seen_ids = set()
    seen_titles = set()
    all_papers = []

    for q in queries:
        # Search both OpenAlex and arXiv
        found = search_openalex(q, max_results=6) + search_arxiv(q, max_results=6)
        for p in found:
            norm_title = p["title"].strip().lower()
            if p["id"] not in seen_ids and norm_title not in seen_titles:
                seen_ids.add(p["id"])
                seen_titles.add(norm_title)
                all_papers.append(p)
                if len(all_papers) >= config.MAX_PAPERS * 2:
                    break
        if len(all_papers) >= config.MAX_PAPERS * 2:
            break

    return {"papers": all_papers}

def filter_relevance(state):
    papers = state.get("papers", [])
    topic = state.get("query", "")
    filtered = []

    for p in papers:
        prompt = (f"Research topic: {topic}\n"
                  f"Paper title: {p['title']}\n"
                  f"Abstract: {p.get('abstract', '')[:1000]}\n"
                  "Is this paper relevant to the research topic?")
        decision = ask_structured(RelevanceDecision, prompt)
        if decision is not None:
            if decision.relevant:
                p["relevance_reason"] = decision.reason
                filtered.append(p)
        else:
            # Fallback if LLM unavailable: simple term overlap check
            q_terms = set(topic.lower().split())
            text = (p["title"] + " " + p.get("abstract", "")).lower()
            if any(t in text for t in q_terms if len(t) > 3):
                p["relevance_reason"] = "Keyword overlap with topic"
                filtered.append(p)
            else:
                p["relevance_reason"] = "Retained as potential candidate"
                filtered.append(p)

        if len(filtered) >= config.MAX_PAPERS:
            break

    # If filter was too aggressive and we have papers, preserve candidates up to MIN_PAPERS
    if len(filtered) < config.MIN_PAPERS and len(papers) > len(filtered):
        for p in papers:
            if p not in filtered:
                p.setdefault("relevance_reason", "Included to meet minimum paper threshold")
                filtered.append(p)
                if len(filtered) >= config.MIN_PAPERS:
                    break

    return {"papers": filtered}
