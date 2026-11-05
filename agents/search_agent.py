def expand_queries(state):
    attempt = state.get("attempts", 0)
    prev = state.get("expanded_queries", [])
    extra = f" Do NOT repeat these: {prev}. Use broader terms." if attempt else ""
    structured = llm.with_structured_output(Queries)
    prompt = (f"Research topic: {state['query']}\n"
              "Write 3 short, different academic search queries (synonyms, related terms)." + extra)
    try:
        qs = structured.invoke(prompt).queries[:3]
    except Exception:
        qs = []
    queries = list(dict.fromkeys(prev + [state["query"]] + qs))
    return {"expanded_queries": queries, "attempts": attempt + 1}