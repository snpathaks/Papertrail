from langgraph.graph import StateGraph, START, END
import config
from state import PaperTrailState
from agents.search_agent import expand_queries, search_papers, filter_relevance
from agents.extraction_agent import extract_papers
from agents.comparison_agent import compare_papers
from agents.gap_agent import analyze_gaps
from agents.writer_agent import write_review
from verify import verify_review

def _route_after_filter(state):
    if len(state["papers"]) < config.MIN_PAPERS and state.get("attempts", 0) < config.MAX_SEARCH_ATTEMPTS:
        return "expand"          # too few papers: retry with broader queries
    return END

def build_search_graph():
    g = StateGraph(PaperTrailState)
    g.add_node("expand", expand_queries)
    g.add_node("search", search_papers)
    g.add_node("filter", filter_relevance)
    g.add_edge(START, "expand")
    g.add_edge("expand", "search")
    g.add_edge("search", "filter")
    g.add_conditional_edges("filter", _route_after_filter, {"expand": "expand", END: END})
    return g.compile()

def _verify_node(state):
    return {"verification": verify_review(state["review"], state["extracted"])}

def build_analysis_graph():
    g = StateGraph(PaperTrailState)
    g.add_node("extract", extract_papers)
    g.add_node("compare", compare_papers)
    g.add_node("gaps", analyze_gaps)
    g.add_node("write", write_review)
    g.add_node("verify", _verify_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "compare")
    g.add_edge("compare", "gaps")
    g.add_edge("gaps", "write")
    g.add_edge("write", "verify")
    g.add_edge("verify", END)
    return g.compile()