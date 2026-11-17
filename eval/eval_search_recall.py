"""
Eval 2: Search Recall
For 3 topics, check how many well-known papers PaperTrail's search discovers.
Uses gold_search_recall.json as ground truth.
"""
import json, sys, time
from pathlib import Path
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_tools import search_openalex, search_arxiv, search_crossref
from agents.search_agent import expand_queries
import config

def normalize(title):
    return title.lower().strip().rstrip(".")

def title_match(candidate, gold_titles, threshold=80):
    """Check if candidate title fuzzy-matches any gold title using token set & partial ratio."""
    c_norm = normalize(candidate)
    for gt in gold_titles:
        g_norm = normalize(gt)
        if fuzz.token_set_ratio(c_norm, g_norm) >= threshold or fuzz.partial_ratio(c_norm, g_norm) >= 85 or g_norm in c_norm or c_norm in g_norm:
            return gt
    return None

def get_expanded_queries(topic):
    """Expand queries purely using live LLM reasoning (NO hardcoded hints)."""
    try:
        state = {"query": topic}
        expansion = expand_queries(state)
        queries = expansion.get("expanded_queries", [])
    except Exception:
        queries = []

    # Fallback only to the original user query if LLM returned nothing
    if not queries:
        queries = [topic]
    return list(dict.fromkeys(queries))

def search_recall_for_topic(topic, gold_titles, max_results=20):
    """Run PaperTrail multi-source search pipeline for a topic and compute recall."""
    queries = get_expanded_queries(topic)
    print(f"  Live queries ({len(queries)}): {queries}")

    # Search all academic sources with all expanded queries
    all_papers = []
    seen_titles = set()
    for q in queries:
        papers_q = search_openalex(q, max_results=max_results)
        papers_q += search_crossref(q, max_results=12)
        papers_q += search_arxiv(q, max_results=min(max_results, 8))
        for paper in papers_q:
            norm = normalize(paper.get("title", ""))
            if norm and norm not in seen_titles:
                seen_titles.add(norm)
                all_papers.append(paper)
        time.sleep(0.4)

    # Step 3: check recall against gold titles
    found = []
    not_found = []
    for gt in gold_titles:
        g_norm = normalize(gt)
        matched = False
        for p in all_papers:
            p_norm = normalize(p.get("title", ""))
            if fuzz.token_set_ratio(p_norm, g_norm) >= 80 or fuzz.partial_ratio(p_norm, g_norm) >= 85 or g_norm in p_norm or p_norm in g_norm:
                found.append(gt)
                matched = True
                break
        if not matched:
            not_found.append(gt)

    recall = len(found) / len(gold_titles) if gold_titles else 0
    return {
        "topic": topic,
        "gold_count": len(gold_titles),
        "found_count": len(found),
        "recall": round(recall, 3),
        "total_candidates": len(all_papers),
        "found_titles": found,
        "missed_titles": not_found
    }

def run_search_recall(gold_path="eval/gold_search_recall.json", out_path="eval/results/search_recall.json"):
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    results = []
    start = time.time()

    for topic, gold_titles in gold.items():
        print(f"\nTopic: {topic}")
        print(f"  Gold papers: {len(gold_titles)}")
        res = search_recall_for_topic(topic, gold_titles)
        print(f"  Found: {res['found_count']}/{res['gold_count']} (recall={res['recall']:.1%})")
        if res["missed_titles"]:
            for mt in res["missed_titles"]:
                print(f"    MISSED: {mt}")
        results.append(res)

    elapsed = time.time() - start
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0

    summary = {
        "avg_recall": round(avg_recall, 3),
        "total_time_s": round(elapsed, 1),
        "topics": results
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Average Search Recall: {avg_recall:.1%}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    run_search_recall()
