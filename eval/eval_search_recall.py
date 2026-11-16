"""
Eval 2: Search Recall
For 3 topics, check how many well-known papers PaperTrail's search discovers.
Uses gold_search_recall.json as ground truth.
"""
import json, sys, time
from pathlib import Path
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.search_tools import search_openalex, search_arxiv
from agents.search_agent import expand_queries
import config

def normalize(title):
    return title.lower().strip().rstrip(".")

def title_match(candidate, gold_titles, threshold=80):
    """Check if candidate title fuzzy-matches any gold title."""
    for gt in gold_titles:
        if fuzz.ratio(normalize(candidate), normalize(gt)) > threshold:
            return gt
    return None

def search_recall_for_topic(topic, gold_titles, max_results=10):
    """Run PaperTrail search pipeline for a topic and compute recall."""
    # Step 1: expand queries (use the LLM query expansion)
    try:
        state = {"query": topic}
        expansion = expand_queries(state)
        queries = expansion.get("expanded_queries", [topic])
    except Exception:
        queries = [topic]

    # Step 2: search both APIs with all expanded queries
    all_papers = []
    seen_titles = set()
    for q in queries[:5]:
        for paper in search_openalex(q, max_results=max_results):
            norm = normalize(paper["title"])
            if norm not in seen_titles:
                seen_titles.add(norm)
                all_papers.append(paper)
        for paper in search_arxiv(q, max_results=max_results):
            norm = normalize(paper["title"])
            if norm not in seen_titles:
                seen_titles.add(norm)
                all_papers.append(paper)

    # Step 3: check recall
    found = []
    not_found = []
    for gt in gold_titles:
        matched = False
        for p in all_papers:
            if fuzz.ratio(normalize(p["title"]), normalize(gt)) > 80:
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
