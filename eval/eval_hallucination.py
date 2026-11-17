"""
Eval 3: Hallucination Baseline
Ask a plain LLM (no retrieval pipeline) for 10 references on a topic,
then verify each title exists on OpenAlex via fuzzy match.
Compare: PaperTrail is 100% by design (only cites retrieved papers).
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from rapidfuzz import fuzz
from llm import ask_text
import config

TOPICS = [
    "transformer models for time-series forecasting",
    "graph neural networks for drug discovery",
    "retrieval augmented generation hallucination mitigation"
]

def exists_on_openalex(title, email=None):
    """User-specified title existence checker on OpenAlex with Crossref fallback."""
    email = email or config.CONTACT_EMAIL
    try:
        r = requests.get("https://api.openalex.org/works",
                         params={"search": title, "per-page": 1, "mailto": email},
                         timeout=15)
        if r.status_code == 200:
            res = r.json().get("results", [])
            if res:
                best_title = res[0].get("title") or ""
                score = fuzz.ratio(title.lower(), best_title.lower())
                if score > 85:
                    return True, best_title
    except Exception:
        pass

    # Fallback to Crossref (indexes 150M+ papers without OpenAlex daily IP rate limits)
    try:
        headers = {"User-Agent": f"PaperTrail/1.0 (mailto:{email})"}
        r = requests.get("https://api.crossref.org/works",
                         params={"query.bibliographic": title, "rows": 1},
                         headers=headers, timeout=12)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                best_title = (items[0].get("title") or [""])[0]
                score = fuzz.ratio(title.lower(), best_title.lower())
                if score > 85:
                    return True, best_title
    except Exception:
        pass

    return False, None

def get_plain_llm_references(topic, n=10):
    """Ask the LLM directly for n paper references on a topic (no RAG pipeline)."""
    prompt = (
        f"List exactly {n} real academic paper titles about '{topic}'. "
        f"Output ONLY the titles, one per line, numbered 1-{n}. "
        f"Do not include authors, years, or any other text. "
        f"Only include papers that actually exist."
    )
    try:
        response = ask_text(prompt)
    except Exception:
        response = ""

    titles = []
    if response:
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            import re
            cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip().strip('"').strip("'")
            if len(cleaned) > 10:
                titles.append(cleaned)

    if len(titles) < 5:
        raise RuntimeError(f"Live LLM provider returned insufficient titles ({len(titles)}) for topic '{topic}'. Check model connectivity or quota.")

    return titles[:n]

def run_hallucination_eval(out_path="eval/results/hallucination_baseline.json"):
    results = []
    start = time.time()

    for topic in TOPICS:
        print(f"\nTopic: {topic}")
        titles = get_plain_llm_references(topic)
        print(f"  LLM returned {len(titles)} titles")

        verified = []
        for title in titles:
            exists, matched_title = exists_on_openalex(title)
            verified.append({
                "llm_title": title,
                "exists_on_openalex": exists,
                "matched_title": matched_title
            })
            status = "✓" if exists else "✗"
            print(f"  {status} {title[:70]}")
            time.sleep(0.3)  # respect OpenAlex rate limits

        real_count = sum(1 for v in verified if v["exists_on_openalex"])
        total = len(verified)
        accuracy = real_count / total if total else 0

        results.append({
            "topic": topic,
            "titles_generated": total,
            "titles_verified": real_count,
            "accuracy": round(accuracy, 3),
            "details": verified
        })

    elapsed = time.time() - start
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0

    # PaperTrail's rate is 100% by design
    summary = {
        "provider": config.LLM_PROVIDER,
        "model": getattr(config, f"{config.LLM_PROVIDER.upper()}_MODEL", config.LLM_MODEL),
        "plain_llm_avg_accuracy": round(avg_accuracy, 3),
        "papertrail_accuracy": 1.0,
        "note": "PaperTrail only cites papers retrieved from academic APIs, so its citation accuracy is 100% by design.",
        "total_time_s": round(elapsed, 1),
        "topics": results
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Provider: {config.LLM_PROVIDER}")
    print(f"Plain LLM citation accuracy: {avg_accuracy:.1%}")
    print(f"PaperTrail citation accuracy: 100.0% (by design)")
    print(f"Results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    run_hallucination_eval()
