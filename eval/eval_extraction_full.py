"""
Eval 1: Extraction Quality (Gold-Set Benchmark)
Runs eval_extraction.py logic against the 18-paper gold set.
Reports per-paper dataset F1, aggregate F1, and quote verification rate.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.extraction_agent import extract_one
import config

def f1(pred, gold):
    pred_s = {x.lower().strip() for x in pred if x.strip()}
    gold_s = {x.lower().strip() for x in gold if x.strip()}
    if not pred_s or not gold_s:
        return 0.0
    tp = len(pred_s & gold_s)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred_s), tp / len(gold_s)
    return 2 * p * r / (p + r)

def run_extraction_eval(gold_path="eval/gold_extraction.json", out_path="eval/results/extraction_results.json"):
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    results = []
    total_f1, total_verified, total_evidence = 0.0, 0, 0
    start = time.time()

    for idx, item in enumerate(gold):
        paper = item["paper"]
        gold_ds = item["datasets"]
        gold_metric = item.get("main_metric", "")

        try:
            rec = extract_one(paper)
        except Exception as e:
            results.append({
                "title": paper["title"],
                "status": "error",
                "error": str(e),
                "dataset_f1": 0.0,
                "quote_verified": 0
            })
            continue

        pred_ds = rec.get("datasets", [])
        score = f1(pred_ds, gold_ds)
        total_f1 += score

        ev_verified = sum(1 for e in rec.get("evidence", []) if e.get("verified"))
        ev_total = len(rec.get("evidence", []))
        total_verified += ev_verified
        total_evidence += ev_total

        results.append({
            "title": paper["title"][:80],
            "gold_datasets": gold_ds,
            "pred_datasets": pred_ds,
            "dataset_f1": round(score, 3),
            "evidence_verified": ev_verified,
            "evidence_total": ev_total
        })
        print(f"  [{idx+1}/{len(gold)}] F1={score:.2f}  {paper['title'][:60]}")

    elapsed = time.time() - start
    n = len(gold)
    summary = {
        "provider": config.LLM_PROVIDER,
        "model": getattr(config, f"{config.LLM_PROVIDER.upper()}_MODEL", config.LLM_MODEL),
        "papers_evaluated": n,
        "avg_dataset_f1": round(total_f1 / n, 3) if n else 0,
        "quote_verification_rate": round(total_verified / total_evidence, 3) if total_evidence else 0,
        "total_time_s": round(elapsed, 1),
        "per_paper_s": round(elapsed / n, 1) if n else 0,
        "details": results
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Provider: {config.LLM_PROVIDER} ({summary['model']})")
    print(f"Papers evaluated: {n}")
    print(f"Average Dataset F1: {summary['avg_dataset_f1']:.3f}")
    print(f"Quote Verification: {summary['quote_verification_rate']:.1%}")
    print(f"Total time: {elapsed:.1f}s ({summary['per_paper_s']:.1f}s/paper)")
    print(f"Results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    run_extraction_eval()
