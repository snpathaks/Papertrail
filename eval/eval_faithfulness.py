"""
Eval 4: Citation Faithfulness
Reports verify_review scores from existing runs, then provides a structured
manual verification framework for 20 sentences.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify import verify_review

def extract_cited_sentences(review):
    """Extract all sentences with citations [Pn] from a review."""
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", review):
        refs = re.findall(r"\[(P\d+)\]", s)
        if refs:
            sentences.append({"sentence": s.strip(), "citations": refs})
    return sentences

def run_faithfulness_eval(run_paths=None, out_path="eval/results/citation_faithfulness.json"):
    if run_paths is None:
        run_paths = [
            "eval/provider_comparison/ollama_run.json",
            "eval/provider_comparison/groq_run.json"
        ]

    results = []
    manual_check_sentences = []

    for rp in run_paths:
        if not Path(rp).exists():
            print(f"  Skipping {rp} (not found)")
            continue

        with open(rp, "r", encoding="utf-8") as f:
            run = json.load(f)

        provider = run.get("provider", "unknown")
        review_text = run.get("review", "")
        verification = run.get("verification", {})

        # Extract all cited sentences for manual audit
        cited = extract_cited_sentences(review_text)

        # Re-run verification if we have the extraction data
        papers = []
        for i, ct in enumerate(run.get("comparison_table", [])):
            papers.append({
                "ref": ct.get("ref", f"P{i+1}"),
                "title": ct.get("title", ""),
                "abstract": ct.get("method", ""),
                "problem": "",
                "method": ct.get("method", ""),
                "datasets": [d.strip() for d in ct.get("datasets", "").split(",") if d.strip()],
                "metrics": [],
                "limitations": [l.strip() for l in ct.get("limitations", "").split(";") if l.strip()],
                "future_work": []
            })

        if review_text and papers:
            fresh_verification = verify_review(review_text, papers)
        else:
            fresh_verification = verification

        # Select up to 20 sentences for manual check template
        for s in cited[:20]:
            manual_check_sentences.append({
                "provider": provider,
                "sentence": s["sentence"],
                "citations": s["citations"],
                "automated_verdict": "supported",  # placeholder
                "manual_verdict": "TODO",
                "manual_notes": ""
            })

        results.append({
            "provider": provider,
            "automated_score": verification.get("score", None),
            "fresh_score": fresh_verification.get("score", None),
            "checked": fresh_verification.get("checked", 0),
            "supported": fresh_verification.get("supported", 0),
            "issues_count": len(fresh_verification.get("issues", [])),
            "issues": fresh_verification.get("issues", []),
            "total_cited_sentences": len(cited)
        })

    # Manual verification template (20 sentences)
    # Pre-fill with automated results so user just marks agree/disagree
    manual_audit = {
        "instructions": (
            "For each sentence below, manually verify whether the citation "
            "supports the claim. Mark manual_verdict as 'correct' if the "
            "automated checker got it right, 'wrong' if the checker missed "
            "an error or flagged a correct sentence. Add notes."
        ),
        "sentences": manual_check_sentences[:20]
    }

    # Compute checker accuracy from any completed manual audits
    completed = [s for s in manual_audit["sentences"] if s["manual_verdict"] != "TODO"]
    checker_accuracy = None
    if completed:
        correct = sum(1 for s in completed if s["manual_verdict"] == "correct")
        checker_accuracy = round(correct / len(completed), 3)

    summary = {
        "automated_results": results,
        "manual_audit": manual_audit,
        "checker_accuracy": checker_accuracy,
        "note": "checker_accuracy is computed from manual_audit sentences where manual_verdict != TODO"
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Citation Faithfulness Results:")
    for r in results:
        print(f"  {r['provider']}: score={r['automated_score']}, "
              f"checked={r['checked']}, supported={r['supported']}, "
              f"issues={r['issues_count']}")
    print(f"  Manual audit template: {len(manual_audit['sentences'])} sentences")
    if checker_accuracy is not None:
        print(f"  Checker accuracy (manual): {checker_accuracy:.1%}")
    print(f"Results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    run_faithfulness_eval()
