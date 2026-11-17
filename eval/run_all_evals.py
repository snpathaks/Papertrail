"""
Master Evaluation Runner
Runs all evaluation scripts and produces a unified results summary.
Usage: python eval/run_all_evals.py [--skip-network] [--skip-extraction]
"""
import json, sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Run PaperTrail evaluation suite")
    parser.add_argument("--skip-network", action="store_true", help="Skip evals that need network")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction eval (needs LLM)")
    parser.add_argument("--skip-hallucination", action="store_true", help="Skip hallucination eval (needs network+LLM)")
    parser.add_argument("--skip-human", action="store_true", help="Skip human eval form generation")
    args = parser.parse_args()

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # 1. Extraction quality
    if not args.skip_extraction:
        print("\n" + "="*60)
        print("EVAL 1: Extraction Quality (Gold-Set Benchmark)")
        print("="*60)
        try:
            from eval_extraction_full import run_extraction_eval
            all_results["extraction"] = run_extraction_eval()
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            all_results["extraction"] = {"status": "skipped", "error": str(e)}
    elif (results_dir / "extraction_results.json").exists():
        with open(results_dir / "extraction_results.json", "r", encoding="utf-8") as f:
            all_results["extraction"] = json.load(f)
    else:
        all_results["extraction"] = {"status": "skipped"}

    # 2. Search recall
    if not args.skip_network:
        print("\n" + "="*60)
        print("EVAL 2: Search Recall")
        print("="*60)
        try:
            from eval_search_recall import run_search_recall
            all_results["search_recall"] = run_search_recall()
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            all_results["search_recall"] = {"status": "skipped", "error": str(e)}
    elif (results_dir / "search_recall.json").exists():
        with open(results_dir / "search_recall.json", "r", encoding="utf-8") as f:
            all_results["search_recall"] = json.load(f)
    else:
        all_results["search_recall"] = {"status": "skipped"}

    # 3. Hallucination baseline
    if not args.skip_hallucination and not args.skip_network:
        print("\n" + "="*60)
        print("EVAL 3: Hallucination Baseline")
        print("="*60)
        try:
            from eval_hallucination import run_hallucination_eval
            all_results["hallucination"] = run_hallucination_eval()
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            all_results["hallucination"] = {"status": "skipped", "error": str(e)}
    elif (results_dir / "hallucination_baseline.json").exists():
        with open(results_dir / "hallucination_baseline.json", "r", encoding="utf-8") as f:
            all_results["hallucination"] = json.load(f)
    else:
        all_results["hallucination"] = {"status": "skipped"}

    # 4. Citation faithfulness
    print("\n" + "="*60)
    print("EVAL 4: Citation Faithfulness")
    print("="*60)
    try:
        from eval_faithfulness import run_faithfulness_eval
        all_results["faithfulness"] = run_faithfulness_eval()
    except Exception as e:
        print(f"  SKIPPED (error: {e})")
        all_results["faithfulness"] = {"status": "skipped", "error": str(e)}

    # 5. Human eval forms
    if not args.skip_human:
        print("\n" + "="*60)
        print("EVAL 5: Human Rating (Form Generation)")
        print("="*60)
        try:
            from eval_human import create_blinded_forms, aggregate_human_scores
            create_blinded_forms("eval/provider_comparison/groq_run.json")
            all_results["human_eval"] = aggregate_human_scores()
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            all_results["human_eval"] = {"status": "skipped", "error": str(e)}
    else:
        all_results["human_eval"] = {"status": "skipped"}

    # 6. Cost/timing
    print("\n" + "="*60)
    print("EVAL 6: Cost & Timing")
    print("="*60)
    try:
        from eval_cost import collect_timing_from_runs
        all_results["cost"] = collect_timing_from_runs()
        if (results_dir / "cost_timing.json").exists():
            with open(results_dir / "cost_timing.json", "r", encoding="utf-8") as f:
                all_results["timing"] = json.load(f)
    except Exception as e:
        print(f"  SKIPPED (error: {e})")
        all_results["cost"] = {"status": "skipped", "error": str(e)}

    # Save unified results
    with open(results_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All evaluation results saved to: eval/results/all_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
