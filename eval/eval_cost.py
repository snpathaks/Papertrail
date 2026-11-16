"""
Eval 6: Cost & Timing
Records wall-clock time per review for each model/provider.
Uses existing run data and can measure new runs.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from graph import build_search_graph, build_analysis_graph

def time_full_run(topic, out_path="eval/results/cost_timing.json"):
    """Run the full pipeline and record timing for each phase."""
    timing = {
        "provider": config.LLM_PROVIDER,
        "model": getattr(config, f"{config.LLM_PROVIDER.upper()}_MODEL", config.LLM_MODEL),
        "vector_backend": config.VECTOR_BACKEND,
        "topic": topic,
        "phases": {}
    }

    # Phase 1: Search
    t0 = time.time()
    try:
        search_result = build_search_graph().invoke({"query": topic})
        papers = search_result.get("papers", [])
    except Exception as e:
        papers = []
        timing["phases"]["search"] = {"time_s": round(time.time() - t0, 1), "error": str(e)}
    timing["phases"]["search"] = {
        "time_s": round(time.time() - t0, 1),
        "papers_found": len(papers)
    }
    print(f"  Search: {timing['phases']['search']['time_s']}s ({len(papers)} papers)")

    if not papers:
        timing["total_time_s"] = round(time.time() - t0, 1)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(timing, f, indent=2)
        return timing

    # Phase 2: Analysis (take first 5 papers)
    selected = papers[:5]
    state = {"query": topic, "selected": selected}

    t1 = time.time()
    try:
        for update in build_analysis_graph().stream(state, stream_mode="updates"):
            for node, out in update.items():
                node_time = time.time() - t1
                timing["phases"][node] = {"time_s": round(node_time, 1)}
                print(f"  {node}: {node_time:.1f}s")
                state.update(out)
                t1 = time.time()
    except Exception as e:
        timing["phases"]["analysis_error"] = {"error": str(e)}

    total = time.time() - t0
    timing["total_time_s"] = round(total, 1)
    timing["papers_analyzed"] = len(selected)
    timing["time_per_paper_s"] = round(total / len(selected), 1) if selected else 0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Provider: {config.LLM_PROVIDER} ({timing['model']})")
    print(f"Total time: {total:.1f}s")
    print(f"Per paper: {timing['time_per_paper_s']:.1f}s")
    print(f"Results saved to: {out_path}")
    return timing

def collect_timing_from_runs(run_dir="eval/provider_comparison", out_path="eval/results/cost_summary.json"):
    """Collect timing data from existing provider comparison runs."""
    run_dir = Path(run_dir)
    summary = []

    for run_file in run_dir.glob("*_run.json"):
        with open(run_file) as f:
            run = json.load(f)
        summary.append({
            "provider": run.get("provider", "unknown"),
            "papers_analyzed": len(run.get("selected_papers", [])),
            "verification_score": run.get("verification", {}).get("score"),
            "gaps_found": len(run.get("gaps", []))
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="transformer models for time-series forecasting")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.collect_only:
        collect_timing_from_runs()
    else:
        time_full_run(args.topic)
