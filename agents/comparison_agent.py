from llm import ask_structured, ask_text
from schemas import ClusterLabel
from tools.cluster import cluster_texts

def _results(metrics):
    return "; ".join(f"{m['name']} {m['value']} ({m['dataset']})" for m in metrics[:2]) or "n/a"

def compare_papers(state):
    papers = state["extracted"]
    for i, p in enumerate(papers, 1):
        p["ref"] = f"P{i}"

    k = 3 if len(papers) >= 6 else 2 if len(papers) >= 3 else 1
    labels = cluster_texts([p["method"] or p["abstract"][:300] for p in papers], k)
    for p, c in zip(papers, labels):
        p["cluster"] = c

    names = {}
    for c in sorted(set(labels)):
        members = [p["method"] for p in papers if p["cluster"] == c]
        out = ask_structured(ClusterLabel,
            "Give a 2-4 word name for the approach family shared by these methods:\n- "
            + "\n- ".join(members))
        names[c] = out.name if out else f"Group {c + 1}"

    rows = [{"ref": p["ref"], "title": p["title"], "year": p["year"],
             "approach": names[p["cluster"]], "method": p["method"],
             "datasets": ", ".join(p["datasets"]), "key results": _results(p["metrics"]),
             "limitations": "; ".join(p["limitations"][:2])} for p in papers]

    # datasets used by 2+ papers = the only places results are directly comparable
    ds_map = {}
    for p in papers:
        for d in {x.lower() for x in p["datasets"]}:
            ds_map.setdefault(d, []).append(p["ref"])
    shared = {d: r for d, r in ds_map.items() if len(r) >= 2}

    table_txt = "\n".join(f"[{r['ref']}] {r['approach']} | {r['method']} | "
                          f"datasets: {r['datasets']} | results: {r['key results']}" for r in rows)
    insights = ask_text(
        "Using ONLY this table, write 3 short bullet-point insights comparing the approaches. "
        "Cite papers like [P2]. Point out when results are on different datasets and cannot be "
        "compared directly.\n\n" + table_txt)
    if not insights:
        insights = "\n".join(f"- [{r['ref']}] utilizes {r['approach']} evaluated on {r['datasets'] or 'various benchmarks'}." for r in rows)

    return {"extracted": papers,
            "comparison": {"table": rows, "approach_names": names,
                           "shared_datasets": shared, "insights": insights}}