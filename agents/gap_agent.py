import pandas as pd
from collections import Counter
from llm import ask_structured
from schemas import GapExplanation
from tools.cluster import cluster_texts

ASPECTS = {
    "robustness": ["robust", "adversarial", "noise"],
    "efficiency": ["efficien", "latency", "memory", "computational cost", "inference time"],
    "fairness": ["fairness", "bias"],
    "interpretability": ["interpretab", "explainab"],
    "generalization": ["generaliz", "out-of-distribution", "cross-domain"],
}
PRIORITY = {"dataset": 0, "stated_limitation": 1, "evaluation": 2, "coverage": 3}

def _text(p):
    return " ".join([p["abstract"], p["method"], " ".join(p["limitations"]),
                     " ".join(p["future_work"])]).lower()

def find_candidates(papers, names):
    n, cands = len(papers), []
    ds = Counter(d.lower() for p in papers for d in set(p["datasets"]))

    # 1. dataset concentration
    if ds:
        top, cnt = ds.most_common(1)[0]
        if cnt >= 2 and cnt / n >= 0.5:
            cands.append({"type": "dataset",
                "finding": f"{cnt} of {n} papers use '{top}'; few other datasets appear.",
                "papers": [p["ref"] for p in papers if top in [d.lower() for d in p["datasets"]]]})

    # 2. coverage matrix: approach x dataset, find empty cells
    rows = [{"approach": names[p["cluster"]], "dataset": d.lower()}
            for p in papers for d in set(p["datasets"])]
    if rows:
        top_ds = [d for d, _ in ds.most_common(4)]
        cov = pd.crosstab(pd.DataFrame(rows)["approach"], pd.DataFrame(rows)["dataset"]) \
                .reindex(columns=top_ds, fill_value=0)
        added = 0
        for approach in cov.index:
            for d in cov.columns:
                if cov.loc[approach, d] == 0 and added < 3:
                    added += 1
                    cands.append({"type": "coverage",
                        "finding": f"Approach '{approach}' was never evaluated on '{d}'.",
                        "papers": [p["ref"] for p in papers if names[p["cluster"]] == approach]})

    # 3. rarely studied evaluation aspects
    for aspect, kws in ASPECTS.items():
        hits = [p["ref"] for p in papers if any(k in _text(p) for k in kws)]
        if len(hits) <= max(1, n // 5):
            cands.append({"type": "evaluation",
                "finding": f"Only {len(hits)} of {n} papers discuss {aspect}.", "papers": hits})

    # 4. repeated stated limitations / future work
    items = [(p["ref"], s) for p in papers for s in p["limitations"] + p["future_work"]]
    if len(items) >= 4:
        labels = cluster_texts([s for _, s in items], min(4, len(items) // 2))
        for c in set(labels):
            grp = [(r, s) for (r, s), l in zip(items, labels) if l == c]
            refs = sorted({r for r, _ in grp})
            if len(refs) >= 2:
                cands.append({"type": "stated_limitation",
                    "finding": "Recurring limitation/future work: " + " | ".join(s for _, s in grp[:3]),
                    "papers": refs})
    return sorted(cands, key=lambda c: PRIORITY[c["type"]])[:8]

def analyze_gaps(state):
    papers = state["extracted"]
    names = state["comparison"]["approach_names"]
    gaps = []
    for c in find_candidates(papers, names):
        out = ask_structured(GapExplanation,
            f"Finding computed from {len(papers)} papers: {c['finding']}\n"
            f"Related papers: {', '.join(c['papers']) or 'none'}\n"
            "Explain in 2 sentences why this is a research gap and propose ONE concrete research "
            "question. Do not invent facts beyond the finding.")
        ratio = len(c["papers"]) / max(1, len(papers))
        conf = "high" if ratio >= 0.5 else "medium" if ratio >= 0.3 else "low"
        if c["type"] in ("evaluation", "coverage") and conf == "high":
            conf = "medium"   # absence of evidence is weaker evidence
        exp = out.explanation if (out and out.explanation) else f"The identified pattern indicates an underexplored dimension regarding {c['finding']}."
        rq = out.research_question if (out and out.research_question) else f"How can future models effectively address {c['finding']}?"
        gaps.append({"type": c["type"], "finding": c["finding"],
                     "explanation": exp,
                     "research_question": rq,
                     "evidence_papers": c["papers"], "confidence": conf})
    return {"gaps": gaps}