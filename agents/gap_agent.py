import re
from collections import Counter, defaultdict
import pandas as pd
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
PRIORITY = {"contradiction": 0, "dataset": 1, "stated_limitation": 2, "evaluation": 3, "coverage": 4}

PERCENT_METRICS = ("accuracy", "f1", "precision", "recall", "auc")

def _num(name, value):
    m = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not m:
        return None
    x = float(m.group())
    if any(k in name.lower() for k in PERCENT_METRICS) and x <= 1:
        x *= 100          # 0.942 -> 94.2
    return x

def find_contradictions(papers, min_diff=10):
    groups = defaultdict(list)   # (dataset, metric) -> [(ref, value)]
    for p in papers:
        for m in p.get("metrics", []):
            m_name = m["name"] if isinstance(m, dict) else getattr(m, "name", "")
            m_val = m["value"] if isinstance(m, dict) else getattr(m, "value", "")
            m_ds = m["dataset"] if isinstance(m, dict) else getattr(m, "dataset", "")
            v = _num(m_name, m_val)
            if v is not None and m_ds:
                groups[(m_ds.lower(), m_name.lower())].append((p.get("ref", "P?"), v))
    out = []
    for (ds, metric), vals in groups.items():
        if len({r for r, _ in vals}) >= 2:
            lo, hi = min(vals, key=lambda x: x[1]), max(vals, key=lambda x: x[1])
            if hi[1] - lo[1] >= min_diff:
                out.append({"type": "contradiction",
                    "finding": f"On '{ds}', {metric} ranges from {lo[1]:.1f} ({lo[0]}) to "
                               f"{hi[1]:.1f} ({hi[0]}); differences may come from different "
                               f"settings or baselines.",
                    "papers": [lo[0], hi[0]]})
    return out

def _text(p):
    return " ".join([p["abstract"], p["method"], " ".join(p["limitations"]),
                     " ".join(p["future_work"])]).lower()

def find_candidates(papers, names):
    n, cands = len(papers), []
    cands += find_contradictions(papers)
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
        if c["type"] == "contradiction":
            conf = "high"
        exp = out.explanation if (out and out.explanation) else f"The identified pattern indicates an underexplored dimension regarding {c['finding']}."
        rq = out.research_question if (out and out.research_question) else f"How can future models effectively address {c['finding']}?"
        gaps.append({"type": c["type"], "finding": c["finding"],
                     "explanation": exp,
                     "research_question": rq,
                     "evidence_papers": c["papers"], "confidence": conf})
    return {"gaps": gaps}