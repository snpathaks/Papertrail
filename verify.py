import re
from rapidfuzz import fuzz

def _evidence_text(p):
    metrics = " ".join(f"{m['name']} {m['value']} {m['dataset']}" for m in p["metrics"])
    return " ".join([p["abstract"], p["problem"], p["method"], " ".join(p["datasets"]),
                     metrics, " ".join(p["limitations"] + p["future_work"])])

def verify_review(review, papers, threshold=45):
    """Checks: (1) cited ref exists, (2) decimals/percentages appear in the source,
    (3) sentence loosely resembles the cited paper's evidence. Tune `threshold` on your eval set."""
    by_ref = {p["ref"]: _evidence_text(p) for p in papers}
    issues, checked, supported = [], 0, 0
    for s in re.split(r"(?<=[.!?])\s+", review):
        refs = re.findall(r"\[(P\d+)\]", s)
        if not refs:
            continue
        checked += 1
        unknown = [r for r in refs if r not in by_ref]
        if unknown:
            issues.append({"sentence": s, "problem": f"unknown citation {unknown}"})
            continue
        clean = re.sub(r"\[P\d+\]", "", s)
        source = " ".join(by_ref[r] for r in refs)
        bad_nums = [n for n in re.findall(r"\d+\.\d+|\d+%", clean) if n not in source]
        if bad_nums:
            issues.append({"sentence": s, "problem": f"numbers not in source: {bad_nums}"})
            continue
        if fuzz.token_set_ratio(clean, source) < threshold:
            issues.append({"sentence": s, "problem": "weak support in cited paper"})
            continue
        supported += 1
    return {"checked": checked, "supported": supported,
            "score": round(supported / checked, 2) if checked else None, "issues": issues}