from rapidfuzz import fuzz
import config
from llm import ask_structured
from schemas import MethodInfo, ResultsInfo, LimitInfo
from tools.pdf_parser import get_chunks
from storage import vectorstore as vs
from storage.db import cache_get, cache_put, make_key

# (field, schema, retrieval query, instruction): one small task per call
TASKS = [
    ("method", MethodInfo, "proposed method architecture approach we propose",
     "Extract the problem and the proposed method."),
    ("results", ResultsInfo, "experimental results datasets accuracy metrics baselines comparison",
     "Extract the datasets used and the reported metrics."),
    ("limits", LimitInfo, "limitations future work conclusion discussion",
     "Extract limitations and future work stated by the authors."),
]

def _verify_quote(quote, chunks):
    if not quote:
        return False, None
    best, page = 0, None
    for c in chunks:
        score = fuzz.partial_ratio(quote.lower(), c["text"].lower())
        if score > best:
            best, page = score, c["page"]
    return best >= 85, page

def extract_one(paper):
    uid = make_key(paper["id"])[:12]
    key = make_key("extract", uid, config.LLM_MODEL)
    hit = cache_get(key)
    if hit:
        hit.setdefault("pdf_fallback", False)
        hit.setdefault("fallback_reason", None)
        return hit

    chunks, issue = get_chunks(paper, uid)
    vs.add_chunks(uid, chunks)

    record = {"id": paper.get("id", uid), "uid": uid, "title": paper["title"], "authors": paper["authors"],
              "year": paper["year"], "abstract": paper["abstract"],
              "pdf_fallback": bool(issue),
              "fallback_reason": issue["reason"] if issue else None,
              "problem": "", "method": "", "datasets": [], "metrics": [],
              "limitations": [], "future_work": [], "evidence": []}

    for field, schema, query, instruction in TASKS:
        ctx = vs.retrieve(uid, query, k=3)
        text = "\n---\n".join(f"[page {c['page']}, {c['section']}] {c['text']}" for c in ctx)
        out = ask_structured(schema,
            f"{instruction}\nUse ONLY the text below. If something is not stated, leave it empty. "
            f"The quote must be copied exactly from the text.\n\nTEXT:\n{text[:5000]}")
        if out is None:
            continue
        data = out.model_dump()
        quote = data.pop("quote", "")
        ok, page = _verify_quote(quote, ctx)
        record["evidence"].append({"field": field, "quote": quote, "page": page, "verified": ok})
        record.update(data)

    if not record["method"]:
        record["method"] = paper["abstract"][:250]
    cache_put(key, record)
    return record

def extract_papers(state):
    selected = state.get("selected", [])
    total = len(selected)
    cb = state.get("progress_callback")
    extracted = []
    issues = []

    for idx, p in enumerate(selected, 1):
        if cb:
            cb(f"Extracting {idx}/{total}")
        rec = extract_one(p)
        extracted.append(rec)
        if rec.get("pdf_fallback"):
            issues.append({
                "ref": rec.get("ref", f"P{idx}"),
                "title": rec.get("title", ""),
                "reason": rec.get("fallback_reason", "Abstract-only fallback")
            })

    return {"extracted": extracted, "issues": issues}