import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
dotenv.load_dotenv()
import json, config

from agents.extraction_agent import extract_one
from agents.comparison_agent import compare_papers
from agents.gap_agent import analyze_gaps
from agents.writer_agent import write_review
from verify import verify_review
from storage.db import cache_get, make_key
import llm
from langchain_groq import ChatGroq

with open("eval/gold_extraction.json", "r", encoding="utf-8") as f:
    gold = json.load(f)[:5]

extracted = []
for i, item in enumerate(gold, 1):
    uid = make_key(item["paper"]["id"])[:12]
    # Check cache for 120b first
    key_120b = make_key("extract", uid, "groq", "openai/gpt-oss-120b")
    rec = cache_get(key_120b)
    if not rec:
        rec = extract_one(item["paper"])
    rec["ref"] = f"P{i}"
    extracted.append(rec)

# Use openai/gpt-oss-20b for writing review to avoid 120b daily token limit
active_model = "openai/gpt-oss-20b"
llm.llm = ChatGroq(model=active_model, api_key=config.GROQ_API_KEY, temperature=0)
llm.provider = "groq"

state = {
    "query": "transformer models for time-series forecasting",
    "extracted": extracted
}

state.update(compare_papers(state))
state.update(analyze_gaps(state))
rev_out = write_review(state)
rev = rev_out["review"]

v = verify_review(rev, extracted)
print(f"Generated review length: {len(rev)} characters")
print(f"Verification score: {v['score']} ({v['supported']}/{v['checked']} supported)")
print(f"Issues: {v['issues']}")

out_data = {
    "provider": "groq",
    "model": active_model,
    "topic": state["query"],
    "review": rev,
    "extracted": extracted,
    "verification": v
}

out_path = Path("eval/provider_comparison/groq_run.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

with open("eval/provider_comparison/groq_review.md", "w", encoding="utf-8") as f:
    f.write(rev)

print(f"Saved live review to {out_path} and eval/provider_comparison/groq_review.md")
