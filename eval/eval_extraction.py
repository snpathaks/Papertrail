import json
from agents.extraction_agent import extract_one

def f1(pred, gold):
    pred, gold = {x.lower() for x in pred}, {x.lower() for x in gold}
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    p, r = tp / len(pred), tp / len(gold)
    return 0 if tp == 0 else 2 * p * r / (p + r)

# gold file: [{"paper": {<search-result dict>}, "datasets": [...]}, ...]
gold = json.load(open("eval/gold_extraction.json"))
scores, verified = [], []
for item in gold:
    rec = extract_one(item["paper"])
    scores.append(f1(rec["datasets"], item["datasets"]))
    verified += [e["verified"] for e in rec["evidence"]]
print(f"Dataset extraction F1: {sum(scores)/len(scores):.2f}")
print(f"Quotes verified: {sum(verified)/len(verified):.0%}")