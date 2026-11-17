from llm import ask_text

RULES = ("Write in a formal academic style. Use ONLY facts from the evidence provided. "
         "After each claim cite its source like [P1] or [P1][P3]. Never cite a paper that is not "
         "listed. Do not invent numbers.")

def _pack(p):
    res = "; ".join(f"{m['name']} {m['value']} on {m['dataset']}" for m in p["metrics"][:2]) or "n/a"
    return (f"[{p['ref']}] {p['title']} ({p['year']}). Method: {p['method']} "
            f"Datasets: {', '.join(p['datasets']) or 'n/a'}. Results: {res}. "
            f"Limitations: {'; '.join(p['limitations'][:2]) or 'n/a'}")

def write_review(state):
    papers, comp, gaps = state["extracted"], state["comparison"], state["gaps"]
    all_ev = "\n".join(_pack(p) for p in papers)

    intro = ask_text(f"{RULES}\nWrite a ~120-word introduction to '{state['query']}' and the "
                     f"scope of this review ({len(papers)} papers).\nEVIDENCE:\n{all_ev}")
    if not intro:
        raise RuntimeError("LLM failed to generate review introduction: check API connection or quota.")

    themes = []
    for c, name in comp.get("approach_names", {}).items():
        grp = [p for p in papers if str(p.get("cluster")) == str(c)]
        if not grp:
            grp = papers
        ev = "\n".join(_pack(p) for p in grp)
        text = ask_text(
            f"{RULES}\nWrite ~100 words describing this group of papers and what they have in "
            f"common.\nEVIDENCE:\n{ev}")
        if not text:
            raise RuntimeError(f"LLM failed to generate thematic section for approach group '{name}'.")
        themes.append(f"### {name}\n" + text)

    comparison = ask_text(f"{RULES}\nIn ~150 words, summarize how the approaches differ, using "
                          f"these insights.\nINSIGHTS:\n{comp['insights']}\nEVIDENCE:\n{all_ev}")
    if not comparison:
        raise RuntimeError("LLM failed to generate comparative methodology section.")

    gap_ev = "\n".join(f"- ({g['confidence']} confidence) {g['finding']} "
                       f"Papers: {''.join('[' + r + ']' for r in g['evidence_papers'])}. "
                       f"Possible question: {g['research_question']}" for g in gaps)
    gap_text = ask_text(f"{RULES}\nWrite a ~150-word 'Research Gaps' section from these findings. "
                        f"Mention confidence levels.\nFINDINGS:\n{gap_ev}")
    if not gap_text:
        raise RuntimeError("LLM failed to generate research gaps section.")

    conclusion = ask_text(f"{RULES}\nWrite a ~80-word conclusion.\nGAPS:\n{gap_ev}")
    if not conclusion:
        raise RuntimeError("LLM failed to generate conclusion section.")

    review = (f"## Introduction\n{intro}\n\n## Thematic Analysis\n" + "\n\n".join(themes) +
              f"\n\n## Comparison of Methodologies\n{comparison}\n\n"
              f"## Research Gaps\n{gap_text}\n\n## Conclusion\n{conclusion}")
    return {"review": review}
