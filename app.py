import streamlit as st
import pandas as pd
import config
from graph import build_search_graph, build_analysis_graph
from export import render_review, to_bibtex

st.set_page_config(page_title="PaperTrail", layout="wide")
st.title("PaperTrail: Research Discovery Agent")

with st.sidebar:
    st.caption(f"LLM: {config.LLM_MODEL}")
    year_from = st.number_input("Only papers from year", 2000, 2026, 2018)

query = st.text_input("Research topic", "transformer models for time-series forecasting")

# ---------- Phase 1: discovery ----------
if st.button("1. Find papers"):
    with st.spinner("Searching, deduplicating and filtering..."):
        result = build_search_graph().invoke({"query": query})
    st.session_state["papers"] = result["papers"]
    st.session_state.pop("result", None)

# ---------- Human approval ----------
if "papers" in st.session_state:
    papers = st.session_state["papers"]
    st.subheader("Review the paper list")
    df = pd.DataFrame([{"keep": True, "title": p["title"], "year": p["year"],
                        "source": p["source"], "why relevant": p.get("relevance_reason", "")}
                       for p in papers])
    edited = st.data_editor(df, hide_index=True,
                            disabled=["title", "year", "source", "why relevant"])
    selected = [p for p, k in zip(papers, edited["keep"])
                if k and (p["year"] or 0) >= year_from]
    st.caption(f"{len(selected)} papers selected")

    # ---------- Phase 2: analysis ----------
    if st.button("2. Analyze selected papers") and selected:
        state = {"query": query, "selected": selected}
        with st.status("Running agents...", expanded=True) as status:
            for update in build_analysis_graph().stream(state, stream_mode="updates"):
                for node, out in update.items():
                    st.write(f"Finished: {node}")
                    state.update(out)
            status.update(label="Done", state="complete")
        st.session_state["result"] = state

# ---------- Results ----------
if "result" in st.session_state:
    r = st.session_state["result"]
    tabs = st.tabs(["Papers", "Comparison", "Gaps", "Literature review", "Evidence"])

    with tabs[0]:
        for p in r["extracted"]:
            st.markdown(f"**{p['ref']}. {p['title']}** ({p['year']})")
            st.caption(p["problem"] or p["abstract"][:200])

    with tabs[1]:
        st.dataframe(pd.DataFrame(r["comparison"]["table"]), use_container_width=True)
        st.markdown(r["comparison"]["insights"])
        st.caption("Datasets shared by 2+ papers (directly comparable): " +
                   (", ".join(f"{d} {v}" for d, v in r["comparison"]["shared_datasets"].items()) or "none"))

    with tabs[2]:
        for g in r["gaps"]:
            st.markdown(f"**[{g['type']} | {g['confidence']} confidence]** {g['finding']}")
            st.write(g["explanation"])
            st.info(f"Research question: {g['research_question']}")
            st.caption("Evidence: " + ", ".join(g["evidence_papers"]))

    with tabs[3]:
        v = r["verification"]
        st.metric("Citation support score", v["score"])
        with st.expander(f"{len(v['issues'])} flagged sentences"):
            for i in v["issues"]:
                st.write(f"- {i['problem']}: {i['sentence']}")
        md = render_review(r["review"], r["extracted"])
        st.markdown(md)
        st.download_button("Download review (.md)", md, "review.md")
        st.download_button("Download BibTeX", to_bibtex(r["extracted"]), "refs.bib")

    with tabs[4]:
        pick = st.selectbox("Paper", [f"{p['ref']}. {p['title']}" for p in r["extracted"]])
        paper = next(p for p in r["extracted"] if p["ref"] == pick.split(".")[0])
        for e in paper["evidence"]:
            icon = "verified" if e["verified"] else "NOT verified"
            st.markdown(f"**{e['field']}** ({icon}, page {e['page']})")
            st.code(e["quote"] or "(none)")