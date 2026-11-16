import json
from datetime import datetime
import streamlit as st
import pandas as pd
import config
from graph import build_search_graph, build_analysis_graph
from export import render_review, to_bibtex

st.set_page_config(page_title="PaperTrail", layout="wide")
st.title("PaperTrail: Research Discovery Agent")

with st.sidebar:
    st.subheader("Configuration")
    st.caption(f"Provider: **{config.LLM_PROVIDER.upper()}**")
    active_model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.LLM_MODEL
    st.caption(f"Model: `{active_model}`")
    st.caption(f"Vector Store: `{config.VECTOR_BACKEND}`")
    year_from = st.number_input("Only papers from year", 2000, 2026, 2018)

    st.divider()
    st.subheader("Previous Runs")
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_files = sorted(list(config.RUNS_DIR.glob("*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_files:
        run_names = ["-- Select run to load --"] + [f.name for f in run_files]
        chosen_run = st.selectbox("Load previous run", run_names, key="previous_run_selector")
        if chosen_run != "-- Select run to load --":
            if st.button("Load Run", key="btn_load_run"):
                target_path = config.RUNS_DIR / chosen_run
                with open(target_path, "r", encoding="utf-8") as f:
                    loaded_run = json.load(f)
                st.session_state["result"] = loaded_run
                st.session_state["query"] = loaded_run.get("query", "")
                st.session_state["papers"] = loaded_run.get("selected", loaded_run.get("papers", []))
                st.session_state["loaded_run_name"] = chosen_run
                st.success(f"Loaded {chosen_run}")
                st.rerun()
    else:
        st.caption("No previous runs saved yet.")

if "loaded_run_name" in st.session_state:
    st.info(f"📂 Viewing loaded run: **{st.session_state['loaded_run_name']}**")

default_query = st.session_state.get("query", "transformer models for time-series forecasting")
query = st.text_input("Research topic", default_query)

# ---------- Phase 1: discovery ----------
if st.button("1. Find papers"):
    with st.spinner("Searching, deduplicating and filtering..."):
        result = build_search_graph().invoke({"query": query})
    st.session_state["papers"] = result["papers"]
    st.session_state.pop("result", None)
    st.session_state.pop("loaded_run_name", None)

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
        progress_slot = st.empty()
        def on_progress(msg):
            status.update(label=f"Running agents... ({msg})")
            progress_slot.info(f"⏳ {msg}")

        state = {"query": query, "selected": selected, "progress_callback": on_progress}
        with st.status("Running agents...", expanded=True) as status:
            for update in build_analysis_graph().stream(state, stream_mode="updates"):
                for node, out in update.items():
                    st.write(f"Finished: {node}")
                    state.update(out)
            progress_slot.empty()
            status.update(label="Done", state="complete")
        
        # Save run to runs/<timestamp>.json
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_file = config.RUNS_DIR / f"{ts}.json"
        save_state = {k: v for k, v in state.items() if k != "progress_callback"}
        try:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(save_state, f, indent=2, default=str)
            st.session_state["loaded_run_name"] = f"{ts}.json"
            st.toast(f"Run saved to runs/{ts}.json")
        except Exception as e:
            st.warning(f"Could not auto-save run to runs/{ts}.json: {e}")

        st.session_state["result"] = state

# ---------- Results ----------
if "result" in st.session_state:
    r = st.session_state["result"]

    # Issues Panel (listing papers that fell back to abstract-only because PDF failed)
    issues = r.get("issues", [])
    if not issues:
        issues = [
            {
                "ref": p.get("ref", "P?"),
                "title": p.get("title", "Unknown"),
                "reason": p.get("fallback_reason", "PDF unavailable or parsing failed; used abstract only.")
            }
            for p in r.get("extracted", []) if p.get("pdf_fallback")
        ]

    if issues:
        with st.expander(f"⚠️ Issues Panel ({len(issues)} paper(s) fell back to abstract-only)", expanded=True):
            st.warning("The following papers could not be downloaded/parsed as full PDFs and fell back to abstract-only extraction:")
            for iss in issues:
                ref_label = f"**{iss.get('ref', '')}** - " if iss.get("ref") else ""
                st.markdown(f"- {ref_label}**{iss.get('title', 'Unknown')}**: `{iss.get('reason', 'PDF failed')}`")

    tabs = st.tabs(["Papers", "Comparison", "Gaps", "Literature review", "Evidence", "Issues"])

    with tabs[0]:
        for p in r["extracted"]:
            fallback_badge = " *(Abstract only)*" if p.get("pdf_fallback") else ""
            st.markdown(f"**{p['ref']}. {p['title']}** ({p['year']}){fallback_badge}")
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

    with tabs[5]:
        st.subheader("Extraction Issues & PDF Fallbacks")
        if issues:
            st.dataframe(pd.DataFrame(issues), use_container_width=True)
        else:
            st.success("All papers were successfully parsed from full PDF text (no fallbacks).")