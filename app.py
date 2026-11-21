import os
import streamlit as st

# Environment / secrets bridge (must remain at top)
try:
    for k in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
              "CONTACT_EMAIL", "VECTOR_BACKEND"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

import json
from datetime import datetime
import pandas as pd
import config
from export import render_review, to_bibtex
from ui import splash, pipeline_tracker, stat_cards, badge, paper_card, inject_theme_css

# Page Configuration
st.set_page_config(
    page_title="PaperTrail: Research Discovery Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Play 10-sec animated splash screen
splash()

# Inject custom research-lab styling
inject_theme_css()

# --- Sidebar: Configuration & Run Management ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Config")
    st.caption(f"Provider: **{config.LLM_PROVIDER.upper()}**")
    active_model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.LLM_MODEL
    st.caption(f"Model: `{active_model}`")
    st.caption(f"Vector Store: `{config.VECTOR_BACKEND}`")
    year_from = st.number_input("Only papers from year", 2000, 2026, 2018)

    st.divider()
    st.markdown("### 🗃️ Previous Runs")
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_files = sorted(list(config.RUNS_DIR.glob("*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_files:
        run_names = ["-- Select run to load --"] + [f.name for f in run_files]
        chosen_run = st.selectbox("Load previous run", run_names, key="previous_run_selector")
        if chosen_run != "-- Select run to load --":
            if st.button("Load Run", key="btn_load_run", use_container_width=True):
                target_path = config.RUNS_DIR / chosen_run
                with open(target_path, "r", encoding="utf-8") as f:
                    loaded_run = json.load(f)
                
                # Normalize schema for backwards compatibility
                if "comparison" not in loaded_run and "comparison_table" in loaded_run:
                    loaded_run["comparison"] = {
                        "table": loaded_run.get("comparison_table", []),
                        "insights": loaded_run.get("insights", "Comparative analysis loaded from archive."),
                        "shared_datasets": loaded_run.get("shared_datasets", {})
                    }
                if "extracted" not in loaded_run and "selected_papers" in loaded_run:
                    loaded_run["extracted"] = [
                        {"ref": f"P{i+1}", "title": p["title"], "year": p.get("year", 2024),
                         "problem": "", "method": "", "abstract": "", "evidence": []}
                        for i, p in enumerate(loaded_run["selected_papers"])
                    ]
                if "verification" not in loaded_run:
                    loaded_run["verification"] = {"score": "N/A", "issues": []}

                st.session_state["result"] = loaded_run
                st.session_state["query"] = loaded_run.get("query", loaded_run.get("topic", ""))
                st.session_state["papers"] = loaded_run.get("selected", loaded_run.get("papers", loaded_run.get("selected_papers", [])))
                st.session_state["loaded_run_name"] = chosen_run
                st.success(f"Loaded {chosen_run}")
                st.rerun()
    else:
        st.caption("No previous runs saved yet.")

    st.divider()
    if st.button("▶️ Replay Intro Animation (10s)", use_container_width=True):
        st.session_state["splash_done"] = False
        st.rerun()

# --- Hero Header ---
st.markdown('<div class="hero-title">Paper<span>Trail</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Multi-Agent Literature Review & Research Discovery Engine</div>', unsafe_allow_html=True)

if "loaded_run_name" in st.session_state:
    st.info(f"📂 Viewing loaded run: **{st.session_state['loaded_run_name']}**")

# Example Topic Chips
st.markdown("**Explore example topics:**")
chip_cols = st.columns(3)
examples = [
    ("⚡ Transformer Time-Series", "transformer models for time-series forecasting"),
    ("🧬 GNN Drug Discovery", "graph neural networks for molecular drug discovery"),
    ("🛡️ RAG Hallucination", "retrieval augmented generation hallucination mitigation")
]
for col, (label, ex_query) in zip(chip_cols, examples):
    if col.button(label, use_container_width=True):
        st.session_state["query"] = ex_query
        st.session_state.pop("papers", None)
        st.session_state.pop("result", None)
        st.session_state.pop("loaded_run_name", None)
        st.rerun()

current_query = st.session_state.get("query", "transformer models for time-series forecasting")
query = st.text_input("Research Topic or Hypothesis", value=current_query, placeholder="Enter a research topic (e.g. state space models vs transformers)...")

# --- Phase 1: Search & Discovery ---
col_search, _ = st.columns([2, 5])
if col_search.button("🔎 1. Find & Curate Papers", type="primary", use_container_width=True):
    st.session_state["query"] = query
    tracker_slot = st.empty()
    pipeline_tracker([], slot=tracker_slot)
    with st.spinner("Search Agent scanning OpenAlex & arXiv, filtering by relevance..."):
        from graph import build_search_graph
        result = build_search_graph().invoke({"query": query})
    st.session_state["papers"] = result["papers"]
    st.session_state.pop("result", None)
    st.session_state.pop("loaded_run_name", None)
    tracker_slot.empty()
    st.rerun()

# --- Human Approval & Curation Stage ---
if "papers" in st.session_state and "result" not in st.session_state:
    papers = st.session_state["papers"]
    st.divider()
    st.markdown("### 📋 Step 1: Curate Retrieved Candidates")
    st.caption("Review candidate papers found by Search Agent. Uncheck any paper you wish to drop from in-depth full-text extraction.")
    
    # Active pipeline tracker showing Search completed
    pipeline_tracker(["Search"])

    selected_flags = {}
    for i, p in enumerate(papers):
        p_year = p.get("year") or 0
        is_year_ok = p_year >= year_from
        card_key = f"keep_paper_{i}_{p.get('id', p.get('title', ''))[:15]}"
        keep = paper_card(p, card_key)
        selected_flags[i] = keep and is_year_ok

    selected = [p for i, p in enumerate(papers) if selected_flags.get(i, True)]
    
    st.markdown(f"**{len(selected)} of {len(papers)} papers selected** for deep extraction and comparative synthesis.")
    
    # --- Phase 2: Multi-Agent Analysis ---
    if st.button("🚀 2. Analyze Selected Papers", type="primary", disabled=len(selected) == 0):
        done_stages = ["Search"]
        tracker_placeholder = st.empty()
        pipeline_tracker(done_stages, slot=tracker_placeholder)
        
        status_box = st.status("Initializing research team...", expanded=True)
        
        def on_progress(msg):
            status_box.update(label=f"Research Team active: {msg}")
            status_box.write(f"⏳ {msg}")

        state = {"query": query, "selected": selected, "progress_callback": on_progress}
        
        # Stage mapping for live tracker updates
        node_to_stage = {
            "extract": "Extract",
            "compare": "Compare",
            "gaps": "Gaps",
            "write": "Write",
            "verify": "Verify"
        }

        from graph import build_analysis_graph
        for update in build_analysis_graph().stream(state, stream_mode="updates"):
            for node, out in update.items():
                stage_name = node_to_stage.get(node)
                if stage_name and stage_name not in done_stages:
                    done_stages.append(stage_name)
                    pipeline_tracker(done_stages, slot=tracker_placeholder)
                status_box.write(f"✅ Finished stage: **{stage_name or node}**")
                state.update(out)
        
        status_box.update(label="Literature review and verification complete!", state="complete")
        
        # Auto-save run
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_file = config.RUNS_DIR / f"{ts}.json"
        save_state = {k: v for k, v in state.items() if k != "progress_callback"}
        try:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(save_state, f, indent=2, default=str)
            st.session_state["loaded_run_name"] = f"{ts}.json"
            st.toast(f"Run saved to runs/{ts}.json")
        except Exception as e:
            st.warning(f"Could not auto-save run: {e}")

        st.session_state["result"] = state
        st.rerun()

# --- Results Presentation ---
if "result" in st.session_state:
    r = st.session_state["result"]
    extracted = r.get("extracted", [])
    gaps = r.get("gaps", [])
    verification = r.get("verification", {})
    comp = r.get("comparison", {})
    issues = r.get("issues", [])
    
    st.divider()
    # Fully completed pipeline tracker
    pipeline_tracker(["Search", "Extract", "Compare", "Gaps", "Write", "Verify"])
    
    # Calculate total verified evidence quotes
    verified_quotes_count = sum(
        1 for p in extracted for e in p.get("evidence", []) if e.get("verified")
    )
    score_val = verification.get("score")
    score_display = "N/A" if score_val is None else str(score_val)
    if score_display != "N/A" and not str(score_display).endswith("%"):
        try:
            score_num = float(score_display)
            score_display = f"{score_num:.0%}" if score_num <= 1.0 else f"{score_num}%"
        except Exception:
            pass

    # High-impact Stat Cards
    stat_cards([
        (str(len(extracted)), "Papers Analyzed"),
        (str(len(gaps)), "Research Gaps Found"),
        (score_display, "Citation Score"),
        (str(verified_quotes_count), "Verified Evidence Quotes")
    ])

    # Issues Panel
    if not issues:
        issues = [
            {
                "ref": p.get("ref", "P?"),
                "title": p.get("title", "Unknown"),
                "reason": p.get("fallback_reason", "PDF unavailable or parsing failed; used abstract only.")
            }
            for p in extracted if p.get("pdf_fallback")
        ]

    if issues:
        with st.expander(f"⚠️ PDF Fallback Notice ({len(issues)} paper(s) fell back to abstract)", expanded=False):
            st.warning("The following papers could not be parsed as full-text PDFs and fell back to abstract-only extraction:")
            for iss in issues:
                ref_label = f"**{iss.get('ref', '')}** - " if iss.get("ref") else ""
                st.markdown(f"- {ref_label}**{iss.get('title', 'Unknown')}**: `{iss.get('reason', 'PDF failed')}`")

    # Tabbed View
    tabs = st.tabs(["📄 Papers", "⚖️ Comparison", "🧭 Gaps", "✍️ Literature Review", "🔍 Evidence Inspector", "⚠️ Issues"])

    # 1. Papers Tab
    with tabs[0]:
        st.markdown("### Extracted Papers & Methodology")
        for p in extracted:
            with st.container(border=True):
                col_meta, col_stat = st.columns([5, 1])
                badge_type = "Abstract Only" if p.get("pdf_fallback") else "Full PDF"
                col_meta.markdown(f"#### {p.get('ref', 'P?')}. {p.get('title', 'Untitled')} ({p.get('year', 'n.d.')})")
                col_stat.markdown(badge("low" if p.get("pdf_fallback") else "high"), unsafe_allow_html=True)
                
                if p.get("authors"):
                    st.caption(f"Authors: {', '.join(p['authors'])}")
                
                p_prob = p.get("problem")
                if p_prob:
                    st.markdown(f"**Problem:** {p_prob}")
                p_meth = p.get("method")
                if p_meth:
                    st.markdown(f"**Method:** {p_meth}")
                p_res = p.get("results") or p.get("key_results")
                if p_res:
                    st.markdown(f"**Key Results:** {p_res}")
                if not p_prob and not p_meth:
                    st.caption(p.get("abstract", "")[:300] + ("..." if len(p.get("abstract", "")) > 300 else ""))

    # 2. Comparison Tab
    with tabs[1]:
        st.markdown("### Comparative Methodological Matrix")
        table_data = comp.get("table", [])
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.info("No comparative table generated.")

        insights = comp.get("insights", "")
        if insights:
            st.markdown("### Comparative Insights")
            st.markdown(insights)

        shared = comp.get("shared_datasets", {})
        if shared:
            st.markdown("### Benchmark Datasets Shared Across Papers")
            st.info(", ".join(f"**{d}** ({v} papers)" for d, v in shared.items()))

    # 3. Gaps Tab
    with tabs[2]:
        st.markdown("### Identified Research Gaps & Open Challenges")
        for g in gaps:
            conf = g.get("confidence", "medium")
            evidence_papers = g.get("evidence_papers", [])
            ev_str = ", ".join(f"`{p}`" for p in evidence_papers) if evidence_papers else "None cited"
            
            st.markdown(
                f'<div class="gap-card">'
                f'<div style="margin-bottom:8px;">{badge(conf)} <strong style="font-size:1.05rem; margin-left:6px;">[{g.get("type", "Gap").title()}] {g.get("finding", "")}</strong></div>'
                f'<p style="margin: 6px 0; color:#CAD5E2;">{g.get("explanation", "")}</p>'
                f'<div class="gap-question">💡 <strong>Research Question:</strong> {g.get("research_question", "")}</div>'
                f'<div style="margin-top:8px; font-size:0.85rem; color:#8FA3BA;">Linked Evidence Papers: {ev_str}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # 4. Literature Review Tab
    with tabs[3]:
        st.markdown("### Synthesized Literature Review")
        
        c_score, c_issues = st.columns([1, 3])
        c_score.metric("Citation Faithfulness Score", score_display)
        
        flagged = verification.get("issues", [])
        with c_issues.expander(f"Auditor Log ({len(flagged)} flagged sentences)", expanded=(len(flagged) > 0)):
            if flagged:
                for iss in flagged:
                    st.markdown(f"- ⚠️ **{iss.get('problem', 'Issue')}**: *\"{iss.get('sentence', '')}\"*")
            else:
                st.success("No sentences were flagged by the rule-based checker.")

        # Render formatted markdown
        raw_review = r.get("review", "")
        md_text = render_review(raw_review, extracted) if raw_review else "No review generated."
        
        st.markdown(md_text)
        
        st.divider()
        c_dl1, c_dl2 = st.columns(2)
        c_dl1.download_button(
            "📥 Download Review (.md)",
            data=md_text,
            file_name=f"literature_review_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        c_dl2.download_button(
            "📚 Download BibTeX (.bib)",
            data=to_bibtex(extracted),
            file_name=f"references_{datetime.now().strftime('%Y%m%d')}.bib",
            mime="text/plain",
            use_container_width=True
        )

    # 5. Evidence Inspector Tab
    with tabs[4]:
        st.markdown("### Interactive Citation & Evidence Inspector")
        st.caption("Inspect exact quote spans extracted from each paper along with verified page numbers.")
        if extracted:
            paper_choices = [f"{p.get('ref', 'P?')}. {p.get('title', 'Unknown')}" for p in extracted]
            pick = st.selectbox("Select Paper to inspect", paper_choices)
            chosen_ref = pick.split(".")[0].strip()
            chosen_paper = next((p for p in extracted if p.get("ref", "").strip() == chosen_ref), extracted[0])
            
            ev_list = chosen_paper.get("evidence", [])
            if ev_list:
                for ev in ev_list:
                    is_ver = ev.get("verified", False)
                    v_badge = badge("verified" if is_ver else "flagged")
                    st.markdown(f"**Field: `{ev.get('field', 'quote')}`** (Page {ev.get('page', '?')}) &nbsp; {v_badge}", unsafe_allow_html=True)
                    st.code(ev.get("quote", "(no direct quote available)"))
            else:
                st.info("No evidence quotes logged for this paper (paper may have fallen back to abstract).")
        else:
            st.info("No papers extracted.")

    # 6. Issues Tab
    with tabs[5]:
        st.markdown("### Extraction Issues & PDF Fallback Logs")
        if issues:
            st.dataframe(pd.DataFrame(issues), use_container_width=True)
        else:
            st.success("All papers were successfully parsed from full-text PDFs with zero fallbacks!")
