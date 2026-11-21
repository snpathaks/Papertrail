import html
import streamlit as st

STAGES = [
    ("🔎", "Search Agent", "scanning OpenAlex and arXiv"),
    ("📄", "Extraction Agent", "reading papers and quoting evidence"),
    ("⚖️", "Comparison Agent", "clustering methods & benchmarks"),
    ("🧭", "Gap Analyst", "hunting for what is missing"),
    ("✍️", "Writer + Verifier", "drafting and citation auditing"),
]
SECONDS = 10
STEP = SECONDS / len(STAGES)  # 2.0s per agent stage

COMMON_PAPERS_CSS = """
/* Floating Small Papers Background (drifting behind splash & dashboard) */
.floating-papers-container {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
  overflow: hidden;
  z-index: 1;
}
.paper-sheet {
  position: absolute;
  bottom: -60px;
  width: 26px;
  height: 34px;
  background: rgba(230, 237, 245, 0.22);
  border: 1.5px solid #2A9D8F;
  border-radius: 2px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45), 0 0 10px rgba(42, 157, 143, 0.45);
  pointer-events: none;
  animation: floatPaperUp linear infinite;
}
.paper-sheet::before {
  content: "";
  position: absolute;
  top: 5px;
  left: 4px;
  right: 4px;
  height: 2px;
  background: #2A9D8F;
  box-shadow: 0 5px 0 #E6EDF5, 0 10px 0 #E6EDF5, 0 15px 0 #2A9D8F;
}
.paper-sheet::after {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  width: 0;
  height: 0;
  border-width: 0 8px 8px 0;
  border-style: solid;
  border-color: #0F1B2D #0F1B2D #2A9D8F #2A9D8F;
}
@keyframes floatPaperUp {
  0% {
    transform: translateY(0) rotate(0deg) scale(0.85);
    opacity: 0;
  }
  10% {
    opacity: 0.9;
  }
  88% {
    opacity: 0.9;
  }
  100% {
    transform: translateY(-120vh) rotate(360deg) scale(1.1);
    opacity: 0;
  }
}
/* Negative animation delays so papers are already visibly floating across the screen immediately */
.paper-sheet.p1  { left: 4%;  animation-duration: 16s; animation-delay: -2s;   width: 28px; height: 36px; }
.paper-sheet.p2  { left: 12%; animation-duration: 14s; animation-delay: -7s;   width: 22px; height: 30px; }
.paper-sheet.p3  { left: 20%; animation-duration: 18s; animation-delay: -12s;  width: 30px; height: 38px; }
.paper-sheet.p4  { left: 29%; animation-duration: 15s; animation-delay: -4s;   width: 24px; height: 32px; }
.paper-sheet.p5  { left: 38%; animation-duration: 17s; animation-delay: -10s;  width: 26px; height: 34px; }
.paper-sheet.p6  { left: 47%; animation-duration: 13s; animation-delay: -1s;   width: 22px; height: 28px; }
.paper-sheet.p7  { left: 56%; animation-duration: 19s; animation-delay: -8s;   width: 28px; height: 36px; }
.paper-sheet.p8  { left: 65%; animation-duration: 16s; animation-delay: -13s;  width: 24px; height: 32px; }
.paper-sheet.p9  { left: 74%; animation-duration: 18s; animation-delay: -5s;   width: 26px; height: 34px; }
.paper-sheet.p10 { left: 83%; animation-duration: 14s; animation-delay: -11s;  width: 23px; height: 30px; }
.paper-sheet.p11 { left: 91%; animation-duration: 20s; animation-delay: -3s;   width: 29px; height: 37px; }
.paper-sheet.p12 { left: 97%; animation-duration: 15s; animation-delay: -9s;   width: 23px; height: 31px; }
"""

SPLASH_CSS = f"""<style>
{COMMON_PAPERS_CSS}
#skip-splash {{ display: none; }}
#skip-splash:checked ~ #splash-screen {{
  display: none !important;
  opacity: 0 !important;
  pointer-events: none !important;
  transform: translateY(-200vh) !important;
}}
#splash-screen {{
  position: fixed; inset: 0; z-index: 99999;
  background: radial-gradient(circle at center, #101F35 0%, #0A1320 100%);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  color: #E6EDF5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  overflow: hidden;
  animation: splashAutoDismiss {SECONDS}s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}}
@keyframes splashAutoDismiss {{
  0%   {{ opacity: 1; pointer-events: all;  transform: translateY(0); }}
  82%  {{ opacity: 1; pointer-events: all;  transform: translateY(0); }}
  96%  {{ opacity: 0; pointer-events: none; transform: translateY(0); }}
  100% {{ opacity: 0; pointer-events: none; transform: translateY(-200vh); }}
}}
.splash-content {{
  position: relative; z-index: 10;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}}
.splash h1 {{
  font-family: Georgia, serif; font-size: 3.4rem; margin: 0; letter-spacing: 2px; font-weight: 700;
  text-shadow: 0 4px 16px rgba(0,0,0,0.6);
}}
.splash h1 span {{ color: #2A9D8F; }}
.splash p.tag {{
  opacity: .85; margin: .4rem 0 1.8rem; font-size: 1.05rem; letter-spacing: 1.5px; text-transform: uppercase;
}}
.stage-box {{
  position: relative; height: 80px; width: 480px; text-align: center; margin: 1rem 0;
}}
.stage {{
  position: absolute; inset: 0; width: 100%; opacity: 0; font-size: 1.35rem; font-weight: 600;
  color: #E6EDF5; text-shadow: 0 2px 10px rgba(0,0,0,0.6);
  animation: stg {STEP}s ease-in-out both;
}}
.stage small {{
  display: block; opacity: .85; font-size: .95rem; margin-top: 6px; font-weight: 400; color: #8FA3BA;
}}
@keyframes stg {{
  0%   {{ opacity: 0; transform: translateY(14px); }}
  18%  {{ opacity: 1; transform: translateY(0); }}
  82%  {{ opacity: 1; transform: translateY(0); }}
  100% {{ opacity: 0; transform: translateY(-12px); }}
}}
.bar {{
  width: 440px; height: 6px; background: #1E3350; border-radius: 4px; overflow: hidden; margin-top: 1.6rem;
  box-shadow: 0 0 12px rgba(42, 157, 143, 0.35);
}}
.bar div {{
  height: 100%; width: 0; background: #2A9D8F; box-shadow: 0 0 10px #2A9D8F;
  animation: fill {SECONDS}s linear forwards;
}}
@keyframes fill {{ to {{ width: 100%; }} }}
.pulse {{
  width: 18px; height: 18px; border-radius: 50%; background: #2A9D8F; margin-bottom: 1.4rem;
  box-shadow: 0 0 12px #2A9D8F; animation: pulse 1.4s infinite;
}}
@keyframes pulse {{
  0%   {{ box-shadow: 0 0 0 0 rgba(42, 157, 143, .8); }}
  100% {{ box-shadow: 0 0 0 26px rgba(42, 157, 143, 0); }}
}}
.skip-link {{
  position: absolute; bottom: 25px; right: 35px; z-index: 100;
  background: rgba(22, 37, 59, 0.85); border: 1px solid #2A9D8F;
  color: #E6EDF5; padding: 8px 18px; border-radius: 20px; font-size: 0.85rem;
  font-weight: 600; cursor: pointer; transition: all 0.2s ease;
  user-select: none;
}}
.skip-link:hover {{
  background: #2A9D8F; color: #ffffff;
}}
</style>"""

CUSTOM_THEME_CSS = f"""<style>
{COMMON_PAPERS_CSS}
/* App-wide research-lab styling */
.stApp {{
  background-color: #0F1B2D;
  color: #E6EDF5;
}}
h1, h2, h3 {{
  font-family: Georgia, serif;
  color: #E6EDF5 !important;
}}
.hero-title {{
  font-family: Georgia, serif;
  font-size: 2.8rem;
  font-weight: 700;
  margin-bottom: 0.2rem;
  letter-spacing: 0.5px;
}}
.hero-title span {{
  color: #2A9D8F;
}}
.hero-subtitle {{
  font-size: 1.15rem;
  color: #8FA3BA;
  margin-bottom: 1.5rem;
}}
.stat-card {{
  background: #16253B;
  border-top: 4px solid #2A9D8F;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 4px 14px rgba(0,0,0,0.3);
  margin-bottom: 1rem;
}}
.stat-num {{
  font-size: 2.2rem;
  font-weight: 700;
  font-family: Georgia, serif;
  color: #E6EDF5;
}}
.stat-label {{
  opacity: 0.75;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 4px;
}}
.gap-card {{
  background: #16253B;
  border-left: 4px solid #2A9D8F;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}
.gap-question {{
  background: rgba(42, 157, 143, 0.12);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.92rem;
  margin-top: 8px;
  border: 1px solid rgba(42, 157, 143, 0.3);
}}
</style>"""

FLOATING_PAPERS_HTML = """<div class="floating-papers-container" aria-hidden="true">
<div class="paper-sheet p1"></div>
<div class="paper-sheet p2"></div>
<div class="paper-sheet p3"></div>
<div class="paper-sheet p4"></div>
<div class="paper-sheet p5"></div>
<div class="paper-sheet p6"></div>
<div class="paper-sheet p7"></div>
<div class="paper-sheet p8"></div>
<div class="paper-sheet p9"></div>
<div class="paper-sheet p10"></div>
<div class="paper-sheet p11"></div>
<div class="paper-sheet p12"></div>
</div>"""

def inject_theme_css():
    """Inject research-lab CSS and the subtle floating papers background across the dashboard."""
    st.markdown(CUSTOM_THEME_CSS, unsafe_allow_html=True)
    st.markdown(FLOATING_PAPERS_HTML, unsafe_allow_html=True)

def warmup():
    """Real warmup work: loads FastEmbed embedder."""
    try:
        from llm import embedder
        embedder.embed_query("warm up")
    except Exception:
        pass

def splash(warmup_fn=None):
    """Render CSS-animated 10s splash screen with floating papers, 5 agent stages, and progress bar."""
    if st.session_state.get("splash_done"):
        return
    
    stages_html = "".join(
        f'<div class="stage" style="animation-delay:{i*STEP:.2f}s">{ic} {name}<small>{sub}</small></div>'
        for i, (ic, name, sub) in enumerate(STAGES)
    )
    
    splash_html = (
        f"{SPLASH_CSS}\n"
        f'<input type="checkbox" id="skip-splash">\n'
        f'<div id="splash-screen" class="splash">\n'
        f'{FLOATING_PAPERS_HTML}\n'
        f'<div class="splash-content">\n'
        f'<div class="pulse"></div>\n'
        f'<h1>Paper<span>Trail</span></h1>\n'
        f'<p class="tag">Autonomous Multi-Agent Literature Review</p>\n'
        f'<div class="stage-box">\n{stages_html}\n</div>\n'
        f'<div class="bar"><div></div></div>\n'
        f'</div>\n'
        f'<label for="skip-splash" class="skip-link">Skip Intro ➔</label>\n'
        f'</div>'
    )
    
    st.markdown(splash_html, unsafe_allow_html=True)
    if warmup_fn:
        import threading
        threading.Thread(target=warmup_fn, daemon=True).start()
    st.session_state["splash_done"] = True

def pipeline_tracker(done: list[str]):
    """Horizontal strip: Search -> Extract -> Compare -> Gaps -> Write -> Verify"""
    steps = ["Search", "Extract", "Compare", "Gaps", "Write", "Verify"]
    done_lower = [str(d).strip().lower() for d in done]
    html_out = '<div style="display:flex; gap:10px; margin:12px 0 20px; flex-wrap:wrap; position:relative; z-index:1;">'
    for s in steps:
        ok = s.lower() in done_lower
        bg, fg, border = ("#2A9D8F", "#ffffff", "1px solid #2A9D8F") if ok else ("#16253B", "#8FA3BA", "1px solid #1E3350")
        icon = "✓ " if ok else "○ "
        html_out += (
            f'<div style="flex:1; min-width:110px; text-align:center; padding:10px 6px; border-radius:8px;'
            f'background:{bg}; color:{fg}; border:{border}; font-weight:600; font-size:.85rem;'
            f'box-shadow: 0 2px 6px rgba(0,0,0,0.2);">'
            f'{icon}{s}</div>'
        )
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)

def stat_cards(items: list[tuple[str, str]]):
    """Render high-impact stat cards with Georgia font numbers and teal borders."""
    cols = st.columns(len(items))
    for c, (num, label) in zip(cols, items):
        c.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-num">{html.escape(str(num))}</div>'
            f'<div class="stat-label">{html.escape(str(label))}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

def badge(level: str) -> str:
    """Return an inline HTML badge styled for confidence or status."""
    lvl = (level or "low").strip().lower()
    color_map = {
        "high": "#2A9D8F",      # teal/green
        "medium": "#E9A23B",    # amber
        "low": "#D1495B",       # red
        "verified": "#2A9D8F",
        "flagged": "#D1495B"
    }
    color = color_map.get(lvl, "#2A9D8F")
    return (
        f'<span style="background:{color}; color:#fff; padding:3px 12px; border-radius:12px;'
        f'font-size:.75rem; font-weight:700; letter-spacing:0.5px; display:inline-block;">'
        f'{html.escape(level.upper())}</span>'
    )

def paper_card(p: dict, key: str) -> bool:
    """Render a paper candidate card with metadata and an active Keep toggle."""
    with st.container(border=True):
        a, b = st.columns([6, 1])
        title = html.escape(p.get("title", "Untitled"))
        year = p.get("year") or "n.d."
        source = html.escape(p.get("source", "scholar"))
        authors = ", ".join(p.get("authors", []))
        if authors:
            authors_str = f" · {html.escape(authors)}"
        else:
            authors_str = ""
        a.markdown(f"**{title}**  \n*{year} · {source}{authors_str}*")
        reason = p.get("relevance_reason", "")
        if reason:
            a.caption(f"💡 {reason}")
        return b.toggle("Keep", value=True, key=key)
