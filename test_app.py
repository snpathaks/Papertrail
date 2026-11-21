import streamlit as st
import time

t0 = time.time()
import config
t_config = round(time.time() - t0, 3)

t0 = time.time()
from export import render_review, to_bibtex
t_export = round(time.time() - t0, 3)

t0 = time.time()
from ui import inject_theme_css
t_ui = round(time.time() - t0, 3)

st.title("Timing Test")
st.write(f"config: {t_config}s, export: {t_export}s, ui: {t_ui}s")
