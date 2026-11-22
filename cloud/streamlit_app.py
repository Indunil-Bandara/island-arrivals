"""Streamlit entry point for the Island Arrivals dashboard."""
from pathlib import Path
import sys

import streamlit as st
from streamlit.components.v1 import declare_component

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import features, models, predict_one, report

st.set_page_config(page_title="Island Arrivals", layout="wide")
dashboard = declare_component("island_arrivals", path=str(HERE / "frontend"))
prior = st.session_state.get("scenario_prior", report["prior"])
inputs = features(1, prior, report["two_back"], report["year_back"])
prediction = predict_one(models, inputs)
selected = dashboard(report=report, prior=prior, prediction=prediction, key="dashboard")
if type(selected) is int and 0 <= selected <= 350000 and selected != prior:
    st.session_state.scenario_prior = selected
    st.rerun()
