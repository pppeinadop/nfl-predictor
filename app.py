"""
App Streamlit para visualizar los picks y props guardados en MongoDB.
También permite generar semanas NFL y NCAAF en flujos separados.

Uso:
    streamlit run app.py
"""
import contextlib
import html
import io
import json
import re
import unicodedata
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

import config
import db
import predict_games
import rebuild_weeks


# ---------- Configuración de página ----------
st.set_page_config(
    page_title="NFL / NCAAF Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Estilos ----------
st.markdown("""
<style>
  /* Base visual: claro, minimalista y deportivo */
  .stApp {
    background: #f7f9fc;
    color: #172033;
  }
  [data-testid="stHeader"] {
    background: rgba(247, 249, 252, 0.92);
  }
  [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5eaf2;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #172033;
  }
  h1, h2, h3 {
    color: #172033;
    letter-spacing: -0.025em;
  }
  h1 {
    font-weight: 750;
    padding-bottom: 0.25rem;
  }
  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
  }
  [data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 600;
  }
  [data-testid="stMetricValue"] {
    color: #172033;
  }
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e2e8f0 !important;
    border-radius: 14px !important;
    background: #ffffff;
  }
  div[data-testid="stPopover"] > button {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #172033 !important;
    padding: 10px 14px !important;
    line-height: 1.3 !important;
    border: 1px solid #dbe3ee !important;
    border-radius: 10px !important;
    background: #ffffff !important;
  }
  div[data-testid="stPopover"] > button p,
  div[data-testid="stPopover"] > button strong {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #172033 !important;
    margin: 0 !important;
  }
  .tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 650;
    margin-right: 6px;
  }
  .tag-value {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
  }
  .tag-neutral {
    background: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
  }
  .tag-warn {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
  }
  .slot-header {
    margin-top: 24px;
    margin-bottom: 10px;
    padding: 0 0 8px 2px;
    border-bottom: 1px solid #dbe3ee;
    color: #334155;
    font-size: 1.05rem;
    font-weight: 750;
    letter-spacing: -0.01em;
  }
  .result-score {
    font-size: 1.05rem;
    font-weight: 750;
    letter-spacing: 0.1px;
    margin-bottom: 10px;
    line-height: 1.25;
  }
  .result-score.hit { color: #047857; }
  .result-score.miss { color: #b91c1c; }
  .result-score.neutral { color: #475569; }
  .badge {
    display: inline-block;
    font-size: 0.92rem;
    font-weight: 750;
    padding: 5px 12px;
    border-radius: 999px;
    letter-spacing: 0.1px;
    line-height: 1.2;
    margin-bottom: 6px;
    border: 1px solid;
  }
  .badge-hit { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
  .badge-miss { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
  .badge-neutral { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }
  .record {
    color: #64748b;
    font-size: 0.72em;
    font-weight: 500;
    letter-spacing: 0;
  }
  .ml-line {
    font-size: 0.92em;
    color: #475569;
    margin-top: 6px;
  }
  .ml-line .v {
    color: #172033;
    font-weight: 700;
  }
  .ml-line .v.edge {
    color: #047857;
    font-weight: 750;
  }
  .ml-line .sep {
    color: #94a3b8;
    margin: 0 4px;
  }
  .at-sep {
    text-align: center;
    font-size: 1.35rem;
    font-weight: 750;
    color: #94a3b8;
    padding-top: 4px;
  }
  [data-testid="stImage"] {
    display: flex;
    justify-content: center;
  }
  .projection-logo-fallback {
    min-height: 52px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #334155;
    font-weight: 750;
    text-align: center;
  }
  .projection-score {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.8rem;
    color: #172033;
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.3;
  }
  .projection-score .separator {
    color: #94a3b8;
    font-size: 1rem;
  }
  .gen-confirm {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }
  .gen-confirm .label {
    color: #64748b;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .gen-confirm .value {
    color: #172033;
    font-size: 1.35rem;
    font-weight: 750;
    letter-spacing: -0.01em;
  }
  .gen-confirm .sub {
    color: #475569;
    font-size: 0.95rem;
    margin-top: 4px;
  }
  div.stButton > button {
    border-radius: 9px;
    font-weight: 650;
  }
  div[data-baseweb="select"] > div {
    border-radius: 9px;
    border-color: #dbe3ee;
  }
  :root {
    --ui-ink: #172033;
    --ui-muted: #64748b;
    --ui-line: #e2e8f0;
    --ui-surface: #ffffff;
    --ui-green: #087f5b;
    --ui-green-dark: #066548;
    --ui-green-soft: #e8f7f0;
    --ui-orange: #e67e22;
  }
  .stApp {
    background: radial-gradient(ellipse at 10% 0%, rgba(8, 127, 91, 0.06), transparent 34rem), #f3f6fa;
  }
  [data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1540px;
    padding: 2.1rem clamp(1.25rem, 3vw, 3rem) 3rem;
  }
  [data-testid="stHeader"] {
    background: rgba(243, 246, 250, 0.88);
    backdrop-filter: blur(12px);
  }
  [data-testid="stSidebar"] {
    background: var(--ui-surface);
    border-right: 1px solid var(--ui-line);
  }
  [data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
  }
  h1, h2, h3 {
    color: var(--ui-ink);
    letter-spacing: -0.035em;
  }
  h1 {
    padding-bottom: 0.7rem;
    margin-bottom: 0.8rem;
    border-bottom: 3px solid var(--ui-green);
    width: fit-content;
    max-width: 100%;
  }
  [data-testid="stTabs"] [role="tablist"] {
    gap: 0.35rem;
    border-bottom: 1px solid var(--ui-line);
  }
  [data-testid="stTabs"] button[role="tab"] {
    color: var(--ui-muted);
    font-weight: 650;
    padding: 0.65rem 0.85rem;
    border-radius: 0.65rem 0.65rem 0 0;
    transition: color 140ms ease, background 140ms ease;
  }
  [data-testid="stTabs"] button[role="tab"]:hover,
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--ui-green-dark);
    background: var(--ui-green-soft);
  }
  [data-testid="stTabs"] button[role="tab"]:focus-visible,
  div.stButton > button:focus-visible,
  div[data-testid="stPopover"] > button:focus-visible {
    outline: 3px solid rgba(8, 127, 91, 0.35);
    outline-offset: 2px;
  }
  [data-testid="stMetric"] {
    min-height: 100%;
    border: 1px solid var(--ui-line);
    border-top: 3px solid var(--ui-green);
    border-radius: 13px;
    padding: 14px 16px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
    transition: transform 140ms ease, box-shadow 140ms ease;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
  }
  [data-testid="stMetricLabel"] { color: var(--ui-muted); }
  [data-testid="stMetricValue"] { color: var(--ui-ink); font-weight: 750; }
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--ui-line) !important;
    border-radius: 16px !important;
    background: var(--ui-surface);
    box-shadow: 0 3px 14px rgba(15, 23, 42, 0.035);
    transition: border-color 140ms ease, box-shadow 140ms ease;
  }
  [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #cbd9d5 !important;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
  }
  div[data-testid="stPopover"] > button {
    border-color: var(--ui-line) !important;
    border-radius: 11px !important;
    background: #f8fafc !important;
    transition: border-color 140ms ease, background 140ms ease;
  }
  div[data-testid="stPopover"] > button:hover {
    border-color: #9acbb9 !important;
    background: var(--ui-green-soft) !important;
  }
  .tag { padding: 4px 11px; margin: 3px 6px 3px 0; }
  .tag-value {
    background: var(--ui-green-soft);
    color: var(--ui-green-dark);
    border-color: #b8e4d1;
  }
  .slot-header {
    margin-top: 28px;
    margin-bottom: 12px;
    padding: 0 0 9px 11px;
    border-bottom: 1px solid #d5e0e9;
    border-left: 3px solid var(--ui-orange);
  }
  .result-score.hit, .ml-line .v.edge { color: var(--ui-green-dark); }
  .badge-hit {
    background: var(--ui-green-soft);
    color: var(--ui-green-dark);
    border-color: #b8e4d1;
  }
  .gen-confirm {
    background: linear-gradient(120deg, #e8f7f0, #f3f9ff);
    border-color: #c8e5d8;
    border-left: 4px solid var(--ui-green);
    border-radius: 13px;
    padding: 17px 21px;
    margin: 8px 0 18px;
  }
  div.stButton > button {
    border-radius: 10px;
    min-height: 2.6rem;
    transition: transform 140ms ease, box-shadow 140ms ease;
  }
  div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.09);
  }
  div.stButton > button[kind="primary"] {
    background: var(--ui-green);
    border-color: var(--ui-green);
  }
  div.stButton > button[kind="primary"]:hover {
    background: var(--ui-green-dark);
    border-color: var(--ui-green-dark);
  }
  div[data-baseweb="select"] > div {
    border-radius: 10px;
    border-color: #d5dee8;
  }
  [data-testid="stDataFrame"] {
    border: 1px solid var(--ui-line);
    border-radius: 12px;
    overflow: hidden;
  }
  @media (max-width: 760px) {
    [data-testid="stAppViewContainer"] .main .block-container {
      padding: 1.25rem 1rem 2rem;
    }
    [data-testid="stTabs"] [role="tablist"] {
      overflow-x: auto;
      flex-wrap: nowrap;
      scrollbar-width: thin;
    }
    [data-testid="stTabs"] button[role="tab"] {
      white-space: nowrap;
      padding: 0.6rem 0.7rem;
      font-size: 0.9rem;
    }
    [data-testid="stMetric"] { padding: 11px 12px; }
    .slot-header { margin-top: 22px; }
    div[data-testid="stPopover"] > button { padding: 8px 9px !important; }
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopover"] > button p,
    div[data-testid="stPopover"] > button strong { font-size: 0.92rem !important; }
  }
  [data-testid="stRadioGroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
  }
  [data-testid="stRadioOption"] {
    padding: 0.42rem 0.8rem;
    border: 1px solid var(--ui-line);
    border-radius: 999px;
    background: var(--ui-surface);
    transition: color 140ms ease, border-color 140ms ease, background 140ms ease;
  }
  [data-testid="stRadioOption"]:hover {
    border-color: #9acbb9;
    background: var(--ui-green-soft);
  }
  [data-testid="stRadioOption"][data-selected="true"] {
    border-color: var(--ui-green);
    background: var(--ui-green);
    color: #ffffff;
  }
  [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] {
    color: #ffffff;
  }
  [data-testid="stRadioOption"]:focus-within {
    outline: 3px solid rgba(8, 127, 91, 0.3);
    outline-offset: 2px;
  }
  [data-testid="stExpander"] summary {
    border-radius: 10px;
    color: var(--ui-green-dark);
    font-weight: 650;
  }
  [data-testid="stExpander"] summary:hover {
    background: var(--ui-green-soft);
  }

  /* Stadium-inspired dark theme */
  :root {
    color-scheme: dark;
    --ui-ink: #edf4fa;
    --ui-muted: #9aabba;
    --ui-line: #2b3b4a;
    --ui-surface: #121e29;
    --ui-green: #42d99a;
    --ui-green-dark: #20b77a;
    --ui-green-soft: #173a32;
    --ui-orange: #ff9f43;
  }
  .stApp {
    color: var(--ui-ink);
    background:
      radial-gradient(ellipse at 12% 0%, rgba(66, 217, 154, 0.1), transparent 34rem),
      radial-gradient(ellipse at 92% 8%, rgba(255, 159, 67, 0.06), transparent 30rem),
      #0a1118;
  }
  [data-testid="stAppViewContainer"] {
    background: transparent;
  }
  [data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 2rem;
  }
  [data-testid="stHeader"] {
    background: rgba(10, 17, 24, 0.88);
    color: var(--ui-ink);
  }
  [data-testid="stSidebar"] {
    background: #0e1821;
    border-right: 1px solid #263644;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  h1, h2, h3, h4 {
    color: var(--ui-ink);
  }
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stCaptionContainer"],
  label,
  [data-testid="stWidgetLabel"] {
    color: var(--ui-muted);
  }
  h1 {
    border-bottom-color: var(--ui-green);
  }
  [data-testid="stTabs"] [role="tablist"] {
    border-bottom-color: var(--ui-line);
  }
  [data-testid="stTabs"] button[role="tab"] {
    color: var(--ui-muted);
  }
  [data-testid="stTabs"] button[role="tab"]:hover,
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #b6ffdc;
    background: #173a32;
  }
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    box-shadow: inset 0 -2px 0 var(--ui-green);
  }
  [data-testid="stMetric"],
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #121e29;
    border-color: var(--ui-line) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  }
  [data-testid="stMetric"] {
    border-top-color: var(--ui-green);
  }
  [data-testid="stMetricLabel"] {
    color: var(--ui-muted);
  }
  [data-testid="stMetricValue"],
  [data-testid="stMetricDelta"] {
    color: var(--ui-ink);
  }
  [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #3e6659 !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
  }
  div[data-testid="stPopover"] > button {
    color: var(--ui-ink) !important;
    background: #192733 !important;
    border-color: #334655 !important;
  }
  [data-testid="stPopoverButton"] {
    color: var(--ui-ink) !important;
    background: #192733 !important;
    border-color: #334655 !important;
  }
  [data-testid="stPopoverButton"]:hover {
    background: #20372f !important;
    border-color: #3e8068 !important;
  }
  div[data-testid="stPopover"] > button p,
  div[data-testid="stPopover"] > button strong {
    color: var(--ui-ink) !important;
  }
  div[data-testid="stPopover"] > button:hover {
    background: #20372f !important;
    border-color: #3e8068 !important;
  }
  .tag-value, .badge-hit {
    color: #9af4c8;
    background: #173a32;
    border-color: #28664f;
  }
  .tag-neutral, .badge-neutral {
    color: #b3c0cc;
    background: #1b2935;
    border-color: #344756;
  }
  .tag-warn {
    color: #ffd08f;
    background: #3b2b1c;
    border-color: #70502d;
  }
  .badge-miss {
    color: #ffb4ad;
    background: #3d2426;
    border-color: #743b3e;
  }
  .result-score.hit, .ml-line .v.edge {
    color: #80edb9;
  }
  .result-score.miss {
    color: #ff9b94;
  }
  .result-score.neutral, .record {
    color: var(--ui-muted);
  }
  .ml-line, .ml-line .v, .projection-score {
    color: var(--ui-ink);
  }
  .ml-line .sep, .at-sep, .projection-score .separator {
    color: #8092a2;
  }
  .projection-logo-fallback {
    color: var(--ui-ink);
  }
  .gen-confirm {
    color: var(--ui-ink);
    background: linear-gradient(120deg, #173a32, #172b3b);
    border-color: #315448;
    border-left-color: var(--ui-green);
  }
  .gen-confirm .label, .gen-confirm .sub {
    color: #b1c2ce;
  }
  .gen-confirm .value {
    color: #edfff6;
  }
  .slot-header {
    color: var(--ui-ink);
    border-bottom-color: var(--ui-line);
    border-left-color: var(--ui-orange);
  }
  div.stButton > button,
  [data-testid="stDownloadButton"] button {
    color: var(--ui-ink);
    background: #192733;
    border-color: #3a4b5a;
  }
  div.stButton > button:hover,
  [data-testid="stDownloadButton"] button:hover {
    color: #ffffff;
    background: #223541;
    border-color: #658071;
  }
  div.stButton > button[kind="primary"] {
    color: #07130e;
    background: var(--ui-green);
    border-color: var(--ui-green);
  }
  div.stButton > button[kind="primary"]:hover {
    color: #07130e;
    background: #69e9ad;
    border-color: #69e9ad;
  }
  div[data-baseweb="select"] > div,
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea {
    color: var(--ui-ink);
    background: #111e29;
    border-color: #354757;
  }
  [data-testid="stSelectbox"] [role="group"],
  [data-testid="stNumberInputContainer"] {
    color: var(--ui-ink);
    background: #111e29 !important;
    border: 1px solid #354757 !important;
  }
  [data-testid="stSelectbox"] input[role="combobox"] {
    color: var(--ui-ink) !important;
  }
  [data-testid="stSelectbox"] [role="group"] button,
  [data-testid="stNumberInputStepDown"],
  [data-testid="stNumberInputStepUp"] {
    color: var(--ui-ink) !important;
    background: #192733 !important;
  }
  [data-baseweb="select"] [role="option"] {
    color: var(--ui-ink);
    background: #15222d;
  }
  [data-baseweb="popover"],
  [data-baseweb="menu"] {
    background: #15222d;
  }
  [data-testid="stRadioOption"] {
    color: var(--ui-muted);
    background: #121e29;
    border-color: #354757;
  }
  [data-testid="stRadioOption"]:hover {
    color: var(--ui-ink);
    background: #1b302a;
    border-color: #3e8068;
  }
  [data-testid="stRadioOption"][data-selected="true"] {
    color: #07130e;
    background: var(--ui-green);
    border-color: var(--ui-green);
  }
  [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] {
    color: #07130e;
  }
  [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] p {
    color: #07130e !important;
  }
  input[type="radio"], input[type="checkbox"] {
    accent-color: var(--ui-green) !important;
  }
  [data-testid="stExpander"] {
    border-color: var(--ui-line);
  }
  [data-testid="stExpander"] summary {
    color: #a7f5cf;
  }
  [data-testid="stExpander"] summary:hover {
    background: #172b27;
  }
  [data-testid="stDataFrame"] {
    border-color: var(--ui-line);
  }
  [data-testid="stAlert"] {
    background: #172633;
    border-color: #334b5d;
    color: var(--ui-ink);
  }
  a {
    color: #78ddb0;
  }
  :focus-visible {
    outline-color: rgba(66, 217, 154, 0.75) !important;
  }
  @media (max-width: 760px) {
    [data-testid="stAppViewContainer"] .main .block-container {
      padding-top: 1.25rem;
    }
  }
  .matchday-filter-kicker,
  .scoreboard-team-kicker,
  .scoreboard-projection-label {
    color: #8fa3b3;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .matchday-filter-kicker {
    margin-bottom: 0.35rem;
    color: var(--ui-green);
  }
  .scoreboard-meta {
    color: #9aabba;
    font-size: 0.86rem;
    font-weight: 650;
  }
  .scoreboard-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }
  .scoreboard-matchup {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(150px, 0.95fr) minmax(0, 1fr);
    align-items: center;
    gap: clamp(0.65rem, 2.5vw, 2rem);
    padding: 1.4rem clamp(0.25rem, 2vw, 1.5rem) 1rem;
  }
  .scoreboard-team-panel {
    display: flex;
    min-width: 0;
    flex-direction: column;
    align-items: center;
  }
  .scoreboard-team-panel.away {
    align-items: flex-start;
  }
  .scoreboard-team-panel.home {
    align-items: flex-end;
  }
  .scoreboard-status {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.32rem 0.65rem;
    border: 1px solid #354757;
    border-radius: 999px;
    background: #192733;
    color: #c5d2dc;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-align: center;
    text-transform: uppercase;
  }
  .scoreboard-status.final {
    color: #a4f5ce;
    background: #173a32;
    border-color: #28664f;
  }
  .scoreboard-team-kicker,
  .scoreboard-projection-label {
    margin-bottom: 0.55rem;
    text-align: center;
  }
  .scoreboard-team-panel.away .scoreboard-team-kicker,
  .scoreboard-team-panel.away .scoreboard-team-name,
  .scoreboard-team-panel.away .scoreboard-team-record {
    text-align: left;
  }
  .scoreboard-team-panel.home .scoreboard-team-kicker,
  .scoreboard-team-panel.home .scoreboard-team-name,
  .scoreboard-team-panel.home .scoreboard-team-record {
    text-align: right;
  }
  .scoreboard-team-logo {
    display: flex;
    min-height: 92px;
    align-items: center;
    justify-content: center;
  }
  .scoreboard-team-logo img {
    display: block;
    width: clamp(64px, 8vw, 92px);
    max-height: 92px;
    object-fit: contain;
    filter: drop-shadow(0 8px 14px rgba(0, 0, 0, 0.3));
  }
  .scoreboard-team-logo span {
    color: var(--ui-ink);
    font-size: 1.35rem;
    font-weight: 900;
  }
  .scoreboard-team-name {
    color: #edf4fa;
    font-size: clamp(1.05rem, 2.1vw, 1.45rem);
    font-weight: 850;
    letter-spacing: -0.02em;
    line-height: 1.15;
    overflow-wrap: anywhere;
  }
  .scoreboard-team-record {
    margin-top: 0.2rem;
    color: #91a3b2;
    font-size: 0.82rem;
  }
  .scoreboard-projection {
    min-width: 0;
    text-align: center;
  }
  .scoreboard-projection-label {
    color: var(--ui-orange);
  }
  .scoreboard-team-logo-fallback {
    min-height: 84px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--ui-ink);
    font-size: 1.15rem;
    font-weight: 850;
    text-align: center;
  }
  .scoreboard-scoreline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: clamp(0.55rem, 2vw, 1.2rem);
    color: #f2f7fb;
    font-size: clamp(2.4rem, 5vw, 3.65rem);
    font-variant-numeric: tabular-nums;
    font-weight: 900;
    letter-spacing: -0.06em;
    line-height: 1.1;
    text-shadow: 0 0 28px rgba(66, 217, 154, 0.12);
  }
  .scoreboard-score-separator {
    color: #708391;
    font-size: 0.68em;
    font-weight: 600;
  }
  .scoreboard-probability {
    margin-top: 0.9rem;
    padding: 0.8rem 1rem 0.85rem;
    border: 1px solid #263846;
    border-radius: 12px;
    background: rgba(8, 17, 24, 0.52);
  }
  .scoreboard-probability-labels {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.55rem;
    color: #c2ced8;
    font-size: 0.84rem;
    font-weight: 700;
  }
  .scoreboard-probability-labels .away {
    color: #76e4b1;
  }
  .scoreboard-probability-labels .home {
    color: #ffbd72;
    text-align: right;
  }
  .scoreboard-probability-track {
    display: flex;
    overflow: hidden;
    height: 8px;
    border-radius: 999px;
    background: #253440;
  }
  .scoreboard-probability-track span {
    display: block;
    height: 100%;
    transition: width 180ms ease;
  }
  .scoreboard-probability-track .away {
    background: linear-gradient(90deg, #20b77a, #64e6ac);
  }
  .scoreboard-probability-track .home {
    background: linear-gradient(90deg, #ffbd72, #e98732);
  }
  .scoreboard-values {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 0.8rem;
  }
  .scoreboard-value-panel {
    min-width: 0;
    padding: 0.8rem 0.95rem;
    border: 1px solid #2b3d4b;
    border-radius: 12px;
    background: linear-gradient(145deg, #15232e, #111c26);
  }
  .scoreboard-value-title {
    margin-bottom: 0.4rem;
    color: #8fa3b3;
    font-size: 0.7rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .scoreboard-value-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.28rem 0;
    color: #a9bac7;
    font-size: 0.84rem;
  }
  .scoreboard-value-row + .scoreboard-value-row {
    border-top: 1px solid rgba(66, 83, 97, 0.45);
  }
  .scoreboard-value-row strong {
    color: #eef5fa;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .scoreboard-value-panel.market .scoreboard-value-title {
    color: var(--ui-orange);
  }
  .scoreboard-result-panel {
    margin-top: 0.8rem;
    padding: 0.8rem 0.95rem 0.7rem;
    border: 1px solid #2b3d4b;
    border-radius: 12px;
    background: #101b25;
  }
  .scoreboard-result-title {
    margin-bottom: 0.4rem;
    color: #8fa3b3;
    font-size: 0.7rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .scoreboard-pending-result {
    color: #d3dee6;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.04em;
  }
  .scoreboard-result-panel .result-score {
    margin-bottom: 0.55rem;
    font-size: 1.15rem;
  }
  .scoreboard-result-panel .badge {
    margin-right: 0.35rem;
  }
  @media (max-width: 760px) {
    .scoreboard-topline {
      align-items: flex-start;
      flex-direction: column;
      gap: 0.45rem;
    }
    .scoreboard-matchup {
      grid-template-columns: minmax(0, 1fr) minmax(110px, 0.9fr) minmax(0, 1fr);
      gap: 0.35rem;
      padding: 1.1rem 0.1rem 0.8rem;
    }
    .scoreboard-team-logo {
      min-height: 70px;
    }
    .scoreboard-team-logo img {
      width: clamp(48px, 14vw, 72px);
      max-height: 72px;
    }
    .scoreboard-team-name {
      font-size: 1rem;
    }
    .scoreboard-team-record {
      font-size: 0.72rem;
    }
    .scoreboard-values {
      grid-template-columns: 1fr;
    }
    .scoreboard-scoreline {
      font-size: clamp(2rem, 8vw, 2.7rem);
    }
    .scoreboard-probability-labels {
      font-size: 0.76rem;
    }
  }
</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def fmt(v, default="—"):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _get(row_like, *keys, default=None):
    for k in keys:
        try:
            v = row_like.get(k)
        except AttributeError:
            v = None
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        return v
    return default


def _bool_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return bool(v)
    except (TypeError, ValueError):
        return None


def has_real_value(text) -> bool:
    if text is None:
        return False
    try:
        if pd.isna(text):
            return False
    except (TypeError, ValueError):
        pass
    return "Sin valor" not in str(text)


def classify_value(text) -> str:
    if not has_real_value(text):
        return "tag-neutral"
    if "⚠" in str(text):
        return "tag-warn"
    return "tag-value"


def count_value_signals(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].apply(has_real_value).sum())


def fmt_datetime(fecha, hora) -> str:
    if not fecha and not hora:
        return ""
    partes = []
    if fecha:
        try:
            d = pd.to_datetime(fecha)
            dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
            meses = ["ene", "feb", "mar", "abr", "may", "jun",
                     "jul", "ago", "sep", "oct", "nov", "dic"]
            partes.append(f"{dias[d.weekday()]} {d.day} {meses[d.month - 1]}")
        except Exception:
            partes.append(str(fecha))
    if hora:
        etiqueta = getattr(config, "TIMEZONE_LABEL", "hora local")
        partes.append(f"{hora} {etiqueta}")
    return " · ".join(partes)


def _team_with_record(name: str, record) -> str:
    rec = fmt(record, default=None)
    if rec and rec != "—":
        return f"{name} ({rec})"
    return name


def _team_md(name: str, record) -> str:
    rec = fmt(record, default=None)
    if rec and rec != "—":
        return f'{name} <span class="record">({rec})</span>'
    return name


def _accuracy_metric(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    valid = df[df[col].notna()]
    if valid.empty:
        return None
    aciertos = sum(1 for v in valid[col] if _bool_value(v) is True)
    total = len(valid)
    if total == 0:
        return None
    return aciertos, total, round(100 * aciertos / total)


def _probability_fraction(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    has_percent = raw.endswith("%")
    try:
        probability = float(raw.rstrip("%").strip())
    except (TypeError, ValueError):
        return None
    if not has_percent and abs(probability) <= 1:
        return min(max(probability, 0.0), 1.0)
    return min(max(probability / 100.0, 0.0), 1.0)


def _normalize_team_name(team: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(team))
    return "".join(
        character.lower()
        for character in normalized
        if character.isalnum() and not unicodedata.combining(character)
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _get_ncaaf_logo_urls() -> dict:
    request = urllib.request.Request(
        "https://site.api.espn.com/apis/site/v2/sports/football/"
        "college-football/teams?limit=1000",
        headers={"User-Agent": "NFL-Predictor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)

    teams = payload["sports"][0]["leagues"][0]["teams"]
    logo_urls = {}
    for entry in teams:
        team = entry.get("team", {})
        logos = team.get("logos", [])
        logo = next(
            (item.get("href") for item in logos if "default" in item.get("rel", [])),
            None,
        )
        if not logo:
            continue
        for name in (
            team.get("location"),
            team.get("displayName"),
            team.get("shortDisplayName"),
            team.get("name"),
            team.get("abbreviation"),
        ):
            if name:
                logo_urls[_normalize_team_name(name)] = logo

    if not logo_urls:
        raise ValueError("ESPN no devolvió logos para equipos colegiales.")
    return logo_urls


def _team_logo_url(team: str, league: str, ncaaf_logos: dict):
    if league == "nfl":
        espn_abbreviations = {
            "JAX": "jac",
            "LA": "lar",
            "WAS": "wsh",
        }
        abbreviation = espn_abbreviations.get(team.upper(), team.lower())
        return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbreviation}.png"
    if league == "ncaaf":
        return ncaaf_logos.get(_normalize_team_name(team))
    return None


def _projected_scores(value, home_team: str, away_team: str):
    match = re.fullmatch(r"\s*(.+?)\s+(\d+)\s+-\s+(\d+)\s+(.+?)\s*", str(value))
    if not match:
        return None
    home_name, home_score, away_score, away_name = match.groups()
    if (
        _normalize_team_name(home_name) != _normalize_team_name(home_team)
        or _normalize_team_name(away_name) != _normalize_team_name(away_team)
    ):
        return None
    return int(home_score), int(away_score)


def _render_result_body(rr: str, ac, acs, act) -> str:
    ac_b = _bool_value(ac)
    if ac_b is True:
        score_class, icon = "hit", "✔"
    elif ac_b is False:
        score_class, icon = "miss", "✘"
    else:
        score_class, icon = "neutral", "•"

    badges = []
    for label, raw in [("ML", ac), ("Spread", acs), ("O/U", act)]:
        val = _bool_value(raw)
        if val is None:
            continue
        if val:
            badges.append(f'<div class="badge badge-hit">✔ {label}</div>')
        else:
            badges.append(f'<div class="badge badge-miss">✘ {label}</div>')

    badges_html = "".join(badges) or '<div class="badge badge-neutral">Sin evaluación</div>'

    return (
        f'<div class="result-score {score_class}">{icon} {rr}</div>'
        f'<div>{badges_html}</div>'
    )


def _implied_prob_from_american(price):
    """Convierte precio americano a probabilidad implícita (0-1)."""
    if price is None or pd.isna(price):
        return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price == 0:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def _market_ml_probs_html(r, home_team: str, away_team: str) -> str:
    """Bloque HTML con probabilidad implícita del ML del mercado + comparación con modelo."""
    ml_h = _get(r, "ml_home", "ml_local")
    ml_a = _get(r, "ml_away", "ml_visita")

    p_h_mkt = _implied_prob_from_american(ml_h)
    p_a_mkt = _implied_prob_from_american(ml_a)

    def _parse_pct(s):
        if s is None or pd.isna(s):
            return None
        try:
            return float(str(s).replace("%", "").strip()) / 100.0
        except (TypeError, ValueError):
            return None

    p_h_mod = _parse_pct(_get(r, "prob_home", "prob_gana_local"))
    p_a_mod = _parse_pct(_get(r, "prob_away", "prob_gana_visita"))

    if p_h_mkt is None and p_a_mkt is None:
        return ""

    parts = ['<div class="ml-line"><b>ML implícito del mercado:</b>']
    if p_h_mkt is not None:
        parts.append(
            f'<span class="sep">·</span> {home_team} '
            f'<span class="v">{p_h_mkt*100:.1f}%</span>'
        )
    if p_a_mkt is not None:
        parts.append(
            f'<span class="sep">·</span> {away_team} '
            f'<span class="v">{p_a_mkt*100:.1f}%</span>'
        )
    parts.append('</div>')

    if p_h_mkt is not None and p_h_mod is not None:
        diff = (p_h_mod - p_h_mkt) * 100
        edge_cls = "edge" if abs(diff) >= 5 else ""
        sign = "+" if diff > 0 else ""
        lado = home_team if diff > 0 else away_team
        parts.append(
            f'<div class="ml-line">Modelo vs mercado (local): '
            f'<span class="v {edge_cls}">{sign}{diff:.1f} pts</span> '
            f'a favor de {lado}</div>'
        )

    return "".join(parts)


# ---------- Records fallback ----------
@st.cache_data(ttl=3600, show_spinner=False)
def _get_records_for(season: int, before_week: int, league: str) -> dict:
    try:
        stats = predict_games._stats_module(league)
        return predict_games._build_records(stats, season, before_week=before_week)
    except Exception:
        return {}


def _resolve_record(row, side: str, season: int, week: int, league: str, fallback_cache: dict) -> str:
    field = f"{side}_record"
    val = _get(row, field)
    if val is not None and not (isinstance(val, float) and pd.isna(val)):
        return val
    team_field = f"{side}_team"
    team = _get(row, team_field)
    if team and fallback_cache:
        return fallback_cache.get(team)
    return None


# ---------- Historial de un equipo ----------
@st.cache_data(ttl=3600, show_spinner=False)
def _get_team_games(season: int, team: str, league: str) -> pd.DataFrame:
    """
    Devuelve SOLO los partidos ya jugados de un equipo en la temporada.
    Columnas: Sem, Cond, Rival, Marcador, Res, Récord.
    """
    try:
        stats = predict_games._stats_module(league)
        sched = stats.get_schedule(season)
    except Exception:
        return pd.DataFrame()

    if sched is None or sched.empty:
        return pd.DataFrame()

    mask = (sched["home_team"] == team) | (sched["away_team"] == team)
    games = sched[mask].copy()
    if games.empty:
        return pd.DataFrame()

    games = games.dropna(subset=["home_score", "away_score"])
    if games.empty:
        return pd.DataFrame()

    if "week" in games.columns:
        games = games.sort_values("week")

    rows = []
    wins = 0
    losses = 0
    ties = 0

    for _, g in games.iterrows():
        is_home = g["home_team"] == team
        opp = g["away_team"] if is_home else g["home_team"]
        hs_i = int(g["home_score"])
        as_i = int(g["away_score"])
        team_pts = hs_i if is_home else as_i
        opp_pts = as_i if is_home else hs_i

        if team_pts > opp_pts:
            res = "G"
            wins += 1
        elif team_pts < opp_pts:
            res = "P"
            losses += 1
        else:
            res = "E"
            ties += 1

        marcador = f"{team_pts}-{opp_pts}"
        running = f"{wins}-{losses}" if ties == 0 else f"{wins}-{losses}-{ties}"

        try:
            wk = int(g.get("week", 0))
        except (TypeError, ValueError):
            wk = 0

        rows.append({
            "Sem": wk,
            "Cond": "Loc" if is_home else "Vis",
            "Rival": opp,
            "Marcador": marcador,
            "Res": res,
            "Récord": running,
        })

    return pd.DataFrame(rows)


def _render_team_history(season: int, team: str, league: str):
    """Contenido del popover: tabla de partidos jugados del equipo."""
    df = _get_team_games(season, team, league)
    if df.empty:
        st.caption(f"Sin partidos jugados para {team} en {season}.")
        return

    w = int((df["Res"] == "G").sum())
    l = int((df["Res"] == "P").sum())
    t = int((df["Res"] == "E").sum())
    rec_txt = f"{w}-{l}" if t == 0 else f"{w}-{l}-{t}"
    st.markdown(f"**Récord {season}:** `{rec_txt}`  ·  Partidos jugados: {len(df)}")

    def _color_res(val):
        if val == "G":
            return "color: #047857; font-weight: 700;"
        if val == "P":
            return "color: #dc2626; font-weight: 700;"
        if val == "E":
            return "color: #6b7280; font-weight: 700;"
        return ""

    try:
        styled = df.style.map(_color_res, subset=["Res"])
    except AttributeError:
        styled = df.style.applymap(_color_res, subset=["Res"])

    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        height=min(400, 40 + 35 * len(df)),
    )


# ---------- Generación desde la app ----------
def _run_generation(season, league, from_week, to_week,
                    skip_odds, refresh_odds, verbose, prediction_mode,
                    log_placeholder, progress_placeholder):
    ok, fail, log_lines = [], [], []
    total = to_week - from_week + 1

    for idx, w in enumerate(range(from_week, to_week + 1), start=1):
        progress_placeholder.progress(
            (idx - 1) / total,
            text=f"Generando {league.upper()} · Temporada {season} · Semana {w} ({idx}/{total})..."
        )

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                success = rebuild_weeks.rebuild_week(
                    season, w, league,
                    skip_odds=skip_odds,
                    refresh_odds=refresh_odds,
                    verbose=verbose,
                    prediction_mode=prediction_mode,
                )
        except Exception as e:
            success = False
            buf.write(f"Excepción: {e}\n")

        captured = buf.getvalue()
        header = f"{'✔' if success else '✘'} SEMANA {w}\n"
        log_lines.append(header + captured + "\n")
        log_placeholder.code("".join(log_lines), language="text")

        (ok if success else fail).append(w)

    progress_placeholder.progress(1.0, text="Listo.")
    return ok, fail


def _preview_generation_week(season, week, league, prediction_mode):
    stats = predict_games._stats_module(league)
    games = stats.get_week_games(season, week)
    saved = db.get_weekly_picks(season, week, league)
    saved_games = set()
    if not saved.empty:
        saved_games = set(zip(saved["home_team"], saved["away_team"]))

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for _, game in games.iterrows():
        home, away = game["home_team"], game["away_team"]
        kickoff = predict_games._kickoff_utc(game, league)
        kickoff_at = pd.to_datetime(kickoff, utc=True, errors="coerce")
        exists = (home, away) in saved_games
        home_score, away_score = game.get("home_score"), game.get("away_score")
        has_score = pd.notna(home_score) and pd.notna(away_score)
        completed_value = game.get("completed")
        completed = (
            bool(completed_value)
            if completed_value is not None and not pd.isna(completed_value)
            else bool(has_score)
        )

        if prediction_mode == "official":
            if exists:
                status = "Ya existe; se conserva el pronóstico oficial"
            elif pd.isna(kickoff_at):
                status = "Kickoff no disponible; no se insertará"
            elif kickoff_at <= now:
                status = "Kickoff iniciado; bloqueado en modo oficial"
            else:
                status = "Listo para pronóstico oficial"
        elif prediction_mode == "update":
            if not exists:
                status = "No hay pronóstico oficial para versionar"
            elif pd.isna(kickoff_at) or kickoff_at <= now:
                status = "Kickoff iniciado o desconocido; actualización bloqueada"
            else:
                status = "Listo para guardar una versión pregame"
        elif exists:
            status = "Ya existe; protegido, no se reemplazará"
        elif pd.isna(kickoff_at):
            status = "Kickoff no disponible; backfill bloqueado"
        elif kickoff_at > now:
            status = "Aún no inicia; fuera de la reconstrucción histórica"
        elif completed and has_score:
            status = "Listo; resultado final disponible para evaluación"
        elif completed:
            status = "Marcado final sin marcador; no se insertará"
        else:
            status = "Listo; se guardará sin marcador parcial"

        rows.append({
            "Partido": f"{away} @ {home}",
            "Kickoff UTC": (
                kickoff_at.strftime("%Y-%m-%d %H:%M")
                if pd.notna(kickoff_at) else "No disponible"
            ),
            "Estado": status,
        })

    market = db.get_market_cache(season, week, league)
    return pd.DataFrame(rows), rebuild_weeks._has_market_lines(market), market.attrs.get(
        "cache_updated_at"
    )


def _preview_ncaaf_flow(season, week, prediction_mode):
    stats = predict_games._stats_module("ncaaf")
    messages = io.StringIO()
    with contextlib.redirect_stdout(messages):
        games = stats.get_week_games(season, week)
        prior_results = stats.get_completed_games(season, before_week=week)

    saved = db.get_weekly_picks(season, week, "ncaaf")
    saved_games = set()
    if not saved.empty:
        saved_games = set(zip(saved["home_team"], saved["away_team"]))

    now = pd.Timestamp.now(tz="UTC")
    preview_rows = []
    for _, game in games.iterrows():
        home, away = game["home_team"], game["away_team"]
        kickoff = predict_games._kickoff_utc(game, "ncaaf")
        kickoff_at = pd.to_datetime(kickoff, utc=True, errors="coerce")
        exists = (home, away) in saved_games
        home_score, away_score = game.get("home_score"), game.get("away_score")
        has_score = pd.notna(home_score) and pd.notna(away_score)
        completed_value = game.get("completed")
        completed = (
            bool(completed_value)
            if completed_value is not None and not pd.isna(completed_value)
            else bool(has_score)
        )

        if prediction_mode == "official":
            if exists:
                status = "Ya está en weekly_picks; no se reemplaza"
            elif pd.isna(kickoff_at):
                status = "Kickoff no disponible; no se insertará"
            elif kickoff_at <= now:
                status = "Kickoff iniciado; bloqueado en modo oficial"
            else:
                status = "Listo para pronóstico oficial"
        elif prediction_mode == "update":
            if not exists:
                status = "No hay oficial para crear una versión"
            elif pd.isna(kickoff_at) or kickoff_at <= now:
                status = "Kickoff iniciado o desconocido; versión bloqueada"
            else:
                status = "Listo para versionar en prediction_versions"
        elif exists:
            status = "Ya está en weekly_picks; protegido"
        elif pd.isna(kickoff_at):
            status = "Kickoff no disponible; reconstrucción bloqueada"
        elif kickoff_at > now:
            status = "Todavía no inicia; fuera del modo histórico"
        elif completed and has_score:
            status = "Falta pick; resultado final disponible"
        elif completed:
            status = "Marcado final sin marcador; requiere revisión"
        else:
            status = "Falta pick; se guarda sin marcador parcial"

        preview_rows.append({
            "Partido": f"{away} @ {home}",
            "Kickoff UTC": (
                kickoff_at.strftime("%Y-%m-%d %H:%M")
                if pd.notna(kickoff_at) else "No disponible"
            ),
            "Estado en Mongo": status,
        })

    market = db.get_market_cache(season, week, "ncaaf")
    ratings = predict_games.elo_model.load_ratings("ncaaf")
    api_key = str(config.CFBD_API_KEY or "").strip()
    api_token = (
        api_key[7:].strip()
        if api_key.lower().startswith("bearer ")
        else api_key
    )
    cfbd_key_configured = bool(
        api_token and api_token not in {"TU_API_KEY_AQUI", "None"}
    )
    market_fields = [
        "consensus_spread_home",
        "consensus_total",
        "consensus_ml_home",
        "consensus_ml_away",
    ]
    market_line_count = (
        int(market[[
            field for field in market_fields if field in market.columns
        ]].notna().any(axis=1).sum())
        if not market.empty and any(field in market.columns for field in market_fields)
        else 0
    )
    return {
        "games": games,
        "preview": pd.DataFrame(preview_rows),
        "provider": games.attrs.get("provider", "No informado por el proveedor"),
        "provider_fallback_reason": games.attrs.get("provider_fallback_reason"),
        "invalid_game_count": int(games.attrs.get("invalid_game_count", 0)),
        "provider_messages": messages.getvalue().strip(),
        "cfbd_key_configured": cfbd_key_configured,
        "prior_result_count": len(prior_results),
        "rating_team_count": len(ratings),
        "saved_pick_count": len(saved),
        "market_line_count": market_line_count,
        "market_updated_at": market.attrs.get("cache_updated_at"),
    }


def _render_match_card(r, season, week, league, records_fallback, ncaaf_logos):
    away_team, home_team = r["partido"].split(" @ ")
    home_rec = _resolve_record(r, "home", season, week, league, records_fallback)
    away_rec = _resolve_record(r, "away", season, week, league, records_fallback)
    away_title = _team_with_record(away_team, away_rec)
    home_title = _team_with_record(home_team, home_rec)

    dt_label = fmt_datetime(r.get("fecha"), r.get("hora"))
    cap = f"{fmt(r.get('liga'))} · Semana {fmt(r.get('semana'))}"
    if dt_label:
        cap = f"{cap} · {dt_label}"

    prob_a = _get(r, "prob_away", "prob_gana_visita")
    prob_h = _get(r, "prob_home", "prob_gana_local")
    fraction_a = _probability_fraction(prob_a)
    fraction_h = _probability_fraction(prob_h)
    display_fraction_a = (
        fraction_a if fraction_a is not None
        else 1 - fraction_h if fraction_h is not None
        else 0.5
    )
    display_fraction_h = 1 - display_fraction_a
    away_probability = (
        f"{fraction_a:.0%}" if fraction_a is not None else fmt(prob_a)
    )
    home_probability = (
        f"{fraction_h:.0%}" if fraction_h is not None else fmt(prob_h)
    )
    rr = r.get("resultado_real")
    model_spread = fmt(r.get("spread_modelo"))
    model_total = fmt(r.get("total_modelo"))
    market_spread_line = _get(r, "market_spread_line")
    market_total_line = _get(r, "market_total_line")
    projected_scores = _projected_scores(
        r.get("marcador_proyectado"), home_team, away_team
    )
    projected_spread_line = r.get("spread_model_line")
    evaluated_spread = r.get("acierto_spread")
    if projected_scores is not None:
        projected_home_score, projected_away_score = projected_scores
        projected_spread_line = predict_games._spread_line_from_projected_score(
            projected_home_score, projected_away_score
        )
        model_spread = predict_games.format_spread(
            home_team, away_team, projected_spread_line
        )
        model_total = str(projected_home_score + projected_away_score)
        try:
            actual_margin = int(r["home_score_real"]) - int(r["away_score_real"])
            if projected_spread_line == 0:
                evaluated_spread = None
            else:
                spread_diff = (
                    actual_margin - projected_spread_line
                    if projected_spread_line > 0
                    else projected_spread_line - actual_margin
                )
                evaluated_spread = None if spread_diff == 0 else spread_diff > 0
        except (KeyError, TypeError, ValueError):
            pass
    cons_spread = fmt(r.get("spread_consenso"))
    if cons_spread == "—":
        market_spread_line = _get(r, "market_spread_line")
        if market_spread_line is not None:
            try:
                market_spread_line = float(market_spread_line)
                if market_spread_line > 0:
                    cons_spread = f"{home_team} -{abs(market_spread_line):g}"
                elif market_spread_line < 0:
                    cons_spread = f"{away_team} -{abs(market_spread_line):g}"
                else:
                    cons_spread = "Pick'em (0)"
            except (TypeError, ValueError):
                pass

    cons_total = fmt(r.get("total_consenso"))
    if cons_total == "—":
        market_total_line = _get(r, "market_total_line")
        if market_total_line is not None:
            try:
                cons_total = f"{float(market_total_line):g}"
            except (TypeError, ValueError):
                pass
    ml_h = _get(r, "ml_home", "ml_local")
    ml_a = _get(r, "ml_away", "ml_visita")

    def _value_row(label, value):
        return (
            '<div class="scoreboard-value-row">'
            f'<span>{html.escape(label)}</span>'
            f'<strong>{html.escape(str(value))}</strong>'
            '</div>'
        )

    with st.container(border=True):
        retrospective = r.get("retrospective")
        is_retrospective = pd.notna(retrospective) and bool(retrospective)
        game_completed = r.get("game_completed")
        final_result_pending = (
            is_retrospective
            and pd.notna(game_completed)
            and not bool(game_completed)
        )
        if rr and pd.notna(rr):
            status_html = (
                '<div class="scoreboard-status final">'
                f'FINAL · {html.escape(str(rr))}</div>'
            )
        elif final_result_pending:
            status_html = (
                '<div class="scoreboard-status">RESULTADO FINAL PENDIENTE</div>'
            )
        else:
            status_html = (
                '<div class="scoreboard-status">PARTIDO PENDIENTE</div>'
            )
        if is_retrospective:
            status_html += (
                '<div class="scoreboard-status retrospective">'
                'RECONSTRUCCIÓN · DESPUÉS DEL KICKOFF</div>'
            )

        def _team_panel(team, record, side, side_key):
            logo_url = _team_logo_url(team, league, ncaaf_logos)
            if logo_url:
                logo_html = (
                    f'<img src="{html.escape(logo_url, quote=True)}" '
                    f'alt="Logo de {html.escape(team, quote=True)}">'
                )
            else:
                logo_html = f'<span>{html.escape(team)}</span>'
            record_text = fmt(record, default="")
            record_html = (
                f'<div class="scoreboard-team-record">'
                f'{html.escape(str(record_text))}</div>'
                if record_text and record_text != "—"
                else ""
            )
            return (
                f'<div class="scoreboard-team-panel {side_key}">'
                f'<div class="scoreboard-team-kicker">{side}</div>'
                f'<div class="scoreboard-team-logo">{logo_html}</div>'
                f'<div class="scoreboard-team-name">{html.escape(team)}</div>'
                f'{record_html}</div>'
            )

        scores = projected_scores
        if scores is None:
            scoreline = html.escape(fmt(r.get("marcador_proyectado")))
        else:
            home_score, away_score = scores
            scoreline = (
                f'<span>{away_score}</span>'
                '<span class="scoreboard-score-separator">–</span>'
                f'<span>{home_score}</span>'
            )
        matchup_html = (
            '<div class="scoreboard-topline">'
            f'<div class="scoreboard-meta">{html.escape(cap)}</div>'
            f'{status_html}</div>'
            '<div class="scoreboard-matchup">'
            f'{_team_panel(away_team, away_rec, "VISITANTE", "away")}'
            '<div class="scoreboard-projection">'
            '<div class="scoreboard-projection-label">MARCADOR PROYECTADO</div>'
            f'<div class="scoreboard-scoreline">{scoreline}</div>'
            '</div>'
            f'{_team_panel(home_team, home_rec, "LOCAL", "home")}'
            '</div>'
        )
        st.markdown(matchup_html, unsafe_allow_html=True)

        history_cols = st.columns(2)
        with history_cols[0]:
            with st.popover("Historial visitante", use_container_width=True):
                st.markdown(f"**{away_title}**")
                _render_team_history(season, away_team, league)
        with history_cols[1]:
            with st.popover("Historial local", use_container_width=True):
                st.markdown(f"**{home_title}**")
                _render_team_history(season, home_team, league)

        probability_html = (
            '<div class="scoreboard-probability">'
            '<div class="scoreboard-probability-labels">'
            f'<span class="away">{html.escape(away_team)} · '
            f'{html.escape(away_probability)} victoria</span>'
            f'<span class="home">{html.escape(home_team)} · '
            f'{html.escape(home_probability)} victoria</span>'
            '</div><div class="scoreboard-probability-track">'
            f'<span class="away" style="width:{display_fraction_a * 100:.2f}%"></span>'
            f'<span class="home" style="width:{display_fraction_h * 100:.2f}%"></span>'
            '</div></div>'
        )
        st.markdown(probability_html, unsafe_allow_html=True)

        values_html = (
            '<div class="scoreboard-values">'
            '<section class="scoreboard-value-panel">'
            '<div class="scoreboard-value-title">Pronóstico del modelo</div>'
            f'{_value_row("Spread", model_spread)}'
            f'{_value_row("Total", model_total)}'
            '</section>'
            '<section class="scoreboard-value-panel market">'
            '<div class="scoreboard-value-title">Líneas de mercado</div>'
            f'{_value_row("Spread", cons_spread)}'
            f'{_value_row("Total", cons_total)}'
            f'{_value_row(f"ML visitante · {away_team}", fmt(ml_a))}'
            f'{_value_row(f"ML local · {home_team}", fmt(ml_h))}'
            '</section></div>'
        )
        st.markdown(values_html, unsafe_allow_html=True)

        if rr and pd.notna(rr):
            result_content = _render_result_body(
                html.escape(str(rr)),
                r.get("acierto_ganador"),
                evaluated_spread,
                r.get("acierto_total"),
            )
        else:
            result_content = (
                '<div class="scoreboard-pending-result">'
                'ESPERANDO EL RESULTADO FINAL'
                '</div>'
                if final_result_pending else
                '<div class="scoreboard-pending-result">POR JUGARSE</div>'
            )
        st.markdown(
            '<div class="scoreboard-result-panel">'
            '<div class="scoreboard-result-title">Resultado y evaluación</div>'
            f'{result_content}</div>',
            unsafe_allow_html=True,
        )

        tags = []
        value_labels = predict_games._market_value_labels(
            home_team,
            away_team,
            projected_spread_line,
            sum(projected_scores) if projected_scores is not None else r.get("total_modelo"),
            market_spread_line,
            market_total_line,
        )

        for label, field in [("Spread", "valor_spread"), ("Over/Under", "valor_total")]:
            val = value_labels.get(field, r.get(field))
            if val and pd.notna(val):
                cls = classify_value(val)
                tags.append(f'<span class="tag {cls}">{label}: {val}</span>')
        if tags:
            st.markdown(" ".join(tags), unsafe_allow_html=True)

        with st.expander("Ver análisis del modelo, mercado y resultado"):
            st.markdown("**📊 Pronóstico**")
            st.markdown(
                f"- Prob. victoria: {_team_md(home_team, home_rec)} `{fmt(prob_h)}` · "
                f"{_team_md(away_team, away_rec)} `{fmt(prob_a)}`",
                unsafe_allow_html=True,
            )
            st.markdown(f"- Spread: `{model_spread}`")
            st.markdown(f"- Over/Under: `{model_total}`")
            st.markdown("**💹 Mercado**")

            odds_updated = r.get("odds_cache_updated_at")
            if odds_updated is not None and pd.notna(odds_updated):
                st.caption(f"Cuotas guardadas: {fmt(odds_updated)}")
            elif all(
                fmt(_get(r, col)) == "—"
                for col in ("spread_consenso", "total_consenso", "ml_home", "ml_away")
            ):
                st.caption("Sin cuotas disponibles; predicción generada solo con el modelo.")

            st.markdown(f"- Spread: `{cons_spread}`")

            st.markdown(f"- Over/Under: `{cons_total}`")

            if ml_h is not None or ml_a is not None:
                st.markdown(
                    f"- ML {_team_md(home_team, home_rec)}: `{fmt(ml_h)}` · "
                    f"ML {_team_md(away_team, away_rec)}: `{fmt(ml_a)}`",
                    unsafe_allow_html=True,
                )

            ml_probs_html = _market_ml_probs_html(r, home_team, away_team)
            if ml_probs_html:
                st.markdown(ml_probs_html, unsafe_allow_html=True)

            st.markdown("**🏁 Resultado**")
            if rr and pd.notna(rr):
                st.markdown(
                    _render_result_body(
                        rr,
                        r.get("acierto_ganador"),
                        r.get("acierto_spread"),
                        r.get("acierto_total"),
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-score neutral">Por jugarse</div>',
                    unsafe_allow_html=True,
                )

            ajustes = r.get("ajustes")
            if ajustes and pd.notna(ajustes):
                st.caption(f"🔧 Ajustes: {ajustes}")


# ---------- Sidebar ----------
st.sidebar.title("🏈 NFL / NCAAF Predictor")

league_options = {"nfl": "NFL", "ncaaf": "NCAAF (Colegial)"}
league_choice = st.sidebar.selectbox(
    "Liga",
    options=list(league_options.keys()),
    format_func=lambda x: league_options[x],
)

season = st.sidebar.number_input(
    "Temporada", min_value=2015, max_value=2030,
    value=config.CURRENT_SEASON, step=1,
)

try:
    available_weeks = db.list_available_weeks(season, league_choice)
except Exception as e:
    st.sidebar.error(f"No se pudo conectar a MongoDB: {e}")
    st.error(f"❌ Error de conexión a MongoDB: {e}")
    st.stop()

week = None
only_value = False

if available_weeks:
    sorted_weeks = sorted(available_weeks)
    last_week = sorted_weeks[-1]

    context_key = f"{league_choice}_{season}"
    if st.session_state.get("_week_context") != context_key:
        st.session_state["_week_context"] = context_key
        st.session_state["_selected_week"] = last_week

    if st.session_state.get("_selected_week") not in sorted_weeks:
        st.session_state["_selected_week"] = last_week

    week = st.sidebar.selectbox("Semana", options=sorted_weeks, key="_selected_week")
    only_value = st.sidebar.checkbox("Mostrar solo con señal de valor", value=False)
else:
    st.sidebar.info(
        f"Sin datos para {league_options[league_choice]} {season}. "
        "Usa **Generar NFL** o **Generar NCAAF** para crear la primera semana."
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"BD: `{config.MONGO_DB_NAME}`")
if available_weeks:
    st.sidebar.caption(f"Semanas existentes: {sorted(available_weeks)}")
else:
    st.sidebar.caption("Semanas existentes: —")


# ---------- Cabecera ----------
league_label = league_options[league_choice]
if week is not None:
    st.title(f"{league_label} · Temporada {season} · Semana {week}")
else:
    st.title(f"{league_label} · Temporada {season}")


# ---------- Tabs ----------
tab_picks, tab_props, tab_nfl_gen, tab_ncaaf_flow = st.tabs(
    [
        "🏟️ Partidos",
        "⭐ Props de jugadores",
        "🏈 Generar NFL",
        "🏈 Generar NCAAF",
    ]
)


# ---------- Carga de datos ----------
if week is not None:
    picks_df = db.get_weekly_picks(season, week, league_choice)
    props_df = db.get_player_props(season, week, league_choice)
else:
    picks_df = pd.DataFrame()
    props_df = pd.DataFrame()

picks_full_df = picks_df.copy()

if week is not None and only_value and not picks_df.empty:
    mask = pd.Series(False, index=picks_df.index)
    for col in ("valor_spread", "valor_total"):
        if col in picks_df.columns:
            mask = mask | picks_df[col].apply(has_real_value)
    picks_df = picks_df[mask].reset_index(drop=True)

records_fallback = {}
if week is not None and not picks_df.empty:
    need_fallback = (
        ("home_record" not in picks_df.columns) or
        ("away_record" not in picks_df.columns) or
        (picks_df.get("home_record", pd.Series(dtype=object)).isna().all())
    )
    if need_fallback:
        with st.spinner("Calculando records…"):
            records_fallback = _get_records_for(season, week, league_choice)


# ---------- Métricas de cabecera ----------
if week is not None and not picks_full_df.empty:
    with st.expander("📊 Resumen de la semana", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Partidos mostrados", len(picks_df))
        with c2:
            st.metric("Señales en spread", count_value_signals(picks_df, "valor_spread"))
        with c3:
            st.metric("Señales en O/U", count_value_signals(picks_df, "valor_total"))
        with c4:
            acc = _accuracy_metric(picks_full_df, "acierto_ganador")
            if acc is None:
                st.metric("Aciertos ML", "—")
            else:
                a, t, p = acc
                st.metric("Aciertos ML", f"{a}/{t}", delta=f"{p}%")
        with c5:
            acc = _accuracy_metric(picks_full_df, "acierto_spread")
            if acc is None:
                st.metric("Aciertos spread", "—")
            else:
                a, t, p = acc
                st.metric("Aciertos spread", f"{a}/{t}", delta=f"{p}%")
        with c6:
            acc = _accuracy_metric(picks_full_df, "acierto_total")
            if acc is None:
                st.metric("Aciertos O/U", "—")
            else:
                a, t, p = acc
                st.metric("Aciertos O/U", f"{a}/{t}", delta=f"{p}%")

        st.caption(
            "**ML** = acertó quién gana. "
            "**Spread** = el resultado cubrió el spread que dio el modelo. "
            "**O/U** = el total real cayó del lado que el modelo eligió respecto a la línea del mercado."
        )


# --- TAB: Partidos ---
with tab_picks:
    if week is None:
        st.info(
            f"No hay datos para **{league_label} {season}**. "
            f"Ve a **Generar {'NFL' if league_choice == 'nfl' else 'NCAAF'}** "
            "para crear la primera semana."
        )
    elif picks_df.empty:
        st.warning("No hay partidos guardados para esta semana (o el filtro los ocultó).")
    else:
        display_picks_df = picks_df
        if "fecha" in display_picks_df.columns:
            display_picks_df = display_picks_df.sort_values(
                by=["fecha", "hora"], na_position="last", kind="stable"
            ).reset_index(drop=True)

        has_slots = "slot" in display_picks_df.columns and display_picks_df["slot"].notna().any()
        slot_options = (
            display_picks_df["slot"].dropna().astype(str).drop_duplicates().tolist()
            if has_slots else []
        )
        selected_slot = "Todos"
        if slot_options:
            with st.container(border=True):
                st.markdown(
                    '<div class="matchday-filter-kicker">CENTRO DE PARTIDOS</div>',
                    unsafe_allow_html=True,
                )
                st.subheader("Marcador de la semana")
                st.caption(
                    "Selecciona un horario para filtrar los enfrentamientos. "
                    "Abre cada partido para consultar modelo, mercado y resultado."
                )
                selected_slot = st.radio(
                    "Horario",
                    options=["Todos", *slot_options],
                    horizontal=True,
                    key=f"match_slot_{league_choice}_{season}_{week}",
                    label_visibility="collapsed",
                )
            if selected_slot != "Todos":
                display_picks_df = display_picks_df[
                    display_picks_df["slot"].astype(str) == selected_slot
                ].reset_index(drop=True)
            st.caption(f"Mostrando {len(display_picks_df)} de {len(picks_df)} partidos.")

        ncaaf_logos = {}
        if league_choice == "ncaaf":
            try:
                ncaaf_logos = _get_ncaaf_logo_urls()
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                ValueError,
            ) as error:
                st.warning(
                    "No se pudieron cargar los logos de equipos colegiales; "
                    f"se mostrarán sus nombres. Detalle: {error}"
                )

        for _, r in display_picks_df.iterrows():
            _render_match_card(
                r, season, week, league_choice, records_fallback, ncaaf_logos
            )

        marcados = False
        for col in ("valor_spread", "valor_total"):
            if col in picks_df.columns and picks_df[col].astype(str).str.contains("⚠").any():
                marcados = True
        if marcados:
            st.warning(
                "⚠ = el modelo y el mercado difieren mucho. Revísalo antes de confiar en esa señal."
            )

    if week is not None:
        versions_df = db.get_prediction_versions(season, week, league_choice)
        if not versions_df.empty:
            st.markdown("### Historial de actualizaciones prepartido")
            st.caption(
                "Estas versiones son independientes. La predicción oficial de cada partido permanece intacta."
            )
            visible_columns = [
                col for col in [
                    "partido", "version", "version_created_at", "marcador_proyectado",
                    "prob_home", "spread_modelo", "total_modelo", "ajustes",
                ] if col in versions_df.columns
            ]
            st.dataframe(
                versions_df[visible_columns],
                use_container_width=True,
                hide_index=True,
            )


# --- TAB: Props ---
with tab_props:
    if week is None:
        st.info(
            "Primero genera alguna semana desde la pestaña "
            f"**Generar {'NFL' if league_choice == 'nfl' else 'NCAAF'}**."
        )
    elif props_df.empty:
        st.info("No hay props guardadas para esta semana.")
    else:
        st.caption(f"{len(props_df)} props encontradas.")

        col_cfg = {}
        for col, label in [
            ("prob_modelo_over", "P(modelo over)"),
            ("prob_implicita_over", "P(implícita over)"),
            ("edge_over", "Edge over"),
            ("edge_under", "Edge under"),
        ]:
            if col in props_df.columns:
                col_cfg[col] = st.column_config.NumberColumn(label, format="%.3f")

        st.dataframe(
            props_df,
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
        )


# --- TAB: Generar NFL ---
with tab_nfl_gen:
    st.subheader("NFL · Generar predicciones")
    st.caption(
        "Generación NFL independiente del flujo NCAAF. Revisa una semana antes "
        "de ejecutar el pipeline."
    )
    st.markdown(
        "**1. Revisa** el calendario y los registros guardados. "
        "**2. Genera** solo después de comprobar el resumen. "
        "**3. Sincroniza** resultados finales por separado."
    )
    last_nfl_run = st.session_state.get("nfl_gen_last_run")
    if last_nfl_run:
        run_state = "completada" if last_nfl_run["success"] else "con errores"
        st.caption(
            f"Última ejecución NFL: temporada {last_nfl_run['season']} · "
            f"semana {last_nfl_run['week']} · {last_nfl_run['mode']} · {run_state}."
        )

    st.session_state.setdefault("nfl_gen_season", int(config.CURRENT_SEASON))
    nfl_gen_season = st.number_input(
        "Temporada NFL",
        min_value=2015,
        max_value=2030,
        step=1,
        key="nfl_gen_season",
    )
    st.session_state.setdefault("nfl_gen_week", 1)
    nfl_gen_week = st.number_input(
        "Semana NFL",
        min_value=1,
        max_value=22,
        step=1,
        key="nfl_gen_week",
    )
    nfl_mode_label = st.radio(
        "Tipo de predicción",
        options=[
            "Oficial (crear una sola vez)",
            "Actualizar antes del partido (guardar versión)",
            "Reconstrucción histórica (después del kickoff)",
        ],
        key="nfl_gen_mode",
        help=(
            "La reconstrucción solo agrega partidos que ya iniciaron y no tienen "
            "registro; no reemplaza picks existentes."
        ),
    )
    nfl_prediction_mode = (
        "update" if nfl_mode_label.startswith("Actualizar")
        else "historical" if nfl_mode_label.startswith("Reconstrucción")
        else "official"
    )
    if nfl_prediction_mode == "historical":
        st.info(
            "Modo retrospectivo: no usa cuotas actuales; solo acepta caché anterior "
            "al kickoff y no guarda marcadores parciales."
        )
        nfl_skip_odds = True
        nfl_refresh_odds = False
    else:
        fetch_missing_market = st.checkbox(
            "Obtener también líneas de mercado automáticamente",
            value=True,
            key="nfl_gen_fetch_missing_market",
            help=(
                "Usa primero las líneas guardadas. Si no hay caché utilizable, "
                "consulta The Odds API; puede consumir cuota."
            ),
        )
        nfl_skip_odds = not fetch_missing_market
        nfl_refresh_odds = False
        if fetch_missing_market:
            st.caption(
                "Se usará la caché si existe; si faltan líneas, se consultará "
                "The Odds API (máximo una llamada para esta semana)."
            )

    nfl_review_token = (
        int(nfl_gen_season),
        int(nfl_gen_week),
        nfl_prediction_mode,
        bool(nfl_skip_odds),
    )
    nfl_review_clicked = st.button(
        "1. Revisar semana (solo lectura)",
        key="nfl_gen_review_button",
        use_container_width=True,
    )
    if nfl_review_clicked:
        st.session_state["nfl_gen_preview"] = None
        st.session_state["nfl_gen_last_run"] = None
        try:
            with st.spinner("Revisando calendario y picks existentes..."):
                preview_df, has_cached_market, market_updated_at = (
                    _preview_generation_week(
                        int(nfl_gen_season), int(nfl_gen_week), "nfl",
                        nfl_prediction_mode,
                    )
                )
            st.session_state["nfl_gen_preview_token"] = nfl_review_token
            st.session_state["nfl_gen_preview"] = preview_df
            st.session_state["nfl_gen_market_available"] = has_cached_market
            st.session_state["nfl_gen_market_updated_at"] = market_updated_at
        except Exception as exc:
            st.error(f"No se pudo revisar la semana: {exc}")

    nfl_preview_is_current = (
        st.session_state.get("nfl_gen_preview_token") == nfl_review_token
        and st.session_state.get("nfl_gen_preview") is not None
    )
    if nfl_preview_is_current:
        preview_df = st.session_state["nfl_gen_preview"]
        st.markdown("#### Resumen de la revisión")
        if preview_df.empty:
            st.warning("El calendario no devolvió partidos para esta semana.")
        else:
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(preview_df)} partidos en el calendario · "
                f"{int(preview_df['Estado'].str.startswith('Listo').sum())} "
                "marcados como listos para este modo."
            )
        if st.session_state.get("nfl_gen_market_available"):
            st.caption(
                "Hay líneas en la caché de mercado. Actualizada: "
                f"{fmt(st.session_state.get('nfl_gen_market_updated_at'))}."
            )
        else:
            if nfl_prediction_mode == "historical":
                st.caption(
                    "No hay líneas utilizables en caché; esta reconstrucción "
                    "histórica continuará solo con el modelo."
                )
            elif nfl_skip_odds:
                st.caption(
                    "No hay líneas utilizables en caché; se generará solo "
                    "el pronóstico del modelo."
                )
            else:
                st.caption(
                    "No hay líneas utilizables en caché; al generar se consultará "
                    "The Odds API para intentar obtenerlas."
                )
        st.caption(
            "La revisión no genera proyecciones ni escribe en MongoDB. "
            "La generación se ejecuta únicamente con el botón siguiente."
        )

    can_generate_nfl = bool(
        nfl_preview_is_current
        and not st.session_state["nfl_gen_preview"].empty
    )
    generate_nfl_clicked = st.button(
        f"2. Generar NFL · semana {int(nfl_gen_week)}",
        key="nfl_gen_execute_button",
        type="primary",
        disabled=not can_generate_nfl,
        help=(
            "Primero pulsa «1. Revisar semana» y confirma que el calendario "
            "muestre partidos."
            if not can_generate_nfl
            else "Genera únicamente esta semana de NFL con el modo seleccionado."
        ),
        use_container_width=True,
    )
    if generate_nfl_clicked:
        st.session_state["nfl_gen_preview_token"] = None
        progress_placeholder = st.empty()
        log_placeholder = st.empty()
        with st.spinner("Generando la semana seleccionada..."):
            generated, failed = _run_generation(
                int(nfl_gen_season),
                "nfl",
                int(nfl_gen_week),
                int(nfl_gen_week),
                skip_odds=nfl_skip_odds,
                refresh_odds=nfl_refresh_odds,
                verbose=True,
                prediction_mode=nfl_prediction_mode,
                log_placeholder=log_placeholder,
                progress_placeholder=progress_placeholder,
            )
        if generated:
            st.success(
                f"NFL temporada {int(nfl_gen_season)}, semana {int(nfl_gen_week)} "
                "procesada. Revisa el resumen de guardado "
                "en la salida anterior."
            )
        if failed:
            st.error(f"No se pudo completar la semana NFL {int(nfl_gen_week)}.")
        st.session_state["nfl_gen_last_run"] = {
            "week": int(nfl_gen_week),
            "season": int(nfl_gen_season),
            "mode": nfl_prediction_mode,
            "success": bool(generated) and not bool(failed),
        }
        st.info(
            "Para volver a ejecutar, revisa la semana otra vez. Los modos oficial "
            "e histórico protegen registros existentes; las actualizaciones crean "
            "una nueva versión por cada ejecución aceptada."
        )

    last_nfl_run = st.session_state.get("nfl_gen_last_run")
    can_sync_nfl_results = bool(
        last_nfl_run
        and last_nfl_run["success"]
        and last_nfl_run["season"] == int(nfl_gen_season)
        and last_nfl_run["week"] == int(nfl_gen_week)
    )
    sync_nfl_results = st.button(
        f"3. Sincronizar resultados NFL · semana {int(nfl_gen_week)}",
        key="nfl_gen_sync_button",
        disabled=not can_sync_nfl_results,
        help=(
            "Actualiza únicamente resultados finales; no vuelve a calcular "
            "proyecciones. Se habilita cuando termina correctamente el paso 2 "
            "para esta temporada y semana NFL."
        ),
        use_container_width=True,
    )
    if sync_nfl_results:
        with st.spinner("Consultando resultados finales..."):
            sync_summary = rebuild_weeks.sync_final_results(
                int(nfl_gen_season), int(nfl_gen_week), "nfl"
            )
        st.success(
            f"Resultados actualizados: {sync_summary['updated']} partidos. "
            f"Sin predicción coincidente: {sync_summary['not_found']}."
        )


# --- TAB: Flujo NCAAF ---
with tab_ncaaf_flow:
    st.subheader("NCAAF · Generar paso a paso")
    st.caption(
        "Este flujo es independiente del generador NFL. Revisa calendario y registros "
        "antes de ejecutar; después verifica lo que quedó guardado en MongoDB."
    )
    st.markdown(
        "**Orden:** calendario y resultados (`ncaaf_fetch.py`) → ratings y proyección "
        "(`elo_model.py`, `team_stats.py`, `predict_games.py`) → mercado "
        "(`data_fetch.py`) → guardado y verificación (`db.py`) → resultado final."
    )

    flow_col_a, flow_col_b = st.columns(2)
    with flow_col_a:
        st.session_state.setdefault("ncaaf_flow_season", int(config.CURRENT_SEASON))
        flow_season = st.number_input(
            "Temporada NCAAF",
            min_value=2015,
            max_value=2030,
            step=1,
            key="ncaaf_flow_season",
        )
    with flow_col_b:
        st.session_state.setdefault(
            "ncaaf_flow_week",
            1,
        )
        flow_week = st.number_input(
            "Semana NCAAF",
            min_value=1,
            max_value=25,
            step=1,
            key="ncaaf_flow_week",
        )

    flow_mode_label = st.radio(
        "Modo de guardado",
        options=[
            "Oficial (una vez, antes del kickoff)",
            "Actualizar (versión pregame separada)",
            "Histórico (partidos tras kickoff faltantes)",
        ],
        horizontal=True,
        key="ncaaf_flow_mode",
    )
    if flow_mode_label.startswith("Actualizar"):
        flow_mode = "update"
    elif flow_mode_label.startswith("Histórico"):
        flow_mode = "historical"
    else:
        flow_mode = "official"

    if flow_mode == "historical":
        flow_skip_odds = True
        st.info(
            "Modo histórico: consulta solo la caché; líneas posteriores al kickoff "
            "se descartan."
        )
    else:
        fetch_flow_market = st.checkbox(
            "Si faltan líneas en caché, consultar The Odds API",
            value=True,
            key="ncaaf_flow_fetch_market",
            help="Puede consumir cuota de The Odds API; se hace como máximo una llamada por semana.",
        )
        flow_skip_odds = not fetch_flow_market

    flow_token = (
        int(flow_season),
        int(flow_week),
        flow_mode,
        bool(flow_skip_odds),
    )
    check_flow_clicked = st.button(
        "1. Revisar entradas y registros (solo lectura)",
        key="ncaaf_flow_review_button",
        use_container_width=True,
    )
    if check_flow_clicked:
        st.session_state["ncaaf_flow_preview"] = None
        st.session_state["ncaaf_flow_run"] = None
        st.session_state["ncaaf_flow_sync"] = None
        try:
            with st.spinner("Leyendo calendario NCAAF, historial y MongoDB..."):
                flow_preview = _preview_ncaaf_flow(
                    int(flow_season), int(flow_week), flow_mode
                )
            st.session_state["ncaaf_flow_preview"] = flow_preview
            st.session_state["ncaaf_flow_preview_token"] = flow_token
        except Exception as exc:
            st.session_state["ncaaf_flow_preview_token"] = None
            st.error(f"Falló la revisión de entradas: {type(exc).__name__}: {exc}")

    flow_preview_is_current = (
        st.session_state.get("ncaaf_flow_preview_token") == flow_token
        and st.session_state.get("ncaaf_flow_preview") is not None
    )
    if flow_preview_is_current:
        flow_preview = st.session_state["ncaaf_flow_preview"]
        flow_games = flow_preview["games"]
        flow_preview_df = flow_preview["preview"]

        st.markdown("#### 1 · Calendario y origen")
        source_col, key_col, results_col, ratings_col, picks_col = st.columns(5)
        source_col.metric("Proveedor", flow_preview["provider"])
        key_col.metric(
            "Clave CFBD",
            "Configurada" if flow_preview["cfbd_key_configured"] else "Falta",
        )
        results_col.metric(
            "Partidos previos esta temporada",
            flow_preview["prior_result_count"],
        )
        ratings_col.metric("Equipos con rating Elo", flow_preview["rating_team_count"])
        picks_col.metric(
            "Registros en `weekly_picks`",
            flow_preview["saved_pick_count"],
        )
        if flow_preview["provider_fallback_reason"]:
            st.warning(
                "Se usó ESPN como respaldo. CFBD reportó: "
                f"{flow_preview['provider_fallback_reason']}."
            )
        if flow_preview["invalid_game_count"]:
            st.warning(
                f"Se descartaron {flow_preview['invalid_game_count']} filas del "
                "calendario porque faltaban equipos o número de semana."
            )
        if not flow_preview["cfbd_key_configured"]:
            st.error(
                "Falta `CFBD_API_KEY`; CFBD no podrá responder. La app intentará "
                "usar ESPN como calendario de respaldo."
            )
        if flow_preview["provider_messages"]:
            st.code(flow_preview["provider_messages"], language="text")
        if flow_games.empty:
            st.error(
                "No hay partidos válidos para esta semana. Revisa el proveedor y "
                "los datos descartados indicados arriba. "
                "La generación queda bloqueada y no escribirá en MongoDB."
            )
        else:
            st.dataframe(
                flow_preview_df,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 2 · Mercado")
        if flow_preview["market_line_count"]:
            st.success(
                f"La caché de `market_cache` contiene líneas para "
                f"{flow_preview['market_line_count']} partidos. "
                f"Actualizada: {fmt(flow_preview['market_updated_at'])}."
            )
        elif flow_mode == "historical" or flow_skip_odds:
            st.warning(
                "No hay líneas utilizables en caché. Con la selección actual no se "
                "consultará la API; el modelo podrá generar sin mercado."
            )
        else:
            st.info(
                "No hay líneas utilizables en caché. Al generar, el flujo intentará "
                "consultar The Odds API y mostrará el resultado en el registro."
            )

        st.markdown("#### 3 · Proyección y escritura")
        st.caption(
            "El botón siguiente ejecuta `rebuild_weeks.rebuild_week`, que actualiza "
            "Elo, carga las estadísticas anteriores a la semana, construye la proyección "
            "y combina resultados de esta temporada con el prior de la anterior, "
            "según el modelo existente. Después guarda según el modo seleccionado y vuelve a leer "
            "`weekly_picks` para verificar lo persistido."
        )

    previous_ncaaf_run = st.session_state.get("ncaaf_flow_run")
    flow_run_attempted = bool(
        previous_ncaaf_run and previous_ncaaf_run.get("token") == flow_token
    )
    can_run_ncaaf = bool(
        flow_preview_is_current
        and not st.session_state["ncaaf_flow_preview"]["games"].empty
        and not flow_run_attempted
    )
    run_ncaaf_clicked = st.button(
        f"2. Generar NCAAF · semana {int(flow_week)}",
        key="ncaaf_flow_run_button",
        type="primary",
        disabled=not can_run_ncaaf,
        help=(
            "Esta revisión ya ejecutó el flujo; vuelve a revisar para habilitar otro intento."
            if flow_run_attempted
            else (
                "Primero completa el paso 1 y confirma que haya partidos en el calendario."
                if not can_run_ncaaf
                else "Ejecuta una vez por revisión; vuelve a revisar para habilitar otro intento."
            )
        ),
        use_container_width=True,
    )
    if run_ncaaf_clicked:
        output = io.StringIO()
        try:
            with st.spinner("Ejecutando el pipeline NCAAF..."):
                with contextlib.redirect_stdout(output):
                    success = rebuild_weeks.rebuild_week(
                        int(flow_season),
                        int(flow_week),
                        "ncaaf",
                        skip_odds=flow_skip_odds,
                        refresh_odds=False,
                        verbose=True,
                        prediction_mode=flow_mode,
                    )
            saved_after = db.get_weekly_picks(
                int(flow_season), int(flow_week), "ncaaf"
            )
            st.session_state["ncaaf_flow_run"] = {
                "token": flow_token,
                "success": bool(success),
                "saved_after": saved_after,
                "output": output.getvalue(),
            }
        except Exception as exc:
            st.session_state["ncaaf_flow_run"] = {
                "token": flow_token,
                "success": False,
                "saved_after": pd.DataFrame(),
                "output": output.getvalue(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    ncaaf_run = st.session_state.get("ncaaf_flow_run")
    flow_run_is_current = bool(
        ncaaf_run and ncaaf_run.get("token") == flow_token
    )
    if flow_run_is_current:
        if ncaaf_run.get("success"):
            st.success(
                f"El pipeline terminó. MongoDB devuelve ahora "
                f"{len(ncaaf_run['saved_after'])} registros en "
                "`weekly_picks` para esta semana."
            )
        else:
            st.error("El pipeline no terminó correctamente; revisa el registro por etapa.")
        if ncaaf_run.get("error"):
            st.error(ncaaf_run["error"])
        st.code(ncaaf_run.get("output", ""), language="text")
        if not ncaaf_run["saved_after"].empty:
            st.dataframe(
                ncaaf_run["saved_after"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "La lectura posterior de `weekly_picks` no encontró documentos "
                "para esta liga, temporada y semana."
            )

    previous_ncaaf_sync = st.session_state.get("ncaaf_flow_sync")
    flow_sync_attempted = bool(
        previous_ncaaf_sync and previous_ncaaf_sync.get("token") == flow_token
    )
    sync_ncaaf_clicked = st.button(
        f"3. Sincronizar resultados finales · semana {int(flow_week)}",
        key="ncaaf_flow_sync_button",
        disabled=not (
            flow_run_is_current
            and ncaaf_run.get("success")
            and not flow_sync_attempted
        ),
        help=(
            "Se habilita después del paso 2 y permite una sincronización por revisión. "
            "Solo sincroniza marcadores finales; no vuelve a calcular proyecciones."
        ),
        use_container_width=True,
    )
    if sync_ncaaf_clicked:
        try:
            with st.spinner("Consultando resultados finales NCAAF..."):
                sync_summary = rebuild_weeks.sync_final_results(
                    int(flow_season), int(flow_week), "ncaaf"
                )
            refreshed_picks = db.get_weekly_picks(
                int(flow_season), int(flow_week), "ncaaf"
            )
            st.session_state["ncaaf_flow_sync"] = {
                "token": flow_token,
                "success": True,
                "summary": sync_summary,
                "picks": refreshed_picks,
            }
        except Exception as exc:
            st.session_state["ncaaf_flow_sync"] = {
                "token": flow_token,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    ncaaf_sync = st.session_state.get("ncaaf_flow_sync")
    if ncaaf_sync and ncaaf_sync.get("token") == flow_token:
        if ncaaf_sync.get("success"):
            sync_summary = ncaaf_sync["summary"]
            st.success(
                f"Marcadores finales sincronizados: {sync_summary['updated']}. "
                f"Partidos sin registro asociado: {sync_summary['not_found']}."
            )
            if not ncaaf_sync["picks"].empty:
                st.dataframe(
                    ncaaf_sync["picks"],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.error(
                f"Falló la sincronización de resultados: {ncaaf_sync['error']}"
            )


# ---------- Footer ----------
st.markdown("---")
st.caption(
    "Los datos se leen de MongoDB. También puedes generar semanas desde la terminal con "
    "`python rebuild_weeks.py --season <AÑO> --league <nfl|ncaaf> --from-week <X> --to-week <Y> --skip-odds`."
)
