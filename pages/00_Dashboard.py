import streamlit as st
import pandas as pd

import db
import config

st.set_page_config(
    page_title="Dashboard",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Predictor Pro")

league = st.sidebar.selectbox(
    "Liga",
    ["nfl", "ncaaf"]
)

season = st.sidebar.number_input(
    "Temporada",
    min_value=2015,
    max_value=2035,
    value=config.CURRENT_SEASON
)

weeks = db.list_available_weeks(
    season,
    league
)

if not weeks:
    st.warning("No hay datos.")
    st.stop()

week = max(weeks)

df = db.get_weekly_picks(
    season,
    week,
    league
)

st.subheader(f"Semana {week}")

value_spread = 0
value_total = 0

if "valor_spread" in df.columns:
    value_spread = (
        df["valor_spread"]
        .astype(str)
        .str.contains("Sin valor", na=False)
        .eq(False)
        .sum()
    )

if "valor_total" in df.columns:
    value_total = (
        df["valor_total"]
        .astype(str)
        .str.contains("Sin valor", na=False)
        .eq(False)
        .sum()
    )

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Games",
    len(df)
)

c2.metric(
    "Spread Edges",
    int(value_spread)
)

c3.metric(
    "Total Edges",
    int(value_total)
)

c4.metric(
    "Season",
    season
)

st.markdown("---")

st.subheader("🔥 Mejores oportunidades")

for _, row in df.head(5).iterrows():

    st.markdown(f"""
    <div class='custom-card'>
        <h3>{row['partido']}</h3>
        <p><b>Probabilidad:</b>
        {row.get('prob_home','?')} /
        {row.get('prob_away','?')}</p>

        <p><b>Spread Modelo:</b>
        {row.get('spread_modelo','?')}</p>

        <p><b>Spread Mercado:</b>
        {row.get('spread_consenso','?')}</p>

        <p><b>Valor:</b>
        {row.get('valor_spread','Sin valor')}</p>
    </div>
    """, unsafe_allow_html=True)