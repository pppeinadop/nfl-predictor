"""
App Streamlit para visualizar los picks y props guardados en MongoDB.
También permite generar / regenerar semanas y descargar reportes.

Uso:
    streamlit run app.py
"""
import contextlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

import config
import db
import predict_games
import rebuild_weeks
import report
import simulate


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
  .calib-note {
    color: #64748b;
    font-size: 0.9em;
    margin-bottom: 12px;
  }
  div.stButton > button {
    border-radius: 9px;
    font-weight: 650;
  }
  div[data-baseweb="select"] > div {
    border-radius: 9px;
    border-color: #dbe3ee;
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


# ---------- Calibración ----------
def _collect_all_played(league: str, from_season: int, to_season: int) -> pd.DataFrame:
    frames = []
    for season in range(from_season, to_season + 1):
        for w in db.list_available_weeks(season, league):
            df = db.get_weekly_picks(season, w, league)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    if "resultado_real" in full.columns:
        full = full[full["resultado_real"].notna()].reset_index(drop=True)
    return full


def _parse_pct(s):
    if s is None or pd.isna(s):
        return None
    try:
        return float(str(s).replace("%", "").strip()) / 100.0
    except (TypeError, ValueError):
        return None


def _calibration_table_ml(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "prob_home" not in df.columns or "acierto_ganador" not in df.columns:
        return pd.DataFrame()

    def fav_prob(row):
        ph = _parse_pct(row.get("prob_home"))
        pa = _parse_pct(row.get("prob_away"))
        if ph is None or pa is None:
            return None
        return max(ph, pa)

    df2 = df.copy()
    df2["_fav_prob"] = df2.apply(fav_prob, axis=1)
    df2 = df2.dropna(subset=["_fav_prob", "acierto_ganador"])

    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.01]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80-90%", "90%+"]
    df2["_bucket"] = pd.cut(df2["_fav_prob"], bins=bins, labels=labels, right=False)

    rows = []
    for label in labels:
        sub = df2[df2["_bucket"] == label]
        if sub.empty:
            continue
        n = len(sub)
        hits = sum(1 for v in sub["acierto_ganador"] if _bool_value(v) is True)
        pct = 100 * hits / n if n > 0 else 0
        pred_mean = 100 * sub["_fav_prob"].mean()
        rows.append({
            "Bucket": label,
            "Partidos": n,
            "Predicho": f"{pred_mean:.1f}%",
            "Real": f"{pct:.1f}%",
            "Diferencia": f"{pct - pred_mean:+.1f}",
        })
    return pd.DataFrame(rows)


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
        f"Ve a la pestaña **🔧 Generar** para crear la primera semana."
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
tab_picks, tab_props, tab_gen, tab_calib, tab_report, tab_console = st.tabs(
    ["🏟️ Partidos", "⭐ Props de jugadores", "🔧 Generar", "📐 Calibración", "📊 Reporte", "🖥️ Salida de consola"]
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
    st.markdown("---")


# --- TAB: Partidos ---
with tab_picks:
    if week is None:
        st.info(
            f"No hay datos para **{league_label} {season}**. "
            f"Ve a la pestaña **🔧 Generar** para crear la primera semana."
        )
    elif picks_df.empty:
        st.warning("No hay partidos guardados para esta semana (o el filtro los ocultó).")
    else:
        if "fecha" in picks_df.columns:
            picks_df = picks_df.sort_values(
                by=["fecha", "hora"], na_position="last", kind="stable"
            ).reset_index(drop=True)

        has_slots = "slot" in picks_df.columns and picks_df["slot"].notna().any()

        current_slot = None
        for _, r in picks_df.iterrows():
            slot = r.get("slot") if has_slots else None

            if has_slots and slot != current_slot:
                st.markdown(
                    f'<div class="slot-header">📅 {slot or "Sin fecha"}</div>',
                    unsafe_allow_html=True,
                )
                current_slot = slot

            away_team, home_team = r["partido"].split(" @ ")
            home_rec = _resolve_record(r, "home", season, week, league_choice, records_fallback)
            away_rec = _resolve_record(r, "away", season, week, league_choice, records_fallback)

            with st.container(border=True):
                away_title = _team_with_record(away_team, away_rec)
                home_title = _team_with_record(home_team, home_rec)

                c_away, c_at, c_home = st.columns([5, 1, 5])
                with c_away:
                    with st.popover(f"**{away_title}**", use_container_width=True):
                        _render_team_history(season, away_team, league_choice)
                with c_at:
                    st.markdown('<div class="at-sep">@</div>', unsafe_allow_html=True)
                with c_home:
                    with st.popover(f"**{home_title}**", use_container_width=True):
                        _render_team_history(season, home_team, league_choice)

                dt_label = fmt_datetime(r.get("fecha"), r.get("hora"))
                cap = f"{fmt(r.get('liga'))} · Semana {fmt(r.get('semana'))}"
                if dt_label:
                    cap = f"{cap} · {dt_label}"
                st.caption(cap)

                c_mod, c_mkt, c_res = st.columns([2, 2, 1.3])

                with c_mod:
                    st.markdown("**📊 Modelo**")
                    st.markdown(f"- Marcador: `{fmt(r.get('marcador_proyectado'))}`")
                    prob_h = _get(r, "prob_home", "prob_gana_local")
                    prob_a = _get(r, "prob_away", "prob_gana_visita")
                    st.markdown(
                        f"- Prob. victoria: {_team_md(home_team, home_rec)} `{fmt(prob_h)}` · "
                        f"{_team_md(away_team, away_rec)} `{fmt(prob_a)}`",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"- Spread: `{fmt(r.get('spread_modelo'))}`")
                    st.markdown(f"- Over/Under: `{fmt(r.get('total_modelo'))}`")

                with c_mkt:
                    st.markdown("**💼 Mercado**")
                    odds_updated = r.get("odds_cache_updated_at")
                    if odds_updated is not None and pd.notna(odds_updated):
                        st.caption(f"Cuotas guardadas: {fmt(odds_updated)}")
                    elif all(
                        fmt(_get(r, col)) == "—"
                        for col in ("spread_consenso", "total_consenso", "ml_home", "ml_away")
                    ):
                        st.caption("Sin cuotas disponibles; predicción generada solo con el modelo.")

                    cons_spread = fmt(r.get("spread_consenso"))
                    if cons_spread == "—":
                        msl = _get(r, "market_spread_line")
                        if msl is not None:
                            try:
                                msl_f = float(msl)
                                if msl_f > 0:
                                    cons_spread = f"{home_team} -{abs(msl_f):g}"
                                elif msl_f < 0:
                                    cons_spread = f"{away_team} -{abs(msl_f):g}"
                                else:
                                    cons_spread = "Pick'em (0)"
                            except (TypeError, ValueError):
                                pass
                    st.markdown(f"- Spread: `{cons_spread}`")

                    cons_total = fmt(r.get("total_consenso"))
                    if cons_total == "—":
                        mtl = _get(r, "market_total_line")
                        if mtl is not None:
                            try:
                                cons_total = f"{float(mtl):g}"
                            except (TypeError, ValueError):
                                pass
                    st.markdown(f"- Over/Under: `{cons_total}`")

                    ml_h = _get(r, "ml_home", "ml_local")
                    ml_a = _get(r, "ml_away", "ml_visita")
                    if ml_h is not None or ml_a is not None:
                        st.markdown(
                            f"- ML {_team_md(home_team, home_rec)}: `{fmt(ml_h)}` · "
                            f"ML {_team_md(away_team, away_rec)}: `{fmt(ml_a)}`",
                            unsafe_allow_html=True,
                        )

                    ml_probs_html = _market_ml_probs_html(r, home_team, away_team)
                    if ml_probs_html:
                        st.markdown(ml_probs_html, unsafe_allow_html=True)

                with c_res:
                    st.markdown("**🏁 Resultado real**")
                    rr = r.get("resultado_real")
                    if rr and pd.notna(rr):
                        ac = r.get("acierto_ganador")
                        acs = r.get("acierto_spread")
                        act = r.get("acierto_total")
                        st.markdown(_render_result_body(rr, ac, acs, act),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="result-score neutral">Por jugarse</div>',
                            unsafe_allow_html=True,
                        )

                tags = []
                for label, col in [("Spread", "valor_spread"), ("Over/Under", "valor_total")]:
                    val = r.get(col)
                    if val and pd.notna(val):
                        cls = classify_value(val)
                        tags.append(f'<span class="tag {cls}">{label}: {val}</span>')
                if tags:
                    st.markdown(" ".join(tags), unsafe_allow_html=True)

                ajustes = r.get("ajustes")
                if ajustes and pd.notna(ajustes):
                    st.caption(f"🔧 Ajustes: {ajustes}")

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
        st.info("Primero genera alguna semana desde la pestaña **🔧 Generar**.")
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


# --- TAB: Generar ---
with tab_gen:
    st.subheader("Generar predicciones")
    st.caption(
        "Las predicciones oficiales se guardan una sola vez. Las actualizaciones prepartido se "
        "registran como versiones independientes; al iniciar el partido, el sistema las bloquea."
    )
    st.caption(
        "La liga y la temporada se toman de la **barra lateral**. "
        "Cambia ahí si quieres generar para NFL o NCAAF."
    )

    col_a, col_b, col_btn = st.columns([1, 1, 1.3])
    with col_a:
        from_week = st.number_input(
            "Desde semana", min_value=1, max_value=25,
            value=1, step=1, key="gen_from_week",
        )
    with col_b:
        to_week = st.number_input(
            "Hasta semana", min_value=1, max_value=25,
            value=max(1, int(from_week)), step=1, key="gen_to_week",
        )
    with col_btn:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        if week is not None:
            if st.button(f"📌 Usar semana actual ({week})", use_container_width=True):
                st.session_state["gen_from_week"] = int(week)
                st.session_state["gen_to_week"] = int(week)
                st.rerun()
        else:
            st.button("📌 Usar semana actual", disabled=True, use_container_width=True)

    prediction_mode_label = st.radio(
        "Tipo de predicción",
        options=["Oficial (crear una sola vez)", "Actualizar antes del partido (guardar versión)"],
        horizontal=True,
        help=(
            "La oficial no se reemplaza. La actualización crea una versión aparte y solo se acepta "
            "si el partido todavía no comenzó."
        ),
    )
    prediction_mode = "update" if prediction_mode_label.startswith("Actualizar") else "official"

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        skip_odds = st.checkbox(
            "Usar caché de mercado (skip-odds)",
            value=True,
            help="Solo cache. NUNCA llama a The Odds API. Ideal para regenerar histórico.",
        )
    with col_d:
        refresh_odds = st.checkbox(
            "Refrescar odds", value=False,
            help="Fuerza llamar a The Odds API aunque ya haya cache (consume cuota).",
        )
    with col_e:
        verbose = st.checkbox("Mostrar detalle", value=False)

    if refresh_odds:
        st.info("Se consultará The Odds API. Esta opción prevalece sobre la caché y puede consumir cuota.")

    if to_week < from_week:
        rango_txt = "(rango inválido)"
    elif to_week == from_week:
        rango_txt = f"Semana {int(from_week)}"
    else:
        rango_txt = f"Semanas {int(from_week)} a {int(to_week)}"

    st.markdown(
        f"""
        <div class="gen-confirm">
          <div class="label">Se va a generar</div>
          <div class="value">{league_label} · Temporada {season}</div>
          <div class="sub">{rango_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_calls = 0
    if to_week >= from_week and (not skip_odds or refresh_odds):
        api_calls = (int(to_week) - int(from_week) + 1)
    if api_calls > 0:
        st.warning(f"⚠ Se harán hasta **{api_calls} llamada(s)** a The Odds API.")
    elif to_week >= from_week:
        st.success("✅ No se llamará a The Odds API (solo caché de Mongo).")

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        generate_clicked = st.button(
            "Generar / actualizar predicciones", type="primary",
            disabled=(to_week < from_week), use_container_width=True,
        )
    with action_col2:
        sync_results_clicked = st.button(
            "Sincronizar resultados finales", use_container_width=True,
            disabled=(to_week < from_week),
            help="Actualiza solo marcadores y métricas de partidos terminados; no recalcula predicciones.",
        )

    if sync_results_clicked:
        with st.spinner("Consultando resultados finales..."):
            result_summary = {"updated": 0, "not_found": 0}
            for result_week in range(int(from_week), int(to_week) + 1):
                week_summary = rebuild_weeks.sync_final_results(
                    int(season), result_week, league_choice
                )
                result_summary["updated"] += week_summary["updated"]
                result_summary["not_found"] += week_summary["not_found"]
        st.success(
            f"Resultados sincronizados: {result_summary['updated']} partidos actualizados. "
            "Las predicciones oficiales no se modificaron."
        )

    progress_placeholder = st.empty()
    log_placeholder = st.empty()

    if generate_clicked:
        with st.spinner("Generando..."):
            ok, fail = _run_generation(
                season, league_choice,
                int(from_week), int(to_week),
                skip_odds=skip_odds,
                refresh_odds=refresh_odds,
                verbose=verbose,
                prediction_mode=prediction_mode,
                log_placeholder=log_placeholder,
                progress_placeholder=progress_placeholder,
            )

        st.markdown("### Resultado")
        if ok:
            st.success(f"✅ Semanas generadas: {ok}")
        if fail:
            st.error(f"❌ Semanas con error: {fail}")

        if st.button("🔄 Recargar datos"):
            st.session_state.pop("_week_context", None)
            st.rerun()


# --- TAB: Calibración ---
with tab_calib:
    st.subheader("📐 Calibración del modelo")
    st.markdown(
        '<div class="calib-note">'
        'Esta tabla te dice si las probabilidades del modelo son <b>honestas</b>. '
        'Cuando el modelo dice "70%", ¿realmente acierta el 70% de las veces? '
        'Si los números de "Predicho" y "Real" son parecidos, las probabilidades '
        'del modelo son confiables.'
        '</div>',
        unsafe_allow_html=True,
    )

    seasons_available_calib = []
    try:
        for s in range(2015, 2031):
            if db.list_available_weeks(s, league_choice):
                seasons_available_calib.append(s)
    except Exception:
        pass

    if not seasons_available_calib:
        st.warning("No hay datos guardados para calibrar.")
        st.stop()

    min_s = min(seasons_available_calib)
    max_s = max(seasons_available_calib)

    col_ca, col_cb = st.columns([1, 1])
    with col_ca:
        calib_from = st.number_input(
            "Desde temporada", min_value=min_s, max_value=max_s,
            value=min_s, step=1, key="calib_from",
        )
    with col_cb:
        calib_to = st.number_input(
            "Hasta temporada", min_value=min_s, max_value=max_s,
            value=max_s, step=1, key="calib_to",
        )

    st.caption(
        f"Analizando temporadas **{int(calib_from)}–{int(calib_to)}** "
        f"para {league_label}."
    )

    with st.spinner("Recopilando datos históricos..."):
        calib_df = _collect_all_played(league_choice, int(calib_from), int(calib_to))

    if calib_df.empty:
        st.warning("No hay partidos jugados en ese rango.")
    else:
        st.caption(f"Partidos analizados: **{len(calib_df)}**")
        st.markdown("---")

        st.markdown("### 🎯 Calibración de probabilidad (ML)")
        st.caption(
            "Cuando el modelo asigna X% al favorito, ¿qué % de esos partidos ganó el favorito? "
            "Si 'Predicho' y 'Real' son parecidos, la probabilidad está bien calibrada."
        )
        tbl_ml = _calibration_table_ml(calib_df)
        if tbl_ml.empty:
            st.info("Sin datos suficientes.")
        else:
            st.dataframe(tbl_ml, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Cómo leer esta tabla:**")
        st.markdown(
            "- **Diferencia** cercana a 0 = bien calibrado.\n"
            "- Si 'Diferencia' es consistentemente positiva, el modelo es "
            "**conservador** (subestima su propia confianza).\n"
            "- Si es consistentemente negativa, es **optimista** (sobreestima).\n"
            "- Buckets con pocos partidos (< 30) pueden tener ruido.\n\n"
            "**Nota:** las probabilidades de spread y over/under no se muestran aquí "
            "porque no están calibradas (dan ~50% real en todos los buckets). "
            "El modelo tiene señal en el ganador (ML), no en quién cubre la línea."
        )


# --- TAB: Reporte ---
with tab_report:
    st.subheader("📊 Reporte de desempeño")
    st.caption(
        "Resumen del desempeño del modelo. Descarga el reporte y súbelo a una IA "
        "(ChatGPT, Claude, Gemini) para que lo analice a fondo."
    )

    seasons_available = []
    try:
        for s in range(2015, 2031):
            if db.list_available_weeks(s, league_choice):
                seasons_available.append(s)
    except Exception:
        pass

    if not seasons_available:
        st.warning("No hay datos guardados para esta liga. Genera primero desde la pestaña **🔧 Generar**.")
        st.stop()

    min_season = min(seasons_available)
    max_season = max(seasons_available)

    col_r1, col_r2, col_r3 = st.columns([1, 1, 1.3])
    with col_r1:
        from_season = st.number_input(
            "Desde temporada",
            min_value=min_season, max_value=max_season,
            value=min_season, step=1, key="rep_from_season",
        )
    with col_r2:
        to_season = st.number_input(
            "Hasta temporada",
            min_value=min_season, max_value=max_season,
            value=max_season, step=1, key="rep_to_season",
        )
    with col_r3:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        if st.button("📚 Usar todo el histórico disponible", use_container_width=True):
            st.session_state["rep_from_season"] = min_season
            st.session_state["rep_to_season"] = max_season
            st.rerun()

    st.caption(
        f"Temporadas con datos: {seasons_available}. "
        f"Rango actual: **{int(from_season)}–{int(to_season)}**."
    )

    col_x, col_y = st.columns([1, 1])
    with col_x:
        only_played_report = st.checkbox(
            "Solo partidos ya jugados", value=True,
            help="Desactívalo para incluir partidos futuros en el CSV crudo.",
        )
    with col_y:
        report_format = st.selectbox(
            "Formato de descarga",
            options=["markdown", "json", "csv"],
            format_func=lambda x: {
                "markdown": "Markdown (.md) — ideal para pegar en una IA",
                "json": "JSON (.json) — estructurado",
                "csv": "CSV (.csv) — crudo para Excel/pandas",
            }[x],
        )

    if st.button("📥 Generar reporte", type="primary"):
        with st.spinner("Recopilando datos de todas las temporadas..."):
            full_df = report.collect_range(
                int(from_season), int(to_season), league_choice,
                only_played=only_played_report,
            )

        if full_df.empty:
            st.warning("No hay partidos para reportar en ese rango.")
        else:
            stats = report.compute_stats(full_df)
            g = stats.get("global") or {}

            st.markdown("### Resumen general")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                acc = g.get("ml")
                if acc and acc.get("total"):
                    st.metric("ML", f"{acc['aciertos']}/{acc['total']}", delta=f"{acc['pct']}%")
                else:
                    st.metric("ML", "—")
            with c2:
                acc = g.get("spread")
                if acc and acc.get("total"):
                    st.metric("Spread", f"{acc['aciertos']}/{acc['total']}", delta=f"{acc['pct']}%")
                else:
                    st.metric("Spread", "—")
            with c3:
                acc = g.get("total")
                if acc and acc.get("total"):
                    st.metric("O/U", f"{acc['aciertos']}/{acc['total']}", delta=f"{acc['pct']}%")
                else:
                    st.metric("O/U", "—")
            with c4:
                st.metric("Partidos", stats["total_partidos"])

            if stats.get("por_temporada"):
                st.markdown("### Por temporada")
                rows = []
                for s in stats["por_temporada"]:
                    def _fmt(a):
                        if not a or not a.get("total"):
                            return "—"
                        return f"{a['aciertos']}/{a['total']} ({a['pct']}%)"
                    rows.append({
                        "Temporada": s["temporada"],
                        "Partidos": s["partidos"],
                        "ML": _fmt(s.get("ml")),
                        "Spread": _fmt(s.get("spread")),
                        "O/U": _fmt(s.get("total")),
                        "Err. margen": s.get("error_margen_medio_pts"),
                        "Err. total": s.get("error_total_medio_pts"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if stats.get("por_semana"):
                st.markdown("### Por semana")
                rows = []
                for s in stats["por_semana"]:
                    def _fmt(a):
                        if not a or not a.get("total"):
                            return "—"
                        return f"{a['aciertos']}/{a['total']} ({a['pct']}%)"
                    rows.append({
                        "Semana": s["semana"],
                        "Partidos": s["partidos"],
                        "ML": _fmt(s.get("ml")),
                        "Spread": _fmt(s.get("spread")),
                        "O/U": _fmt(s.get("total")),
                        "Err. margen": s.get("error_margen_medio_pts"),
                        "Err. total": s.get("error_total_medio_pts"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("### Por rango de favoritismo")
            if stats.get("por_rango_spread"):
                rows = []
                for s in stats["por_rango_spread"]:
                    def _fmt(a):
                        if not a or not a.get("total"):
                            return "—"
                        return f"{a['aciertos']}/{a['total']} ({a['pct']}%)"
                    rows.append({
                        "Rango": s["rango"],
                        "Partidos": s["partidos"],
                        "ML": _fmt(s.get("ml")),
                        "Spread": _fmt(s.get("spread")),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("### Por nivel de confianza del modelo")
            if stats.get("por_confianza_ml"):
                rows = []
                for s in stats["por_confianza_ml"]:
                    def _fmt(a):
                        if not a or not a.get("total"):
                            return "—"
                        return f"{a['aciertos']}/{a['total']} ({a['pct']}%)"
                    rows.append({
                        "Confianza": s["confianza"],
                        "Partidos": s["partidos"],
                        "ML": _fmt(s.get("ml")),
                        "Spread": _fmt(s.get("spread")),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Descargar reporte")

            label = f"{int(from_season)}" if from_season == to_season else f"{int(from_season)}-{int(to_season)}"
            base_name = f"reporte_{league_choice}_{label}"

            if report_format == "markdown":
                md = report.to_markdown(full_df, stats)
                st.download_button(
                    "⬇️ Descargar Markdown (.md)",
                    data=md.encode("utf-8"),
                    file_name=f"{base_name}.md",
                    mime="text/markdown",
                )
                with st.expander("Vista previa (Markdown)"):
                    st.markdown(md)

            elif report_format == "json":
                data = report.to_json_dict(full_df, stats)
                js = json.dumps(data, ensure_ascii=False, indent=2, default=str)
                st.download_button(
                    "⬇️ Descargar JSON (.json)",
                    data=js.encode("utf-8"),
                    file_name=f"{base_name}.json",
                    mime="application/json",
                )
                with st.expander("Vista previa (JSON, primeros 3000 caracteres)"):
                    st.code(js[:3000] + ("\n..." if len(js) > 3000 else ""), language="json")

            else:
                csv_bytes = full_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar CSV (.csv)",
                    data=csv_bytes,
                    file_name=f"{base_name}.csv",
                    mime="text/csv",
                )
                with st.expander("Vista previa (primeras filas)"):
                    st.dataframe(full_df.head(20), use_container_width=True, hide_index=True)


# --- TAB: Consola ---
with tab_console:
    if week is None:
        st.info("No hay datos para mostrar. Genera alguna semana primero.")
    else:
        st.caption("Salida literal de `predict_games.print_picks()`.")
        if picks_df.empty:
            st.info("Sin datos.")
        else:
            try:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    predict_games.print_picks(picks_df)
                st.code(f.getvalue(), language="text")
            except Exception as e:
                st.error(f"No se pudo generar la salida de consola: {e}")


# ---------- Footer ----------
st.markdown("---")
st.caption(
    "Los datos se leen de MongoDB. También puedes generar semanas desde la terminal con "
    "`python rebuild_weeks.py --season <AÑO> --league <nfl|ncaaf> --from-week <X> --to-week <Y> --skip-odds`."
)
