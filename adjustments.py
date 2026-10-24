"""
Ajustes adicionales que se le aplican al marcador ya combinado (Elo + promedios
de puntos), en este orden: lesiones -> descanso -> tope -> rivalidad -> clima.
Cada uno se puede activar/desactivar por separado en config.py.

Devuelve, junto con el marcador ajustado, una lista de notas en texto plano para
que en el reporte se vea por qué se movió la predicción (transparencia).
"""
import requests
import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None

import config

_injury_cache = {}


def _is_true(value) -> bool:
    """Comprueba si un valor es 'verdadero de verdad', sin caer en bool(nan)==True."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


# ---------- Lesiones ----------

def _get_injuries(season: int) -> pd.DataFrame:
    if season in _injury_cache:
        return _injury_cache[season]
    try:
        df = nfl.import_injuries([season])
    except Exception as e:
        print(f"[Aviso] No se pudo traer el reporte de lesiones: {e}")
        df = pd.DataFrame()
    _injury_cache[season] = df
    return df


def _last_starter(sched, team: str, before_week=None):
    need = {"home_qb_id", "away_qb_id", "home_team", "away_team",
            "home_score", "away_score", "week"}
    if sched is None or sched.empty or not need.issubset(sched.columns):
        return None
    played = sched["home_score"].notna() & sched["away_score"].notna()
    g = sched[((sched["home_team"] == team) | (sched["away_team"] == team)) & played]
    if before_week is not None:
        g = g[g["week"] < before_week]
    if "game_type" in g.columns:
        g = g[g["game_type"] == "REG"]
    if g.empty:
        return None
    last = g.sort_values("week").iloc[-1]
    qb = last["home_qb_id"] if last["home_team"] == team else last["away_qb_id"]
    return None if pd.isna(qb) else str(qb)


def _likely_starting_qb(season: int, team: str, week: int, schedule):
    starter = _last_starter(schedule, team, before_week=week)
    if starter:
        return starter
    try:
        import data_fetch
        return _last_starter(data_fetch.get_schedule(season - 1), team)
    except Exception:
        return None


def _qb_injury_penalty(season: int, team: str, week: int, schedule=None):
    if not getattr(config, "ENABLE_INJURY_ADJUSTMENT", False) or nfl is None:
        return 0.0, None
    injuries = _get_injuries(season)
    if injuries.empty:
        return 0.0, None
    qb_rows = injuries[
        (injuries["team"] == team) & (injuries["week"] == week) & (injuries["position"] == "QB")
    ]
    if qb_rows.empty:
        return 0.0, None

    starter = _likely_starting_qb(season, team, week, schedule)
    factor, tag = 1.0, "titular"
    if starter is not None and "gsis_id" in qb_rows.columns:
        qb_rows = qb_rows[qb_rows["gsis_id"].astype(str) == starter]
        if qb_rows.empty:
            return 0.0, None
    else:
        factor = getattr(config, "INJURY_UNKNOWN_STARTER_FACTOR", 0.5)
        tag = "titular no verificado"

    statuses = qb_rows["report_status"].dropna().str.lower().tolist()
    if any("out" in s for s in statuses):
        return getattr(config, "INJURY_QB_OUT_PENALTY", -7.0) * factor, f"{team} QB {tag} reportado Out"
    if any("doubtful" in s for s in statuses):
        return getattr(config, "INJURY_QB_DOUBTFUL_PENALTY", -3.5) * factor, f"{team} QB {tag} reportado Doubtful"
    if any("questionable" in s for s in statuses):
        return getattr(config, "INJURY_QB_QUESTIONABLE_PENALTY", -1.0) * factor, f"{team} QB {tag} reportado Questionable"
    return 0.0, None


_OL_POSITIONS = {"OL", "T", "OT", "G", "OG", "C"}


def _ol_injury_penalty(season: int, team: str, week: int):
    if not getattr(config, "ENABLE_INJURY_ADJUSTMENT", False) or nfl is None:
        return 0.0, None
    injuries = _get_injuries(season)
    if injuries.empty:
        return 0.0, None
    rows = injuries[
        (injuries["team"] == team) & (injuries["week"] == week)
        & (injuries["position"].isin(_OL_POSITIONS))
    ]
    if rows.empty:
        return 0.0, None
    status = rows["report_status"].fillna("").str.lower()
    n_out = int((status.str.contains("out") | status.str.contains("doubtful")).sum())
    min_cluster = getattr(config, "INJURY_OL_CLUSTER_MIN", 3)
    if n_out < min_cluster:
        return 0.0, None
    per_extra = getattr(config, "INJURY_OL_PENALTY_PER_EXTRA", -1.5)
    max_pen = getattr(config, "INJURY_OL_MAX_PENALTY", -6.0)
    penalty = max(max_pen, per_extra * (n_out - 1))
    return penalty, f"{team} {n_out} linieros ofensivos fuera"


# ---------- Descanso ----------

def _rest_days(schedule: pd.DataFrame, team: str, game_date, date_col: str) -> int:
    if game_date is None or pd.isna(game_date) or date_col not in schedule.columns:
        return 7
    game_dt = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(game_dt):
        return 7
    dates = pd.to_datetime(schedule[date_col], errors="coerce")
    team_games = schedule[
        ((schedule["home_team"] == team) | (schedule["away_team"] == team))
        & (dates < game_dt)
    ]
    if team_games.empty:
        return 7
    last_date = pd.to_datetime(team_games[date_col], errors="coerce").max()
    if pd.isna(last_date):
        return 7
    return max(int((game_dt - last_date).days), 0)


def _rest_adjustment(rest_days: int):
    if not getattr(config, "ENABLE_REST_ADJUSTMENT", False):
        return 0.0, None
    if rest_days <= getattr(config, "SHORT_REST_DAYS_THRESHOLD", 4):
        return getattr(config, "SHORT_REST_PENALTY", -1.0), f"descanso corto ({rest_days}d)"
    if rest_days >= getattr(config, "EXTRA_REST_DAYS_THRESHOLD", 9):
        return getattr(config, "EXTRA_REST_BONUS", 1.0), f"descanso extra ({rest_days}d)"
    return 0.0, None


# ---------- Rivalidad ----------

def _rivalry_shrink(is_rivalry: bool) -> float:
    if not getattr(config, "ENABLE_RIVALRY_ADJUSTMENT", False) or not is_rivalry:
        return 1.0
    return getattr(config, "DIVISIONAL_MARGIN_SHRINK", 0.85)


# ---------- Clima ----------

def _weather_total_adjustment(home_team: str, game_date):
    if not getattr(config, "ENABLE_WEATHER_ADJUSTMENT", False):
        return 0.0, None
    if game_date is None or pd.isna(game_date):
        return 0.0, None
    stadium = getattr(config, "NFL_STADIUMS", {}).get(home_team)
    if not stadium or stadium.get("dome"):
        return 0.0, None
    try:
        date_str = str(pd.to_datetime(game_date).date())
        params = {
            "latitude": stadium["lat"], "longitude": stadium["lon"],
            "daily": "windspeed_10m_max,precipitation_probability_max",
            "start_date": date_str, "end_date": date_str,
            "timezone": "auto",
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        wind = (daily.get("windspeed_10m_max") or [None])[0]
        rain_prob = (daily.get("precipitation_probability_max") or [None])[0]
    except Exception:
        return 0.0, None

    adj, notes = 0.0, []
    wind_strong = getattr(config, "WEATHER_WIND_STRONG_THRESHOLD_KMH", 32)
    wind_ok = getattr(config, "WEATHER_WIND_THRESHOLD_KMH", 24)
    wind_strong_pen = getattr(config, "WEATHER_WIND_STRONG_TOTAL_PENALTY", -4.0)
    wind_pen = getattr(config, "WEATHER_WIND_TOTAL_PENALTY", -2.0)
    rain_th = getattr(config, "WEATHER_RAIN_PROB_THRESHOLD", 60)
    rain_pen = getattr(config, "WEATHER_RAIN_TOTAL_PENALTY", -2.0)

    if wind is not None and wind >= wind_strong:
        adj += wind_strong_pen
        notes.append(f"viento fuerte {round(wind)}km/h")
    elif wind is not None and wind >= wind_ok:
        adj += wind_pen
        notes.append(f"viento {round(wind)}km/h")
    if rain_prob is not None and rain_prob >= rain_th:
        adj += rain_pen
        notes.append(f"lluvia {round(rain_prob)}%")
    return adj, (", ".join(notes) if notes else None)


# ---------- Punto de entrada ----------

def apply_extra_adjustments(league: str, season: int, week: int, home: str, away: str,
                             game_row: pd.Series, schedule: pd.DataFrame, date_col: str,
                             home_score: float, away_score: float):
    """
    Aplica, en orden:
      1. Lesiones (QB + línea ofensiva)
      2. Descanso
      3. TOPE a los ajustes acumulados
      4. Rivalidad
      5. Clima
    """
    notes = []
    base_home = home_score
    base_away = away_score

    # --- 1. Lesiones ---
    if league == "nfl":
        pen_home, note_home = _qb_injury_penalty(season, home, week, schedule)
        pen_away, note_away = _qb_injury_penalty(season, away, week, schedule)
        home_score += pen_home
        away_score += pen_away
        if note_home:
            notes.append(note_home)
        if note_away:
            notes.append(note_away)

        ol_home, ol_note_home = _ol_injury_penalty(season, home, week)
        ol_away, ol_note_away = _ol_injury_penalty(season, away, week)
        home_score += ol_home
        away_score += ol_away
        if ol_note_home:
            notes.append(ol_note_home)
        if ol_note_away:
            notes.append(ol_note_away)

    # --- 2. Descanso ---
    game_date = game_row.get(date_col)
    home_rest = _rest_days(schedule, home, game_date, date_col)
    away_rest = _rest_days(schedule, away, game_date, date_col)
    pen_home, note_home = _rest_adjustment(home_rest)
    pen_away, note_away = _rest_adjustment(away_rest)
    home_score += pen_home
    away_score += pen_away
    if note_home:
        notes.append(f"{home} {note_home}")
    if note_away:
        notes.append(f"{away} {note_away}")

    # --- 3. TOPE a los ajustes acumulados ---
    max_neg = getattr(config, "MAX_TOTAL_ADJUSTMENT_NEG", -6.0)
    max_pos = getattr(config, "MAX_TOTAL_ADJUSTMENT_POS", 4.0)

    delta_home = home_score - base_home
    delta_away = away_score - base_away
    clipped_home = min(max(delta_home, max_neg), max_pos)
    clipped_away = min(max(delta_away, max_neg), max_pos)

    if abs(clipped_home - delta_home) > 0.01:
        notes.append(f"{home} ajustes recortados al tope ({delta_home:+.1f} → {clipped_home:+.1f})")
    if abs(clipped_away - delta_away) > 0.01:
        notes.append(f"{away} ajustes recortados al tope ({delta_away:+.1f} → {clipped_away:+.1f})")

    home_score = base_home + clipped_home
    away_score = base_away + clipped_away

    # --- 4. Rivalidad ---
    is_rivalry = _is_true(game_row.get("div_game")) or _is_true(game_row.get("conference_game"))
    shrink = _rivalry_shrink(is_rivalry)
    if shrink != 1.0:
        margin = home_score - away_score
        total = home_score + away_score
        home_score = total / 2 + margin * shrink / 2
        away_score = total / 2 - margin * shrink / 2
        notes.append("partido de rivalidad (margen reducido)")

    # --- 5. Clima ---
    if league == "nfl":
        adj, note = _weather_total_adjustment(home, game_date)
        if adj != 0.0:
            home_score += adj / 2
            away_score += adj / 2
            notes.append(f"clima: {note}")

    return round(home_score, 1), round(away_score, 1), notes