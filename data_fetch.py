"""
Trae:
 1) Calendario y resultados (nfl_data_py) para actualizar Elo y promedios de equipo.
 2) Estadísticas semanales de jugadores (nfl_data_py) para las props.
 3) Líneas de casas de apuestas (The Odds API) para partidos y props.

nfl_data_py es gratuito y no requiere API key. The Odds API sí requiere key (config.py).
"""
import difflib
import math
import re

import requests
import numpy as np
import pandas as pd
import nfl_data_py as nfl

import team_stats

import config


# ---------- Estadísticas (nfl_data_py) ----------

_SCHEDULE_CACHE = {}


def get_schedule(season: int = config.CURRENT_SEASON) -> pd.DataFrame:
    """Calendario completo de la temporada, con resultados de los partidos ya jugados."""
    if season not in _SCHEDULE_CACHE:  # se descarga una sola vez por corrida
        _SCHEDULE_CACHE[season] = nfl.import_schedules([season])
    return _SCHEDULE_CACHE[season].copy()


def get_completed_games(season: int = config.CURRENT_SEASON, before_week: int = None) -> pd.DataFrame:
    """Partidos ya jugados. Con before_week, solo los de semanas anteriores a esa."""
    sched = get_schedule(season)
    games = sched.dropna(subset=["home_score", "away_score"])
    if before_week is not None:
        games = games[games["week"] < before_week]
    return games


def get_week_games(season: int, week: int) -> pd.DataFrame:
    """Todos los partidos de esa semana, hayan terminado o no."""
    sched = get_schedule(season)
    return sched[sched["week"] == week]


def _previous_season_games(season: int):
    """Partidos de la temporada anterior (sirven como punto de partida). None si falla."""
    if not config.CARRY_OVER_PREVIOUS_SEASON:
        return None
    try:
        return get_completed_games(season - 1)
    except Exception as e:
        print(f"[Aviso] No se pudo cargar la temporada anterior: {e}")
        return None


def get_team_scoring_averages(season: int = config.CURRENT_SEASON,
                               window: int = config.ROLLING_WINDOW_TEAM,
                               before_week: int = None) -> pd.DataFrame:
    """
    Promedios de puntos anotados/permitidos por equipo (ver team_stats.py): con más peso
    a lo reciente, mezclados con la temporada anterior, ajustados por calendario y con
    splits local/visitante. 'before_week' limita los datos a semanas anteriores a esa
    (así una predicción nunca "ve" el resultado del partido que está prediciendo).
    """
    games = get_completed_games(season, before_week=before_week)
    return team_stats.build_team_averages(games, _previous_season_games(season),
                                          default_league_avg=22.0, window=window)


def get_weekly_player_stats(season: int = config.CURRENT_SEASON) -> pd.DataFrame:
    """Estadísticas semana a semana por jugador (para props)."""
    return nfl.import_weekly_data([season])


# ---------- Líneas de casas de apuestas (The Odds API) ----------

def _round_half(value: float) -> float:
    """
    Redondea al múltiplo de 0.5 más cercano, como muestran las casas de apuestas
    (41.6 -> 41.5, 40.2 -> 40.0, 43.8 -> 44.0). Evita el 'banker's rounding' de Python.
    """
    return math.floor(float(value) * 2 + 0.5) / 2


def get_game_odds(sport_key: str = config.SPORT_KEY) -> list:
    """
    Devuelve, por partido, las líneas (moneyline, spread, total) agregadas de
    varias casas de apuestas (DraftKings, FanDuel, BetMGM, etc. según 'REGIONS').
    'sport_key' permite pedir otra liga, ej. config.LEAGUES["ncaaf"]["odds_sport_key"].
    """
    if not config.ODDS_API_KEY or config.ODDS_API_KEY == "TU_API_KEY_AQUI":
        raise ValueError("Falta configurar ODDS_API_KEY; no se llamó a The Odds API.")
    url = f"{config.ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.REGIONS,
        "markets": config.GAME_MARKETS,
        "oddsFormat": config.ODDS_FORMAT,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _normalize_team(name: str) -> str:
    """Minúsculas, sin puntuación, espacios colapsados. Conserva siglas entre paréntesis."""
    s = name.lower().strip()
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def fuzzy_match_team(name: str, candidates) -> str:
    """
    Empareja un nombre de equipo con el candidato más probable, evitando falsos
    positivos por substring arbitrario (ej. 'Miami' no debe matchear 'Miami (OH)'
    cuando el correcto es 'Miami (FL)'). Estrategia:

      1) Match exacto (normalizado).
      2) Igualdad de CONJUNTO de palabras (más seguro que substring).
      3) Contención por palabra completa.
      4) difflib con umbral alto.

    Devuelve None si no hay match razonable.
    """
    if not name or candidates is None:
        return None

    name_norm = _normalize_team(name)
    name_words = set(name_norm.split())
    cand_pairs = [(c, _normalize_team(c)) for c in candidates if isinstance(c, str)]

    # 1) exacto
    for c, cn in cand_pairs:
        if cn == name_norm:
            return c

    # 2) igualdad de conjunto de palabras
    for c, cn in cand_pairs:
        cn_words = set(cn.split())
        if name_words == cn_words:
            return c

    # 3) contención por palabra completa
    for c, cn in cand_pairs:
        cn_words = set(cn.split())
        if name_words and name_words.issubset(cn_words):
            return c
        if cn_words and cn_words.issubset(name_words):
            return c

    # 4) difflib con umbral alto, sobre la cadena normalizada
    matches = difflib.get_close_matches(
        name_norm, [cn for _, cn in cand_pairs], n=1, cutoff=0.85
    )
    if matches:
        target = matches[0]
        for c, cn in cand_pairs:
            if cn == target:
                return c
    return None


def get_consensus_lines(events: list) -> pd.DataFrame:
    """
    En vez de quedarse con la línea de una sola casa, promedia el spread, el total
    y el moneyline entre TODAS las casas de apuestas devueltas por la API, para
    tener un número de "consenso de mercado" más confiable que una sola casa.

    El spread y el total se redondean al entero o medio punto más cercano
    (como muestran las casas), para que el reporte y la comparación con el modelo
    sean consistentes. También devuelve cuántas casas se usaron y qué tan
    dispersas están sus líneas (spread_std / total_std).
    """
    rows = []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        spreads_home, totals, ml_home, ml_away = [], [], [], []
        books_with_spread = []

        for bk in ev.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market["key"] == "spreads":
                    for oc in market["outcomes"]:
                        if oc["name"] == home:
                            spreads_home.append(oc["point"])
                            books_with_spread.append(bk["title"])
                elif market["key"] == "totals":
                    for oc in market["outcomes"]:
                        if oc["name"] == "Over":
                            totals.append(oc["point"])
                elif market["key"] == "h2h":
                    for oc in market["outcomes"]:
                        price = oc.get("price")
                        if price is None or abs(price) > config.MAX_ABS_MONEYLINE:
                            continue  # dato corrupto de la fuente, se descarta
                        if oc["name"] == home:
                            ml_home.append(price)
                        elif oc["name"] == away:
                            ml_away.append(price)

        rows.append({
            "home_team": home,
            "away_team": away,
            "consensus_spread_home": _round_half(float(np.mean(spreads_home))) if spreads_home else None,
            "spread_books_n": len(spreads_home),
            "spread_std": round(float(np.std(spreads_home)), 2) if len(spreads_home) > 1 else 0.0,
            "consensus_total": _round_half(float(np.mean(totals))) if totals else None,
            "total_books_n": len(totals),
            "total_std": round(float(np.std(totals)), 2) if len(totals) > 1 else 0.0,
            "consensus_ml_home": round(float(np.mean(ml_home))) if ml_home else None,
            "consensus_ml_away": round(float(np.mean(ml_away))) if ml_away else None,
            "casas_usadas": ", ".join(sorted(set(books_with_spread))) if books_with_spread else None,
        })
    return pd.DataFrame(rows)


def get_player_prop_odds(event_id: str, markets: list = None) -> dict:
    """
    Líneas de props de jugador para un partido puntual (se necesita el event_id
    que devuelve get_game_odds()).
    """
    markets = markets or config.PLAYER_PROP_MARKETS
    url = f"{config.ODDS_API_BASE}/sports/{config.SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": config.ODDS_FORMAT,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()