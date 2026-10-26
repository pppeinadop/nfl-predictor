"""
Equivalente a data_fetch.py pero para fútbol americano colegial (NCAAF),
usando la API de collegefootballdata.com (paquete 'cfbd'). Necesitas una
API key gratis en https://collegefootballdata.com/key, puesta en config.py
como CFBD_API_KEY.

Nota: a diferencia de nfl_data_py, aquí los nombres de equipo son el nombre
completo de la escuela (ej. "Ohio State", "Georgia", "Alabama"), no siglas.
"""
import cfbd
import pandas as pd

import config
import team_stats


def _games_api():
    configuration = cfbd.Configuration()
    # Compatibilidad segura para asignar la API key según la versión de cfbd
    try:
        configuration.access_token = config.CFBD_API_KEY
    except AttributeError:
        try:
            configuration.api_key['Authorization'] = f"Bearer {config.CFBD_API_KEY}"
        except Exception:
            configuration.api_key['api_key'] = config.CFBD_API_KEY
            configuration.api_key_prefix['api_key'] = 'Bearer'
            
    api_client = cfbd.ApiClient(configuration)
    return cfbd.GamesApi(api_client)


def _fetch_schedule(season: int, week: int = None, division: str = "fbs") -> pd.DataFrame:
    """
    Calendario de la temporada (o de una semana puntual) para la división FBS
    (la que cubren la mayoría de casas de apuestas).
    """
    # Se remueve 'classification' para evitar incompatibilidades de versión
    games = _games_api().get_games(
        year=season, week=week, season_type="regular"
    )
    
    rows = []
    for g in games:
        # Filtrar por división FBS si viene especificada en el objeto del juego
        div_home = getattr(g, "home_division", "fbs")
        if division and div_home and div_home.lower() != division.lower():
            continue
            
        rows.append({
            "week": g.week,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_score": g.home_points,
            "away_score": g.away_points,
            "start_date": getattr(g, "start_date", None),
            "completed": getattr(g, "completed", None),
            "conference_game": getattr(g, "conference_game", False),
        })
        
    return pd.DataFrame(rows)


_SCHEDULE_CACHE = {}


def get_schedule(season: int, week: int = None, division: str = "fbs") -> pd.DataFrame:
    """Igual que _fetch_schedule pero con caché: cada consulta a CFBD se hace una sola vez por corrida."""
    key = (season, week, division)
    if key not in _SCHEDULE_CACHE:
        _SCHEDULE_CACHE[key] = _fetch_schedule(season, week, division)
    return _SCHEDULE_CACHE[key].copy()


def get_completed_games(season: int, before_week: int = None) -> pd.DataFrame:
    """Partidos ya jugados. Con before_week, solo los de semanas anteriores a esa."""
    sched = get_schedule(season)
    games = sched.dropna(subset=["home_score", "away_score"])
    if before_week is not None:
        games = games[games["week"] < before_week]
    return games


def get_week_games(season: int, week: int) -> pd.DataFrame:
    """Todos los partidos de esa semana, hayan terminado o no."""
    return get_schedule(season, week=week)


def _previous_season_games(season: int):
    """Partidos de la temporada anterior (sirven como punto de partida). None si falla."""
    if not config.CARRY_OVER_PREVIOUS_SEASON:
        return None
    try:
        return get_completed_games(season - 1)
    except Exception as e:
        print(f"[Aviso] No se pudo cargar la temporada anterior: {e}")
        return None


def get_team_scoring_averages(season: int, window: int = config.ROLLING_WINDOW_TEAM,
                             before_week: int = None) -> pd.DataFrame:
    """Igual que en data_fetch.py (ver team_stats.py)."""
    games = get_completed_games(season, before_week=before_week)
    return team_stats.build_team_averages(games, _previous_season_games(season),
                                        default_league_avg=24.0, window=window)
