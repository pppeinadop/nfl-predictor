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
import requests

import config
import team_stats


ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/scoreboard"
)


def _games_api():
    configuration = cfbd.Configuration()
    api_key = str(config.CFBD_API_KEY).strip()
    token = api_key[7:].strip() if api_key.lower().startswith("bearer ") else api_key
    if not token or token == "TU_API_KEY_AQUI":
        raise ValueError(
            "CFBD_API_KEY no esta configurada. Agregala a los secretos "
            "de la aplicacion para consultar datos NCAAF."
        )

    configuration.api_key["Authorization"] = token
    configuration.api_key_prefix["Authorization"] = "Bearer"
    return cfbd.GamesApi(cfbd.ApiClient(configuration))

def _espn_score_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_espn_games(payload, requested_week=None):
    rows = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        competitors = competition.get("competitors") or []
        home = next(
            (item for item in competitors if item.get("homeAway") == "home"),
            None,
        )
        away = next(
            (item for item in competitors if item.get("homeAway") == "away"),
            None,
        )
        if home is None or away is None:
            continue

        week_info = event.get("week") or {}
        game_week = week_info.get("number", requested_week)
        if game_week is None:
            continue

        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        home_name = home_team.get("location") or home_team.get("displayName")
        away_name = away_team.get("location") or away_team.get("displayName")
        if not home_name or not away_name:
            continue

        status = (competition.get("status") or {}).get("type") or {}
        completed = bool(status.get("completed"))
        rows.append({
            "week": int(game_week),
            "home_team": home_name,
            "away_team": away_name,
            "home_score": _espn_score_value(home.get("score")) if completed else None,
            "away_score": _espn_score_value(away.get("score")) if completed else None,
            "start_date": competition.get("date") or event.get("date"),
            "completed": completed,
            "conference_game": bool(competition.get("conferenceCompetition")),
        })

    return rows


def _valid_team_name(value):
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "none", "nan", "null"}


def _clean_schedule(frame):
    if frame.empty:
        frame.attrs["invalid_game_count"] = 0
        return frame

    required = {"week", "home_team", "away_team"}
    if not required.issubset(frame.columns):
        cleaned = frame.iloc[0:0].copy()
        cleaned.attrs.update(frame.attrs)
        cleaned.attrs["invalid_game_count"] = len(frame)
        return cleaned

    valid_rows = (
        frame["week"].notna()
        & frame["home_team"].map(_valid_team_name)
        & frame["away_team"].map(_valid_team_name)
    )
    cleaned = frame.loc[valid_rows].copy()
    cleaned.attrs.update(frame.attrs)
    cleaned.attrs["invalid_game_count"] = (
        int(frame.attrs.get("invalid_game_count", 0))
        + int((~valid_rows).sum())
    )
    return cleaned


def _fetch_espn_schedule(season: int, week: int = None) -> pd.DataFrame:
    """Fetch FBS games and results from ESPN's public scoreboard as a CFBD fallback."""
    weeks = [int(week)] if week is not None else range(1, 16)
    rows = []
    for requested_week in weeks:
        response = requests.get(
            ESPN_SCOREBOARD_URL,
            params={
                "year": int(season),
                "seasontype": 2,
                "week": requested_week,
                "groups": 80,
                "limit": 500,
            },
            headers={"User-Agent": "nfl-predictor/1.0"},
            timeout=(5, 20),
        )
        response.raise_for_status()
        rows.extend(_parse_espn_games(response.json(), requested_week))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["week", "home_team", "away_team"], keep="last"
        )
        frame["week"] = pd.to_numeric(
            frame["week"], errors="coerce"
        ).astype("Int64")
    return _clean_schedule(frame)


def _error_summary(error):
    status = getattr(error, "status", None)
    if status is None and isinstance(error, requests.HTTPError):
        status = getattr(error.response, "status_code", None)
    suffix = f" (HTTP {status})" if status else ""
    return f"{type(error).__name__}{suffix}"


def _fetch_schedule(season: int, week: int = None, division: str = "fbs") -> pd.DataFrame:
    """
    Calendario de la temporada (o de una semana puntual) para la división FBS
    (la que cubren la mayoría de casas de apuestas).
    """
    kwargs = {"year": season, "season_type": "regular"}
    if week is not None:
        kwargs["week"] = int(week)
        
    games = _games_api().get_games(**kwargs)
    
    rows = []
    for g in games:
        div_home = getattr(g, "home_division", "fbs")
        if division and div_home and div_home.lower() != division.lower():
            continue
            
        g_week = getattr(g, "week", None)
        if g_week is None:
            continue
            
        rows.append({
            "week": int(g_week),
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_score": g.home_points,
            "away_score": g.away_points,
            "start_date": getattr(g, "start_date", None),
            "completed": getattr(g, "completed", None),
            "conference_game": getattr(g, "conference_game", False),
        })
        
    df = pd.DataFrame(rows)
    if not df.empty and "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    return _clean_schedule(df)


_SCHEDULE_CACHE = {}


def get_schedule(season: int, week: int = None, division: str = "fbs") -> pd.DataFrame:
    """Fetch CFBD first, falling back to ESPN for FBS schedules and results."""
    key = (season, week, division)
    if key not in _SCHEDULE_CACHE:
        cfbd_error = None
        provider = "CFBD"
        invalid_game_count = 0
        try:
            schedule = _fetch_schedule(season, week, division)
            invalid_game_count += int(
                schedule.attrs.get("invalid_game_count", 0)
            )
        except Exception as error:
            cfbd_error = error
            schedule = pd.DataFrame()

        if schedule.empty and division == "fbs":
            provider = "ESPN"
            detail = (
                f"CFBD no disponible ({_error_summary(cfbd_error)}); "
                if cfbd_error else "CFBD no devolvió partidos; "
            )
            try:
                schedule = _fetch_espn_schedule(season, week)
                invalid_game_count += int(
                    schedule.attrs.get("invalid_game_count", 0)
                )
            except requests.RequestException as error:
                raise RuntimeError(
                    f"{detail}tampoco se pudo consultar el calendario FBS de ESPN "
                    f"({_error_summary(error)})."
                ) from error
            if not schedule.empty:
                print(
                    f"[Aviso] {detail}se usó el calendario FBS de ESPN "
                    f"para {season}"
                    f"{f', semana {week}' if week is not None else ''}."
                )
            else:
                print(
                    f"[Aviso] {detail}ESPN tampoco devolvió partidos "
                    f"para {season}"
                    f"{f', semana {week}' if week is not None else ''}."
                )
        elif schedule.empty and cfbd_error is not None:
            raise RuntimeError(
                f"No se pudo consultar el calendario {division} de CFBD "
                f"({_error_summary(cfbd_error)}); no hay proveedor alternativo "
                f"configurado para esa división."
            ) from cfbd_error

        schedule.attrs["provider"] = provider
        schedule.attrs["invalid_game_count"] = invalid_game_count
        schedule.attrs["provider_fallback_reason"] = (
            _error_summary(cfbd_error) if provider == "ESPN" and cfbd_error else None
        )
        _SCHEDULE_CACHE[key] = schedule
    return _SCHEDULE_CACHE[key].copy()


def get_completed_games(season: int, before_week: int = None) -> pd.DataFrame:
    """Partidos ya jugados. Con before_week, solo los de semanas anteriores a esa."""
    sched = get_schedule(season)
    if sched.empty:
        return pd.DataFrame()
        
    games = sched.dropna(subset=["home_score", "away_score"]).copy()
    if not games.empty and "week" in games.columns:
        games["week"] = pd.to_numeric(games["week"], errors="coerce").fillna(0).astype(int)
        
    if before_week is not None and not games.empty:
        try:
            bw = int(before_week)
            games = games[games["week"] < bw]
        except (ValueError, TypeError):
            pass
            
    return games


def get_week_games(season: int, week: int) -> pd.DataFrame:
    """Obtiene la semana pedida; usa el calendario completo si falla el filtro de CFBD."""
    week = int(week)
    games = get_schedule(season, week=week)
    if not games.empty:
        return games

    season_schedule = get_schedule(season)
    if season_schedule.empty or "week" not in season_schedule.columns:
        print(
            f"[Aviso] CFBD no devolvió calendario para la temporada {season}."
        )
        return pd.DataFrame()

    season_schedule = season_schedule.copy()
    season_schedule["week"] = pd.to_numeric(
        season_schedule["week"], errors="coerce"
    )
    games = season_schedule[season_schedule["week"] == week].copy()
    games.attrs.update(season_schedule.attrs)
    if not games.empty:
        print(
            f"[Aviso] CFBD no encontró la semana {week} con filtro directo; "
            "se recuperó desde el calendario completo."
        )
        return games

    available_weeks = sorted(
        season_schedule["week"].dropna().astype(int).unique().tolist()
    )
    print(
        f"[Aviso] CFBD no tiene partidos en la semana {week} de {season}. "
        f"Semanas disponibles: {available_weeks or 'ninguna'}."
    )
    return pd.DataFrame()


def _previous_season_games(season: int):
    """Partidos de la temporada anterior (sirven como punto de partida). None si falla."""
    if not config.CARRY_OVER_PREVIOUS_SEASON:
        return None
    try:
        prev_season = int(season) - 1
        sched = get_schedule(prev_season)
        if sched.empty:
            return None
        games = sched.dropna(subset=["home_score", "away_score"]).copy()
        if not games.empty and "week" in games.columns:
            games["week"] = pd.to_numeric(games["week"], errors="coerce").fillna(0).astype(int)
        return games
    except Exception as e:
        print(f"[Aviso] No se pudo cargar la temporada anterior: {e}")
        return None


def get_team_scoring_averages(season: int, window: int = config.ROLLING_WINDOW_TEAM,
                             before_week: int = None) -> pd.DataFrame:
    """Igual que en data_fetch.py (ver team_stats.py)."""
    games = get_completed_games(season, before_week=before_week)
    return team_stats.build_team_averages(games, _previous_season_games(season),
                                        default_league_avg=24.0, window=window)
