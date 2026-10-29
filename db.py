"""
Toda la interaccion con MongoDB vive aqui. Colecciones usadas:

  - elo_ratings   : rating actual de cada equipo (1 doc por liga+equipo)
  - elo_history   : snapshot del rating de cada equipo, por semana
  - weekly_picks  : una fila por partido/semana/liga
  - player_props  : una fila por jugador/mercado/semana (NFL)
  - market_cache  : consenso de mercado (odds) por liga/semana, para reusar sin
                    gastar cuota de The Odds API al regenerar semanas.

Todo queda separado por 'league' ("nfl" o "ncaaf") para que no se mezclen
los ratings ni los picks de ambas ligas.
"""
from datetime import datetime, timezone
from copy import deepcopy
import re

import pandas as pd
from pymongo import MongoClient, ASCENDING

import config

_client = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(config.MONGO_URI)
    return _client


def get_db():
    return get_client()[config.MONGO_DB_NAME]


def ensure_indexes():
    db = get_db()
    db.weekly_picks.create_index(
        [("league", ASCENDING), ("season", ASCENDING), ("week", ASCENDING),
         ("home_team", ASCENDING), ("away_team", ASCENDING)],
        unique=True,
    )
    db.player_props.create_index(
        [("league", ASCENDING), ("season", ASCENDING), ("week", ASCENDING),
         ("jugador", ASCENDING), ("mercado", ASCENDING)],
        unique=True,
    )
    db.elo_history.create_index(
        [("league", ASCENDING), ("season", ASCENDING), ("week", ASCENDING), ("team", ASCENDING)],
        unique=True,
    )
    db.elo_ratings.create_index([("league", ASCENDING), ("team", ASCENDING)], unique=True)
    db.market_cache.create_index(
        [("league", ASCENDING), ("season", ASCENDING), ("week", ASCENDING)],
        unique=True,
    )
    db.prediction_versions.create_index(
        [("league", ASCENDING), ("season", ASCENDING), ("week", ASCENDING),
         ("home_team", ASCENDING), ("away_team", ASCENDING), ("version", ASCENDING)],
        unique=True,
    )


# ---------- Elo ----------

def load_ratings(league: str = "nfl") -> dict:
    db = get_db()
    return {doc["team"]: doc["rating"] for doc in db.elo_ratings.find({"league": league})}


def save_ratings(ratings: dict, league: str = "nfl", season: int = None, week: int = None):
    db = get_db()
    now = datetime.now(timezone.utc)
    for team, rating in ratings.items():
        db.elo_ratings.update_one(
            {"league": league, "team": team},
            {"$set": {"rating": rating, "updated_at": now}},
            upsert=True,
        )
        if season is not None and week is not None:
            db.elo_history.update_one(
                {"league": league, "season": season, "week": week, "team": team},
                {"$set": {"rating": rating, "updated_at": now}},
                upsert=True,
            )


def get_elo_history(league: str = "nfl", team: str = None) -> pd.DataFrame:
    db = get_db()
    query = {"league": league}
    if team:
        query["team"] = team
    docs = list(db.elo_history.find(query, {"_id": 0}).sort([("season", 1), ("week", 1)]))
    return pd.DataFrame(docs)


# ---------- Picks de partidos ----------

def _split_matchup(partido: str) -> tuple:
    """Fallback: extrae (away, home) del string 'AWAY @ HOME'."""
    try:
        away, home = partido.split(" @ ", 1)
        return away.strip(), home.strip()
    except (ValueError, AttributeError):
        return None, None


def _mongo_safe_record(record: dict) -> dict:
    """Convierte escalares de pandas/numpy a tipos compatibles con BSON."""
    clean = {}
    for key, value in record.items():
        if value is pd.NA or value is pd.NaT:
            clean[key] = None
            continue
        if isinstance(value, pd.Timestamp):
            clean[key] = value.to_pydatetime()
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        try:
            if pd.isna(value):
                clean[key] = None
                continue
        except (TypeError, ValueError):
            pass
        clean[key] = value
    return clean


def _to_utc_datetime(value):
    """Convierte fechas almacenadas a datetime UTC; devuelve None si no puede."""
    if value is None:
        return None
    try:
        stamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(stamp):
            return None
        return stamp.to_pydatetime()
    except Exception:
        return None


def _projected_model_lines(record):
    projected_score = record.get("marcador_proyectado")
    match = re.fullmatch(
        r"\s*(.+?)\s+(\d+)\s+-\s+(\d+)\s+(.+?)\s*",
        str(projected_score or ""),
    )
    if not match:
        return {}
    home_name, home_score, away_score, away_name = match.groups()
    home_team = record.get("home_team")
    away_team = record.get("away_team")
    if home_name != home_team or away_name != away_team:
        return {}

    home_score = int(home_score)
    away_score = int(away_score)
    margin = home_score - away_score
    spread_line = (
        float(margin) - 0.5 if margin > 0
        else float(margin) + 0.5 if margin < 0
        else 0.0
    )
    if spread_line > 0:
        spread = f"{home_team} -{abs(spread_line):g}"
    elif spread_line < 0:
        spread = f"{away_team} -{abs(spread_line):g}"
    else:
        spread = "Pick'em (0)"

    return {
        "spread_modelo": spread,
        "spread_model_line": spread_line,
        "total_modelo": float(home_score + away_score),
    }


def _kickoff_from_record(record):
    kickoff = _to_utc_datetime(record.get("kickoff_utc"))
    if kickoff is not None:
        return kickoff
    fecha = record.get("fecha")
    hora = record.get("hora")
    if not fecha:
        return None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(getattr(config, "TIMEZONE", "America/New_York"))
        local = pd.to_datetime(f"{fecha} {hora or '23:59'}", errors="coerce")
        if pd.isna(local):
            return None
        return local.to_pydatetime().replace(tzinfo=tz).astimezone(timezone.utc)
    except Exception:
        return None


def _result_fields(official, record):
    """Calcula resultados usando la predicción oficial, nunca la versión recién generada."""
    if record.get("home_score_real") is None or record.get("away_score_real") is None:
        return {}
    try:
        home_pts = int(record["home_score_real"])
        away_pts = int(record["away_score_real"])
    except (TypeError, ValueError):
        return {}
    actual_margin = home_pts - away_pts
    actual_total = home_pts + away_pts
    model_margin = official.get("margin_final")
    projected_lines = _projected_model_lines(official)
    model_spread_line = record.get("spread_model_line")
    if model_spread_line is None:
        model_spread_line = projected_lines.get("spread_model_line")
    if model_spread_line is None:
        model_spread_line = official.get("spread_model_line")
    if model_spread_line is None:
        model_spread_line = model_margin
    model_total = record.get("total_modelo", official.get("total_modelo"))
    if "total_modelo" not in record or record["total_modelo"] is None:
        model_total = projected_lines.get(
            "total_modelo", official.get("total_modelo")
        )
    result = {
        "home_score_real": home_pts,
        "away_score_real": away_pts,
        "resultado_real": f"{official.get('home_team')} {home_pts} - {away_pts} {official.get('away_team')}",
        "resultado_actualizado_at": datetime.now(timezone.utc),
    }
    try:
        if model_margin is not None and float(model_margin) != 0 and actual_margin != 0:
            result["acierto_ganador"] = bool((actual_margin > 0) == (float(model_margin) > 0))
        else:
            result["acierto_ganador"] = None
        if model_margin is not None:
            result["error_margen"] = round(float(model_margin) - actual_margin, 1)
            if model_spread_line is not None and float(model_spread_line) != 0:
                diff = (
                    actual_margin - float(model_spread_line)
                    if float(model_spread_line) > 0
                    else float(model_spread_line) - actual_margin
                )
                result["acierto_spread"] = None if diff == 0 else bool(diff > 0)
            else:
                result["acierto_spread"] = None
        if model_total is not None:
            result["error_total"] = round(float(model_total) - actual_total, 1)
        market_total = official.get("market_total_line")
        if model_total is not None and market_total is not None and float(model_total) != float(market_total):
            if float(model_total) > float(market_total):
                diff_total = actual_total - float(market_total)
            else:
                diff_total = float(market_total) - actual_total
            result["acierto_total"] = None if diff_total == 0 else bool(diff_total > 0)
        else:
            result["acierto_total"] = None
    except (TypeError, ValueError):
        pass
    return result


def save_weekly_picks(df: pd.DataFrame, season: int, week: int, league: str = "nfl",
                      prediction_mode: str = "official") -> dict:
    """Guarda pronósticos sin sobrescribir el oficial.

    official: crea el pronóstico una sola vez antes del kickoff; las líneas de
        mercado pueden actualizarse después sin cambiar los valores del modelo.
    update: agrega una versión prepartido inmutable en prediction_versions.
    """
    if prediction_mode not in {"official", "update"}:
        raise ValueError("prediction_mode debe ser 'official' o 'update'.")
    db = get_db()
    now = datetime.now(timezone.utc)
    summary = {
        "inserted": 0,
        "updated_versions": 0,
        "updated_market": 0,
        "updated_projection": 0,
        "locked": 0,
        "skipped": 0,
    }
    market_fields = (
        "spread_consenso",
        "total_consenso",
        "ml_home",
        "ml_away",
        "casas_usadas",
        "market_spread_line",
        "market_total_line",
        "odds_cache_updated_at",
    )
    projection_fields = (
        "spread_modelo",
        "spread_model_line",
        "total_modelo",
    )
    value_fields = (
        "valor_spread",
        "valor_total",
    )

    for source in df.to_dict("records"):
        record = _mongo_safe_record(deepcopy(source))
        record.pop("_id", None)
        record["season"] = int(season)
        record["week"] = int(week)
        record["league"] = league
        home_team = record.get("home_team")
        away_team = record.get("away_team")
        if not home_team or not away_team:
            away_parsed, home_parsed = _split_matchup(record.get("partido", ""))
            home_team = home_team or home_parsed
            away_team = away_team or away_parsed
        if not home_team or not away_team:
            summary["skipped"] += 1
            continue

        query = {"league": league, "season": season, "week": week,
                 "home_team": home_team, "away_team": away_team}
        kickoff = _kickoff_from_record(record)
        record["home_team"] = home_team
        record["away_team"] = away_team
        record["kickoff_utc"] = kickoff
        record["prediction_generated_at"] = now
        record["prediction_mode"] = prediction_mode

        official = db.weekly_picks.find_one(query)
        if prediction_mode == "official":
            if official:
                # Preserve the official forecast while refreshing independently sourced market lines.
                same_projection = (
                    record.get("marcador_proyectado")
                    == official.get("marcador_proyectado")
                )
                result_record = record.copy()
                official_projection = _projected_model_lines(official)
                if official_projection:
                    result_record.update(official_projection)
                elif not same_projection:
                    result_record.pop("spread_model_line", None)
                updates = _result_fields(official, result_record)
                market_updates = {
                    field: record[field]
                    for field in market_fields
                    if record.get(field) is not None
                }
                if market_updates:
                    updates.update(market_updates)
                    summary["updated_market"] += 1
                projection_source = official_projection or (
                    record if same_projection else {}
                )
                projection_updates = {
                    field: projection_source[field]
                    for field in projection_fields
                    if projection_source.get(field) is not None
                }
                if same_projection:
                    projection_updates.update({
                        field: record[field]
                        for field in value_fields
                        if record.get(field) is not None
                    })
                if projection_updates:
                    updates.update(projection_updates)
                    summary["updated_projection"] += 1
                if updates:
                    db.weekly_picks.update_one(query, {"$set": updates})
                else:
                    summary["skipped"] += 1
                continue
            if kickoff is None or kickoff <= now:
                summary["locked"] += 1
                continue
            record["official_created_at"] = now
            record["official_version"] = 1
            db.weekly_picks.insert_one(record)
            summary["inserted"] += 1
            continue

        # Una actualización requiere predicción oficial y kickoff futuro.
        if not official:
            summary["skipped"] += 1
            continue
        official_kickoff = _kickoff_from_record(official)
        if (kickoff is None or kickoff <= now or
                official_kickoff is None or official_kickoff <= now):
            summary["locked"] += 1
            continue
        version = db.prediction_versions.count_documents(query) + 1
        record["version"] = version
        record["official_created_at"] = official.get("official_created_at")
        record["official_prediction_id"] = str(official.get("_id"))
        record["version_created_at"] = now
        db.prediction_versions.insert_one(record)
        summary["updated_versions"] += 1

    return summary


def sync_final_score(league: str, season: int, week: int, home_team: str,
                     away_team: str, home_score: int, away_score: int) -> bool:
    """Actualiza únicamente el marcador y la evaluación de una predicción oficial (con búsqueda flexible)."""
    db = get_db()
    
    # Intento 1: Búsqueda exacta estándar
    query = {
        "league": league,
        "season": int(season),
        "week": int(week),
        "home_team": home_team,
        "away_team": away_team,
    }
    official = db.weekly_picks.find_one(query)
    
    # Intento 2: Si falla, buscar de forma flexible por subcadena (por si la API trae apodos)
    if not official:
        candidates = list(db.weekly_picks.find({
            "league": league,
            "season": int(season),
            "week": int(week)
        }))
        for doc in candidates:
            db_home = str(doc.get("home_team", "")).lower()
            db_away = str(doc.get("away_team", "")).lower()
            if (home_team.lower() in db_home or db_home in home_team.lower()) and \
               (away_team.lower() in db_away or db_away in away_team.lower()):
                official = doc
                query = {"_id": doc["_id"]}
                break

    if not official:
        return False

    result = _result_fields(official, {
        "home_score_real": home_score,
        "away_score_real": away_score,
    })
    if not result:
        return False
    result.update(_projected_model_lines(official))
        
    db.weekly_picks.update_one(query, {"$set": result})
    return True


def get_prediction_versions(season: int, week: int, league: str = "nfl",
                           home_team: str = None, away_team: str = None) -> pd.DataFrame:
    query = {"league": league, "season": season, "week": week}
    if home_team:
        query["home_team"] = home_team
    if away_team:
        query["away_team"] = away_team
    docs = list(get_db().prediction_versions.find(query, {"_id": 0}).sort("version", 1))
    return pd.DataFrame(docs)


def get_weekly_picks(season: int, week: int, league: str = "nfl") -> pd.DataFrame:
    db = get_db()
    docs = list(
        db.weekly_picks
        .find({"league": league, "season": season, "week": week}, {"_id": 0})
        .sort([("partido", 1)])
    )
    return pd.DataFrame(docs)


# ---------- Props de jugadores ----------

def save_player_props(df: pd.DataFrame, season: int, week: int, league: str = "nfl"):
    if df.empty:
        return
    db = get_db()
    now = datetime.now(timezone.utc)
    for record in df.to_dict("records"):
        record.pop("_id", None)
        record["season"] = season
        record["week"] = week
        record["league"] = league
        record["updated_at"] = now
        db.player_props.update_one(
            {"league": league, "season": season, "week": week,
             "jugador": record["jugador"], "mercado": record["mercado"]},
            {"$set": record},
            upsert=True,
        )


def get_player_props(season: int, week: int, league: str = "nfl") -> pd.DataFrame:
    db = get_db()
    docs = list(
        db.player_props
        .find({"league": league, "season": season, "week": week}, {"_id": 0})
        .sort([("jugador", 1), ("mercado", 1)])
    )
    return pd.DataFrame(docs)


# ---------- Caché de mercado (odds) ----------

def save_market_cache(df: pd.DataFrame, season: int, week: int, league: str = "nfl"):
    """
    Guarda el consenso de mercado (spread/total/ML) de una semana, para poder
    regenerar los picks sin volver a llamar a The Odds API (útil con --skip-odds).
    """
    if df is None or df.empty:
        return
    db = get_db()
    now = datetime.now(timezone.utc)
    records = df.to_dict("records")
    for rec in records:
        for k, v in list(rec.items()):
            if hasattr(v, "item"):
                try:
                    rec[k] = v.item()
                except Exception:
                    pass
    db.market_cache.update_one(
        {"league": league, "season": season, "week": week},
        {"$set": {"records": records, "updated_at": now}},
        upsert=True,
    )


def get_market_cache(season: int, week: int, league: str = "nfl") -> pd.DataFrame:
    """Lee el consenso de mercado guardado. DataFrame vacío si no hay cache."""
    db = get_db()
    doc = db.market_cache.find_one(
        {"league": league, "season": season, "week": week}, {"_id": 0}
    )
    if not doc or not doc.get("records"):
        return pd.DataFrame()
    frame = pd.DataFrame(doc["records"])
    frame.attrs["cache_updated_at"] = doc.get("updated_at")
    return frame


def has_market_cache(season: int, week: int, league: str = "nfl") -> bool:
    db = get_db()
    return db.market_cache.count_documents(
        {"league": league, "season": season, "week": week}, limit=1
    ) > 0


# ---------- Utilidad ----------

def list_available_weeks(season: int = None, league: str = "nfl") -> list:
    db = get_db()
    query = {"league": league}
    if season:
        query["season"] = season
    return sorted(db.weekly_picks.distinct("week", query))
