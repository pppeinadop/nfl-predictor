"""
Proyecta estadísticas individuales (yardas de pase, carrera, recepciones, etc.)
usando el promedio móvil reciente de cada jugador, y las compara contra las
líneas de props de las casas de apuestas para sugerir Over/Under.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import config
import data_fetch

STAT_COLUMN_MAP = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}


def american_to_implied_prob(price) -> float:
    """
    Convierte un precio moneyline americano a probabilidad implícita.
    Positivo (+150): prob = 100 / (price + 100).
    Negativo (-120): prob = -price / (-price + 100).
    Devuelve None si el precio no es válido.
    """
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


def player_projection(weekly_stats: pd.DataFrame, player_name: str, stat_col: str,
                       window: int = config.ROLLING_WINDOW_PLAYER) -> dict:
    """Media y desviación estándar de las últimas N actuaciones del jugador."""
    hist = weekly_stats[weekly_stats["player_display_name"] == player_name].sort_values("week")
    recent = hist[stat_col].tail(window)
    if recent.empty:
        return {"mean": None, "std": None, "n_games": 0}
    return {
        "mean": round(float(recent.mean()), 1),
        "std": round(float(recent.std(ddof=0)) if len(recent) > 1 else 0.0, 1),
        "n_games": len(recent),
    }


def _model_over_prob(mean: float, std: float, line: float) -> float:
    """
    Probabilidad (según una normal con esa media y desviación) de que el jugador
    supere la línea. Si la std es 0, se usa una std mínima razonable para no
    dividir por cero ni decir siempre 0%/100%.
    """
    if mean is None or line is None:
        return None
    s = std if std and std > 0 else 1.0
    return float(1.0 - norm.cdf(line, loc=mean, scale=s))


def recommend_prop(projection: dict, line: float, over_price=None, under_price=None) -> dict:
    """
    Compara la proyección del modelo contra la línea y el precio de la casa.
    Devuelve un dict con la recomendación y toda la información usada.

    Criterio: si la probabilidad que da el modelo (normal con media y std de las
    últimas N actuaciones) supera a la probabilidad implícita del precio por al
    menos EDGE_THRESHOLD_PROP, se marca OVER o UNDER. Si no, "Sin valor claro".
    """
    out = {
        "recomendacion": "Datos insuficientes",
        "prob_modelo_over": None,
        "prob_implicita_over": None,
        "prob_implicita_under": None,
        "edge_over": None,
        "edge_under": None,
        "z_score": None,
    }
    if projection["mean"] is None or line is None:
        return out

    std = projection["std"] if projection["std"] and projection["std"] > 0 else 1.0
    z = (projection["mean"] - line) / std
    p_over_model = _model_over_prob(projection["mean"], projection["std"], line)
    p_under_model = 1.0 - p_over_model

    p_over_imp = american_to_implied_prob(over_price)
    p_under_imp = american_to_implied_prob(under_price)

    out["prob_modelo_over"] = round(p_over_model, 3)
    out["prob_implicita_over"] = round(p_over_imp, 3) if p_over_imp is not None else None
    out["prob_implicita_under"] = round(p_under_imp, 3) if p_under_imp is not None else None
    out["z_score"] = round(z, 2)

    edge_over = p_over_model - p_over_imp if p_over_imp is not None else None
    edge_under = p_under_model - p_under_imp if p_under_imp is not None else None
    out["edge_over"] = round(edge_over, 3) if edge_over is not None else None
    out["edge_under"] = round(edge_under, 3) if edge_under is not None else None

    # Si no hay precios, no se puede juzgar el valor: se informa solo la probabilidad del modelo.
    if p_over_imp is None and p_under_imp is None:
        out["recomendacion"] = (
            f"Sin precio (modelo: over {out['prob_modelo_over']:.0%}, proyección {projection['mean']} vs línea {line})"
        )
        return out

    # Se compara contra el umbral de edge (diferencia mínima de probabilidad).
    best = None
    if edge_over is not None and edge_over >= config.EDGE_THRESHOLD_PROP:
        best = ("OVER", edge_over)
    if edge_under is not None and edge_under >= config.EDGE_THRESHOLD_PROP:
        if best is None or edge_under > best[1]:
            best = ("UNDER", edge_under)

    if best is None:
        out["recomendacion"] = (
            f"Sin valor claro (proyección {projection['mean']} vs línea {line}, "
            f"z={out['z_score']})"
        )
        return out

    side, edge = best
    out["recomendacion"] = (
        f"{side} {line} (proyección {projection['mean']}, "
        f"z={out['z_score']}, edge={edge:.1%})"
    )
    return out


def _best_lines_for_event(prop_odds: dict) -> dict:
    """
    Devuelve un dict {(jugador, mercado): {"point": ..., "over_price": ..., "under_price": ...}}
    con la línea que tenga el mejor edge potencial. Cuando hay varias casas, se
    promedia el point entre casas y se toman los precios Over/Under promedio.
    """
    acc = {}
    for bk in prop_odds.get("bookmakers", []):
        for market in bk.get("markets", []):
            mkey = market["key"]
            for oc in market.get("outcomes", []):
                player = oc.get("description")
                if not player:
                    continue
                point = oc.get("point")
                if point is None:
                    continue
                key = (player, mkey)
                slot = acc.setdefault(key, {"points": [], "over_prices": [], "under_prices": []})
                slot["points"].append(point)
                name = (oc.get("name") or "").lower()
                price = oc.get("price")
                if "over" in name:
                    slot["over_prices"].append(price)
                elif "under" in name:
                    slot["under_prices"].append(price)

    out = {}
    for key, slot in acc.items():
        if not slot["points"]:
            continue
        out[key] = {
            "point": round(float(np.mean(slot["points"])), 1),
            "over_price": round(float(np.mean(slot["over_prices"]))) if slot["over_prices"] else None,
            "under_price": round(float(np.mean(slot["under_prices"]))) if slot["under_prices"] else None,
            "n_books": len(slot["points"]),
        }
    return out


def build_prop_recommendations(season: int, event_id: str, players_of_interest: list) -> pd.DataFrame:
    """
    players_of_interest: lista de dicts como
      {"player_display_name": "Patrick Mahomes", "market": "player_pass_yds"}
    """
    weekly_stats = data_fetch.get_weekly_player_stats(season)

    try:
        prop_odds = data_fetch.get_player_prop_odds(event_id)
    except Exception as e:
        print(f"[Aviso] No se pudieron traer props de casas de apuestas: {e}")
        prop_odds = {}

    lines_by_player_market = _best_lines_for_event(prop_odds)

    rows = []
    for item in players_of_interest:
        name, market = item["player_display_name"], item["market"]
        stat_col = STAT_COLUMN_MAP.get(market)
        if stat_col is None or stat_col not in weekly_stats.columns:
            continue
        proj = player_projection(weekly_stats, name, stat_col)
        entry = lines_by_player_market.get((name, market), {})
        line = entry.get("point")
        rec = recommend_prop(proj, line,
                             over_price=entry.get("over_price"),
                             under_price=entry.get("under_price"))
        rows.append({
            "jugador": name,
            "mercado": market,
            "proyeccion_modelo": proj["mean"],
            "desviacion_std": proj["std"],
            "partidos_usados": proj["n_games"],
            "linea_casa": line,
            "precio_over": entry.get("over_price"),
            "precio_under": entry.get("under_price"),
            "prob_modelo_over": rec["prob_modelo_over"],
            "prob_implicita_over": rec["prob_implicita_over"],
            "edge_over": rec["edge_over"],
            "edge_under": rec["edge_under"],
            "recomendacion": rec["recomendacion"],
        })

    return pd.DataFrame(rows)