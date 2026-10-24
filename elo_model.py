"""
Modelo Elo para NFL/NCAAF: mantiene un rating por equipo, lo actualiza partido a
partido y lo traduce a probabilidad de victoria / spread estimado.

Los ratings se guardan y se leen de MongoDB (ver db.py), así que el histórico
semana a semana queda registrado y se puede consultar después.
"""
import numpy as np

import config
import db


def load_ratings(league: str = "nfl") -> dict:
    return db.load_ratings(league=league)


def save_ratings(ratings: dict, league: str = "nfl", season: int = None, week: int = None):
    db.save_ratings(ratings, league=league, season=season, week=week)


def get_rating(ratings: dict, team: str) -> float:
    return ratings.get(team, config.ELO_INITIAL)


def win_probability(elo_a: float, elo_b: float) -> float:
    """Probabilidad de que el equipo A gane, dado el rating de ambos (A ya incluye home adv si aplica)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def predicted_spread(elo_a: float, elo_b: float, points_per_spread: float = None) -> float:
    """
    Spread estimado a favor de A (positivo = A favorito por X puntos).

    points_per_spread: cuántos puntos Elo equivalen a 1 punto de margen.
    Si es None, se usa config.ELO_POINTS_PER_SPREAD (default NFL).
    """
    if points_per_spread is None:
        points_per_spread = getattr(config, "ELO_POINTS_PER_SPREAD", 25.0)
    if points_per_spread <= 0:
        points_per_spread = 25.0
    return (elo_a - elo_b) / points_per_spread


def update_ratings(ratings: dict, home_team: str, away_team: str,
                    home_score: int, away_score: int) -> dict:
    """
    Actualiza los ratings Elo tras conocer el resultado real de un partido.
    Los empates se tratan como 0.5 (poco frecuentes en NFL, comunes en NCAAF).
    """
    elo_home = get_rating(ratings, home_team) + config.ELO_HOME_ADVANTAGE
    elo_away = get_rating(ratings, away_team)

    exp_home = win_probability(elo_home, elo_away)
    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    k = config.ELO_K
    if config.ELO_MOV_MULTIPLIER:
        margin = abs(home_score - away_score)
        if home_score > away_score:
            elo_diff_winner = elo_home - elo_away
        elif away_score > home_score:
            elo_diff_winner = elo_away - elo_home
        else:
            elo_diff_winner = 0.0
        mov_mult = np.log(margin + 1) * (2.2 / ((elo_diff_winner * 0.001) + 2.2))
        k = config.ELO_K * mov_mult

    delta = k * (actual_home - exp_home)

    base_home = get_rating(ratings, home_team)
    base_away = get_rating(ratings, away_team)
    ratings[home_team] = round(base_home + delta, 2)
    ratings[away_team] = round(base_away - delta, 2)
    return ratings


def matchup_prediction(ratings: dict, home_team: str, away_team: str) -> dict:
    """Devuelve prob. de victoria y spread estimado para un partido futuro."""
    elo_home = get_rating(ratings, home_team) + config.ELO_HOME_ADVANTAGE
    elo_away = get_rating(ratings, away_team)
    prob_home = win_probability(elo_home, elo_away)
    spread_home = predicted_spread(elo_home, elo_away)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(prob_home, 3),
        "away_win_prob": round(1 - prob_home, 3),
        "model_spread_home": round(spread_home, 1),
    }