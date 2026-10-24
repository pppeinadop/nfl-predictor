"""
Promedios de puntos anotados/permitidos por equipo (compartido por NFL y NCAAF).

Protecciones contra el ruido de pocos partidos:
  1) Peso a lo reciente (decaimiento exponencial, RECENCY_HALFLIFE_GAMES).
  2) Prior de la temporada anterior: el promedio actual se mezcla con lo que el equipo
     hizo el año pasado (regresado a la media), con un peso de PRIOR_STRENGTH_GAMES
     "partidos virtuales". Con 2 partidos jugados manda el prior; con 12, casi solo los
     datos nuevos. Así hacen las casas: nadie arranca de cero en la semana 1.
  3) Ajuste por calendario (SOS), limitado y activado gradualmente.
  4) Los splits local/visitante solo aportan la DIFERENCIA casa-vs-visita (encogida según
     cuántos partidos hay), no el nivel del equipo, para que 1-2 partidos en casa no
     definan por sí solos cómo juega un equipo de local.
"""
import pandas as pd

import config

COLUMNS = ["team", "pf_avg", "pa_avg", "pf_avg_home", "pa_avg_home",
           "pf_avg_away", "pa_avg_away", "home_games_n", "away_games_n"]


def _smooth(s, window):
    """Promedio con más peso a los partidos recientes (ver RECENCY_HALFLIFE_GAMES)."""
    hl = config.RECENCY_HALFLIFE_GAMES
    if hl:
        return s.ewm(halflife=hl, adjust=True).mean()
    return s.rolling(window, min_periods=1).mean()


def _long(games: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido: puntos a favor (pf) y en contra (pa)."""
    home = games[["week", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})
    home["is_home"] = True
    away = games[["week", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})
    away["is_home"] = False
    out = pd.concat([home, away], ignore_index=True)
    out[["pf", "pa"]] = out[["pf", "pa"]].astype(float)
    return out.sort_values(["team", "week"], kind="stable").reset_index(drop=True)


def build_team_averages(games, prev_games=None, default_league_avg: float = 22.0,
                        window: int = config.ROLLING_WINDOW_TEAM) -> pd.DataFrame:
    have_cur = games is not None and not games.empty
    have_prev = prev_games is not None and not prev_games.empty
    if not have_cur and not have_prev:
        return pd.DataFrame(columns=COLUMNS)

    cur = _long(games) if have_cur else None
    prev = _long(prev_games) if have_prev else None

    # --- promedio de liga (base de referencia) ---
    if have_cur and len(cur) >= 60:
        league_avg = float(cur["pf"].mean())
    elif have_prev:
        league_avg = float(prev["pf"].mean())
    elif have_cur:
        league_avg = float(cur["pf"].mean())
    else:
        league_avg = default_league_avg

    # --- datos de esta temporada (suavizados hacia lo reciente) ---
    if have_cur:
        cur["pf_s"] = cur.groupby("team")["pf"].transform(lambda s: _smooth(s, window))
        cur["pa_s"] = cur.groupby("team")["pa"].transform(lambda s: _smooth(s, window))
        cur_stats = cur.groupby("team").agg(n=("pf", "size"), pf_s=("pf_s", "last"), pa_s=("pa_s", "last"))
    else:
        cur_stats = pd.DataFrame(columns=["n", "pf_s", "pa_s"], dtype=float)

    # --- prior: lo que hizo cada equipo la temporada anterior, regresado a la media ---
    if have_prev:
        pm = prev.groupby("team").agg(pf=("pf", "mean"), pa=("pa", "mean"))
        prev_lg = float(prev["pf"].mean())
        w = config.SEASON_CARRYOVER_WEIGHT
        prior = pd.DataFrame({"prior_pf": league_avg + (pm["pf"] - prev_lg) * w,
                              "prior_pa": league_avg + (pm["pa"] - prev_lg) * w})
    else:
        prior = pd.DataFrame(columns=["prior_pf", "prior_pa"], dtype=float)

    teams = sorted(set(cur_stats.index) | set(prior.index))
    n = cur_stats["n"].reindex(teams).fillna(0.0).astype(float)
    pf_s = cur_stats["pf_s"].reindex(teams).astype(float)
    pa_s = cur_stats["pa_s"].reindex(teams).astype(float)
    prior_pf = prior["prior_pf"].reindex(teams).astype(float).fillna(league_avg)
    prior_pa = prior["prior_pa"].reindex(teams).astype(float).fillna(league_avg)

    K = config.PRIOR_STRENGTH_GAMES
    pf_est = (n * pf_s.fillna(0.0) + K * prior_pf) / (n + K)
    pa_est = (n * pa_s.fillna(0.0) + K * prior_pa) / (n + K)

    # --- ajuste por calendario (SOS), usando las estimaciones ya estabilizadas ---
    off_f = pd.Series(1.0, index=teams)
    def_f = pd.Series(1.0, index=teams)
    if have_cur:
        opp = pd.concat([
            games[["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"}),
            games[["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opponent"}),
        ], ignore_index=True)
        opp["opp_pa"] = opp["opponent"].map(pa_est).fillna(league_avg)
        opp["opp_pf"] = opp["opponent"].map(pf_est).fillna(league_avg)
        sos = opp.groupby("team").agg(opp_def=("opp_pa", "mean"), opp_off=("opp_pf", "mean"))
        raw_off = (league_avg / sos["opp_def"].replace(0, league_avg)).reindex(teams).fillna(1.0)
        raw_def = (league_avg / sos["opp_off"].replace(0, league_avg)).reindex(teams).fillna(1.0)
        shrink = (n / config.SOS_FULL_STRENGTH_GAMES).clip(upper=1.0)
        off_f = (1 + (raw_off - 1) * shrink).clip(config.SOS_MIN_FACTOR, config.SOS_MAX_FACTOR)
        def_f = (1 + (raw_def - 1) * shrink).clip(config.SOS_MIN_FACTOR, config.SOS_MAX_FACTOR)
    pf_adj = pf_est * off_f
    pa_adj = pa_est * def_f

    # --- splits local / visitante ---
    # Se calcula la media SIMPLE de los últimos `window` partidos en esa condición y se
    # resta contra el baseline ajustado (pf_adj / pa_adj). Eso da la "diferencia de jugar
    # en casa / fuera" limpia, que luego se encoge según cuántos partidos hay.
    Ks = config.SPLIT_PRIOR_GAMES

    def _split(is_home: bool):
        n_s = pd.Series(0.0, index=teams)
        pf_v, pa_v = pf_adj.copy(), pa_adj.copy()
        if have_cur:
            d = cur[cur["is_home"] == is_home].sort_values(["team", "week"])
            if not d.empty:
                d_win = d.groupby("team").tail(window)
                g = (d_win.groupby("team")
                     .agg(n=("pf", "size"), pf_m=("pf", "mean"), pa_m=("pa", "mean"))
                     .reindex(teams))
                n_s = g["n"].fillna(0.0).astype(float)
                eff_pf = (g["pf_m"] - pf_adj).fillna(0.0)
                eff_pa = (g["pa_m"] - pa_adj).fillna(0.0)
                weight = n_s / (n_s + Ks)
                pf_v = pf_adj + eff_pf * weight
                pa_v = pa_adj + eff_pa * weight
        return pf_v, pa_v, n_s

    pf_h, pa_h, n_h = _split(True)
    pf_a, pa_a, n_a = _split(False)

    res = pd.DataFrame({
        "pf_avg": pf_adj.round(2), "pa_avg": pa_adj.round(2),
        "pf_avg_home": pf_h.round(2), "pa_avg_home": pa_h.round(2),
        "pf_avg_away": pf_a.round(2), "pa_avg_away": pa_a.round(2),
        "home_games_n": n_h.astype(int), "away_games_n": n_a.astype(int),
    }, index=teams)
    res.index.name = "team"
    return res.reset_index()[COLUMNS]