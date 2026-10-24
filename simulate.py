"""
Simulación Monte Carlo de partidos.

Cada partido se simula N veces a partir de la media del modelo y una
desviación estándar empírica (calibrada por liga). Se generan:
  - percentiles del margen y del total
  - probabilidad de victoria del local (simulada)
  - (en la app) probabilidad de cubrir una línea y de over/under
"""
import numpy as np
from scipy.stats import norm

import config


def _std_margin(league: str) -> float:
    if league == "ncaaf":
        return getattr(config, "SIM_MARGIN_STD_NCAAF", 18.9)
    return getattr(config, "SIM_MARGIN_STD_NFL", 13.2)


def _std_total(league: str) -> float:
    if league == "ncaaf":
        return getattr(config, "SIM_TOTAL_STD_NCAAF", 19.9)
    return getattr(config, "SIM_TOTAL_STD_NFL", 14.0)


def simulate_and_summarize(league: str, margin_mean: float,
                            total_mean: float, n: int = None) -> dict:
    """
    Simula N partidos con margen ~ N(margin_mean, std) y total ~ N(total_mean, std).
    Devuelve percentiles y probabilidades listos para guardar en Mongo.
    """
    if n is None:
        n = getattr(config, "SIM_N_SAMPLES", 10000)

    margin_std = _std_margin(league)
    total_std = _std_total(league)

    rng = np.random.default_rng()
    margins = rng.normal(margin_mean, margin_std, n)
    totals = rng.normal(total_mean, total_std, n)

    return {
        "sim_margin_mean": round(float(margin_mean), 2),
        "sim_margin_std": round(float(margin_std), 2),
        "sim_margin_p10": round(float(np.percentile(margins, 10)), 1),
        "sim_margin_p50": round(float(np.percentile(margins, 50)), 1),
        "sim_margin_p90": round(float(np.percentile(margins, 90)), 1),
        "sim_total_mean": round(float(total_mean), 2),
        "sim_total_std": round(float(total_std), 2),
        "sim_total_p10": round(float(np.percentile(totals, 10)), 1),
        "sim_total_p50": round(float(np.percentile(totals, 50)), 1),
        "sim_total_p90": round(float(np.percentile(totals, 90)), 1),
        "sim_home_win_prob": round(float((margins > 0).mean()), 4),
        "sim_n_samples": n,
    }


def prob_cover(margin_mean: float, margin_std: float, line_home: float) -> dict:
    """
    Con N(margin_mean, margin_std), devuelve P(local cubre | línea del local).

    line_home en convención "positivo = local favorito":
        +3.5 → el local debe ganar por > 3.5
        -3.5 → el local debe ganar por > -3.5 (o perder por < 3.5)
    """
    if margin_std <= 0:
        margin_std = 1.0
    p_home = float(1.0 - norm.cdf(line_home, loc=margin_mean, scale=margin_std))
    p_away = float(norm.cdf(line_home, loc=margin_mean, scale=margin_std))
    return {"home": p_home, "away": p_away, "push": 0.0}


def prob_over(total_mean: float, total_std: float, line: float) -> dict:
    """Con N(total_mean, total_std), devuelve P(over) y P(under)."""
    if total_std <= 0:
        total_std = 1.0
    p_over = float(1.0 - norm.cdf(line, loc=total_mean, scale=total_std))
    p_under = float(norm.cdf(line, loc=total_mean, scale=total_std))
    return {"over": p_over, "under": p_under, "push": 0.0}
