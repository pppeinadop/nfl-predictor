"""
Genera, para la semana indicada, en la liga indicada (nfl o ncaaf):
 - Marcador proyectado de cada partido
 - Spread y total proyectados por el modelo
 - Simulación Monte Carlo (percentiles de margen/total, prob. victoria)
 - Comparación contra las lineas de mercado -> señala "valor"
"""
import math

import pandas as pd

import config
import db
import elo_model
import data_fetch
import ncaaf_fetch
import adjustments
import simulate


def _stats_module(league: str):
    if league == "nfl":
        return data_fetch
    elif league == "ncaaf":
        return ncaaf_fetch
    raise ValueError(f"Liga no soportada: {league}. Usa 'nfl' o 'ncaaf'.")


def _margin_shrink_for(league: str) -> float:
    if league == "ncaaf":
        return getattr(config, "MARGIN_SHRINK_NCAAF", 1.25)
    return getattr(config, "MARGIN_SHRINK_NFL", 0.75)


def _elo_scale_for(league: str) -> float:
    if league == "ncaaf":
        return getattr(config, "ELO_POINTS_PER_SPREAD_NCAAF", 12.0)
    return getattr(config, "ELO_POINTS_PER_SPREAD_NFL", 25.0)


def _round_half(value: float) -> float:
    return math.floor(float(value) * 2 + 0.5) / 2


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


# ---------- Fecha / hora / slot ----------

def _game_date_time(game, league: str):
    tz = getattr(config, "TIMEZONE", "America/New_York")

    if league == "nfl":
        st_utc = game.get("start_time")
        if st_utc is not None and pd.notna(st_utc):
            try:
                dt = pd.to_datetime(st_utc, utc=True).tz_convert(tz)
                return dt.date().isoformat(), dt.strftime("%H:%M")
            except Exception:
                pass

        gameday = game.get("gameday")
        gametime = game.get("gametime")
        fecha = None
        hora = None
        if pd.notna(gameday):
            try:
                if pd.notna(gametime):
                    naive = pd.to_datetime(f"{gameday} {gametime}")
                    dt = naive.tz_localize("America/New_York").tz_convert(tz)
                    return dt.date().isoformat(), dt.strftime("%H:%M")
                else:
                    fecha = pd.to_datetime(gameday).date().isoformat()
            except Exception:
                fecha = None if not pd.notna(gameday) else str(gameday)
        if pd.notna(gametime):
            hora = str(gametime)[:5]
        return fecha, hora

    sd = game.get("start_date")
    if sd is None or (not isinstance(sd, str) and pd.isna(sd)):
        return None, None
    try:
        dt = pd.to_datetime(sd, utc=True).tz_convert(tz)
        return dt.date().isoformat(), dt.strftime("%H:%M")
    except Exception:
        return None, None


def _kickoff_utc(game, league: str):
    """Obtiene el inicio programado en UTC para bloquear pronósticos al kickoff."""
    field = "start_time" if league == "nfl" else "start_date"
    value = game.get(field)
    if value is None or pd.isna(value):
        return None
    try:
        stamp = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(stamp) else stamp.to_pydatetime()
    except Exception:
        return None


def _slot_label(fecha_iso, hora_hhmm) -> str:
    if not fecha_iso:
        return "Sin fecha"
    try:
        date_only = pd.to_datetime(fecha_iso, errors="coerce")
        if pd.isna(date_only):
            return "Sin fecha"
        dow = date_only.weekday()
    except Exception:
        return "Sin fecha"

    hour = 12
    if hora_hhmm:
        try:
            hour = int(str(hora_hhmm).split(":")[0])
        except (ValueError, IndexError):
            hour = 12

    if dow == 3:
        return "Jueves por la noche"
    if dow == 4:
        return "Viernes por la noche"
    if dow == 5:
        if hour < 15:
            return "Sábado temprano"
        if hour < 18:
            return "Sábado tarde"
        return "Sábado por la noche"
    if dow == 6:
        if hour < 15:
            return "Domingo temprano"
        if hour < 19:
            return "Domingo tarde"
        return "Domingo por la noche"
    if dow == 0:
        return "Lunes por la noche"
    if dow == 1:
        return "Martes"
    if dow == 2:
        return "Miércoles"
    return "Sin fecha"


# ---------- Records ----------

def _build_records(stats, season: int, before_week: int) -> dict:
    records = {}
    try:
        sched = stats.get_schedule(season)
    except Exception:
        return records
    if sched is None or sched.empty:
        return records
    if "home_score" not in sched.columns or "away_score" not in sched.columns:
        return records

    played = sched.dropna(subset=["home_score", "away_score"])
    if before_week is not None and "week" in played.columns:
        played = played[played["week"] < before_week]

    for _, g in played.iterrows():
        home = g["home_team"]
        away = g["away_team"]
        hs = int(g["home_score"])
        as_ = int(g["away_score"])
        hw, hl, ht = records.get(home, (0, 0, 0))
        aw, al, at_ = records.get(away, (0, 0, 0))
        if hs > as_:
            records[home] = (hw + 1, hl, ht)
            records[away] = (aw, al + 1, at_)
        elif hs < as_:
            records[home] = (hw, hl + 1, ht)
            records[away] = (aw + 1, al, at_)
        else:
            records[home] = (hw, hl, ht + 1)
            records[away] = (aw, al, at_ + 1)

    out = {}
    for team, (w, l, t) in records.items():
        out[team] = f"{w}-{l}" if t == 0 else f"{w}-{l}-{t}"
    return out


# ---------- Proyección ----------

def project_score(team_avgs: pd.DataFrame, home_team: str, away_team: str) -> tuple:
    league_avg = team_avgs["pf_avg"].mean() if not team_avgs.empty else 22.0

    def get_stats(team, side):
        r = team_avgs[team_avgs["team"] == team]
        if r.empty:
            return league_avg, league_avg
        row = r.iloc[0]
        n_col = "home_games_n" if side == "home" else "away_games_n"
        pf_col = "pf_avg_home" if side == "home" else "pf_avg_away"
        pa_col = "pa_avg_home" if side == "home" else "pa_avg_away"
        if pd.notna(row.get(n_col)) and row[n_col] >= config.MIN_SPLIT_GAMES:
            return float(row[pf_col]), float(row[pa_col])
        return float(row["pf_avg"]), float(row["pa_avg"])

    home_pf, home_pa = get_stats(home_team, "home")
    away_pf, away_pa = get_stats(away_team, "away")

    home_off_strength = home_pf / league_avg if league_avg > 0 else 1.0
    home_def_strength = home_pa / league_avg if league_avg > 0 else 1.0
    away_off_strength = away_pf / league_avg if league_avg > 0 else 1.0
    away_def_strength = away_pa / league_avg if league_avg > 0 else 1.0

    hfa = getattr(config, "HOME_FIELD_POINTS", 2.5)
    home_proj = (league_avg * home_off_strength * away_def_strength) + (hfa / 2)
    away_proj = (league_avg * away_off_strength * home_def_strength) - (hfa / 2)

    return round(home_proj, 1), round(away_proj, 1)


# ---------- Formato ----------

def format_spread(home_team: str, away_team: str, home_favorite_value) -> str:
    if home_favorite_value is None or pd.isna(home_favorite_value):
        return None
    value = _round_half(float(home_favorite_value))
    if value == 0:
        return "Pick'em (0)"
    line = format(abs(value), "g")
    favorite = home_team if value > 0 else away_team
    return f"{favorite} -{line}"


def format_pct(value) -> str:
    if value is None or pd.isna(value):
        return None
    return f"{round(float(value) * 100)}%"


def _int_scores(home_score: float, away_score: float, spread_home: float) -> tuple:
    h = int(math.floor(home_score + 0.5))
    a = int(math.floor(away_score + 0.5))
    if h == a:
        if spread_home > 0:
            h += 1
        else:
            a += 1
    return h, a


def _spread_line_from_projected_score(home_score: int, away_score: int) -> float:
    """Return a half-point spread the projected integer score would cover."""
    margin = home_score - away_score
    if margin > 0:
        return float(margin) - 0.5
    if margin < 0:
        return float(margin) + 0.5
    return 0.0


def _match_market_row(market: pd.DataFrame, home: str, away: str, league: str):
    if market is None or market.empty:
        return None
    if league == "nfl":
        home_full = config.TEAM_ABBR_TO_FULL.get(home, home)
        away_full = config.TEAM_ABBR_TO_FULL.get(away, away)
        m = market[(market["home_team"] == home_full) & (market["away_team"] == away_full)]
    else:
        home_match = data_fetch.fuzzy_match_team(home, market["home_team"].tolist())
        away_match = data_fetch.fuzzy_match_team(away, market["away_team"].tolist())
        if not home_match or not away_match:
            return None
        m = market[(market["home_team"] == home_match) & (market["away_team"] == away_match)]
    return m.iloc[0] if not m.empty else None


def _spread_cover_hit(model_spread_home: float, actual_margin: float):
    if model_spread_home is None or pd.isna(model_spread_home):
        return None
    if actual_margin is None or pd.isna(actual_margin):
        return None
    if model_spread_home == 0:
        return None

    if model_spread_home > 0:
        diff = actual_margin - model_spread_home
    else:
        diff = model_spread_home - actual_margin

    if diff == 0:
        return None
    return bool(diff > 0)


def _market_value_labels(
    home_team, away_team, model_spread_home, model_total,
    market_spread_home, market_total,
) -> dict:
    """Compare model-derived lines with market consensus in home-margin convention."""
    labels = {}
    if (
        model_spread_home is not None
        and market_spread_home is not None
        and pd.notna(model_spread_home)
        and pd.notna(market_spread_home)
    ):
        diff_spread = float(model_spread_home) - float(market_spread_home)
        edge_spread = _round_half(abs(diff_spread))
        labels["valor_spread"] = (
            f"{home_team} +{edge_spread}"
            if diff_spread > config.EDGE_THRESHOLD_SPREAD
            else (
                f"{away_team} +{edge_spread}"
                if diff_spread < -config.EDGE_THRESHOLD_SPREAD
                else "Sin valor claro"
            )
        )
        if abs(diff_spread) >= config.EDGE_SUSPICIOUS_SPREAD:
            labels["valor_spread"] += " ⚠"

    if (
        model_total is not None
        and market_total is not None
        and pd.notna(model_total)
        and pd.notna(market_total)
    ):
        diff_total = float(model_total) - float(market_total)
        edge_total = _round_half(abs(diff_total))
        if diff_total > config.EDGE_THRESHOLD_TOTAL:
            labels["valor_total"] = f"Over (+{edge_total})"
        elif diff_total < -config.EDGE_THRESHOLD_TOTAL:
            labels["valor_total"] = f"Under ({edge_total})"
        else:
            labels["valor_total"] = "Sin valor claro"
        if abs(diff_total) >= config.EDGE_SUSPICIOUS_TOTAL:
            labels["valor_total"] += " ⚠"

    return labels


def _total_cover_hit(model_total: float, market_total: float, actual_total: float):
    if model_total is None or pd.isna(model_total):
        return None
    if market_total is None or pd.isna(market_total):
        return None
    if actual_total is None or pd.isna(actual_total):
        return None
    if model_total == market_total:
        return None

    if model_total > market_total:
        diff = actual_total - market_total
    else:
        diff = market_total - actual_total

    if diff == 0:
        return None
    return bool(diff > 0)


# ---------- Carga de mercado ----------

def _load_market(season: int, week: int, league: str, market_df=None) -> pd.DataFrame:
    if market_df is not None:
        return market_df
    try:
        cached = db.get_market_cache(season, week, league)
        if not cached.empty:
            return cached
    except Exception:
        pass
    try:
        sport_key = config.LEAGUES[league]["odds_sport_key"]
        odds_events = data_fetch.get_game_odds(sport_key)
        market = data_fetch.get_consensus_lines(odds_events)
        try:
            db.save_market_cache(market, season, week, league)
        except Exception:
            pass
        return market
    except Exception as e:
        print(f"[Aviso] No se pudieron traer lineas de casas de apuestas: {e}")
        return pd.DataFrame()


# ---------- Constructor principal ----------

def build_weekly_picks(season: int, week: int, league: str = "nfl",
                       market_df=None) -> pd.DataFrame:
    stats = _stats_module(league)
    ratings = elo_model.load_ratings(league)
    team_avgs = stats.get_team_scoring_averages(season, before_week=week)
    upcoming = stats.get_week_games(season, week)
    full_schedule = stats.get_schedule(season)
    date_col = "gameday" if league == "nfl" else "start_date"

    market = _load_market(season, week, league, market_df=market_df)
    market_cache_updated_at = market.attrs.get("cache_updated_at") if market is not None else None
    records = _build_records(stats, season, before_week=week)

    shrink = _margin_shrink_for(league)
    elo_scale = _elo_scale_for(league)
    ml_scale = getattr(config, "ML_PROB_SCALE", 6.5)

    results = []
    for _, game in upcoming.iterrows():
        home, away = game["home_team"], game["away_team"]

        if config.DISCARD_GAMES_UNKNOWN_TEAMS and (home not in ratings or away not in ratings):
            continue

        home_score, away_score = project_score(team_avgs, home, away)

        elo_home_val = elo_model.get_rating(ratings, home) + config.ELO_HOME_ADVANTAGE
        elo_away_val = elo_model.get_rating(ratings, away)
        elo_margin = elo_model.predicted_spread(
            elo_home_val, elo_away_val, points_per_spread=elo_scale
        )
        score_margin = home_score - away_score

        raw_margin = round(
            config.BLEND_WEIGHT_ELO * elo_margin + (1 - config.BLEND_WEIGHT_ELO) * score_margin, 1
        )
        blended_margin = round(raw_margin * shrink, 1)

        combined_total = home_score + away_score
        home_score_final = combined_total / 2 + blended_margin / 2
        away_score_final = combined_total / 2 - blended_margin / 2

        home_score_final, away_score_final, ajustes_notas = adjustments.apply_extra_adjustments(
            league, season, week, home, away, game, full_schedule, date_col,
            home_score_final, away_score_final
        )

        expected_model_total = _round_half(home_score_final + away_score_final)

        final_margin = home_score_final - away_score_final
        tiebreak = final_margin if final_margin != 0 else (elo_margin if elo_margin != 0 else 1)

        model_spread_home = round(final_margin, 1)
        if _round_half(model_spread_home) == 0:
            model_spread_home = 0.5 if tiebreak > 0 else -0.5

        home_final_int, away_final_int = _int_scores(home_score_final, away_score_final, model_spread_home)
        spread_model_line = _spread_line_from_projected_score(
            home_final_int, away_final_int
        )
        model_total = float(home_final_int + away_final_int)

        ajustes_delta = final_margin - blended_margin
        ml_margin = raw_margin + ajustes_delta
        home_win_prob = 1.0 / (1.0 + math.exp(-ml_margin / ml_scale))

        fecha, hora = _game_date_time(game, league)
        slot = _slot_label(fecha, hora)

        row = {
            "liga": config.LEAGUES[league]["label"],
            "semana": week,
            "partido": f"{away} @ {home}",
            "home_team": home,
            "away_team": away,
            "home_record": records.get(home, "0-0"),
            "away_record": records.get(away, "0-0"),
            "fecha": fecha,
            "hora": hora,
            "kickoff_utc": _kickoff_utc(game, league),
            "odds_cache_updated_at": market_cache_updated_at,
            "slot": slot,
            "marcador_proyectado": f"{home} {home_final_int} - {away_final_int} {away}",
            "prob_home": format_pct(home_win_prob),
            "prob_away": format_pct(1 - home_win_prob),
            "spread_modelo": format_spread(home, away, spread_model_line),
            "spread_model_line": spread_model_line,
            "total_modelo": model_total,
            "ajustes": "; ".join(ajustes_notas) if ajustes_notas else None,
            "margin_elo": round(elo_margin, 2),
            "margin_scores": round(score_margin, 2),
            "margin_blend_raw": round(raw_margin, 2),
            "margin_blend_shrunk": round(blended_margin, 2),
            "margin_final": round(final_margin, 2),
        }

        # --- Simulación Monte Carlo ---
        sim_out = simulate.simulate_and_summarize(
            league=league,
            margin_mean=model_spread_home,
            total_mean=expected_model_total,
        )
        row.update(sim_out)

        m = _match_market_row(market, home, away, league)

        has_result = pd.notna(game.get("home_score")) and pd.notna(game.get("away_score"))
        if has_result:
            home_pts, away_pts = int(game["home_score"]), int(game["away_score"])
            actual_margin = home_pts - away_pts
            actual_total = home_pts + away_pts
            row["resultado_real"] = f"{home} {home_pts} - {away_pts} {away}"
            row["home_score_real"] = home_pts
            row["away_score_real"] = away_pts
            row["resultado_actualizado_at"] = pd.Timestamp.now(tz="UTC").to_pydatetime()
            row["acierto_ganador"] = (
                bool((actual_margin > 0) == (model_spread_home > 0)) if actual_margin != 0 else None
            )
            row["error_margen"] = round(float(model_spread_home - actual_margin), 1)
            row["error_total"] = round(float(model_total - actual_total), 1)
            row["acierto_spread"] = _spread_cover_hit(spread_model_line, actual_margin)

            if m is not None and pd.notna(m.get("consensus_total")):
                row["acierto_total"] = _total_cover_hit(
                    model_total, m["consensus_total"], actual_total
                )

        if m is None:
            if config.DISCARD_GAMES_WITHOUT_MARKET and not has_result:
                continue
        else:
            row["spread_consenso"] = format_spread(home, away, -m["consensus_spread_home"]) \
                if pd.notna(m["consensus_spread_home"]) else None
            row["total_consenso"] = m["consensus_total"]
            row["ml_home"] = m["consensus_ml_home"]
            row["ml_away"] = m["consensus_ml_away"]
            row["casas_usadas"] = m["casas_usadas"]

            # Guardar línea de mercado numérica (positivo = local favorito)
            if pd.notna(m["consensus_spread_home"]):
                row["market_spread_line"] = -float(m["consensus_spread_home"])
            if pd.notna(m["consensus_total"]):
                row["market_total_line"] = float(m["consensus_total"])

            value_labels = _market_value_labels(
                home,
                away,
                spread_model_line,
                model_total,
                -m["consensus_spread_home"],
                m["consensus_total"],
            )
            row.update(value_labels)
        results.append(row)

    df = pd.DataFrame(results)
    if not df.empty and "fecha" in df.columns:
        df = df.sort_values(
            by=["fecha", "hora"], na_position="last", kind="stable"
        ).reset_index(drop=True)
    return df


# ---------- Impresión en consola ----------

def print_picks(df: pd.DataFrame):
    if df.empty:
        print("No hay partidos para mostrar.")
        return

    current_slot = None
    for _, r in df.iterrows():
        slot = r.get("slot")
        if slot and slot != current_slot:
            print(f"\n=== {slot} ===")
            current_slot = slot

        print("─" * 60)
        fecha = r.get("fecha") or ""
        hora = r.get("hora") or ""
        cuando = f"  [{fecha} {hora}]" if fecha or hora else ""
        print(f"{r['partido']}   ({r.get('liga', '')} - Semana {r.get('semana', '')}){cuando}")
        print(f"Proyección: {r['marcador_proyectado']}")
        resultado_real = r.get("resultado_real")
        if pd.notna(resultado_real) and resultado_real:
            ac = r.get("acierto_ganador")
            acs = r.get("acierto_spread")
            act = r.get("acierto_total")
            marcas = []
            if pd.notna(ac):
                marcas.append("✔ ML" if ac else "✘ ML")
            if pd.notna(acs):
                marcas.append("✔ spread" if acs else "✘ spread")
            if pd.notna(act):
                marcas.append("✔ O/U" if act else "✘ O/U")
            marca = ("   " + " · ".join(marcas)) if marcas else ""
            print(f"Resultado real: {resultado_real}{marca}")

        away_team, home_team = r["partido"].split(" @ ")
        home_rec = _get(r, "home_record")
        away_rec = _get(r, "away_record")
        home_disp = f"{home_team} ({home_rec})" if home_rec else home_team
        away_disp = f"{away_team} ({away_rec})" if away_rec else away_team

        prob_h = _get(r, "prob_home", "prob_gana_local")
        prob_a = _get(r, "prob_away", "prob_gana_visita")
        print(f"Prob. victoria: {home_disp} {prob_h}  |  {away_disp} {prob_a}")

        p10 = _get(r, "sim_margin_p10")
        p90 = _get(r, "sim_margin_p90")
        if p10 is not None and p90 is not None:
            print(f"Margen probable (80%): {p10} a {p90}")

        print(f"Modelo   -> Spread: {r['spread_modelo']:<18} Over/Under: {r['total_modelo']}")

        ajustes = r.get("ajustes")
        if pd.notna(ajustes) and ajustes:
            print(f"Ajustes  -> {ajustes}")

        spread_consenso = r.get("spread_consenso")
        total_consenso = r.get("total_consenso")
        if pd.notna(spread_consenso) or pd.notna(total_consenso):
            print(f"Mercado  -> Spread: {str(spread_consenso):<18} Over/Under: {total_consenso}")

        ml_h = _get(r, "ml_home", "ml_local")
        ml_a = _get(r, "ml_away", "ml_visita")
        if ml_h is not None or ml_a is not None:
            print(f"ML       -> {home_disp}: {ml_h}  |  {away_disp}: {ml_a}")

        valor_spread = r.get("valor_spread")
        valor_total = r.get("valor_total")
        if pd.notna(valor_spread) or pd.notna(valor_total):
            print(f"Valor    -> Spread: {str(valor_spread):<18} Over/Under: {valor_total}")
    print("─" * 60)

    marcados = False
    for col in ("valor_spread", "valor_total"):
        if col in df.columns and df[col].astype(str).str.contains("⚠").any():
            marcados = True
    if marcados:
        print("⚠ = el modelo y el mercado difieren mucho.")

    if "acierto_ganador" in df.columns:
        jugados = df[df["acierto_ganador"].notna()]
        if not jugados.empty:
            aciertos_ml = int((jugados["acierto_ganador"] == True).sum())  # noqa: E712
            err_margen = pd.to_numeric(jugados["error_margen"]).abs().mean()
            err_total = pd.to_numeric(jugados["error_total"]).abs().mean()

            resumen = (
                f"Resumen de partidos ya jugados: "
                f"ML acertado {aciertos_ml}/{len(jugados)} "
                f"({round(100 * aciertos_ml / len(jugados))}%)"
            )

            if "acierto_spread" in df.columns:
                con_spread = df[df["acierto_spread"].notna()]
                if not con_spread.empty:
                    aciertos_sp = int((con_spread["acierto_spread"] == True).sum())  # noqa: E712
                    resumen += (
                        f" | Spread cubierto {aciertos_sp}/{len(con_spread)} "
                        f"({round(100 * aciertos_sp / len(con_spread))}%)"
                    )

            if "acierto_total" in df.columns:
                con_total = df[df["acierto_total"].notna()]
                if not con_total.empty:
                    aciertos_ou = int((con_total["acierto_total"] == True).sum())  # noqa: E712
                    resumen += (
                        f" | O/U acertado {aciertos_ou}/{len(con_total)} "
                        f"({round(100 * aciertos_ou / len(con_total))}%)"
                    )

            resumen += (
                f" | error medio del margen: {err_margen:.1f} pts "
                f"| error medio del total: {err_total:.1f} pts"
            )
            print(resumen)


# ---------- Actualización de Elo ----------

def update_elo_with_results(season: int, week: int, league: str = "nfl"):
    stats = _stats_module(league)
    ratings = {}

    if config.CARRY_OVER_PREVIOUS_SEASON:
        try:
            prev_games = stats.get_completed_games(season - 1).sort_values("week")
            if not prev_games.empty:
                for _, g in prev_games.iterrows():
                    ratings = elo_model.update_ratings(
                        ratings, g["home_team"], g["away_team"], int(g["home_score"]), int(g["away_score"])
                    )
                ratings = {
                    team: config.ELO_INITIAL + (rating - config.ELO_INITIAL) * config.SEASON_CARRYOVER_WEIGHT
                    for team, rating in ratings.items()
                }
        except Exception as e:
            print(f"[Aviso] No se pudo usar la temporada anterior como base del Elo: {e}")

    games = stats.get_completed_games(season, before_week=week).sort_values("week")
    if games.empty and ratings:
        elo_model.save_ratings(ratings, league=league, season=season, week=max(week - 1, 0))
    for game_week, group in games.groupby("week"):
        for _, g in group.iterrows():
            ratings = elo_model.update_ratings(
                ratings, g["home_team"], g["away_team"], int(g["home_score"]), int(g["away_score"])
            )
        elo_model.save_ratings(ratings, league=league, season=season, week=int(game_week))
    return ratings