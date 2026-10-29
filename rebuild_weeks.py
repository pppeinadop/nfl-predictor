"""
Regenera varias semanas de un jalón. Se puede usar desde la terminal o
importarse desde app.py.

Uso desde terminal:
    # Regenera y consume cuota de The Odds API (una llamada por semana):
    python rebuild_weeks.py --season 2026 --league nfl --from-week 1 --to-week 4

    # Solo cache, 0 llamadas a la API. Si no hay cache, la semana queda sin odds:
    python rebuild_weeks.py --season 2020 --league nfl --from-week 1 --to-week 18 --skip-odds

    # Fuerza refrescar el cache aunque ya exista:
    python rebuild_weeks.py --season 2026 --league nfl --from-week 5 --to-week 5 --refresh-odds
"""
import argparse
import os
import traceback

import pandas as pd

import config
import data_fetch
import db
import predict_games

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _market_empty():
    return pd.DataFrame()


def _has_market_lines(market: pd.DataFrame) -> bool:
    if market is None or market.empty:
        return False
    fields = (
        "consensus_spread_home",
        "consensus_total",
        "consensus_ml_home",
        "consensus_ml_away",
    )
    return any(
        field in market.columns and market[field].notna().any()
        for field in fields
    )


def _resolve_market(season: int, week: int, league: str,
                    skip_odds: bool, refresh_odds: bool) -> pd.DataFrame:
    """
    Resuelve el mercado para una semana.

    - skip_odds=True  y no refresh_odds → SOLO cache. Si no hay, vacío. NUNCA pega a la API.
    - refresh_odds=True                 → siempre pega a la API (y guarda cache).
    - skip_odds=False                   → intenta cache, si no hay pega a la API.
    """
    if refresh_odds:
        return _fetch_and_cache(season, week, league)

    cached = db.get_market_cache(season, week, league)
    if _has_market_lines(cached):
        print(f"  Mercado desde cache: {len(cached)} partidos")
        return cached
    if not cached.empty:
        print(f"  Cache de semana {week} sin líneas utilizables; buscando cuotas actuales")

    if skip_odds:
        print(f"  Sin cache para semana {week} (skip-odds, no se llama a la API)")
        return _market_empty()

    print(f"  Sin cache para semana {week}, pidiendo a la API (consume cuota)")
    return _fetch_and_cache(season, week, league)


def _fetch_and_cache(season: int, week: int, league: str) -> pd.DataFrame:
    try:
        sport_key = config.LEAGUES[league]["odds_sport_key"]
        odds_events = data_fetch.get_game_odds(sport_key)
        market = data_fetch.get_consensus_lines(odds_events)
        if not _has_market_lines(market):
            print(
                "  The Odds API respondió, pero no devolvió líneas utilizables "
                "de moneyline, spread o total."
            )
            return _market_empty()
        if not market.empty:
            try:
                db.save_market_cache(market, season, week, league)
                market.attrs["cache_updated_at"] = pd.Timestamp.now(tz="UTC").to_pydatetime()
                print(f"  Mercado cacheado: {len(market)} partidos")
            except Exception as e:
                print(f"  [Aviso] No se pudo cachear el mercado: {e}")
        return market
    except Exception as e:
        print(f"  No se pudo traer mercado: {e}")
        return _market_empty()


def rebuild_week(season: int, week: int, league: str,
                 skip_odds: bool = False, refresh_odds: bool = False,
                 verbose: bool = False, prediction_mode: str = "official") -> bool:
    """Procesa una sola semana. Devuelve True si terminó bien, False si falló."""
    print(f"\n{'─' * 60}")
    print(f"  {league.upper()} · Temporada {season} · Semana {week}")
    print(f"{'─' * 60}")

    try:
        db.ensure_indexes()
        predict_games.update_elo_with_results(season, week, league)
        # Fuerza una lectura fresca del reporte de lesiones en cada ejecución.
        try:
            import adjustments
            adjustments._injury_cache.clear()
        except Exception:
            pass
        market = _resolve_market(season, week, league, skip_odds, refresh_odds)
        picks = predict_games.build_weekly_picks(
            season, week, league, market_df=market
        )

        if picks.empty:
            print(f"  Sin partidos para la semana {week}.")
            return True

        save_summary = db.save_weekly_picks(
            picks, season, week, league, prediction_mode=prediction_mode
        )
        print(
            "  Guardado: "
            f"{save_summary['inserted']} oficiales nuevos, "
            f"{save_summary['updated_market']} mercados actualizados, "
            f"{save_summary['updated_projection']} proyecciones sincronizadas, "
            f"{save_summary['updated_versions']} actualizaciones, "
            f"{save_summary['locked']} bloqueados por kickoff, "
            f"{save_summary['skipped']} omitidos/protegidos."
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if prediction_mode == "official":
            export_picks = db.get_weekly_picks(season, week, league)
            path = os.path.join(OUTPUT_DIR, f"picks_{league}_semana_{week}_oficial.csv")
        else:
            export_picks = db.get_prediction_versions(season, week, league)
            path = os.path.join(OUTPUT_DIR, f"picks_{league}_semana_{week}_actualizaciones.csv")
        export_picks.to_csv(path, index=False)
        print(f"  OK: {len(export_picks)} registros exportados (Mongo + {os.path.basename(path)})")

        if verbose:
            predict_games.print_picks(export_picks)
        return True

    except Exception as e:
        print(f"  Error en la semana {week}: {e}")
        traceback.print_exc()
        return False


def sync_final_results(season: int, week: int, league: str = "nfl") -> dict:
    """Sincroniza marcadores finales sin volver a calcular ni modificar predicciones."""
    stats = predict_games._stats_module(league)
    schedule = stats.get_schedule(season)
    if schedule is None or schedule.empty:
        return {"updated": 0, "not_found": 0}
    
    # Blindaje seguro para convertir la columna 'week' a numérica y evitar errores de comparación
    if "week" in schedule.columns:
        schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce").fillna(0).astype(int)
        games = schedule[schedule["week"] == int(week)]
    else:
        return {"updated": 0, "not_found": 0}

    updated = 0
    for _, game in games.iterrows():
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        completed = game.get("completed")
        if completed is not None and not pd.isna(completed) and not bool(completed):
            continue
        status = str(game.get("game_status", game.get("status", ""))).lower()
        if any(term in status for term in ("progress", "live", "scheduled", "pre-game", "pregame")):
            continue
        # En NFLverse, 'result' se completa cuando el partido ya tiene resultado final.
        if league == "nfl" and "result" in schedule.columns and pd.isna(game.get("result")):
            continue
        try:
            ok = db.sync_final_score(
                league, season, week,
                str(game["home_team"]), str(game["away_team"]),
                int(home_score), int(away_score),
            )
            if ok:
                updated += 1
        except Exception as exc:
            print(f"No se pudo sincronizar {game.get('away_team')} @ {game.get('home_team')}: {exc}")
    return {"updated": updated, "not_found": max(0, len(games) - updated)}


def main():
    parser = argparse.ArgumentParser(description="Regenera varias semanas de un jalón.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league", choices=["nfl", "ncaaf"], default="nfl")
    parser.add_argument("--from-week", type=int, required=True, dest="from_week")
    parser.add_argument("--to-week", type=int, required=True, dest="to_week")
    parser.add_argument("--skip-odds", action="store_true", dest="skip_odds",
                        help="Solo cache. NUNCA llama a la API.")
    parser.add_argument("--refresh-odds", action="store_true", dest="refresh_odds",
                        help="Fuerza llamada a la API aunque ya haya cache.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mode", choices=["official", "update"], default="official",
        help="official crea la predicción oficial una sola vez; update guarda una versión prepartido sin reemplazarla.",
    )
    args = parser.parse_args()

    if args.from_week > args.to_week:
        parser.error("--from-week no puede ser mayor que --to-week")
    if args.skip_odds and args.refresh_odds:
        parser.error("--skip-odds y --refresh-odds son mutuamente excluyentes")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db.ensure_indexes()

    print(f"\nGenerando {args.league.upper()} temporada {args.season}, "
          f"semanas {args.from_week} a {args.to_week} · modo={args.mode}")
    if args.skip_odds:
        print("   (modo skip-odds: solo cache, sin llamadas a la API)")

    ok, fail = [], []
    for week in range(args.from_week, args.to_week + 1):
        success = rebuild_week(
            args.season, week, args.league,
            skip_odds=args.skip_odds, refresh_odds=args.refresh_odds,
            verbose=args.verbose, prediction_mode=args.mode,
        )
        (ok if success else fail).append(week)

    print(f"\n{'=' * 60}")
    print("  RESUMEN")
    print(f"{'=' * 60}")
    print(f"  Semanas OK     : {ok if ok else '—'}")
    print(f"  Semanas con err: {fail if fail else '—'}")


if __name__ == "__main__":
    main()
