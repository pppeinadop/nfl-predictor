"""
Ejecuta el pipeline semanal completo para la liga indicada (nfl o ncaaf):
  1. Actualiza el Elo con los resultados mas recientes y lo guarda en MongoDB.
  2. Genera predicciones de marcador, spread y over/under para la semana
     y las guarda en la coleccion 'weekly_picks'.
  3. Genera recomendaciones de props (solo NFL por ahora) y las guarda en 'player_props'.
  4. Ademas deja una copia en CSV en /output/ por si quieres revisarlo sin Mongo.

Uso (desde la terminal de PyCharm):
    python main.py --season 2026 --week 4 --league nfl
    python main.py --season 2026 --week 4 --league ncaaf
    python main.py --season 2026 --week 4 --league nfl --props-file props.json --event-id ABC123

Formato de props.json:
    [
      {"player_display_name": "Patrick Mahomes", "market": "player_pass_yds"},
      {"player_display_name": "Travis Kelce",    "market": "player_receptions"}
    ]
El --event-id se obtiene de la API de The Odds API (campo 'id' de cada partido).
"""
import argparse
import json
import os

import db
import predict_games
import player_props
import rebuild_weeks

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _load_props_file(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El archivo de props debe ser una lista de objetos "
                         "con 'player_display_name' y 'market'.")
    for i, item in enumerate(data):
        if "player_display_name" not in item or "market" not in item:
            raise ValueError(f"Elemento {i} del archivo de props no tiene "
                             f"'player_display_name' y/o 'market': {item}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Predicciones de futbol americano semanales")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--league", choices=["nfl", "ncaaf"], default="nfl",
                         help="Liga: 'nfl' o 'ncaaf' (colegial). Por defecto: nfl")
    parser.add_argument("--props-file", type=str, default=None,
                         help="JSON con la lista de jugadores/mercados a proyectar (solo NFL).")
    parser.add_argument("--event-id", type=str, default=None,
                         help="ID del evento (partido) en The Odds API, necesario para props.")
    parser.add_argument(
        "--mode", choices=["official", "update"], default="official",
        help="official crea la predicción oficial una sola vez; update guarda una versión prepartido.",
    )
    parser.add_argument(
        "--refresh-odds", action="store_true",
        help="Consulta The Odds API y consume cuota; por defecto se usa solo la caché.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db.ensure_indexes()

    print(f"[{args.league.upper()}] Actualizando ratings Elo con resultados hasta la semana {args.week - 1}...")
    predict_games.update_elo_with_results(args.season, args.week, args.league)

    print(f"[{args.league.upper()}] Resolviendo cuotas de mercado...")
    market = rebuild_weeks._resolve_market(
        args.season, args.week, args.league,
        skip_odds=not args.refresh_odds,
        refresh_odds=args.refresh_odds,
    )

    print(f"[{args.league.upper()}] Generando picks para la semana {args.week}...")
    # Las lesiones se vuelven a consultar en cada ejecución.
    try:
        import adjustments
        adjustments._injury_cache.clear()
    except Exception:
        pass
    picks = predict_games.build_weekly_picks(
        args.season, args.week, args.league, market_df=market
    )

    save_summary = db.save_weekly_picks(
        picks, args.season, args.week, args.league,
        prediction_mode=args.mode,
    )
    print(f"Resumen de guardado: {save_summary}")
    if args.mode == "official":
        export_picks = db.get_weekly_picks(args.season, args.week, args.league)
        picks_path = os.path.join(
            OUTPUT_DIR, f"picks_{args.league}_semana_{args.week}_oficial.csv"
        )
    else:
        export_picks = db.get_prediction_versions(args.season, args.week, args.league)
        picks_path = os.path.join(
            OUTPUT_DIR, f"picks_{args.league}_semana_{args.week}_actualizaciones.csv"
        )
    export_picks.to_csv(picks_path, index=False)
    predict_games.print_picks(export_picks)
    print(f"\nGuardado en MongoDB y exportado a {picks_path}")

    # --- Props de jugador: solo aplica para NFL por ahora ---
    if args.league == "nfl":
        if args.props_file and args.event_id:
            try:
                players_of_interest = _load_props_file(args.props_file)
            except Exception as e:
                print(f"[Aviso] No se pudo leer --props-file: {e}")
                players_of_interest = []

            if players_of_interest:
                print(f"[NFL] Generando props para {len(players_of_interest)} jugador(es)...")
                props = player_props.build_prop_recommendations(
                    args.season, args.event_id, players_of_interest
                )
                db.save_player_props(props, args.season, args.week, args.league)
                props_path = os.path.join(OUTPUT_DIR, f"props_semana_{args.week}.csv")
                props.to_csv(props_path, index=False)
                print(props.to_string(index=False))
                print(f"\nGuardado en MongoDB (player_props) y en {props_path}")
        else:
            print("\n(Sin props: usa --props-file y --event-id para generarlas)")

    print(f"\nListo. Para ver esta semana: "
          f"python view_week.py --season {args.season} --week {args.week} --league {args.league}")


if __name__ == "__main__":
    main()