"""
Muestra las predicciones ya guardadas en MongoDB para una semana dada.
No vuelve a calcular nada ni consulta APIs externas: solo lee lo que 'main.py'
ya guardó.

Uso:
    python view_week.py --season 2026 --week 4
    python view_week.py --season 2026 --week 4 --html            # abre reporte en el navegador
    python view_week.py --season 2026 --week 4 --html --no-open  # solo genera el archivo
    python view_week.py --season 2026                            # lista semanas guardadas
"""
import argparse
import os
import webbrowser

import pandas as pd

import config
import db
import predict_games

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

HTML_TEMPLATE = """
<html>
<head>
<meta charset="utf-8">
<title>{league_label} Picks - Semana {week}</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 30px; background: #0f1115; color: #e8e8e8; }}
  h1, h2 {{ color: #f5f5f5; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
  th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #1c1f26; }}
  tr:nth-child(even) {{ background: #171a20; }}
  .valor {{ color: #4ade80; font-weight: bold; }}
</style>
</head>
<body>
<h1>Predicciones {league_label} - Temporada {season}, Semana {week}</h1>
<h2>Partidos</h2>
{picks_table}
<h2>Props de jugadores</h2>
{props_table}
</body>
</html>
"""


def show_week(season: int, week: int, league: str = "nfl", as_html: bool = False,
              open_browser: bool = True):
    picks = db.get_weekly_picks(season, week, league)
    props = db.get_player_props(season, week, league)
    league_label = config.LEAGUES.get(league, {}).get("label", league.upper())

    if picks.empty:
        print(f"No hay picks guardados para {league_label}, temporada {season}, semana {week}. "
              f"Corre primero: python main.py --season {season} --week {week} --league {league}")
        return

    if not as_html:
        print(f"\n=== {league_label} - Partidos - Semana {week} ===")
        predict_games.print_picks(picks)
        if not props.empty:
            print(f"\n=== Props de jugadores - Semana {week} ===")
            print(props.to_string(index=False))
        return

    picks_html = picks.to_html(index=False, na_rep="-")
    props_html = props.to_html(index=False, na_rep="-") if not props.empty else "<p>Sin props guardadas.</p>"
    html = HTML_TEMPLATE.format(league_label=league_label, season=season, week=week,
                                 picks_table=picks_html, props_table=props_html)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"reporte_{league}_semana_{week}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    should_open = open_browser and config.REPORT_AUTO_OPEN
    if should_open:
        try:
            webbrowser.open(f"file://{os.path.abspath(path)}")
            print(f"Reporte abierto en el navegador: {path}")
        except Exception as e:
            print(f"No se pudo abrir el navegador ({e}). Reporte en: {path}")
    else:
        print(f"Reporte generado en: {path}")


def main():
    parser = argparse.ArgumentParser(description="Ver predicciones guardadas en MongoDB")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=False)
    parser.add_argument("--league", choices=["nfl", "ncaaf"], default="nfl")
    parser.add_argument("--html", action="store_true", help="Genera y abre un reporte HTML en vez de consola")
    parser.add_argument("--no-open", action="store_true",
                         help="Con --html, genera el archivo pero no abre el navegador")
    args = parser.parse_args()

    if args.week is None:
        weeks = db.list_available_weeks(args.season, args.league)
        if weeks:
            print(f"Semanas guardadas para {args.league} {args.season}: {weeks}")
        else:
            print("Todavia no hay nada guardado.")
        return

    show_week(args.season, args.week, args.league,
              as_html=args.html, open_browser=not args.no_open)


if __name__ == "__main__":
    main()