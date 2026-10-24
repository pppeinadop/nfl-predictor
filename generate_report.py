"""
Genera un reporte de desempeño del modelo desde la terminal.

Ejemplos:
    # Una sola temporada:
    python generate_report.py --season 2026 --league nfl

    # Rango de temporadas (una sola pasada, un solo reporte):
    python generate_report.py --from-season 2020 --to-season 2026 --league nfl

    # Con formatos específicos:
    python generate_report.py --from-season 2020 --to-season 2026 --league nfl --formats md,json

Los archivos se guardan en la carpeta output/.
"""
import argparse
import json
import os

import report

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    parser = argparse.ArgumentParser(description="Genera un reporte de desempeño.")
    parser.add_argument("--season", type=int, default=None,
                        help="Una sola temporada. Si se usa, ignora --from-season / --to-season.")
    parser.add_argument("--from-season", type=int, default=None, dest="from_season")
    parser.add_argument("--to-season", type=int, default=None, dest="to_season")
    parser.add_argument("--league", choices=["nfl", "ncaaf"], required=True)
    parser.add_argument("--formats", default="md,json,csv",
                        help="Formatos separados por coma: md,json,csv")
    parser.add_argument("--all-games", action="store_true",
                        help="Incluye también partidos sin resultado real.")
    args = parser.parse_args()

    # Resolver rango de temporadas
    if args.season is not None:
        from_season = to_season = args.season
    else:
        if args.from_season is None or args.to_season is None:
            parser.error("Debes indicar --season O bien --from-season y --to-season")
        if args.from_season > args.to_season:
            parser.error("--from-season no puede ser mayor que --to-season")
        from_season = args.from_season
        to_season = args.to_season

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label = f"{from_season}" if from_season == to_season else f"{from_season}-{to_season}"
    print(f"\n📊 Generando reporte {args.league.upper()} temporadas {label}...")

    df = report.collect_range(
        from_season, to_season, args.league,
        only_played=not args.all_games,
    )

    if df.empty:
        print("  ⚠ No hay partidos para reportar.")
        return

    stats = report.compute_stats(df)
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    base = os.path.join(OUTPUT_DIR, f"reporte_{args.league}_{label}")

    # Resumen en consola
    g = stats.get("global") or {}
    print(f"  Partidos: {stats['total_partidos']}")
    for lab, key in [("ML", "ml"), ("Spread", "spread"), ("O/U", "total")]:
        acc = g.get(key)
        if acc and acc.get("total"):
            print(f"  {lab}: {acc['aciertos']}/{acc['total']} ({acc['pct']}%)")
        else:
            print(f"  {lab}: —")

    if stats.get("por_temporada"):
        print("\n  Por temporada:")
        for s in stats["por_temporada"]:
            ml = s.get("ml") or {}
            sp = s.get("spread") or {}
            ml_txt = f"{ml.get('pct')}%" if ml.get("total") else "—"
            sp_txt = f"{sp.get('pct')}%" if sp.get("total") else "—"
            print(f"    {s['temporada']}: {s['partidos']} partidos | ML {ml_txt} | Spread {sp_txt}")

    if "md" in formats:
        path = base + ".md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown(df, stats))
        print(f"\n  ✔ Markdown: {path}")

    if "json" in formats:
        path = base + ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_json_dict(df, stats), f, ensure_ascii=False, indent=2, default=str)
        print(f"  ✔ JSON: {path}")

    if "csv" in formats:
        path = base + ".csv"
        df.to_csv(path, index=False)
        print(f"  ✔ CSV: {path}")


if __name__ == "__main__":
    main()