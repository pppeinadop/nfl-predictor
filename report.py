"""
Genera reportes de desempeño del modelo comparando predicciones con resultados reales.

Uso desde app.py o desde generate_report.py.

Los formatos de salida son:
  - DataFrame (crudo, para CSV)
  - dict / JSON (estructurado, ideal para subir a una IA)
  - str Markdown (legible, ideal para copiar/pegar en un chat de IA)
"""
import re
from datetime import datetime, timezone

import pandas as pd

import config
import db


# ---------- Recolección de datos ----------

def collect_all(season: int, league: str, only_played: bool = True) -> pd.DataFrame:
    """Junta TODOS los picks de UNA temporada/liga."""
    return collect_range(season, season, league, only_played=only_played)


def collect_range(from_season: int, to_season: int, league: str,
                  only_played: bool = True) -> pd.DataFrame:
    """
    Junta TODOS los picks guardados entre from_season y to_season (inclusive)
    para una liga dada en un solo DataFrame.

    Si only_played=True, filtra solo partidos con resultado real.
    """
    frames = []
    for season in range(from_season, to_season + 1):
        weeks = db.list_available_weeks(season, league)
        for w in weeks:
            df = db.get_weekly_picks(season, w, league)
            if df.empty:
                continue
            for col, val in (("season", season), ("week", w), ("league", league)):
                if col not in df.columns:
                    df[col] = val
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    full = _normalize_bools(full)

    if only_played and "resultado_real" in full.columns:
        full = full[full["resultado_real"].notna()].reset_index(drop=True)
    return full


def _normalize_bools(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("acierto_ganador", "acierto_spread", "acierto_total"):
        if col in df.columns:
            df[col] = df[col].apply(_to_bool_or_na)
    return df


def _to_bool_or_na(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return bool(v)
    except (TypeError, ValueError):
        return None


# ---------- Estadísticas ----------

def _acc(series) -> dict:
    valid = series.dropna()
    total = len(valid)
    if total == 0:
        return {"aciertos": 0, "total": 0, "pct": None}
    aciertos = int((valid == True).sum())  # noqa: E712
    return {"aciertos": aciertos, "total": total, "pct": round(100 * aciertos / total, 1)}


def _mae(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna().abs()
    if s.empty:
        return None
    return round(float(s.mean()), 2)


def _block(g: pd.DataFrame) -> dict:
    """Estadísticas de un subgrupo (una semana, una temporada, un rango)."""
    return {
        "partidos": len(g),
        "ml": _acc(g["acierto_ganador"]) if "acierto_ganador" in g.columns else None,
        "spread": _acc(g["acierto_spread"]) if "acierto_spread" in g.columns else None,
        "total": _acc(g["acierto_total"]) if "acierto_total" in g.columns else None,
        "error_margen_medio_pts": _mae(g, "error_margen"),
        "error_total_medio_pts": _mae(g, "error_total"),
    }


def compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_partidos": 0}

    stats = {
        "total_partidos": len(df),
        "ligas": sorted(df["liga"].dropna().unique().tolist()) if "liga" in df.columns else [],
        "temporadas": sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else [],
        "semanas": sorted(df["week"].dropna().unique().tolist()) if "week" in df.columns else [],
        "global": _block(df),
        "por_temporada": [],
        "por_semana": [],
        "por_rango_spread": [],
        "por_confianza_ml": [],
        "por_tipo_valor": [],
    }

    # ---- Por temporada ----
    if "season" in df.columns:
        for season, g in df.groupby("season"):
            b = _block(g)
            b["temporada"] = int(season)
            stats["por_temporada"].append(b)

    # ---- Por semana ----
    if "week" in df.columns:
        for wk, g in df.groupby("week"):
            b = _block(g)
            b["semana"] = int(wk)
            stats["por_semana"].append(b)

    # ---- Por rango de favoritismo ----
    spread_numeric = _extract_spread_numeric(df)
    if spread_numeric is not None:
        df2 = df.copy()
        df2["_abs_spread"] = spread_numeric.abs()
        bins = [0, 3, 7, 14, 999]
        labels = ["0-3 pts", "3-7 pts", "7-14 pts", "14+ pts"]
        df2["_bucket"] = pd.cut(df2["_abs_spread"], bins=bins, labels=labels, right=False)
        for bucket, g in df2.groupby("_bucket", observed=True):
            if pd.isna(bucket):
                continue
            b = _block(g)
            b["rango"] = str(bucket)
            stats["por_rango_spread"].append(b)

    # ---- Por nivel de confianza ----
    prob_fav = _extract_favorite_prob(df)
    if prob_fav is not None:
        df3 = df.copy()
        df3["_prob_fav"] = prob_fav
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90%+"]
        df3["_bucket"] = pd.cut(df3["_prob_fav"], bins=bins, labels=labels, right=False)
        for bucket, g in df3.groupby("_bucket", observed=True):
            if pd.isna(bucket):
                continue
            b = _block(g)
            b["confianza"] = str(bucket)
            stats["por_confianza_ml"].append(b)

    # ---- Con/sin señal de valor ----
    if "valor_spread" in df.columns:
        with_val = df[df["valor_spread"].astype(str).str.contains("⚠|\\+", regex=True, na=False)]
        sin_val = df[~df.index.isin(with_val.index)]

        b1 = _block(with_val)
        b1["tipo"] = "Con señal de valor (spread)"
        b2 = _block(sin_val)
        b2["tipo"] = "Sin señal de valor (spread)"
        stats["por_tipo_valor"] = [b1, b2]

    return stats


def _extract_spread_numeric(df: pd.DataFrame):
    if "spread_modelo" not in df.columns:
        return None

    def parse(s):
        if s is None or pd.isna(s):
            return None
        s = str(s)
        if "Pick'em" in s:
            return 0.0
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if not m:
            return None
        return float(m.group(1))

    out = df["spread_modelo"].apply(parse)
    return out if out.notna().any() else None


def _extract_favorite_prob(df: pd.DataFrame):
    if "prob_home" not in df.columns and "prob_away" not in df.columns:
        return None

    def to_float(s):
        if s is None or pd.isna(s):
            return None
        s = str(s).replace("%", "").strip()
        try:
            return float(s) / 100.0
        except ValueError:
            return None

    ph = df.get("prob_home", pd.Series([None] * len(df))).apply(to_float)
    pa = df.get("prob_away", pd.Series([None] * len(df))).apply(to_float)
    out = pd.concat([ph, pa], axis=1).max(axis=1)
    return out if out.notna().any() else None


# ---------- Markdown ----------

def _fmt_acc(acc) -> str:
    if not acc or acc.get("total", 0) == 0:
        return "—"
    return f"{acc['aciertos']}/{acc['total']} ({acc['pct']}%)"


def to_markdown(df: pd.DataFrame, stats: dict) -> str:
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Reporte de desempeño del modelo — {ts}")
    lines.append("")
    lines.append("## Qué es esto")
    lines.append("")
    lines.append(
        "Reporte de un modelo propio de predicción de fútbol americano (NFL y NCAAF). "
        "Genera marcador, spread, total (Over/Under), probabilidad de victoria y señales "
        "de valor contra el consenso de casas. Aquí se compara con resultados reales."
    )
    lines.append("")
    lines.append("**Métricas clave:**")
    lines.append("- **ML** = acertó quién gana.")
    lines.append("- **Spread** = el resultado cubrió el spread que el propio modelo predijo.")
    lines.append("- **O/U** = el resultado cayó del lado (Over/Under) que el modelo eligió respecto a la línea del mercado.")
    lines.append("- Los empates exactos con la línea no cuentan ni como acierto ni como fallo (push).")
    lines.append("")

    lines.append("## Cobertura")
    lines.append("")
    lines.append(f"- **Partidos analizados:** {stats.get('total_partidos', 0)}")
    lines.append(f"- **Ligas:** {', '.join(stats.get('ligas', []))}")
    temporadas = stats.get("temporadas", [])
    if temporadas:
        lines.append(f"- **Temporadas:** {temporadas[0]}–{temporadas[-1]} ({len(temporadas)} en total)")
        lines.append(f"- **Detalle temporadas:** {', '.join(map(str, temporadas))}")
    lines.append(f"- **Semanas:** {', '.join(map(str, stats.get('semanas', [])))}")
    lines.append("")

    g = stats.get("global") or {}
    lines.append("## Resultados globales (todas las temporadas)")
    lines.append("")
    lines.append("| Métrica | Aciertos / Total | % |")
    lines.append("|---|---|---|")
    for label, key in [("ML", "ml"), ("Spread (vs modelo)", "spread"), ("Over/Under", "total")]:
        acc = g.get(key)
        lines.append(f"| {label} | {_fmt_acc(acc)} | {acc.get('pct') if acc else '—'} |")
    lines.append("")
    lines.append(f"- Error medio del margen: {g.get('error_margen_medio_pts', '—')} pts")
    lines.append(f"- Error medio del total: {g.get('error_total_medio_pts', '—')} pts")
    lines.append("")

    # ---- Por temporada (la clave del análisis histórico) ----
    if stats.get("por_temporada"):
        lines.append("## Desglose por temporada")
        lines.append("")
        lines.append("Esta sección muestra si el modelo es consistente año tras año.")
        lines.append("")
        lines.append("| Temporada | Partidos | ML | Spread | O/U | Err. margen | Err. total |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in stats["por_temporada"]:
            lines.append(
                f"| {s['temporada']} | {s['partidos']} | {_fmt_acc(s.get('ml'))} | "
                f"{_fmt_acc(s.get('spread'))} | {_fmt_acc(s.get('total'))} | "
                f"{s.get('error_margen_medio_pts', '—')} | {s.get('error_total_medio_pts', '—')} |"
            )
        lines.append("")

    # ---- Por semana ----
    if stats.get("por_semana"):
        lines.append("## Desglose por semana (agregado entre temporadas)")
        lines.append("")
        lines.append("| Semana | Partidos | ML | Spread | O/U | Err. margen | Err. total |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in stats["por_semana"]:
            lines.append(
                f"| {s['semana']} | {s['partidos']} | {_fmt_acc(s.get('ml'))} | "
                f"{_fmt_acc(s.get('spread'))} | {_fmt_acc(s.get('total'))} | "
                f"{s.get('error_margen_medio_pts', '—')} | {s.get('error_total_medio_pts', '—')} |"
            )
        lines.append("")

    if stats.get("por_rango_spread"):
        lines.append("## Por rango de favoritismo (|spread modelo|)")
        lines.append("")
        lines.append("| Rango | Partidos | ML | Spread |")
        lines.append("|---|---|---|---|")
        for s in stats["por_rango_spread"]:
            lines.append(
                f"| {s['rango']} | {s['partidos']} | {_fmt_acc(s.get('ml'))} | {_fmt_acc(s.get('spread'))} |"
            )
        lines.append("")

    if stats.get("por_confianza_ml"):
        lines.append("## Por nivel de confianza del modelo (prob. favorito)")
        lines.append("")
        lines.append("| Confianza | Partidos | ML | Spread |")
        lines.append("|---|---|---|---|")
        for s in stats["por_confianza_ml"]:
            lines.append(
                f"| {s['confianza']} | {s['partidos']} | {_fmt_acc(s.get('ml'))} | {_fmt_acc(s.get('spread'))} |"
            )
        lines.append("")

    if stats.get("por_tipo_valor"):
        lines.append("## Cuando el modelo marcó valor, ¿acertó más?")
        lines.append("")
        lines.append("| Tipo | Partidos | ML | Spread |")
        lines.append("|---|---|---|---|")
        for s in stats["por_tipo_valor"]:
            lines.append(
                f"| {s['tipo']} | {s['partidos']} | {_fmt_acc(s.get('ml'))} | {_fmt_acc(s.get('spread'))} |"
            )
        lines.append("")

    # ---- Detalle partido a partido ----
    lines.append("## Detalle partido a partido")
    lines.append("")
    lines.append("| Temp | Sem | Partido | Marcador modelo | Real | Spread mod. | Spread mkt | Total mod. | Total mkt | ML | Spread | O/U |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    sort_cols = [c for c in ("season", "week", "partido") if c in df.columns]
    df_sorted = df.sort_values(sort_cols) if sort_cols else df

    for _, r in df_sorted.iterrows():
        def _f(v):
            return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        def _b(v):
            b = _to_bool_or_na(v)
            if b is None:
                return "—"
            return "✔" if b else "✘"
        lines.append(
            f"| {_f(r.get('season'))} | {_f(r.get('week'))} | {_f(r.get('partido'))} | "
            f"{_f(r.get('marcador_proyectado'))} | {_f(r.get('resultado_real'))} | "
            f"{_f(r.get('spread_modelo'))} | {_f(r.get('spread_consenso'))} | "
            f"{_f(r.get('total_modelo'))} | {_f(r.get('total_consenso'))} | "
            f"{_b(r.get('acierto_ganador'))} | {_b(r.get('acierto_spread'))} | {_b(r.get('acierto_total'))} |"
        )
    lines.append("")

    lines.append("## Preguntas sugeridas para el análisis")
    lines.append("")
    lines.append("1. ¿El modelo tiene un edge consistente año tras año, o hay mucha variabilidad entre temporadas?")
    lines.append("2. ¿El 63% de ML es sostenible o fue un año afortunado? Compara con las temporadas previas.")
    lines.append("3. ¿En qué tipo de partidos (rango de favoritismo, confianza) el modelo acierta más y menos?")
    lines.append("4. ¿Hay un sesgo por temporada? ¿Empeora o mejora con el tiempo?")
    lines.append("5. ¿El error de margen es estable o ha crecido/mejorado?")
    lines.append("6. ¿Qué sugerencias concretas harías para mejorar el modelo?")
    lines.append("")

    return "\n".join(lines)


# ---------- JSON ----------

def to_json_dict(df: pd.DataFrame, stats: dict) -> dict:
    records = []
    for _, r in df.iterrows():
        rec = {}
        for k, v in r.items():
            if v is None:
                rec[k] = None
                continue
            try:
                if pd.isna(v):
                    rec[k] = None
                    continue
            except (TypeError, ValueError):
                pass
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            rec[k] = v
        records.append(rec)

    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "partidos": records,
    }