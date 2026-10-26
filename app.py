import os
import streamlit as st
import pandas as pd
import datetime

import config
import db
import predict_games
import rebuild_weeks

st.set_page_config(
    page_title="NFL & NCAAF Predictor",
    page_icon="🏈",
    layout="wide"
)

def main():
    st.title("🏈 Predictor de Fútbol Americano (NFL & NCAAF)")
    st.sidebar.header("Configuración de la App")

    # Selección de Liga
    league_choice = st.sidebar.selectbox(
        "Selecciona la liga",
        options=["nfl", "ncaaf"],
        format_func=lambda x: "NFL" if x == "nfl" else "NCAAF (Colegial)"
    )

    season = st.sidebar.number_input("Temporada", min_value=2020, max_value=2030, value=2026, step=1)
    
    # Selector de semana seguro con manejo correcto del session_state
    if "gen_from_week" not in st.session_state:
        st.session_state["gen_from_week"] = 8

    week = st.sidebar.slider("Semana", min_value=1, max_value=22, value=st.session_state["gen_from_week"], step=1)
    
    # Actualización segura del session_state sin sobreescribir el widget instanciado
    st.session_state["gen_from_week"] = week

    skip_odds = st.sidebar.checkbox("Usar solo caché de MongoDB (No llamar a The Odds API)", value=True)
    refresh_odds = st.sidebar.checkbox("Forzar actualización de cuotas (Odds API)", value=False)

    st.sidebar.markdown("---")
    
    col1, col2 = st.columns(2)
    
    run_prediction = col1.button("Generar / actualizar predicciones", type="primary", use_container_width=True)
    run_sync = col2.button("Sincronizar resultados finales", use_container_width=True)

    if skip_odds:
        st.success("No se llamará a The Odds API (solo caché de Mongo).")
    elif refresh_odds:
        st.warning("Se forzará una llamada nueva a The Odds API (consumirá cuota).")

    if run_prediction:
        with st.spinner(f"Procesando predicciones para {league_choice.upper()} - Semana {week}..."):
            try:
                db.ensure_indexes()
                success = rebuild_weeks.rebuild_week(
                    season=season,
                    week=week,
                    league=league_choice,
                    skip_odds=skip_odds,
                    refresh_odds=refresh_odds,
                    verbose=False,
                    prediction_mode="official"
                )
                if success:
                    st.success(f"¡Semana {week} procesada y guardada exitosamente!")
                else:
                    st.error(f"Hubo un error al procesar la semana {week}. Revisa los registros.")
            except Exception as e:
                st.error(f"Error crítico durante la ejecución: {e}")

    if run_sync:
        with st.spinner(f"Sincronizando marcadores finales para {league_choice.upper()} - Semana {week}..."):
            try:
                db.ensure_indexes()
                week_summary = rebuild_weeks.sync_final_results(
                    int(season), int(week), league_choice
                )
                st.success(f"Sincronización completa: {week_summary.get('updated', 0)} partidos actualizados.")
            except Exception as e:
                st.error(f"Error al sincronizar resultados: {e}")

    st.markdown("---")
    st.subheader(f"📊 Reporte de Picks: {league_choice.upper()} · Semana {week}")

    try:
        picks_df = db.get_weekly_picks(season, week, league_choice)
        if not picks_df.empty:
            st.dataframe(picks_df, use_container_width=True)
        else:
            st.info(f"No hay picks registrados o generados todavía para la semana {week}.")
    except Exception as e:
        st.info(f"Aún no hay datos disponibles para mostrar en la tabla: {e}")

if __name__ == "__main__":
    main()
