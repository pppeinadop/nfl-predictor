"""
Configuración central. Copia este archivo como .env o exporta las variables
de entorno antes de correr el sistema.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _setting(name: str, default: str = "") -> str:
    """Lee primero variables de entorno y luego secretos de Streamlit Cloud."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


# --- The Odds API ---
ODDS_API_KEY = _setting("ODDS_API_KEY", "TU_API_KEY_AQUI")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

REGIONS = "us"
ODDS_FORMAT = "american"
GAME_MARKETS = "h2h,spreads,totals"

PLAYER_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
]

# --- Modelo Elo ---
ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_ADVANTAGE = 55
ELO_MOV_MULTIPLIER = True

ELO_POINTS_PER_SPREAD_NFL = 25.0
ELO_POINTS_PER_SPREAD_NCAAF = 12.0
ELO_POINTS_PER_SPREAD = ELO_POINTS_PER_SPREAD_NFL  # compat

HOME_FIELD_POINTS = 2.5

RECENCY_HALFLIFE_GAMES = 3
PRIOR_STRENGTH_GAMES = 6
SPLIT_PRIOR_GAMES = 4

ROLLING_WINDOW_TEAM = 8
ROLLING_WINDOW_PLAYER = 6

BLEND_WEIGHT_ELO = 0.5

MIN_SPLIT_GAMES = 2

SOS_FULL_STRENGTH_GAMES = 4
SOS_MIN_FACTOR = 0.8
SOS_MAX_FACTOR = 1.25

CARRY_OVER_PREVIOUS_SEASON = True
SEASON_CARRYOVER_WEIGHT = 0.67

# Calibración del margen
MARGIN_SHRINK_NFL = 0.75
MARGIN_SHRINK_NCAAF = 1.25
MARGIN_SHRINK = MARGIN_SHRINK_NFL  # compat

ML_PROB_SCALE = 6.5

MAX_TOTAL_ADJUSTMENT_NEG = -6.0
MAX_TOTAL_ADJUSTMENT_POS = 4.0

# ============================================================
# Simulación Monte Carlo
# ============================================================
# Desviación estándar del error del modelo (en puntos).
# Calibradas a partir del RMSE histórico de los sweeps:
#   NFL  margin_std ≈ 13.2 (RMSE a shrink=0.75)
#   NCAAF margin_std ≈ 18.9 (RMSE a shrink=1.25)
# Para total: RMSE similar al de margen (los totales son un poco más ruidosos).
SIM_MARGIN_STD_NFL = 13.2
SIM_MARGIN_STD_NCAAF = 18.9
SIM_TOTAL_STD_NFL = 14.0
SIM_TOTAL_STD_NCAAF = 19.9

# Número de simulaciones por partido.
SIM_N_SAMPLES = 10000

# ============================================================
# Ajustes adicionales
# ============================================================

ENABLE_INJURY_ADJUSTMENT = True
INJURY_QB_OUT_PENALTY = -7.0
INJURY_QB_DOUBTFUL_PENALTY = -3.5
INJURY_QB_QUESTIONABLE_PENALTY = -1.0
INJURY_UNKNOWN_STARTER_FACTOR = 0.5

INJURY_OL_CLUSTER_MIN = 3
INJURY_OL_PENALTY_PER_EXTRA = -1.5
INJURY_OL_MAX_PENALTY = -6.0

ENABLE_REST_ADJUSTMENT = True
SHORT_REST_DAYS_THRESHOLD = 4
SHORT_REST_PENALTY = -1.0
EXTRA_REST_DAYS_THRESHOLD = 9
EXTRA_REST_BONUS = 1.0

ENABLE_RIVALRY_ADJUSTMENT = True
DIVISIONAL_MARGIN_SHRINK = 0.85

ENABLE_WEATHER_ADJUSTMENT = True
WEATHER_WIND_THRESHOLD_KMH = 24
WEATHER_WIND_TOTAL_PENALTY = -2.0
WEATHER_WIND_STRONG_THRESHOLD_KMH = 32
WEATHER_WIND_STRONG_TOTAL_PENALTY = -4.0
WEATHER_RAIN_PROB_THRESHOLD = 60
WEATHER_RAIN_TOTAL_PENALTY = -2.0

NFL_STADIUMS = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "dome": True},
    "ATL": {"lat": 33.7554, "lon": -84.4008, "dome": True},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "dome": False},
    "BUF": {"lat": 42.7738, "lon": -78.7870, "dome": False},
    "CAR": {"lat": 35.2258, "lon": -80.8528, "dome": False},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "dome": False},
    "CIN": {"lat": 39.0955, "lon": -84.5160, "dome": False},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "dome": False},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "dome": True},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "dome": False},
    "DET": {"lat": 42.3400, "lon": -83.0456, "dome": True},
    "GB":  {"lat": 44.5013, "lon": -88.0622, "dome": False},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "dome": True},
    "IND": {"lat": 39.7601, "lon": -86.1639, "dome": True},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "dome": False},
    "KC":  {"lat": 39.0489, "lon": -94.4839, "dome": False},
    "LV":  {"lat": 36.0909, "lon": -115.1833, "dome": True},
    "LAC": {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "LA":  {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "dome": False},
    "MIN": {"lat": 44.9737, "lon": -93.2577, "dome": True},
    "NE":  {"lat": 42.0909, "lon": -71.2643, "dome": False},
    "NO":  {"lat": 29.9511, "lon": -90.0812, "dome": True},
    "NYG": {"lat": 40.8135, "lon": -74.0745, "dome": False},
    "NYJ": {"lat": 40.8135, "lon": -74.0745, "dome": False},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "dome": False},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "dome": False},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "dome": False},
    "SF":  {"lat": 37.4030, "lon": -121.9700, "dome": False},
    "TB":  {"lat": 27.9759, "lon": -82.5033, "dome": False},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "dome": False},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "dome": False},
}

EDGE_THRESHOLD_SPREAD = 1.5
EDGE_THRESHOLD_TOTAL = 2.0
EDGE_THRESHOLD_PROP = 0.05

EDGE_SUSPICIOUS_SPREAD = 6.0
EDGE_SUSPICIOUS_TOTAL = 8.0

DISCARD_GAMES_WITHOUT_MARKET = False
MAX_ABS_MONEYLINE = 20000
DISCARD_GAMES_UNKNOWN_TEAMS = True

CURRENT_SEASON = 2026
REPORT_AUTO_OPEN = True

TEAM_ABBR_TO_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

MONGO_URI = _setting("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = _setting("MONGO_DB_NAME", "nfl_predictor")

LEAGUES = {
    "nfl": {"odds_sport_key": "americanfootball_nfl", "label": "NFL"},
    "ncaaf": {"odds_sport_key": "americanfootball_ncaaf", "label": "NCAAF (Colegial)"},
}

CFBD_API_KEY = _setting("CFBD_API_KEY", "TU_API_KEY_AQUI")

TIMEZONE = "America/Ciudad_Juarez"
TIMEZONE_LABEL = "hora local"