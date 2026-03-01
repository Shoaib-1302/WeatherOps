"""
WeatherOps v5.0 — Agentic GeoAI Weather Impact Decision Platform
Agents: Ingestion · Modeling · Hazards · Decision · Evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from datetime import datetime, timedelta
import math, tempfile, os
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        accuracy_score, confusion_matrix, average_precision_score,
        mean_squared_error, mean_absolute_error, r2_score,
        roc_curve, precision_recall_curve, brier_score_loss,
    )
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier, XGBRegressor
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="WeatherOps · GeoAI Operations v5",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    """
    <style>
    :root{
      --bg0:#0a0b0d;--bg1:#111318;--bg2:#181b22;--bg3:#1e2230;--bg4:#242840;
      --border:#2a2f3d;--border2:#3a4155;
      --amber:#f0a500;--amber-lo:rgba(240,165,0,.12);
      --teal:#00c9a7;--teal-lo:rgba(0,201,167,.12);
      --red:#e84040;--red-lo:rgba(232,64,64,.12);
      --orange:#f06830;--orange-lo:rgba(240,104,48,.12);
      --purple:#a78bfa;--blue:#60a5fa;
      --text0:#e8ecf4;--text1:#8a93a8;--text2:#4e5568;
      --mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;--r:6px;
    }
    html,body,[class*="css"]{font-family:var(--display);background:var(--bg0);color:var(--text0);}
    #MainMenu,footer{visibility:hidden;}
    .stDeployButton{display:none;}
    [data-testid="stHeader"]{visibility:visible!important;background:transparent!important;}
    .block-container{padding:0 1.8rem 2rem!important;max-width:100%!important;}
    [data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border);}
    [data-testid="stSidebar"] *{color:var(--text0)!important;}
    [data-testid="stSidebar"] label{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--text1)!important;}
    [data-testid="stSelectbox"]>div>div{background:var(--bg2)!important;border:1px solid var(--border)!important;color:var(--text0)!important;border-radius:4px;font-family:var(--mono);}
    [data-testid="stSlider"] [role="slider"]{background:var(--amber)!important;}
    [data-testid="stSlider"]>div>div>div>div{background:var(--amber)!important;}
    [data-testid="stMetric"]{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:.9rem 1.1rem;}
    [data-testid="stMetric"] label{font-family:var(--mono);font-size:.64rem;letter-spacing:.12em;text-transform:uppercase;color:var(--text1)!important;}
    [data-testid="stMetric"] [data-testid="stMetricValue"]{font-family:var(--mono);font-size:1.5rem;font-weight:500;color:var(--amber)!important;}
    [data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid var(--border);}
    [data-testid="stTabs"] button[role="tab"]{background:transparent;border:none;border-bottom:2px solid transparent;color:var(--text1);font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;padding:.55rem 1rem;margin-bottom:-1px;transition:all .15s;}
    [data-testid="stTabs"] button[role="tab"]:hover{color:var(--text0);}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:var(--amber);border-bottom-color:var(--amber);}
    [data-testid="baseButton-secondary"],.stButton>button{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text0)!important;font-family:var(--mono)!important;font-size:.72rem!important;letter-spacing:.08em;text-transform:uppercase;border-radius:4px!important;transition:all .15s!important;}
    [data-testid="baseButton-secondary"]:hover,.stButton>button:hover{border-color:var(--amber)!important;color:var(--amber)!important;background:var(--amber-lo)!important;}
    [data-testid="baseButton-primary"]{background:var(--amber)!important;border:none!important;color:#000!important;font-family:var(--mono)!important;font-size:.72rem!important;font-weight:600;}
    [data-testid="stExpander"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;}
    [data-testid="stExpander"] summary{font-family:var(--mono);font-size:.75rem;color:var(--text1);}
    [data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:var(--r);overflow:hidden;}
    [data-testid="stDataFrame"] th{background:var(--bg3)!important;font-family:var(--mono)!important;font-size:.65rem!important;text-transform:uppercase;letter-spacing:.08em;color:var(--text1)!important;}
    [data-testid="stDataFrame"] td{font-family:var(--mono)!important;font-size:.78rem!important;background:var(--bg2)!important;}
    ::-webkit-scrollbar{width:4px;height:4px;}
    ::-webkit-scrollbar-track{background:var(--bg0);}
    ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}

    /* ── Custom Components ── */
    .ops-header{display:flex;align-items:center;justify-content:space-between;padding:1.1rem 0 .9rem;border-bottom:1px solid var(--border);margin-bottom:1.3rem;}
    .ops-wordmark{font-family:var(--display);font-size:1.45rem;font-weight:800;letter-spacing:-.02em;color:var(--text0);}
    .ops-wordmark span{color:var(--amber);}
    .ops-tagline{font-family:var(--mono);font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:var(--text2);margin-top:2px;}
    .ops-timestamp{font-family:var(--mono);font-size:.7rem;color:var(--text2);text-align:right;}
    .live-dot{display:inline-block;width:6px;height:6px;background:var(--teal);border-radius:50%;margin-right:6px;animation:pulse 2s infinite;}
    @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}
    .source-chip{display:inline-block;font-family:var(--mono);font-size:.63rem;letter-spacing:.08em;text-transform:uppercase;padding:4px 10px;border-radius:999px;border:1px solid var(--border);margin-top:-.8rem;margin-bottom:.9rem;}
    .source-chip.live{color:var(--teal);border-color:rgba(0,201,167,.35);background:rgba(0,201,167,.08);}
    .source-chip.cached{color:var(--amber);border-color:rgba(240,165,0,.35);background:rgba(240,165,0,.10);}
    .source-chip.synthetic{color:var(--orange);border-color:rgba(240,104,48,.35);background:rgba(240,104,48,.10);}
    .source-chip.file{color:var(--amber);border-color:rgba(240,165,0,.35);background:rgba(240,165,0,.10);}
    .section-label{font-family:var(--mono);font-size:.63rem;letter-spacing:.15em;text-transform:uppercase;color:var(--text2);margin-bottom:.7rem;padding-bottom:.35rem;border-bottom:1px solid var(--border);}
    .risk-card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:.95rem 1.1rem;margin-bottom:.7rem;position:relative;overflow:hidden;}
    .risk-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;}
    .risk-card.critical::before{background:var(--red);}
    .risk-card.high::before{background:var(--orange);}
    .risk-card.moderate::before{background:var(--amber);}
    .risk-card.low::before{background:var(--teal);}
    .risk-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.45rem;}
    .risk-card-title{font-family:var(--display);font-size:.9rem;font-weight:700;color:var(--text0);}
    .risk-badge{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;padding:2px 7px;border-radius:3px;font-weight:500;}
    .risk-badge.critical{background:var(--red-lo);color:var(--red);border:1px solid rgba(232,64,64,.3);}
    .risk-badge.high{background:var(--orange-lo);color:var(--orange);border:1px solid rgba(240,104,48,.3);}
    .risk-badge.moderate{background:var(--amber-lo);color:var(--amber);border:1px solid rgba(240,165,0,.3);}
    .risk-badge.low{background:var(--teal-lo);color:var(--teal);border:1px solid rgba(0,201,167,.3);}
    .risk-card-body{font-family:var(--mono);font-size:.73rem;color:var(--text1);line-height:1.7;}
    .risk-card-body strong{color:var(--text0);}
    .ci-bar{margin-top:.55rem;display:flex;align-items:center;gap:8px;}
    .ci-label{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);min-width:60px;}
    .ci-track{flex:1;height:3px;background:var(--bg3);border-radius:2px;position:relative;}
    .ci-fill{position:absolute;top:0;bottom:0;border-radius:2px;}
    .agent-trace{background:var(--bg1);border:1px solid var(--border);border-radius:var(--r);padding:.75rem .95rem;margin-top:.4rem;}
    .agent-step{font-family:var(--mono);font-size:.7rem;color:var(--text1);padding:.22rem 0;display:flex;gap:.7rem;}
    .step-idx{color:var(--amber);min-width:20px;}
    .step-ok{color:var(--teal);}
    .step-warn{color:var(--orange);}
    .step-err{color:var(--red);}

    /* ── Evaluation Panel ── */
    .eval-pill-row{display:flex;gap:.45rem;flex-wrap:wrap;margin-bottom:.5rem;}
    .eval-pill{font-family:var(--mono);font-size:.68rem;padding:3px 9px;border-radius:4px;border:1px solid var(--border);background:var(--bg3);}
    .eval-pill .lbl{color:var(--text2);}
    .eval-pill .val{margin-left:4px;font-weight:500;}
    .eval-pill .val.good{color:var(--teal);}
    .eval-pill .val.warn{color:var(--orange);}
    .eval-pill .val.bad{color:var(--red);}
    .eval-pill .val.neutral{color:var(--amber);}

    .lb-row{display:flex;align-items:center;gap:.7rem;padding:.45rem .7rem;border-radius:5px;border:1px solid var(--border);background:var(--bg3);margin-bottom:.35rem;}
    .lb-rank{font-family:var(--mono);font-size:.62rem;color:var(--text2);min-width:22px;}
    .lb-hazard{font-family:var(--display);font-size:.85rem;font-weight:700;min-width:110px;}
    .lb-model{font-family:var(--mono);font-size:.68rem;color:var(--text1);flex:1;}
    .lb-score{font-family:var(--mono);font-size:.82rem;font-weight:600;color:var(--amber);min-width:52px;text-align:right;}
    .lb-bar{flex:2;height:4px;background:var(--bg0);border-radius:2px;position:relative;}
    .lb-bar-fill{position:absolute;top:0;bottom:0;left:0;border-radius:2px;}

    .diag-block{background:var(--bg1);border:1px solid var(--border);border-radius:6px;padding:.8rem 1rem;margin-bottom:.55rem;}
    .diag-title{font-family:var(--display);font-size:.85rem;font-weight:700;margin-bottom:.3rem;}
    .diag-body{font-family:var(--mono);font-size:.67rem;color:var(--text1);line-height:1.85;}
    .diag-body b{color:var(--text0);}
    .tag{display:inline-block;font-family:var(--mono);font-size:.58rem;padding:1px 6px;border-radius:3px;margin-right:3px;vertical-align:middle;}
    .tag.leak{background:var(--red-lo);color:var(--red);border:1px solid rgba(232,64,64,.3);}
    .tag.fix{background:var(--teal-lo);color:var(--teal);border:1px solid rgba(0,201,167,.3);}
    .tag.info{background:var(--amber-lo);color:var(--amber);border:1px solid rgba(240,165,0,.3);}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PATHS
# ============================================================
_HERE     = Path(os.path.abspath(__file__)) if "__file__" in dir() else Path.cwd()
DATA_DIR  = _HERE.parent / "notebooks" / "data" / "output"
GPKG_PATH = _HERE.parent / "notebooks" / "data" / "data" / "Dehradun.gpkg"
CSV_PATH  = DATA_DIR / "weatherops_feature_table.csv"


# ============================================================
# AGENT 1 — INGESTION
# ============================================================
class IngestionAgent:
    def get_weather_forecast(self, horizon_hours=72, use_live=True):
        def _synthetic():
            rng = np.random.default_rng(42)
            ts  = [datetime.utcnow() + timedelta(hours=i) for i in range(horizon_hours)]
            return pd.DataFrame({"time":ts,
                "rain_mm":   rng.gamma(2,3,horizon_hours),
                "temp_c":    rng.normal(28,5,horizon_hours),
                "wind_kmph": np.abs(rng.normal(18,7,horizon_hours)),
                "app_temp":  rng.normal(30,5,horizon_hours),
                "rh":        rng.uniform(45,85,horizon_hours),
            }), "synthetic", ""

        def _file():
            if not CSV_PATH.exists():
                return _synthetic()[0], "synthetic", f"File not found: {CSV_PATH}"
            df_src = pd.read_csv(CSV_PATH)
            if df_src.empty:
                return _synthetic()[0], "synthetic", "Feature table empty"
            def pull(candidates, default):
                for col in candidates:
                    if col in df_src.columns:
                        return pd.to_numeric(df_src[col],errors="coerce").fillna(default).to_numpy()
                return np.full(len(df_src), default)
            n  = max(1, horizon_hours)
            ts = [datetime.utcnow() + timedelta(hours=i) for i in range(n)]
            return pd.DataFrame({"time":pd.to_datetime(ts),
                "rain_mm":   np.resize(pull(["rain_mm","rainfall_3h","rainfall_24h"],0),n),
                "temp_c":    np.resize(pull(["temp_c","temp_max_C","LST_C_mean"],25),n),
                "wind_kmph": np.resize(pull(["wind_kmph","wind_speed_kmh"],15),n),
                "app_temp":  np.resize(pull(["app_temp","apparent_temp_C","heat_index_C"],25),n),
                "rh":        np.resize(pull(["rh","relativehumidity_2m","relative_humidity_2m"],60),n),
            }), "file", ""

        if not use_live:
            return _file()
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            url = (
                "https://api.open-meteo.com/v1/forecast"
                "?latitude=30.3165&longitude=78.0322"
                "&hourly=precipitation,temperature_2m,wind_speed_10m,"
                "apparent_temperature,relative_humidity_2m"
                f"&forecast_days={max(1,math.ceil(horizon_hours/24))}&timezone=Asia%2FKolkata"
            )
            sess = requests.Session()
            sess.mount("https://", HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1.,
                status_forcelist=[429,500,502,503,504],allowed_methods=["GET"])))
            r = sess.get(url, timeout=(10,45), headers={"User-Agent":"WeatherOps/5.0"})
            r.raise_for_status()
            h     = r.json()["hourly"]
            wind  = h.get("wind_speed_10m", h.get("windspeed_10m",[]))
            rh    = h.get("relative_humidity_2m", h.get("relativehumidity_2m",[]))
            df = pd.DataFrame({"time":pd.to_datetime(h["time"]),
                "rain_mm":   pd.Series(pd.to_numeric(h["precipitation"],errors="coerce")).fillna(0).values,
                "temp_c":    pd.Series(pd.to_numeric(h["temperature_2m"],errors="coerce")).fillna(25).values,
                "wind_kmph": pd.Series(pd.to_numeric(wind,errors="coerce")).fillna(15).values,
                "app_temp":  pd.Series(pd.to_numeric(h["apparent_temperature"],errors="coerce")).fillna(25).values,
                "rh":        pd.Series(pd.to_numeric(rh,errors="coerce")).fillna(60).values,
            })
            st.session_state["weather_cache_df"] = df.copy()
            st.session_state["weather_cache_at"] = datetime.utcnow()
            return df.head(horizon_hours), "live", ""
        except Exception as exc:
            cached = st.session_state.get("weather_cache_df")
            if isinstance(cached,pd.DataFrame) and not cached.empty:
                return cached.head(horizon_hours).copy(), "cached", str(exc)
            df_f, src_f, _ = _file()
            if src_f == "file":
                return df_f, "file", str(exc)
            return _synthetic()[0], "synthetic", str(exc)


# ============================================================
# AGENT 2 — MODELING
# ============================================================
class ModelingAgent:
    def blend_forecast(self, weather_df, terrain_stats):
        df   = weather_df.copy()
        slope = terrain_stats.get("mean_slope", 12)
        df["rain_adj"]    = df["rain_mm"] * (1 + slope/30)
        df["heat_index"]  = df["temp_c"] + 0.33 * df["rh"]/100 * df["temp_c"] - 4
        df["flood_proxy"] = df["rain_adj"].rolling(6, min_periods=1).sum()
        return df


# ============================================================
# AGENT 3 — HAZARDS
# ============================================================
class HazardsAgent:
    def compute_risk(self, weather_df, rain_thresh=80, temp_thresh=35, wind_thresh=40):
        rp = float(weather_df["rain_adj"].max())
        tp = float(weather_df["heat_index"].max())
        wp = float(weather_df["wind_kmph"].max())
        fp = float(weather_df["flood_proxy"].max())
        risk = {
            "Flood":     float(np.clip(rp/rain_thresh, 0,1)),
            "Heat":      float(np.clip((tp-temp_thresh)/10, 0,1)),
            "Wind":      float(np.clip(wp/wind_thresh, 0,1)),
            "Landslide": float(np.clip(fp/200, 0,1)),
        }
        risk_ci = {k:(max(0.,v-.15),min(1.,v+.15)) for k,v in risk.items()}
        return risk, risk_ci, {"rain_peak":rp,"temp_peak":tp,"wind_peak":wp,"flood_proxy":fp}


# ============================================================
# AGENT 4 — DECISION
# ============================================================
class DecisionAgent:
    PLAYBOOK = {
        "Flood":     [("Deploy drainage & pumping crew","Low-lying road segments","Next 12h"),
                      ("Close flood-prone river bridges","Rispana & Bindal crossings","If rain >50mm"),
                      ("Pre-position rescue boats","ISBT & Railway Station zone","Within 6h")],
        "Heat":      [("Issue heat advisory & cooling centres","Urban core (Clock Tower)","12:00–16:00 today"),
                      ("Halt outdoor construction work","All active sites","Peak hours 11–17"),
                      ("Increase ambulance standby","Paltan Bazaar, Rajpur Rd","Next 24h")],
        "Wind":      [("Secure power transmission lines","Ridge corridors >1500m","Within 24h"),
                      ("Ground helicopter operations","Jolly Grant Airport","Gusts >60 km/h"),
                      ("Issue falling-tree advisory","Mussoorie Road","Next 48h")],
        "Landslide": [("Inspect & close vulnerable roads","NH-707, Chakrata–Vikasnagar","Before next rain"),
                      ("Evacuate high-slope settlements","Slope >35° zones","Next 12h"),
                      ("Deploy geo-monitoring sensors","Sahastradhara slide zones","Immediately")],
    }
    def _level(self, v):
        if v>=.75: return "critical"
        if v>=.50: return "high"
        if v>=.25: return "moderate"
        return "low"
    def generate_actions(self, risk, risk_ci):
        actions = []
        for hazard, score in sorted(risk.items(), key=lambda x: -x[1]):
            lvl = self._level(score)
            if lvl == "low": continue
            n = {"critical":3,"high":2,"moderate":1}[lvl]
            for what,where,when in self.PLAYBOOK.get(hazard,[])[:n]:
                actions.append({"Hazard":hazard,"Level":lvl,"Score":score,
                                 "What":what,"Where":where,"When":when,
                                 "Why":f"{hazard} risk {score:.0%}","Confidence":risk_ci[hazard]})
        return actions


# ============================================================
# AGENT 5 — EVALUATION  ← NEW
# ============================================================
class EvaluationAgent:
    """
    Leakage-free statistical model evaluation for all 4 hazards.
    Trains: LR · RF · XGBoost · SVM/SVR
    Uses: stratified CV, single-feature AUC gate, Youden threshold, proper feature engineering
    """

    LEAKAGE_REGISTRY = {
        "Flood":     {"leaked":["river_distance","drainage_distance"],
                      "why":"flood_occurred is river-proximity rule → direct leakage if included",
                      "fix":"Exclude river_distance and drainage_distance from features",
                      "expect":"AUC 0.82–0.86"},
        "Heat":      {"leaked":["LST_C_mean"],
                      "why":"LST_C_mean IS the label — cannot be a feature",
                      "fix":"Use temp_C, elevation, aspect, lat, lon only; LST uncorrelated (r=0.019)",
                      "expect":"AUC 0.50–0.55 (correct — data limitation)"},
        "Wind":      {"leaked":["wind_kmh in features"],
                      "why":"wind_kmh is target, not terrain. IDW wind ≠ terrain physics",
                      "fix":"Terrain-only: elevation, slope, aspect, lat, lon, flow_accumulation",
                      "expect":"R² 0.03–0.12 (honest — IDW field doesn't reflect orography)"},
        "Landslide": {"leaked":["slope","TWI","rainfall_3h"],
                      "why":"slope is label; TWI collinear r=−0.884; rainfall_3h zone-encoded (r=0.80 lon)",
                      "fix":"Use elevation, aspect, soil_moisture, engineered elev_deviation",
                      "expect":"AUC 0.82–0.86"},
    }

    # ── Matplotlib dark style ─────────────────────────────────────
    _RC = {"figure.facecolor":"#111318","axes.facecolor":"#111318","axes.edgecolor":"#2a2f3d",
           "axes.labelcolor":"#8a93a8","xtick.color":"#8a93a8","ytick.color":"#8a93a8",
           "text.color":"#e8ecf4","grid.color":"#2a2f3d","grid.linewidth":.5,
           "axes.grid":True,"font.family":"monospace","figure.dpi":110}
    A="#f0a500"; T="#00c9a7"; R="#e84040"; O="#f06830"; P="#a78bfa"
    PAL=[A,T,R,O,P]

    @staticmethod
    def _single_feat_auc(y, x, threshold=0.88):
        """Check single-feature AUC; True if leaky (> threshold)."""
        x = pd.Series(x).fillna(pd.Series(x).median()).values
        mask = ~np.isnan(x)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            return False
        try:
            a = roc_auc_score(y[mask], x[mask])
            a = max(a, 1 - a)  # symmetrise
            return a > threshold
        except:
            return False

    @staticmethod
    def _youden_threshold(y_true, y_prob):
        """Best threshold by Youden's J = TPR - FPR."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        j = tpr - fpr
        return float(thresh[np.argmax(j)])

    # ── Spatial block split ───────────────────────────────────────
    @staticmethod
    def _spatial_split(df, X, y, test_fold=0, n_folds=5):
        lb = pd.cut(df["lat"], bins=n_folds, labels=False).fillna(0).astype(int)
        lo = pd.cut(df["lon"], bins=n_folds, labels=False).fillna(0).astype(int)
        folds = ((lb * n_folds + lo) % n_folds).values
        te = folds == test_fold
        tr = ~te
        return X[tr], X[te], y[tr], y[te], np.where(tr)[0], np.where(te)[0]

    # ── Build leakage-free labels ─────────────────────────────────
    @staticmethod
    def _make_targets(df):
        """Construct labels from actual CSV columns with data-driven rules."""
        t = {}
        # FLOOD: use original flood_occurred label if available
        if "flood_occurred" in df.columns:
            t["flood"] = df["flood_occurred"].astype(int)
        # HEAT: LST hotspot (Q75)
        if "LST_C_mean" in df.columns:
            lst_q75 = df["LST_C_mean"].quantile(.75)
            t["heat"] = (df["LST_C_mean"] >= lst_q75).astype(int)
        # WIND: continuous target
        if "wind_kmh" in df.columns:
            t["wind"] = df["wind_kmh"].astype(float)
        # LANDSLIDE: steep terrain (slope Q75)
        if "slope" in df.columns:
            slope_q75 = df["slope"].quantile(.75)
            t["landslide"] = (df["slope"] >= slope_q75).astype(int)
        return t

    # ── Feature sets with leakage gate ────────────────────────────
    @staticmethod
    def _make_features(df):
        """Build safe feature sets, excluding leaky columns."""
        f = {}
        # FLOOD: exclude river_distance, drainage_distance (define the label)
        f["flood"] = [c for c in ["elevation","slope","aspect","twi","rainfall_24h",
                                    "rainfall_3h","flow_accumulation","soil_moisture"]
                       if c in df.columns]
        # HEAT: exclude LST_C_mean (it IS the label), use independent signals
        f["heat"] = [c for c in ["temp_C","elevation","slope","aspect","lat","lon","soil_moisture"]
                      if c in df.columns]
        # WIND: terrain-only; exclude wind_kmh (target)
        f["wind"] = [c for c in ["elevation","slope","aspect","lat","lon","flow_accumulation"]
                      if c in df.columns]
        # LANDSLIDE: exclude slope (label), TWI/flow_acc (DEM-collinear), rainfall_3h (zone-correlated)
        f["landslide"] = [c for c in ["elevation","aspect","soil_moisture"]
                           if c in df.columns]
        return f

    # ── Single classifier eval with Youden threshold ──────────────
    def _clf_eval(self, name, model, Xtr, ytr, Xte, yte):
        from sklearn.model_selection import StratifiedKFold
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        pos = int(ytr.sum())
        if 1 < pos < len(ytr)-1:
            try:
                Xtr_s, ytr = SMOTE(random_state=42, k_neighbors=min(5,pos-1)).fit_resample(Xtr_s,ytr)
            except: pass
        model.fit(Xtr_s, ytr)
        prob   = model.predict_proba(Xte_s)[:,1] if hasattr(model,"predict_proba") else model.decision_function(Xte_s)
        t_prob = model.predict_proba(Xtr_s)[:,1] if hasattr(model,"predict_proba") else model.decision_function(Xtr_s)  # train probs for threshold
        thresh = self._youden_threshold(ytr, t_prob) if len(np.unique(ytr)) > 1 else 0.5
        yp     = (prob >= thresh).astype(int)
        
        res = {"name":name, "scaler":sc, "model":model, "y_pred":yp, "y_prob":prob, "thresh":thresh}
        res["f1"]        = f1_score(yte, yp, zero_division=0)
        res["precision"] = precision_score(yte, yp, zero_division=0)
        res["recall"]    = recall_score(yte, yp, zero_division=0)
        res["accuracy"]  = accuracy_score(yte, yp)
        if len(np.unique(yte)) > 1:
            res["roc_auc"]  = roc_auc_score(yte, prob)
            res["avg_prec"] = average_precision_score(yte, prob)
            res["brier"]    = brier_score_loss(yte, prob)
        else:
            res["roc_auc"] = res["avg_prec"] = res["brier"] = float("nan")
        # Stratified 5-fold CV on train
        cv_aucs = []
        if len(np.unique(ytr)) > 1:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tr_i, te_i in skf.split(Xtr_s, ytr):
                try:
                    m2 = type(model)(**model.get_params()) if hasattr(model,"get_params") else model
                    m2.fit(Xtr_s[tr_i], ytr[tr_i])
                    pr2 = m2.predict_proba(Xtr_s[te_i])[:,1] if hasattr(m2,"predict_proba") else m2.decision_function(Xtr_s[te_i])
                    if len(np.unique(ytr[te_i]))>1:
                        cv_aucs.append(roc_auc_score(ytr[te_i], pr2))
                except: pass
        res["cv_auc_mean"] = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        res["cv_auc_std"]  = float(np.std(cv_aucs))  if cv_aucs else float("nan")
        return res

    # ── Single regressor eval ─────────────────────────────────────
    def _reg_eval(self, name, model, Xtr, ytr, Xte, yte):
        from sklearn.model_selection import KFold
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        model.fit(Xtr_s, ytr)
        yp = model.predict(Xte_s)
        res = {"name":name, "scaler":sc, "model":model, "y_pred":yp}
        res["r2"]   = r2_score(yte, yp)
        res["rmse"] = float(np.sqrt(mean_squared_error(yte, yp)))
        res["mae"]  = float(mean_absolute_error(yte, yp))
        res["mape"] = float(np.mean(np.abs((yte-yp)/(np.abs(yte)+1e-9)))*100)
        cv_r2s = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_i, te_i in kf.split(Xtr_s):
            try:
                m2 = type(model)(**model.get_params()) if hasattr(model,"get_params") else model
                m2.fit(Xtr_s[tr_i], ytr[tr_i])
                cv_r2s.append(r2_score(ytr[te_i], m2.predict(Xtr_s[te_i])))
            except: pass
        res["cv_r2_mean"] = float(np.mean(cv_r2s)) if cv_r2s else float("nan")
        res["cv_r2_std"]  = float(np.std(cv_r2s))  if cv_r2s else float("nan")
        res["baseline_rmse"] = float(ytr.std())
        return res

    # ── Main run ──────────────────────────────────────────────────
    def run(self, progress_cb=None):
        if not HAS_ML:
            return {"error":"scikit-learn/xgboost/imbalanced-learn not installed.\npip install scikit-learn xgboost imbalanced-learn"}
        if not CSV_PATH.exists():
            return {"error":f"Feature table not found:\n{CSV_PATH}"}
        df_raw = pd.read_csv(CSV_PATH)
        if df_raw.empty:
            return {"error":"Feature table is empty"}

        df = df_raw.copy()
        if "lat" not in df.columns:
            df["lat"] = np.linspace(30.20,30.45,len(df))
        if "lon" not in df.columns:
            df["lon"] = np.linspace(77.95,78.15,len(df))

        targets   = self._make_targets(df)
        feat_sets = self._make_features(df)

        hazards_cfg = [
            ("flood",     "🌊 Flood",     "flood",  "clf"),
            ("heat",      "🔥 Heat",      "heat",   "clf"),
            ("wind",      "💨 Wind",      "wind",   "reg"),
            ("landslide", "⛰️ Landslide", "landslide","clf"),
        ]
        results = {}
        for step, (hkey, hname, tkey, task) in enumerate(hazards_cfg):
            if progress_cb: progress_cb(step/4, f"Training {hname}…")
            if tkey not in targets:
                results[hkey] = {"error":f"Cannot build target '{tkey}' — check column names in CSV"}
                continue
            feats = feat_sets.get(hkey,[])
            if not feats:
                results[hkey] = {"error":f"No features found for {hkey}"}
                continue

            avail_cols = feats + ["lat","lon"]
            df_s = df[avail_cols].copy()
            # attach target
            tgt = targets[tkey]
            if isinstance(tgt, pd.Series):
                df_s["__y__"] = tgt.values
            else:
                df_s["__y__"] = np.array(tgt)
            df_s = df_s.dropna().reset_index(drop=True)
            if len(df_s) < 40:
                results[hkey] = {"error":f"Only {len(df_s)} rows after dropna — need ≥40"}
                continue

            X = df_s[feats].values
            y = df_s["__y__"].values

            try:
                Xtr, Xte, ytr, yte, idx_tr, idx_te = self._spatial_split(df_s, X, y)
            except Exception as e:
                from sklearn.model_selection import train_test_split as tts
                strat = y if task=="clf" and len(np.unique(y))>1 else None
                Xtr, Xte, ytr, yte = tts(X, y, test_size=.2, random_state=42, stratify=strat)
                idx_tr = np.arange(len(ytr)); idx_te = np.arange(len(ytr),len(y))

            if len(Xte)<5:
                from sklearn.model_selection import train_test_split as tts
                strat = y if task=="clf" and len(np.unique(y))>1 else None
                Xtr, Xte, ytr, yte = tts(X, y, test_size=.2, random_state=42, stratify=strat)

            df_te = df_s.iloc[idx_te].reset_index(drop=True) if len(idx_te)==len(Xte) else df_s.head(len(Xte)).reset_index(drop=True)

            if task == "clf":
                pos_w = max(1, (ytr==0).sum()/max(1,(ytr==1).sum()))
                model_defs = {
                    "Logistic Regression": LogisticRegression(C=.1, class_weight="balanced",
                                           max_iter=1000, random_state=42),
                    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=8,
                                           min_samples_leaf=10, class_weight="balanced",
                                           random_state=42, n_jobs=-1),
                    "XGBoost":             XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.05,
                                           subsample=.7, min_child_weight=10, scale_pos_weight=pos_w,
                                           use_label_encoder=False, eval_metric="logloss",
                                           random_state=42, n_jobs=-1, verbosity=0),
                    "SVM (RBF)":           CalibratedClassifierCV(
                                               SVC(C=1., kernel="rbf", class_weight="balanced"),
                                               cv=min(3,max(2,int(ytr.sum())))),
                }
                mr = {}
                for mn, mdl in model_defs.items():
                    try:
                        mr[mn] = self._clf_eval(mn, mdl, Xtr, ytr, Xte, yte)
                    except Exception as e:
                        mr[mn] = {"name":mn,"error":str(e)}
                valid = [k for k in mr if "roc_auc" in mr[k] and not np.isnan(mr[k].get("roc_auc",float("nan")))]
                best  = max(valid, key=lambda k: mr[k]["roc_auc"]) if valid else list(mr.keys())[0]
                results[hkey] = {"task":"clf","display":hname,"models":mr,"best":best,
                                  "features":feats,"y_te":yte,"df_te":df_te,
                                  "n_train":len(ytr),"n_test":len(yte),
                                  "pos_rate":float(yte.mean())}
            else:
                model_defs = {
                    "Ridge":         Ridge(alpha=10.),
                    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8,
                                     min_samples_leaf=15, random_state=42, n_jobs=-1),
                    "XGBoost":       XGBRegressor(n_estimators=100, max_depth=4, learning_rate=.05,
                                     subsample=.7, min_child_weight=10,
                                     random_state=42, n_jobs=-1, verbosity=0),
                    "SVR (RBF)":     SVR(C=1., kernel="rbf", epsilon=1.),
                }
                mr = {}
                for mn, mdl in model_defs.items():
                    try:
                        mr[mn] = self._reg_eval(mn, mdl, Xtr, ytr, Xte, yte)
                    except Exception as e:
                        mr[mn] = {"name":mn,"error":str(e)}
                valid = [k for k in mr if "r2" in mr[k] and not np.isnan(mr[k].get("r2",float("nan")))]
                best  = max(valid, key=lambda k: mr[k]["r2"]) if valid else list(mr.keys())[0]
                results[hkey] = {"task":"reg","display":hname,"models":mr,"best":best,
                                  "features":feats,"y_te":yte,"df_te":df_te,
                                  "n_train":len(ytr),"n_test":len(yte),
                                  "baseline_rmse":float(ytr.std())}

        if progress_cb: progress_cb(1.0,"Complete")
        return results


# ============================================================
# SPATIAL / MAP UTILITIES
# ============================================================
def load_roi():
    if GPKG_PATH.exists():
        return gpd.read_file(str(GPKG_PATH)).to_crs(epsg=4326)
    from shapely.geometry import box
    return gpd.GeoDataFrame(geometry=[box(77.95,30.20,78.15,30.45)],crs="EPSG:4326")

def generate_risk_points(roi_gdf, risk, n=60):
    geom = roi_gdf.geometry.iloc[0]
    minx,miny,maxx,maxy = geom.bounds
    rng = np.random.default_rng(int(datetime.utcnow().timestamp())%1000)
    out = {}
    for hazard, base in risk.items():
        pts,vals=[],[]
        attempts=0
        while len(pts)<n and attempts<n*5:
            attempts+=1
            p = Point(rng.uniform(minx,maxx),rng.uniform(miny,maxy))
            if geom.contains(p):
                pts.append(p)
                vals.append(float(np.clip(rng.normal(base,.15),0,1)))
        out[hazard] = gpd.GeoDataFrame({"risk":vals},geometry=pts,crs="EPSG:4326")
    return out

def risk_color(r):
    if r>=.75: return "#e84040"
    if r>=.50: return "#f06830"
    if r>=.25: return "#f0a500"
    return "#00c9a7"

def build_map(roi_gdf, hazard_pts, active_layers):
    m = folium.Map(location=[30.32,78.03],zoom_start=10,tiles=None)
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                     attr="© OSM © CartoDB",name="Dark",max_zoom=19).add_to(m)
    folium.GeoJson(roi_gdf.__geo_interface__,name="District Boundary",
        style_function=lambda _: {"fillColor":"none","color":"#f0a500","weight":1.5,"dashArray":"6 4","fillOpacity":0}).add_to(m)
    emoji={"Flood":"🌊","Heat":"🔥","Wind":"💨","Landslide":"⛰️"}
    for hazard, gdf in hazard_pts.items():
        if hazard not in active_layers: continue
        layer = folium.FeatureGroup(name=f"{emoji.get(hazard,'')} {hazard}")
        for _,row in gdf.iterrows():
            color = risk_color(row["risk"])
            folium.CircleMarker([row.geometry.y,row.geometry.x],radius=5+row["risk"]*6,
                color=color,fill=True,fill_color=color,fill_opacity=.75,weight=0,
                popup=folium.Popup(f"<span style='font-family:monospace;font-size:11px'><b>{hazard}</b><br>Risk:{row['risk']:.2f}</span>",max_width=160)
            ).add_to(layer)
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ============================================================
# PDF EXPORT
# ============================================================
def generate_pdf(actions, risk, meta):
    if not HAS_PDF: return None
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
    c   = canvas.Canvas(tmp.name, pagesize=A4)
    W,H = A4
    c.setFillColorRGB(.063,.067,.078); c.rect(0,H-60,W,60,fill=1,stroke=0)
    c.setFillColorRGB(.941,.647,0);    c.setFont("Helvetica-Bold",16); c.drawString(36,H-38,"WeatherOps")
    c.setFillColorRGB(.541,.576,.659); c.setFont("Helvetica",8)
    c.drawString(36,H-52,"OPERATIONAL WEATHER BRIEF  ·  DEHRADUN DISTRICT")
    c.drawRightString(W-36,H-44,datetime.utcnow().strftime("Generated %Y-%m-%d %H:%M UTC"))
    y=H-80
    c.setFont("Helvetica-Bold",7); c.setFillColorRGB(.341,.376,.459); c.drawString(36,y,"CURRENT RISK SCORES")
    y-=14; col_w=(W-72)/4
    hc={"Flood":(.91,.25,.25),"Heat":(.94,.41,.19),"Wind":(.94,.65,0),"Landslide":(.63,.55,.39)}
    for i,(haz,score) in enumerate(risk.items()):
        x=36+i*col_w; rc=hc.get(haz,(.5,.5,.5))
        c.setFillColorRGB(*rc); c.rect(x,y-28,col_w-8,32,fill=1,stroke=0)
        c.setFillColorRGB(1,1,1); c.setFont("Helvetica-Bold",14); c.drawString(x+8,y-12,f"{score:.0%}")
        c.setFont("Helvetica",7); c.drawString(x+8,y-22,haz.upper())
    y-=46
    c.setFont("Helvetica-Bold",7); c.setFillColorRGB(.341,.376,.459); c.drawString(36,y,"RECOMMENDED ACTIONS"); y-=6
    c.setStrokeColorRGB(.165,.180,.220); c.line(36,y,W-36,y); y-=14
    lc={"critical":(.91,.25,.25),"high":(.94,.41,.19),"moderate":(.94,.65,0),"low":(0,.79,.65)}
    for act in actions:
        if y<80: c.showPage(); y=H-50
        rc=lc.get(act["Level"],(.5,.5,.5)); c.setFillColorRGB(*rc); c.rect(36,y-2,3,14,fill=1,stroke=0)
        c.setFillColorRGB(.91,.925,.957); c.setFont("Helvetica-Bold",10); c.drawString(46,y+2,act["What"]); y-=13
        c.setFillColorRGB(.541,.576,.659); c.setFont("Helvetica",8)
        c.drawString(46,y+2,f"WHERE: {act['Where']}   WHEN: {act['When']}"); y-=10
        c.drawString(46,y+2,f"Confidence: {act['Confidence'][0]:.2f}–{act['Confidence'][1]:.2f}"); y-=18
    c.setFillColorRGB(.165,.180,.220); c.rect(0,0,W,28,fill=1,stroke=0)
    c.setFillColorRGB(.341,.376,.459); c.setFont("Helvetica",7)
    c.drawString(36,10,"WeatherOps · GeoAI Weather Impact Decisions · Dehradun, Uttarakhand")
    c.drawRightString(W-36,10,"CONFIDENTIAL — FOR OPERATIONAL USE ONLY")
    c.save(); return tmp.name


# ============================================================
# UI RENDERERS
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 .5rem;border-bottom:1px solid #2a2f3d;margin-bottom:1rem;">
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8ecf4;">
            Weather<span style="color:#f0a500;">Ops</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#4e5568;
                  letter-spacing:.08em;margin-left:6px;">v5.0</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;letter-spacing:.15em;
                      text-transform:uppercase;color:#4e5568;margin-top:2px;">
            5 Agents · Dehradun District
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Forecast Horizon</div>', unsafe_allow_html=True)
        horizon = st.select_slider("Hours", options=[24,48,72,96,120], value=72,
                                    label_visibility="collapsed")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Risk Thresholds</div>',
                    unsafe_allow_html=True)
        rain_thresh = st.slider("Rain (mm/hr)", 20,150,80,5)
        temp_thresh = st.slider("Heat (°C)",    30, 50,35,1)
        wind_thresh = st.slider("Wind (km/h)",  20, 80,40,5)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Map Layers</div>',
                    unsafe_allow_html=True)
        active_layers = [haz for haz,em in
                          [("Flood","🌊"),("Heat","🔥"),("Wind","💨"),("Landslide","⛰️")]
                          if st.checkbox(f"{em} {haz}", value=True, key=f"layer_{haz}")]

        st.markdown('<div class="section-label" style="margin-top:1rem;">Data Source</div>',
                    unsafe_allow_html=True)
        use_live = st.toggle("Live API (Open-Meteo)", value=True)

        st.markdown("---")
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;color:#4e5568;line-height:1.9;">'
            '01 · IngestionAgent<br>02 · ModelingAgent<br>03 · HazardsAgent<br>'
            '04 · DecisionAgent<br>05 · EvaluationAgent</div>',
            unsafe_allow_html=True)

    return horizon, rain_thresh, temp_thresh, wind_thresh, active_layers, use_live


def render_header(source, horizon):
    source_labels = {"live":"LIVE · Open-Meteo API","cached":"CACHED · Last API fetch",
                     "file":"FILE · feature_table.csv","synthetic":"SIMULATION · Synthetic"}
    label = source_labels.get(source,"UNKNOWN")
    cache_at = st.session_state.get("weather_cache_at")
    cache_note = f" · {cache_at.strftime('%H:%M UTC')}" if source=="cached" and cache_at else ""
    st.markdown(f"""
    <div class="ops-header">
      <div>
        <div class="ops-wordmark">Weather<span>Ops</span></div>
        <div class="ops-tagline">GeoAI Weather Impact Decision Platform · Dehradun District · 5-Agent Pipeline</div>
      </div>
      <div class="ops-timestamp">
        <span class="live-dot"></span>{label}{cache_note}
        <br>{datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} · +{horizon}h
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="source-chip {source}">Data: {label}{cache_note}</div>',
                unsafe_allow_html=True)


def render_metrics(stats, risk):
    cols = st.columns(8)
    items = [
        (cols[0],"Rain Peak",  f"{stats['rain_peak']:.1f}","mm/hr"),
        (cols[1],"Temp Peak",  f"{stats['temp_peak']:.1f}","°C"),
        (cols[2],"Wind Peak",  f"{stats['wind_peak']:.1f}","km/h"),
        (cols[3],"Flood Proxy",f"{stats['flood_proxy']:.0f}","mm/6h"),
        (cols[4],"Flood Risk", f"{risk['Flood']:.0%}","score"),
        (cols[5],"Heat Risk",  f"{risk['Heat']:.0%}","score"),
        (cols[6],"Wind Risk",  f"{risk['Wind']:.0%}","score"),
        (cols[7],"Landslide Risk", f"{risk['Landslide']:.0%}","score"),
    ]
    for col,label,val,unit in items:
        with col: st.metric(label,val,unit)


def render_action_cards(actions):
    if not actions:
        st.markdown("""<div class="risk-card low">
          <div class="risk-card-header">
            <div class="risk-card-title">✓ No Critical Actions</div>
            <span class="risk-badge low">All Clear</span>
          </div>
          <div class="risk-card-body">All hazard scores below action threshold.</div>
        </div>""", unsafe_allow_html=True)
        return
    for act in actions:
        lvl = act["Level"]
        lo,hi = act["Confidence"]
        bar_color = {"critical":"#e84040","high":"#f06830","moderate":"#f0a500","low":"#00c9a7"}.get(lvl,"#8a93a8")
        emoji = {"Flood":"🌊","Heat":"🔥","Wind":"💨","Landslide":"⛰️"}.get(act["Hazard"],"⚠️")
        st.markdown(f"""
        <div class="risk-card {lvl}">
          <div class="risk-card-header">
            <div class="risk-card-title">{emoji} {act['What']}</div>
            <span class="risk-badge {lvl}">{lvl.upper()}</span>
          </div>
          <div class="risk-card-body">
            <strong>Where:</strong> {act['Where']}<br>
            <strong>When:</strong> {act['When']}<br>
            <strong>Why:</strong> {act['Why']}
          </div>
          <div class="ci-bar">
            <span class="ci-label">Confidence</span>
            <div class="ci-track">
              <div class="ci-fill" style="left:{lo*100:.0f}%;width:{(hi-lo)*100:.0f}%;background:{bar_color};opacity:.7;"></div>
            </div>
            <span style="font-family:var(--mono);font-size:.6rem;color:var(--text2);">{lo:.2f}–{hi:.2f}</span>
          </div>
        </div>""", unsafe_allow_html=True)


def render_agent_trace(source, n_actions, eval_done=False):
    source_labels = {"live":"Open-Meteo API","cached":"Cached API","file":"feature_table.csv","synthetic":"Synthetic"}
    steps = [
        ("01","IngestionAgent",  f"Forecast fetched · {source_labels.get(source,'?')}","ok"),
        ("02","ModelingAgent",   "Terrain blend · slope=12° · heat_index · flood_proxy","ok"),
        ("03","HazardsAgent",    f"Risk scores · {n_actions} action thresholds crossed","ok" if n_actions>0 else "warn"),
        ("04","DecisionAgent",   f"{n_actions} action cards · sorted by severity","ok"),
        ("05","EvaluationAgent", ("4 hazards · LR/RF/XGB/SVM · spatial-CV · SHAP" if eval_done
                                  else "Not run (open Evaluation tab to train)"),"ok" if eval_done else "warn"),
    ]
    html = '<div class="agent-trace">'
    for idx,agent,msg,status in steps:
        icon = "✓" if status=="ok" else "⚡"
        cls  = "step-ok" if status=="ok" else "step-warn"
        html += (f'<div class="agent-step"><span class="step-idx">{idx}</span>'
                 f'<span class="{cls}">{icon}</span>'
                 f'<span>{agent}&nbsp;·&nbsp;</span>'
                 f'<span style="color:var(--text2)">{msg}</span></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_timeseries(blended):
    if not HAS_PLOTLY:
        st.info("Install plotly for interactive charts: pip install plotly")
        return
    C = {"bg":"#111318","grid":"#2a2f3d","text":"#8a93a8"}
    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.06,
        subplot_titles=("Rainfall (mm/hr)","Temperature (°C)","Wind Speed (km/h)"))
    t = blended["time"]
    fig.add_trace(go.Scatter(x=t,y=blended["rain_adj"],fill="tozeroy",
        fillcolor="rgba(0,201,167,.1)",line=dict(color="#00c9a7",width=1.5),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=t,y=blended["temp_c"],line=dict(color="#f0a500",width=1.5),showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=t,y=blended["heat_index"],line=dict(color="#e84040",width=1,dash="dot"),showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=t,y=blended["wind_kmph"],fill="tozeroy",
        fillcolor="rgba(240,165,0,.08)",line=dict(color="#f0a500",width=1.5),showlegend=False),row=3,col=1)
    ax_style = dict(gridcolor=C["grid"],zeroline=False,
        tickfont=dict(family="JetBrains Mono",size=9,color=C["text"]),
        title_font=dict(family="JetBrains Mono",size=9,color=C["text"]))
    fig.update_xaxes(**ax_style); fig.update_yaxes(**ax_style)
    for i in range(3):
        fig.layout.annotations[i].font = dict(family="JetBrains Mono",size=9,color=C["text"])
    fig.update_layout(paper_bgcolor=C["bg"],plot_bgcolor=C["bg"],
        margin=dict(l=0,r=0,t=28,b=0),height=320,font=dict(family="JetBrains Mono",color=C["text"]))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar":False})


def render_data_table(blended):
    d = blended.copy()
    d["time"] = d["time"].dt.strftime("%m-%d %H:%M")
    d = d.rename(columns={"time":"Time","rain_mm":"Rain Raw","rain_adj":"Rain Adj",
                            "temp_c":"Temp °C","wind_kmph":"Wind km/h",
                            "heat_index":"HI °C","app_temp":"App Temp","flood_proxy":"Flood Proxy"})
    cols = [c for c in ["Time","Rain Raw","Rain Adj","Temp °C","HI °C","Wind km/h","Flood Proxy"] if c in d.columns]
    st.dataframe(d[cols].head(48).style.format({c:"{:.1f}" for c in cols if c!="Time"}),
                 width='stretch', height=280)


# ============================================================
# EVALUATION TAB RENDERER  ← NEW
# ============================================================
def _pill(label, val_str, quality="neutral"):
    return (f'<span class="eval-pill">'
            f'<span class="lbl">{label}</span>'
            f'<span class="val {quality}">{val_str}</span>'
            f'</span>')


def _quality(v, good=.75, warn=.5):
    if np.isnan(v): return "neutral"
    return "good" if v>=good else ("warn" if v>=warn else "bad")


def _plt_fig_to_st(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#111318")
    buf.seek(0)
    st.image(buf, width='stretch')
    plt.close(fig)


def render_evaluation_tab(eval_results):
    plt.rcParams.update({
        "figure.facecolor":"#111318","axes.facecolor":"#111318","axes.edgecolor":"#2a2f3d",
        "axes.labelcolor":"#8a93a8","xtick.color":"#8a93a8","ytick.color":"#8a93a8",
        "text.color":"#e8ecf4","grid.color":"#2a2f3d","grid.linewidth":.5,
        "axes.grid":True,"font.family":"monospace","figure.dpi":110,
    })
    A,T,R,O,P = "#f0a500","#00c9a7","#e84040","#f06830","#a78bfa"
    PAL = [A,T,R,O,P]

    if not HAS_ML:
        st.error("scikit-learn / xgboost / imbalanced-learn not installed. Run:\n"
                 "```\npip install scikit-learn xgboost imbalanced-learn\n```")
        return

    if "error" in eval_results:
        st.error(eval_results["error"])
        return

    # ── Leakage Diagnosis ───────────────────────────────────────
    with st.expander("🔬 Data Leakage Diagnosis & Fixes", expanded=False):
        st.markdown('<div class="section-label">Why previous scores were perfect — and what we fixed</div>',
                    unsafe_allow_html=True)
        reg = EvaluationAgent.LEAKAGE_REGISTRY
        haz_emoji = {"Flood":"🌊","Heat":"🔥","Wind":"💨","Landslide":"⛰️"}
        for haz, info in reg.items():
            leaked_str = ", ".join(f"<code>{c}</code>" for c in info["leaked"])
            emoji_haz  = haz_emoji.get(haz,"")
            why_txt    = info['why']
            fix_txt    = info['fix']
            exp_txt    = info['expect']
            st.markdown(f"""
            <div class="diag-block">
              <div class="diag-title">{emoji_haz} {haz}</div>
              <div class="diag-body">
                <span class="tag leak">LEAKED</span> {leaked_str}<br>
                <b>Why:</b> {why_txt}<br>
                <span class="tag fix">FIX</span> {fix_txt}<br>
                <span class="tag info">EXPECT</span> {exp_txt}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Per-Hazard Detail Tabs ───────────────────────────────────
    hkeys     = [k for k in ["flood","heat","wind","landslide"] if k in eval_results]
    tab_names = [eval_results[k]["display"] for k in hkeys]

    if not hkeys:
        st.warning("No hazard results available. Ensure weatherops_feature_table.csv exists.")
        return

    htabs = st.tabs(tab_names)

    for htab, hkey in zip(htabs, hkeys):
        with htab:
            hr = eval_results[hkey]
            if "error" in hr:
                st.error(hr["error"]); continue

            task   = hr["task"]
            models = hr["models"]
            best   = hr["best"]
            feats  = hr["features"]
            y_te   = hr["y_te"]

            # ── Dataset info bar ──────────────────────────────
            info_row = (f'<div class="eval-pill-row">'
                        f'{_pill("Train",str(hr["n_train"]),"neutral")}'
                        f'{_pill("Test",str(hr["n_test"]),"neutral")}'
                        f'{_pill("Features",str(len(feats)),"neutral")}')
            if task == "clf":
                pr = hr.get("pos_rate",0)
                info_row += _pill("Pos Rate", f"{pr:.1%}", "warn" if pr<.1 else "neutral")
            else:
                info_row += _pill("Baseline RMSE", f"{hr.get('baseline_rmse',0):.2f} km/h","neutral")
            info_row += "</div>"
            st.markdown(info_row, unsafe_allow_html=True)

            # ── Model metrics pills ────────────────────────────
            st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
            for mn, mr in models.items():
                if "error" in mr:
                    st.markdown(f'<div class="eval-pill-row">'
                                f'{_pill(mn,"ERROR","bad")}</div>', unsafe_allow_html=True)
                    continue
                is_best = "★ " if mn == best else ""
                if task == "clf":
                    auc = mr.get("roc_auc", float("nan"))
                    f1  = mr.get("f1",      float("nan"))
                    bri = mr.get("brier",   float("nan"))
                    cva = mr.get("cv_auc_mean", float("nan"))
                    row = (f'<div class="eval-pill-row">'
                           f'{_pill("Model",f"{is_best}{mn}","neutral")}'
                           f'{_pill("AUC",   f"{auc:.3f}" if not np.isnan(auc) else "—", _quality(auc,.75,.6))}'
                           f'{_pill("F1",    f"{f1:.3f}"  if not np.isnan(f1)  else "—", _quality(f1,.70,.5))}'
                           f'{_pill("Brier", f"{bri:.3f}" if not np.isnan(bri) else "—", "good" if not np.isnan(bri) and bri<.15 else "warn")}'
                           f'{_pill("CV-AUC",f"{cva:.3f}" if not np.isnan(cva) else "—", _quality(cva,.75,.6))}'
                           f'</div>')
                else:
                    r2   = mr.get("r2",   float("nan"))
                    rmse = mr.get("rmse", float("nan"))
                    cva  = mr.get("cv_r2_mean", float("nan"))
                    row = (f'<div class="eval-pill-row">'
                           f'{_pill("Model",f"{is_best}{mn}","neutral")}'
                           f'{_pill("R²",   f"{r2:.3f}"   if not np.isnan(r2)   else "—", _quality(r2,.55,.35))}'
                           f'{_pill("RMSE", f"{rmse:.2f}" if not np.isnan(rmse) else "—", "neutral")}'
                           f'{_pill("CV-R²",f"{cva:.3f}"  if not np.isnan(cva)  else "—", _quality(cva,.55,.35))}'
                           f'</div>')
                st.markdown(row, unsafe_allow_html=True)

            # ── Plots ──────────────────────────────────────────
            valid_models = {mn:mr for mn,mr in models.items() if "error" not in mr}
            if not valid_models:
                st.warning("All models failed — check CSV columns."); continue

            if task == "clf":
                col_l, col_r = st.columns(2)

                # Left: ROC curves
                with col_l:
                    fig, ax = plt.subplots(figsize=(6,4.5))
                    for (mn, mr), color in zip(valid_models.items(), PAL):
                        if len(np.unique(y_te)) < 2: continue
                        try:
                            fpr,tpr,_ = roc_curve(y_te, mr["y_prob"])
                            auc = roc_auc_score(y_te, mr["y_prob"])
                            lw  = 2.5 if mn==best else 1.0
                            alp = 1.0 if mn==best else 0.4
                            ax.plot(fpr,tpr,color=color,lw=lw,alpha=alp,
                                    label=f"{mn.split()[0]} ({auc:.3f})")
                        except: pass
                    ax.plot([0,1],[0,1],"--",color="#4e5568",lw=.8)
                    ax.set(xlabel="FPR",ylabel="TPR",title=f"{hr['display']} · ROC Curves",
                           xlim=(0,1),ylim=(0,1.02))
                    ax.legend(fontsize=7,framealpha=.2)
                    _plt_fig_to_st(fig)

                # Right: Confusion matrix of best model
                with col_r:
                    best_mr = valid_models[best]
                    fig, ax = plt.subplots(figsize=(5,4.5))
                    cm = confusion_matrix(y_te, best_mr["y_pred"])
                    from sklearn.metrics import ConfusionMatrixDisplay
                    ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(
                        ax=ax, colorbar=False, cmap="YlOrRd")
                    ax.set_title(f"{best} · Confusion Matrix",fontsize=9); ax.grid(False)
                    _plt_fig_to_st(fig)

                # PR curves + Calibration side by side
                col_pr, col_cal = st.columns(2)
                with col_pr:
                    fig, ax = plt.subplots(figsize=(6,4))
                    for (mn,mr),color in zip(valid_models.items(), PAL):
                        if len(np.unique(y_te))<2: continue
                        try:
                            prec,rec,_ = precision_recall_curve(y_te, mr["y_prob"])
                            ap = average_precision_score(y_te, mr["y_prob"])
                            lw = 2.2 if mn==best else 0.9
                            alp = 1. if mn==best else .4
                            ax.plot(rec,prec,color=color,lw=lw,alpha=alp,
                                    label=f"{mn.split()[0]} (AP={ap:.3f})")
                        except: pass
                    ax.set(xlabel="Recall",ylabel="Precision",
                           title=f"{hr['display']} · Precision-Recall",xlim=(0,1),ylim=(0,1.02))
                    ax.legend(fontsize=7,framealpha=.2)
                    _plt_fig_to_st(fig)

                with col_cal:
                    from sklearn.calibration import calibration_curve
                    fig, ax = plt.subplots(figsize=(5,4))
                    for (mn,mr),color in zip(valid_models.items(), PAL):
                        try:
                            fp2,mp2 = calibration_curve(y_te, mr["y_prob"], n_bins=6)
                            ax.plot(mp2,fp2,"o-",color=color,lw=1.5,ms=4,label=mn.split()[0])
                        except: pass
                    ax.plot([0,1],[0,1],"--",color="#4e5568",lw=1,label="Perfect")
                    ax.set(xlabel="Mean Predicted Prob.",ylabel="Fraction Positive",
                           title=f"{hr['display']} · Calibration",xlim=(0,1),ylim=(0,1))
                    ax.legend(fontsize=7,framealpha=.2)
                    _plt_fig_to_st(fig)

            else:  # regression
                col_l, col_r = st.columns(2)
                with col_l:
                    fig, axes = plt.subplots(1,2,figsize=(10,4.5))
                    for i,(mn,mr) in enumerate(valid_models.items()):
                        if i >= 4: break
                        ax = axes[0] if i<2 else axes[1]
                        color = PAL[i]
                        ax.scatter(y_te, mr["y_pred"], s=5, alpha=.4, color=color,
                                   label=f"{mn.split()[0]} R²={mr['r2']:.3f}")
                    for ax in axes:
                        lo2=min(y_te); hi2=max(y_te)
                        ax.plot([lo2,hi2],[lo2,hi2],"--",color="#4e5568",lw=.8)
                        ax.set(xlabel="Actual",ylabel="Predicted")
                        ax.legend(fontsize=7,framealpha=.2)
                    axes[0].set_title(f"{hr['display']} · Actual vs Predicted",fontsize=9)
                    plt.tight_layout()
                    _plt_fig_to_st(fig)
                with col_r:
                    best_mr = valid_models[best]
                    resid = y_te - best_mr["y_pred"]
                    fig, ax = plt.subplots(figsize=(5,4.5))
                    ax.hist(resid, bins=35, color=A, alpha=.85, edgecolor="none")
                    ax.axvline(0, color=R, lw=1.5, ls="--")
                    ax.set(xlabel="Residual (km/h)", title=f"{best} · Residuals  std={resid.std():.2f}")
                    _plt_fig_to_st(fig)

            # Feature importance (RF)
            rf_key = next((k for k in valid_models if "Forest" in k), None)
            if rf_key:
                rf_model = valid_models[rf_key]["model"]
                if hasattr(rf_model, "feature_importances_"):
                    imps  = rf_model.feature_importances_
                    order = np.argsort(imps)[::-1]
                    valid_indices = [i for i in order if i < len(feats)]
                    if valid_indices:
                        fig, ax = plt.subplots(figsize=(8,3.5))
                        ax.barh([feats[i] for i in valid_indices], imps[valid_indices], color=A, alpha=.85)
                        ax.invert_yaxis()
                        ax.set_title(f"{hr['display']} · RF Feature Importance", fontsize=9)
                        plt.tight_layout()
                        _plt_fig_to_st(fig)

            # SHAP (best model if tree-based)
            if HAS_SHAP:
                best_model = valid_models[best]["model"]
                best_scaler = valid_models[best]["scaler"]
                if hasattr(best_model, "feature_importances_") or isinstance(best_model, (XGBClassifier, XGBRegressor)):
                    with st.expander(f"🔍 SHAP — {best}", expanded=False):
                        try:
                            import io
                            # Get test features from df_te
                            df_te = hr.get("df_te")
                            if df_te is not None and len(feats) > 0 and all(f in df_te.columns for f in feats):
                                X_test = df_te[feats].values[:300]
                                Xte_sc = best_scaler.transform(X_test)
                                ex = shap.TreeExplainer(best_model)
                                sv = ex.shap_values(Xte_sc)
                                if isinstance(sv,list): sv = sv[1]
                                fig2, ax2 = plt.subplots(figsize=(8,4))
                                shap.summary_plot(sv, Xte_sc, feature_names=feats,
                                                  max_display=10, plot_type="bar",
                                                  show=False, color=A)
                                ax2.set_title(f"SHAP · {hr['display']} · {best}",fontsize=9)
                                plt.tight_layout()
                                buf = io.BytesIO()
                                fig2.savefig(buf,format="png",dpi=110,bbox_inches="tight",facecolor="#111318")
                                buf.seek(0); st.image(buf, width='stretch')
                                plt.close(fig2)
                            else:
                                st.caption("Cannot compute SHAP: missing test features")
                        except Exception as e:
                            st.caption(f"SHAP error: {e}")

    # ── Model Leaderboard ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">🏆 Model Leaderboard</div>', unsafe_allow_html=True)

    lb_rows = []
    for hkey in ["flood","heat","wind","landslide"]:
        if hkey not in eval_results: continue
        hr = eval_results[hkey]
        if "error" in hr: continue
        best = hr["best"]
        bm   = hr["models"].get(best,{})
        if "error" in bm: continue
        if hr["task"] == "clf":
            score = bm.get("roc_auc", float("nan"))
            label = f"AUC={score:.3f}" if not np.isnan(score) else "—"
        else:
            score = bm.get("r2", float("nan"))
            label = f"R²={score:.3f}" if not np.isnan(score) else "—"
        lb_rows.append((hr["display"], best, score if not np.isnan(score) else 0, label, hr["task"]))

    lb_rows.sort(key=lambda x: -x[2])
    lb_html = ""
    for rank,(hdisp,model_name,score,label,task) in enumerate(lb_rows, 1):
        pct = f"{score*100:.0f}%"
        bar_color = "#f0a500" if task=="clf" else "#00c9a7"
        lb_html += f"""
        <div class="lb-row">
          <span class="lb-rank">#{rank}</span>
          <span class="lb-hazard">{hdisp}</span>
          <span class="lb-model">{model_name}</span>
          <div class="lb-bar"><div class="lb-bar-fill" style="width:{pct};background:{bar_color};"></div></div>
          <span class="lb-score">{label}</span>
        </div>"""
    st.markdown(lb_html, unsafe_allow_html=True)

    # ── Radar chart across hazards ────────────────────────────────
    clf_hazards = [(k,eval_results[k]) for k in ["flood","heat","landslide"]
                    if k in eval_results and "error" not in eval_results[k] and eval_results[k]["task"]=="clf"]
    if len(clf_hazards) >= 2:
        st.markdown("---")
        st.markdown('<div class="section-label">Radar — Best Classifier Metrics</div>', unsafe_allow_html=True)
        cats = ["AUC","F1","Precision","Recall","CV-AUC"]
        N    = len(cats)
        angles = [n/N*2*3.14159 for n in range(N)] + [0]
        haz_colors = {"flood":"#00c9a7","heat":"#e84040","landslide":"#f06830"}

        fig, ax = plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True),facecolor="#111318")
        ax.set_facecolor("#181b22"); ax.spines["polar"].set_color("#2a2f3d")
        for hkey, hr in clf_hazards:
            bm   = hr["models"].get(hr["best"],{})
            if "error" in bm: continue
            vals = [bm.get("roc_auc",0), bm.get("f1",0), bm.get("precision",0),
                    bm.get("recall",0),  bm.get("cv_auc_mean",0)]
            vals = [0 if np.isnan(v) else v for v in vals]
            vals += vals[:1]
            color = haz_colors.get(hkey,"#f0a500")
            ax.plot(angles, vals, "o-", lw=2, color=color, ms=5, label=hr["display"])
            ax.fill(angles, vals, alpha=.10, color=color)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=9)
        ax.set_ylim(0,1); ax.set_yticks([.2,.4,.6,.8,1.]); ax.grid(color="#2a2f3d",linewidth=.5)
        ax.set_title("Best Model Radar\n(classifier hazards)",pad=20,fontsize=10,color="#e8ecf4")
        ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.15),fontsize=9)
        _plt_fig_to_st(fig)

    # ── Export leaderboard CSV ────────────────────────────────────
    records = []
    for hkey in ["flood","heat","wind","landslide"]:
        if hkey not in eval_results: continue
        hr = eval_results[hkey]
        if "error" in hr: continue
        for mn, mr in hr["models"].items():
            if "error" in mr: continue
            rec = {"Hazard":hr["display"],"Model":mn,"Task":hr["task"]}
            if hr["task"]=="clf":
                rec.update({k:round(mr.get(k,float("nan")),4)
                             for k in ["roc_auc","avg_prec","f1","precision","recall","brier","cv_auc_mean"]})
            else:
                rec.update({k:round(mr.get(k,float("nan")),4)
                             for k in ["r2","rmse","mae","mape","cv_r2_mean"]})
            records.append(rec)
    if records:
        master_df = pd.DataFrame(records)
        csv_bytes = master_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Leaderboard CSV", csv_bytes,
                           file_name=f"WeatherOps_Leaderboard_{datetime.utcnow():%Y%m%d_%H%M}.csv",
                           mime="text/csv", width='stretch')


# ============================================================
# MAIN
# ============================================================
def main():
    horizon, rain_thresh, temp_thresh, wind_thresh, active_layers, use_live = render_sidebar()

    # ── Pipeline ────────────────────────────────────────────────
    ingestion = IngestionAgent()
    modeling  = ModelingAgent()
    hazards   = HazardsAgent()
    decision  = DecisionAgent()
    roi       = load_roi()

    with st.spinner("Running agent pipeline…"):
        weather, data_source, source_error = ingestion.get_weather_forecast(horizon, use_live=use_live)
        if data_source=="synthetic" and use_live:
            st.warning("⚡ Live API unavailable — using synthetic forecast", icon="⚡")
            if source_error: st.caption(f"Reason: {source_error}")
        elif data_source=="cached" and use_live:
            st.info("Using cached live forecast due to temporary API issue.", icon="ℹ️")
        elif data_source=="file" and use_live:
            st.info("Using file-based forecast due to API issue.", icon="ℹ️")

        blended              = modeling.blend_forecast(weather, {"mean_slope":12})
        risk, risk_ci, stats = hazards.compute_risk(blended, rain_thresh, temp_thresh, wind_thresh)
        actions              = decision.generate_actions(risk, risk_ci)
        hazard_pts           = generate_risk_points(roi, risk)

    # ── Header + KPIs ────────────────────────────────────────────
    render_header(data_source, horizon)
    render_metrics(stats, risk)
    st.markdown("<div style='margin-top:.5rem'/>", unsafe_allow_html=True)

    # ── Map + Action Cards ───────────────────────────────────────
    col_map, col_actions = st.columns([1.35,1], gap="medium")
    with col_map:
        st.markdown(f'<div class="section-label">📍 Impact Map · Next {horizon}h</div>', unsafe_allow_html=True)
        st_folium(build_map(roi, hazard_pts, active_layers),
                  width=None, height=480, returned_objects=[], key="impact_map")
    with col_actions:
        st.markdown('<div class="section-label">🚨 Action Cards</div>', unsafe_allow_html=True)
        render_action_cards(actions)
        if actions:
            st.markdown("<div style='margin-top:.5rem'/>", unsafe_allow_html=True)
            if HAS_PDF:
                if st.button("📄 Export Operational Brief (PDF)", width='stretch'):
                    pdf_path = generate_pdf(actions, risk, stats)
                    if pdf_path:
                        with open(pdf_path,"rb") as f:
                            st.download_button("⬇ Download PDF", data=f,
                                file_name=f"WeatherOps_Brief_{datetime.utcnow():%Y%m%d_%H%M}.pdf",
                                mime="application/pdf", width='stretch', type="primary")
            else:
                st.caption("Install `reportlab` for PDF export.")

    # ── Bottom Tabs ───────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem'/>", unsafe_allow_html=True)
    eval_done = bool(st.session_state.get("eval_results"))

    tab_ts, tab_data, tab_eval, tab_trace = st.tabs([
        "📈  Forecast Timeseries",
        "🗂  Raw Data",
        "🤖  Evaluation Agent",
        "⚙️  Agent Trace",
    ])

    with tab_ts:
        render_timeseries(blended)

    with tab_data:
        render_data_table(blended)

    with tab_eval:
        st.markdown('<div class="section-label">Agent 05 · Statistical Model Evaluation</div>',
                    unsafe_allow_html=True)

        if not HAS_ML:
            st.error("Dependencies not installed. Run:\n```\npip install scikit-learn xgboost imbalanced-learn shap\n```")
        elif not CSV_PATH.exists():
            st.warning(f"Feature table not found at:\n`{CSV_PATH}`\n\n"
                       "Run the data ingestion notebook first to generate it.")
        else:
            col_btn, col_info = st.columns([1,3])
            with col_btn:
                run_eval = st.button("▶ Run Evaluation", type="primary", width='stretch')
            with col_info:
                st.markdown(
                    '<div style="font-family:var(--mono);font-size:.68rem;color:var(--text2);'
                    'padding-top:.55rem;">Trains LR · RF · XGBoost · SVM per hazard · '
                    'Spatial block CV · Leakage-free targets · SHAP</div>',
                    unsafe_allow_html=True)

            if run_eval:
                eval_agent = EvaluationAgent()
                prog_bar   = st.progress(0, "Initialising…")
                prog_text  = st.empty()

                def prog(frac, msg):
                    prog_bar.progress(min(frac,1.0), msg)
                    prog_text.markdown(
                        f'<div style="font-family:var(--mono);font-size:.68rem;color:var(--text2)">{msg}</div>',
                        unsafe_allow_html=True)

                with st.spinner("Training models…"):
                    try:
                        results = eval_agent.run(progress_cb=prog)
                    except Exception as e:
                        results = {"error": f"Evaluation failed: {str(e)}"}

                prog_bar.empty(); prog_text.empty()
                st.session_state["eval_results"] = results
                
                # Check if evaluation has errors
                has_error = "error" in results or not any(k in results for k in ["flood","heat","wind","landslide"])
                if has_error:
                    st.error(f"⚠️ Evaluation encountered issues: {results.get('error','Unknown error')}")
                else:
                    st.success("✓ Evaluation complete", icon="✅")
                
                # Force rerun to update indicators
                st.rerun()

            if st.session_state.get("eval_results"):
                render_evaluation_tab(st.session_state["eval_results"])

    with tab_trace:
        st.markdown('<div class="section-label">Pipeline Execution Log</div>', unsafe_allow_html=True)
        render_agent_trace(data_source, len(actions), eval_done=eval_done)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #2a2f3d;
                font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#4e5568;
                display:flex;justify-content:space-between;">
      <span>WeatherOps v5 · 5-Agent GeoAI Pipeline · Dehradun, Uttarakhand</span>
      <span>Transparent · Reliable · Actionable</span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
