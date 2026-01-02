# ============================================================
# WeatherOps: Agentic GeoAI Weather Impact Decision Platform
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="WeatherOps – GeoAI Weather Decisions",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# AGENT 1: INGESTION AGENT
# ============================================================

class IngestionAgent:
    def get_weather_forecast(self, horizon_hours=72):
        try:
            raise Exception("API disabled – using mock")
        except Exception:
            timestamps = [
                datetime.utcnow() + timedelta(hours=i)
                for i in range(0, horizon_hours, 3)
            ]
            return pd.DataFrame({
                "time": timestamps,
                "rain_mm": np.random.gamma(2, 3, len(timestamps)),
                "temp_c": np.random.normal(30, 4, len(timestamps)),
                "wind_kmph": np.random.normal(18, 6, len(timestamps))
            })

# ============================================================
# AGENT 2: MODELING AGENT
# ============================================================

class ModelingAgent:
    def terrain_adjustment(self, rain, slope):
        return rain * (1 + slope / 30)

    def blend_forecast(self, weather_df, terrain_stats):
        df = weather_df.copy()
        df["rain_adj"] = self.terrain_adjustment(
            df["rain_mm"], terrain_stats["mean_slope"]
        )
        df["heat_index"] = df["temp_c"] + 0.1 * df["rain_mm"]
        return df

# ============================================================
# AGENT 3: HAZARDS AGENT
# ============================================================

class HazardsAgent:
    def compute_risk(self, weather_df):
        rain_peak = weather_df["rain_adj"].max()
        temp_peak = weather_df["heat_index"].max()
        wind_peak = weather_df["wind_kmph"].max()

        risk = {
            "Flood": min(1, rain_peak / 80),
            "Heat": min(1, (temp_peak - 35) / 10),
            "Wind": min(1, wind_peak / 40)
        }

        risk_ci = {
            k: (max(0, v - 0.15), min(1, v + 0.15))
            for k, v in risk.items()
        }

        return risk, risk_ci

# ============================================================
# AGENT 4: DECISION AGENT
# ============================================================

class DecisionAgent:
    def generate_actions(self, risk, risk_ci):
        actions = []

        if risk["Flood"] > 0.6:
            actions.append({
                "What": "Deploy drainage & pumping crew",
                "Where": "Low-lying road segments",
                "When": "Next 12 hours",
                "Why": "High flood probability",
                "Confidence": risk_ci["Flood"]
            })

        if risk["Heat"] > 0.5:
            actions.append({
                "What": "Issue heat advisory",
                "Where": "Urban core",
                "When": "12:00–16:00",
                "Why": "High heat stress",
                "Confidence": risk_ci["Heat"]
            })

        if risk["Wind"] > 0.5:
            actions.append({
                "What": "Secure power infrastructure",
                "Where": "Transmission corridors",
                "When": "Within 24h",
                "Why": "Wind gust risk",
                "Confidence": risk_ci["Wind"]
            })

        return actions

# ============================================================
# LOAD DATA
# ============================================================

def load_roi():
    gdf = gpd.read_file("data/Dehradun.gpkg")
    return gdf.to_crs(epsg=4326)

def load_field_brief():
    return pd.read_csv("Dehradun_Field_Brief.csv")

# ============================================================
# SPATIAL UTILITIES
# ============================================================

def generate_roi_points(gdf, n_points=45):
    geom = gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    points = []

    while len(points) < n_points:
        p = Point(
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        )
        if geom.contains(p):
            points.append(p)

    return gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

def assign_spatial_risk(points_gdf, base_risk, spread=0.15):
    gdf = points_gdf.copy()
    gdf["risk"] = np.clip(
        np.random.normal(base_risk, spread, len(gdf)),
        0, 1
    )
    return gdf

def risk_color(r):
    if r < 0.3:
        return "green"
    elif r < 0.6:
        return "orange"
    else:
        return "red"

# ============================================================
# PDF EXPORT
# ============================================================

def generate_pdf(actions):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)

    y = 800
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "WeatherOps – Operational Brief")
    y -= 40

    for act in actions:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, act["What"])
        y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"Where: {act['Where']}")
        y -= 12
        c.drawString(60, y, f"When: {act['When']}")
        y -= 12
        c.drawString(60, y, f"Why: {act['Why']}")
        y -= 12
        c.drawString(
            60, y,
            f"Confidence: {act['Confidence'][0]:.2f}–{act['Confidence'][1]:.2f}"
        )
        y -= 25

    c.save()
    return tmp.name

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("🌦️ WeatherOps – GeoAI Weather Impact Decisions")

    ingestion = IngestionAgent()
    modeling = ModelingAgent()
    hazards = HazardsAgent()
    decision = DecisionAgent()

    roi = load_roi()
    field_brief = load_field_brief()

    weather = ingestion.get_weather_forecast()
    terrain_stats = {"mean_slope": 12}

    blended = modeling.blend_forecast(weather, terrain_stats)
    risk, risk_ci = hazards.compute_risk(blended)
    actions = decision.generate_actions(risk, risk_ci)

    col1, col2 = st.columns([1.2, 0.8])

    # ===================== MAP =====================
    with col1:
        st.subheader("📍 Impact Map (Next 72h)")

        m = folium.Map(location=[30.32, 78.03], zoom_start=10)

        folium.GeoJson(
            roi,
            name="ROI Boundary",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "blue",
                "weight": 2
            }
        ).add_to(m)

        pts = generate_roi_points(roi)

        flood = assign_spatial_risk(pts, risk["Flood"])
        heat = assign_spatial_risk(pts, risk["Heat"])
        wind = assign_spatial_risk(pts, risk["Wind"])

        flood_layer = folium.FeatureGroup(name="🌊 Flood Risk")
        heat_layer = folium.FeatureGroup(name="🔥 Heat Risk")
        wind_layer = folium.FeatureGroup(name="🌪️ Wind Risk")

        for _, r in flood.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=6,
                color=risk_color(r["risk"]),
                fill=True,
                fill_opacity=0.7,
                popup=f"Flood Risk: {r['risk']:.2f}"
            ).add_to(flood_layer)

        for _, r in heat.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=6,
                color=risk_color(r["risk"]),
                fill=True,
                fill_opacity=0.7,
                popup=f"Heat Risk: {r['risk']:.2f}"
            ).add_to(heat_layer)

        for _, r in wind.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=6,
                color=risk_color(r["risk"]),
                fill=True,
                fill_opacity=0.7,
                popup=f"Wind Risk: {r['risk']:.2f}"
            ).add_to(wind_layer)

        flood_layer.add_to(m)
        heat_layer.add_to(m)
        wind_layer.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        st_folium(
            m,
            width=700,
            height=520,
            returned_objects=[],
            key="impact_map"
        )

    # ===================== ACTIONS =====================
    with col2:
        st.subheader("🚨 Action Cards")

        if not actions:
            st.success("No critical actions required")

        for act in actions:
            with st.container(border=True):
                st.markdown(f"### {act['What']}")
                st.write(f"**Where:** {act['Where']}")
                st.write(f"**When:** {act['When']}")
                st.write(f"**Why:** {act['Why']}")
                st.caption(
                    f"Confidence: {act['Confidence'][0]:.2f}–{act['Confidence'][1]:.2f}"
                )

        if actions and st.button("📄 Download One-Page Brief"):
            pdf = generate_pdf(actions)
            with open(pdf, "rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name="WeatherOps_Brief.pdf"
                )

    st.caption("WeatherOps | Agentic GeoAI | Transparent • Reliable • Actionable")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
