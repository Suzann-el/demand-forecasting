"""Application Streamlit : prévision, intervalle et recommandation de stock par magasin x famille.

Lancer :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandforecast import serve  # noqa: E402

st.set_page_config(page_title="Prévision de la demande", layout="wide")
ART = Path(__file__).resolve().parents[1] / "models" / "artifacts.joblib"


@st.cache_resource
def load():
    return serve.load_artifacts(ART)


if not ART.exists():
    st.error("Modèle introuvable : lancez d'abord `python -m demandforecast.pipeline`.")
    st.stop()

art = load()
st.title("Prévision de la demande & stock de sécurité")
if art["source"] == "synthetic":
    st.warning("Modèle entraîné sur des données synthétiques (démonstration).")

series = serve.list_series(art)
c1, c2, c3, c4 = st.columns(4)
store = c1.selectbox("Magasin", sorted(series["store_nbr"].unique()))
family = c2.selectbox("Famille", sorted(series[series["store_nbr"] == store]["family"].unique()))
lead_time = c3.slider("Délai de réappro. (jours)", 1, art["horizon"], 3)
service = c4.slider("Niveau de service", 0.80, 0.99, 0.95, 0.01)

fc = serve.forecast_series(art, store, family)
rec = serve.recommend_stock(art, store, family, lead_time, service)
hist = art["history"][(art["history"]["store_nbr"] == store) & (art["history"]["family"] == family)].tail(90)

m1, m2, m3 = st.columns(3)
m1.metric(f"Demande prévue sur {lead_time} j", f"{rec['expected_demand_lt']:,.0f}")
m2.metric("Stock de sécurité", f"{rec['safety_stock']:,.0f}")
m3.metric("Point de commande", f"{rec['reorder_point']:,.0f}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(hist["date"], hist["sales"], color="#444", lw=1, label="Ventes observées")
ax.plot(fc["date"], fc["yhat"], color="#d1495b", lw=2, label="Prévision")
ax.fill_between(fc["date"], fc["yhat_lo"], fc["yhat_hi"], color="#d1495b", alpha=0.18, label="Intervalle 90 %")
ax.grid(alpha=0.3); ax.legend(loc="upper left")
st.pyplot(fig)

with st.expander("Détail de la prévision"):
    st.dataframe(fc.assign(date=fc["date"].dt.date).round(1), hide_index=True)
st.caption("Stock de sécurité = z(niveau de service) × écart-type de l'erreur cumulée sur le délai (mesuré au backtest).")
