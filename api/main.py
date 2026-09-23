"""API FastAPI : prévisions et recommandations de stock.

Lancer :  uvicorn api.main:app --reload
Docs   :  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from demandforecast import serve

ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "models/artifacts.joblib")
app = FastAPI(title="Demand Forecasting API", version="0.1.0")


@lru_cache(maxsize=1)
def artifacts() -> dict:
    try:
        return serve.load_artifacts(ARTIFACT_PATH)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Modèle introuvable ({ARTIFACT_PATH}) : lancez d'abord le pipeline.") from e


class ForecastRequest(BaseModel):
    store_nbr: int
    family: str
    horizon: int | None = Field(None, ge=1, description="Nb de jours (<= horizon d'entraînement)")
    onpromotion: list[float] | None = Field(None, description="Plan promo futur, 1 valeur par jour")


class ReorderRequest(BaseModel):
    store_nbr: int
    family: str
    lead_time: int = Field(3, ge=1, description="Délai de réapprovisionnement (jours)")
    service_level: float = Field(0.95, gt=0.5, lt=1)
    onpromotion: list[float] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/series")
def series():
    art = artifacts()
    return {"horizon": art["horizon"], "source": art["source"],
            "series": serve.list_series(art).to_dict(orient="records")}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        fc = serve.forecast_series(artifacts(), req.store_nbr, req.family, req.horizon, req.onpromotion)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    fc["date"] = fc["date"].dt.strftime("%Y-%m-%d")
    return {"store_nbr": req.store_nbr, "family": req.family,
            "forecast": fc.round(1).to_dict(orient="records")}


@app.post("/reorder")
def reorder(req: ReorderRequest):
    try:
        return serve.recommend_stock(artifacts(), req.store_nbr, req.family, req.lead_time,
                                     req.service_level, req.onpromotion)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
