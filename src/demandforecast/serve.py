"""Prévision et recommandation de stock à partir de l'artefact entraîné (utilisé par l'API et Streamlit)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import inventory
from .data import KEY
from .features import build_features


def load_artifacts(path: str | Path = "models/artifacts.joblib") -> dict:
    return joblib.load(path)


def list_series(art: dict) -> pd.DataFrame:
    return art["history"][KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)


def forecast_series(art: dict, store_nbr: int, family: str, horizon: int | None = None,
                    onpromotion: list[float] | None = None) -> pd.DataFrame:
    """Prévision des `horizon` prochains jours (<= horizon d'entraînement) avec intervalle à 90 %.

    `onpromotion` : plan promo futur (une valeur par jour) ; 0 par défaut.
    """
    H = art["horizon"]
    horizon = horizon or H
    if not 1 <= horizon <= H:
        raise ValueError(f"horizon doit être compris entre 1 et {H}")
    hist = art["history"]
    hist = hist[(hist["store_nbr"] == store_nbr) & (hist["family"] == family)]
    if hist.empty:
        raise KeyError(f"série inconnue : magasin {store_nbr}, famille {family}")
    last = hist["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=H)
    promo = np.zeros(H)
    if onpromotion is not None:
        promo[: len(onpromotion)] = onpromotion[:H]
    future = pd.DataFrame({"date": dates, "store_nbr": store_nbr, "family": family,
                           "sales": np.nan, "onpromotion": promo})
    frame = pd.concat([hist[KEY + ["date", "sales", "onpromotion"]], future], ignore_index=True)
    feats = build_features(frame, H, art["holidays"], art["store_map"], art["family_map"])
    fut = feats[feats["date"] > last]
    pred = art["model"].predict(fut)
    out = pd.concat([fut[["date"]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    return out.head(horizon)


def recommend_stock(art: dict, store_nbr: int, family: str, lead_time: int = 3,
                    service_level: float = 0.95, onpromotion: list[float] | None = None) -> dict:
    """Demande attendue pendant le délai, stock de sécurité et point de commande."""
    fc = forecast_series(art, store_nbr, family, horizon=None, onpromotion=onpromotion)
    if not 1 <= lead_time <= len(fc):
        raise ValueError(f"lead_time doit être compris entre 1 et {len(fc)}")
    if not 0.5 < service_level < 1:
        raise ValueError("service_level doit être compris entre 0.5 et 1 (exclu)")
    err = art["backtest_errors"]
    err = err[(err["store_nbr"] == store_nbr) & (err["family"] == family)].sort_values("date")["error"]
    sigma = inventory.sigma_lead_time(err, lead_time)
    rec = inventory.reorder_point(fc["yhat"].values, lead_time, sigma, service_level)
    return {**{k: round(float(v), 1) for k, v in rec.items()},
            "sigma_lead_time": round(sigma, 1), "lead_time": lead_time, "service_level": service_level}
