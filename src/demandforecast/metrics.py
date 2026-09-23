"""Métriques de prévision : WAPE (métrique principale), MAE, RMSE, biais, MASE, pinball, couverture."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY


def wape(y, yhat) -> float:
    """Somme des erreurs absolues / somme des ventes. Robuste aux zéros, lisible par le métier."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, yhat) -> float:
    """Biais relatif : >0 = sur-prévision (risque de surstock), <0 = sous-prévision (risque de rupture)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / max(np.abs(y).sum(), 1e-9))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def seasonal_scale(y_train, m: int = 7) -> float:
    """Échelle de MASE : MAE du naïf saisonnier en échantillon d'apprentissage."""
    y_train = np.asarray(y_train, float)
    return float(max(np.mean(np.abs(y_train[m:] - y_train[:-m])), 1e-9))


def pinball(y, q_pred, alpha: float) -> float:
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    diff = y - q_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def coverage(y, lo, hi) -> float:
    """Part des observations tombant dans l'intervalle [lo, hi]."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def score_table(preds: pd.DataFrame, scales: pd.Series, by: list[str] | None = None) -> pd.DataFrame:
    """Métriques par modèle (et éventuellement par `by`). MASE = moyenne des MAE/échelle par série."""
    by = by or []
    rows = []
    for keys, g in preds.groupby(["model"] + by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        per_series = g.groupby(KEY).apply(
            lambda s: mae(s["y"], s["yhat"]) / scales.loc[(s.name[0], s.name[1])], include_groups=False
        )
        rows.append({
            **dict(zip(["model"] + by, keys)),
            "WAPE": wape(g["y"], g["yhat"]),
            "MAE": mae(g["y"], g["yhat"]),
            "RMSE": rmse(g["y"], g["yhat"]),
            "Biais": bias(g["y"], g["yhat"]),
            "MASE": float(per_series.mean()),
        })
    return pd.DataFrame(rows)
