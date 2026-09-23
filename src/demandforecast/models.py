"""Modèles : baselines, statistiques (ETS, SARIMA, Prophet) et LightGBM direct avec intervalles.

Les modèles « par série » ont tous la même signature :
    predict(y: np.ndarray, dates: DatetimeIndex, horizon: int, **ctx) -> np.ndarray
et retournent des prévisions >= 0. Toute panne d'un modèle retombe sur le naïf saisonnier.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import CAT_FEATURES, feature_columns


# ------------------------------------------------------------------ baselines
def seasonal_naive(y: np.ndarray, horizon: int, m: int = 7, **_) -> np.ndarray:
    """Répète la dernière semaine observée : la baseline à battre."""
    return np.resize(np.asarray(y, float)[-m:], horizon)


def moving_average(y: np.ndarray, horizon: int, window: int = 28, **_) -> np.ndarray:
    """Moyenne des 28 derniers jours, prévision plate."""
    return np.full(horizon, float(np.mean(np.asarray(y, float)[-window:])))


# ------------------------------------------------------------------ statistiques
def ets(y: np.ndarray, horizon: int, max_len: int = 365, **_) -> np.ndarray:
    """Lissage exponentiel de Holt-Winters : tendance amortie + saisonnalité hebdomadaire additive."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.asarray(y, float)[-max_len:]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                initialization_method="estimated",
            ).fit()
            return np.clip(fit.forecast(horizon), 0, None)
    except Exception:
        return seasonal_naive(y, horizon)


def sarima(y: np.ndarray, horizon: int, max_len: int = 365, **_) -> np.ndarray:
    """SARIMA(1,0,1)(1,1,1)_7 : saisonnalité hebdomadaire différenciée."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.asarray(y, float)[-max_len:]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(
                y, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=60)
            return np.clip(fit.forecast(horizon), 0, None)
    except Exception:
        return seasonal_naive(y, horizon)


def prophet(y: np.ndarray, horizon: int, dates: pd.DatetimeIndex | None = None,
            holidays: pd.DatetimeIndex | None = None, **_) -> np.ndarray:
    """Prophet : saisonnalités hebdo + annuelle et jours fériés. Modèle optionnel (dépendance lourde)."""
    import logging

    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    try:
        hol = None
        if holidays is not None and len(holidays):
            hol = pd.DataFrame({"holiday": "ferie", "ds": pd.DatetimeIndex(holidays)})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    holidays=hol, changepoint_prior_scale=0.05)
        m.fit(pd.DataFrame({"ds": dates, "y": np.asarray(y, float)}))
        future = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)})
        return np.clip(m.predict(future)["yhat"].values, 0, None)
    except Exception:
        return seasonal_naive(y, horizon)


STAT_MODELS = {
    "naive_saisonnier": seasonal_naive,
    "moyenne_mobile": moving_average,
    "ets": ets,
    "sarima": sarima,
    "prophet": prophet,
}


# ------------------------------------------------------------------ LightGBM direct
class LGBMDirect:
    """Un modèle global (toutes séries) pour prévoir H jours d'un coup.

    - point : objectif Tweedie (ventes >= 0, asymétriques, avec zéros)
    - intervalle : deux régressions quantiles (5 % / 95 %) -> intervalle de prévision à 90 %
    """

    def __init__(self, horizon: int, quantiles: tuple[float, float] = (0.05, 0.95),
                 n_estimators: int = 400, learning_rate: float = 0.05, seed: int = 42):
        self.horizon = horizon
        self.quantiles = quantiles
        self.features = feature_columns(horizon)
        base = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=31,
                    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=seed, verbose=-1, n_jobs=-1)
        self.point = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **base)
        self.lo = LGBMRegressor(objective="quantile", alpha=quantiles[0], **{**base, "n_estimators": n_estimators // 2})
        self.hi = LGBMRegressor(objective="quantile", alpha=quantiles[1], **{**base, "n_estimators": n_estimators // 2})

    def fit(self, feats: pd.DataFrame) -> "LGBMDirect":
        train = feats.dropna(subset=["rmean_28", f"lag_{self.horizon}"])
        X, y = train[self.features], train["sales"]
        for model in (self.point, self.lo, self.hi):
            model.fit(X, y, categorical_feature=CAT_FEATURES)
        return self

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = feats[self.features]
        point = np.clip(self.point.predict(X), 0, None)
        lo, hi = np.clip(self.lo.predict(X), 0, None), np.clip(self.hi.predict(X), 0, None)
        lo, hi = np.minimum(lo, point), np.maximum(hi, point)  # cohérence : lo <= point <= hi
        return pd.DataFrame({"yhat": point, "yhat_lo": lo, "yhat_hi": hi}, index=feats.index)

    def importance(self) -> pd.Series:
        return pd.Series(self.point.booster_.feature_importance("gain"), index=self.features).sort_values()
