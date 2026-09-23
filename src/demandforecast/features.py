"""Construction des variables pour le modèle LightGBM « direct ».

Principe anti-fuite : pour un horizon H, toute variable issue des ventes passées utilise
un décalage >= H. Une ligne datée `d` n'utilise donc jamais de ventes postérieures à `d - H`,
ce qui permet de prévoir H jours d'un coup depuis la dernière date connue (stratégie directe).
Les variables connues à l'avance (calendrier, jours fériés, promotions planifiées) sont autorisées.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import KEY

CAT_FEATURES = ["store_code", "family_code"]


def make_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    """Codes entiers stables pour magasins et familles (sauvegardés avec le modèle)."""
    store_map = {s: i for i, s in enumerate(sorted(df["store_nbr"].unique()))}
    family_map = {f: i for i, f in enumerate(sorted(df["family"].unique()))}
    return store_map, family_map


def same_weekday_lags(horizon: int) -> list[int]:
    """Décalages >= H tombant sur le même jour de semaine que la date cible (multiples de 7)."""
    first = 7 * math.ceil(horizon / 7)
    return [first, first + 7, first + 14, first + 21, 364 if first <= 364 else first + 28]


def feature_columns(horizon: int) -> list[str]:
    lags = sorted({horizon, *same_weekday_lags(horizon)})
    return (
        CAT_FEATURES
        + ["dow", "dom", "month", "weekofyear", "is_payday", "is_holiday", "hol_prev", "hol_next", "onpromotion"]
        + [f"lag_{k}" for k in lags]
        + ["rmean_7", "rmean_28", "rmean_91", "rstd_28", "trend_ratio"]
    )


def build_features(
    df: pd.DataFrame,
    horizon: int,
    holidays: pd.DatetimeIndex,
    store_map: dict,
    family_map: dict,
) -> pd.DataFrame:
    """Retourne df + variables. Accepte des lignes futures (sales = NaN) pour le service."""
    out = df.sort_values(KEY + ["date"]).reset_index(drop=True).copy()
    d = out["date"]

    # --- calendrier (connu à l'avance) ---
    out["store_code"] = out["store_nbr"].map(store_map).astype(int)
    out["family_code"] = out["family"].map(family_map).astype(int)
    out["dow"] = d.dt.dayofweek
    out["dom"] = d.dt.day
    out["month"] = d.dt.month
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["is_payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype(int)
    out["is_holiday"] = d.isin(holidays).astype(int)
    out["hol_prev"] = (d - pd.Timedelta(days=1)).isin(holidays).astype(int)
    out["hol_next"] = (d + pd.Timedelta(days=1)).isin(holidays).astype(int)

    # --- décalages et fenêtres glissantes, tous >= H ---
    g = out.groupby(KEY, sort=False)["sales"]
    for k in sorted({horizon, *same_weekday_lags(horizon)}):
        out[f"lag_{k}"] = g.shift(k)

    def rolling(fn: str, w: int) -> pd.Series:
        return g.transform(lambda s: getattr(s.shift(horizon).rolling(w, min_periods=max(2, w // 2)), fn)())

    out["rmean_7"] = rolling("mean", 7)
    out["rmean_28"] = rolling("mean", 28)
    out["rmean_91"] = rolling("mean", 91)
    out["rstd_28"] = rolling("std", 28)
    out["trend_ratio"] = out["rmean_28"] / out["rmean_91"].replace(0, np.nan)
    return out
