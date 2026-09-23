import numpy as np
import pandas as pd
import pytest

from demandforecast.data import KEY
from demandforecast.features import build_features, feature_columns, make_maps, same_weekday_lags
from demandforecast.pipeline import make_cutoffs

H = 14
LAG_COLS = [c for c in feature_columns(H) if c.startswith(("lag_", "rmean_", "rstd_", "trend_"))]


def test_no_leakage_from_future_sales(ds):
    """Une ligne datée d ne doit dépendre d'aucune vente postérieure à d - H."""
    maps = make_maps(ds.df)
    base = build_features(ds.df, H, ds.holidays, *maps)
    row_date = ds.df["date"].max() - pd.Timedelta(days=40)

    tampered = ds.df.copy()
    future = tampered["date"] > row_date - pd.Timedelta(days=H)
    tampered.loc[future, "sales"] = np.random.default_rng(0).uniform(1e4, 1e5, future.sum())
    alt = build_features(tampered, H, ds.holidays, *maps)

    a = base[base["date"] == row_date].sort_values(KEY)[LAG_COLS].reset_index(drop=True)
    b = alt[alt["date"] == row_date].sort_values(KEY)[LAG_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_all_lags_respect_horizon():
    for h in (1, 7, 10, 14, 28):
        lags = [int(c.split("_")[1]) for c in feature_columns(h) if c.startswith("lag_")]
        assert min(lags) >= h


def test_same_weekday_lags_are_multiples_of_7():
    for h in (1, 10, 14, 20):
        assert all(k % 7 == 0 and k >= h for k in same_weekday_lags(h))


def test_cutoffs_are_contiguous_and_end_at_last_date():
    last = pd.Timestamp("2017-08-15")
    cuts = make_cutoffs(last, 14, 4)
    assert all((b - a).days == 14 for a, b in zip(cuts, cuts[1:]))
    assert cuts[-1] + pd.Timedelta(days=14) == last
