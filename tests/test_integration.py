"""Tests d'intégration : pipeline, cohérence entraînement/service (train-serve skew), API."""
import joblib
import numpy as np
import pandas as pd
import pytest

from demandforecast import serve
from demandforecast.data import KEY, make_synthetic
from demandforecast.features import build_features


def test_pipeline_outputs_exist(trained):
    for f in ("metrics_by_model.csv", "predictions.csv", "summary.md", "inventory_simulation.csv"):
        assert (trained / "reports" / f).exists()
    m = pd.read_csv(trained / "reports" / "metrics_by_model.csv").set_index("model")
    assert m.loc["lightgbm", "WAPE"] < m.loc["naive_saisonnier", "WAPE"]  # le modèle doit battre la baseline


def test_intervals_are_ordered_and_predictions_nonnegative(trained):
    p = pd.read_csv(trained / "reports" / "predictions.csv")
    lgb = p[p["model"] == "lightgbm"]
    assert (lgb["yhat_lo"] <= lgb["yhat"]).all() and (lgb["yhat"] <= lgb["yhat_hi"]).all()
    assert (p["yhat"] >= 0).all()


def test_serving_matches_batch_features(trained):
    """Le service (historique tronqué + lignes futures) doit redonner les prévisions du batch."""
    art = joblib.load(trained / "models" / "artifacts.joblib")
    full = make_synthetic()  # même graine => même données que le pipeline
    H = art["horizon"]
    store, family = 1, "BEVERAGES"
    cutoff = full.df["date"].max() - pd.Timedelta(days=H)

    truncated = {**art, "history": full.df[full.df["date"] <= cutoff]}
    g = full.df[(full.df["store_nbr"] == store) & (full.df["family"] == family) & (full.df["date"] > cutoff)]
    served = serve.forecast_series(truncated, store, family, onpromotion=g["onpromotion"].tolist())

    feats = build_features(full.df, H, art["holidays"], art["store_map"], art["family_map"])
    rows = feats[(feats["store_nbr"] == store) & (feats["family"] == family) & (feats["date"] > cutoff)]
    batch = art["model"].predict(rows)
    np.testing.assert_allclose(served["yhat"].values, batch["yhat"].values, rtol=1e-9)


def test_recommend_stock_is_coherent(trained):
    art = joblib.load(trained / "models" / "artifacts.joblib")
    lo = serve.recommend_stock(art, 1, "BEVERAGES", lead_time=3, service_level=0.90)
    hi = serve.recommend_stock(art, 1, "BEVERAGES", lead_time=3, service_level=0.99)
    assert hi["safety_stock"] > lo["safety_stock"] > 0
    assert hi["reorder_point"] == pytest.approx(hi["expected_demand_lt"] + hi["safety_stock"], abs=0.2)


def test_api(trained, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as m

    monkeypatch.setattr(m, "ARTIFACT_PATH", str(trained / "models" / "artifacts.joblib"))
    m.artifacts.cache_clear()
    c = TestClient(m.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/forecast", json={"store_nbr": 1, "family": "BEVERAGES", "horizon": 7})
    assert r.status_code == 200 and len(r.json()["forecast"]) == 7
    assert c.post("/forecast", json={"store_nbr": 999, "family": "X"}).status_code == 404
    assert c.post("/forecast", json={"store_nbr": 1, "family": "BEVERAGES", "horizon": 500}).status_code == 422
    assert c.post("/reorder", json={"store_nbr": 1, "family": "BEVERAGES", "lead_time": 3}).status_code == 200
    m.artifacts.cache_clear()
