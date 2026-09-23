import numpy as np
import pytest

from demandforecast import inventory


def test_safety_stock_matches_z_score():
    assert inventory.safety_stock(100, 0.95) == pytest.approx(164.485, rel=1e-4)


def test_reorder_point_adds_lead_time_demand_and_safety():
    r = inventory.reorder_point([10, 10, 10, 10], lead_time=3, sigma_lt=20, service_level=0.95)
    assert r["expected_demand_lt"] == 30
    assert r["reorder_point"] == pytest.approx(30 + r["safety_stock"])


def test_sigma_lead_time_grows_with_lead_time():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 10, 500)
    assert inventory.sigma_lead_time(e, 5) > inventory.sigma_lead_time(e, 1)


def test_perfect_forecast_constant_demand_never_stocks_out():
    res = inventory.simulate_base_stock(np.full(60, 10.0), np.full(60, 10.0), lead_time=3, safety=0.0)
    assert res["lost"] == 0 and res["stockout_days"] == 0


def test_underforecast_without_safety_loses_sales():
    res = inventory.simulate_base_stock(np.full(60, 10.0), np.full(60, 5.0), lead_time=3, safety=0.0)
    assert res["lost"] > 0


def test_more_safety_stock_means_higher_fill_rate_and_inventory():
    rng = np.random.default_rng(3)
    demand = rng.poisson(50, 200).astype(float)
    forecast = np.full(200, 50.0)
    low = inventory.simulate_base_stock(demand, forecast, 3, safety=0.0)
    high = inventory.simulate_base_stock(demand, forecast, 3, safety=150.0)
    assert high["lost"] < low["lost"]
    assert high["avg_inventory"] > low["avg_inventory"]
