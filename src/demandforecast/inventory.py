"""Couche métier : de la prévision au stock de sécurité, au point de commande, et simulation de politique.

Politique simulée : « base-stock » à revue quotidienne. Chaque soir on commande pour ramener la
position de stock (en stock + en commande) au niveau cible
    S_t = somme des prévisions sur les L jours suivants + stock de sécurité,
la commande arrivant L jours plus tard. Simplification assumée : demande observée = ventes
(les ruptures censurent la demande réelle), coût de commande nul, pas de contrainte de lot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import KEY


def sigma_lead_time(errors: pd.Series, lead_time: int) -> float:
    """Écart-type de l'erreur de prévision CUMULÉE sur `lead_time` jours (série d'erreurs ordonnée par date).

    Plus fidèle que sigma_jour * sqrt(L), car les erreurs successives sont autocorrélées.
    """
    e = pd.Series(np.asarray(errors, float))
    if lead_time <= 1:
        return float(e.std(ddof=1))
    return float(e.rolling(lead_time).sum().dropna().std(ddof=1))


def safety_stock(sigma_lt: float, service_level: float) -> float:
    """Stock de sécurité = z(niveau de service) x sigma de l'erreur cumulée sur le délai."""
    return float(norm.ppf(service_level) * sigma_lt)


def reorder_point(forecast, lead_time: int, sigma_lt: float, service_level: float) -> dict:
    """Point de commande = demande prévue pendant le délai + stock de sécurité."""
    expected = float(np.asarray(forecast, float)[:lead_time].sum())
    ss = safety_stock(sigma_lt, service_level)
    return {"expected_demand_lt": expected, "safety_stock": ss, "reorder_point": expected + ss}


def simulate_base_stock(demand, forecast, lead_time: int, safety: float) -> dict:
    """Simule une série. `demand` et `forecast` alignés sur la même période (n jours)."""
    demand, forecast = np.asarray(demand, float), np.asarray(forecast, float)
    n, L = len(demand), lead_time
    inv = forecast[:L].sum() + safety
    arrivals = np.zeros(n + L + 1)
    sold = lost = inv_sum = 0.0
    stockout_days = counted = 0
    for t in range(n - L):
        inv += arrivals[t]
        d = demand[t]
        s = min(inv, d)
        inv -= s
        if t >= L:  # période de chauffe ignorée
            sold += s
            lost += d - s
            stockout_days += int(d > s)
            inv_sum += inv
            counted += 1
        position = inv + arrivals[t + 1:].sum()
        target = forecast[t + 1:t + L + 1].sum() + safety
        arrivals[t + L] += max(0.0, target - position)
    return {"sold": sold, "lost": lost, "stockout_days": stockout_days,
            "avg_inventory": inv_sum / max(counted, 1), "days": counted}


def simulate_policy(preds: pd.DataFrame, model: str, lead_time: int, service_level: float) -> dict:
    """Agrège la simulation sur toutes les séries pour un modèle donné.

    Le stock de sécurité de chaque série est calibré sur les erreurs du modèle lui-même
    (calibration en échantillon sur la période de test : légèrement optimiste, à signaler).
    """
    p = preds[preds["model"] == model].sort_values(KEY + ["date"])
    sold = lost = avg_inv = 0.0
    stockout_days = days = n_series = 0
    for _, g in p.groupby(KEY):
        sig = sigma_lead_time(g["yhat"] - g["y"], lead_time)
        res = simulate_base_stock(g["y"].values, g["yhat"].values, lead_time, safety_stock(sig, service_level))
        sold += res["sold"]; lost += res["lost"]; avg_inv += res["avg_inventory"]
        stockout_days += res["stockout_days"]; days += res["days"]; n_series += 1
    total = sold + lost
    days_per_series = days / max(n_series, 1)
    daily_demand = total / days_per_series if days_per_series else np.nan  # demande quotidienne, toutes séries
    return {
        "model": model,
        "fill_rate": sold / total if total else np.nan,
        "stockout_day_rate": stockout_days / days if days else np.nan,
        "avg_inventory": avg_inv,
        "avg_inventory_days": avg_inv / daily_demand if daily_demand else np.nan,
    }
