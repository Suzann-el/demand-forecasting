"""Chargement des données (Favorita/Kaggle) et générateur synthétique au même schéma.

Schéma commun retourné : date, store_nbr, family, sales, onpromotion
+ un DatetimeIndex `holidays` (jours fériés/évènements nationaux, connus à l'avance).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["store_nbr", "family"]
DEFAULT_FAMILIES = ["GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY", "BREAD/BAKERY"]


@dataclass
class Dataset:
    df: pd.DataFrame            # historique journalier complet (calendrier sans trou)
    holidays: pd.DatetimeIndex  # jours fériés connus (peut dépasser la fin de l'historique)
    source: str                 # "favorita" ou "synthetic"


def complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Force un calendrier journalier sans trou par série (ex. 25/12 absent chez Favorita).

    Les jours manquants sont remplis à 0 (magasin fermé = pas de vente).
    """
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product(
        [sorted(df["store_nbr"].unique()), sorted(df["family"].unique()), full],
        names=KEY + ["date"],
    )
    out = df.set_index(KEY + ["date"]).reindex(idx)
    out[["sales", "onpromotion"]] = out[["sales", "onpromotion"]].fillna(0.0)
    return out.reset_index().sort_values(KEY + ["date"]).reset_index(drop=True)


def load_holidays(path: str | Path) -> pd.DatetimeIndex:
    """Jours fériés nationaux effectivement chômés/célébrés (holidays_events.csv)."""
    h = pd.read_csv(path, parse_dates=["date"])
    h = h[(h["locale"] == "National") & (~h["transferred"].astype(bool)) & (h["type"] != "Work Day")]
    return pd.DatetimeIndex(sorted(h["date"].unique()))


def load_favorita(
    data_dir: str | Path,
    n_stores: int = 6,
    families: list[str] | None = None,
    start: str = "2015-01-01",
) -> Dataset:
    """Charge Favorita (Kaggle: store-sales-time-series-forecasting) et garde un sous-ensemble.

    On retient les `n_stores` magasins aux plus fortes ventes sur les familles choisies,
    à partir de `start` (2013-2014 contient beaucoup de séries à zéro : produit pas encore vendu).
    """
    data_dir = Path(data_dir)
    families = families or DEFAULT_FAMILIES
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    train = train[(train["family"].isin(families)) & (train["date"] >= start)]
    top = train.groupby("store_nbr")["sales"].sum().nlargest(n_stores).index
    train = train[train["store_nbr"].isin(top)][["date", "store_nbr", "family", "sales", "onpromotion"]]
    holidays = load_holidays(data_dir / "holidays_events.csv")
    return Dataset(complete_calendar(train), holidays, "favorita")


def _synthetic_holidays(start: str, end: str) -> pd.DatetimeIndex:
    """Jours fériés fixes (1/1, 1/5, 10/8, 2/11, 25/12) — équivalent simplifié du calendrier réel."""
    days = pd.date_range(start, end, freq="D")
    fixed = {(1, 1), (5, 1), (8, 10), (11, 2), (12, 25)}
    return pd.DatetimeIndex([d for d in days if (d.month, d.day) in fixed])


def make_synthetic(
    n_stores: int = 4,
    families: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2017-08-15",
    seed: int = 42,
) -> Dataset:
    """Génère des ventes réalistes au schéma Favorita : tendance, saisonnalité hebdo/annuelle,
    promotions, jours fériés, jours de paie, sur-dispersion (binomiale négative) et ruptures.

    Sert à tester le pipeline sans télécharger Kaggle. Les résultats obtenus sur ces données
    ne doivent PAS être présentés comme des résultats réels.
    """
    rng = np.random.default_rng(seed)
    families = families or DEFAULT_FAMILIES[:5]
    dates = pd.date_range(start, end, freq="D")
    holidays = _synthetic_holidays(start, "2017-12-31")
    is_hol = dates.isin(holidays).astype(float)
    t = np.arange(len(dates))
    payday = ((dates.day == 15) | dates.is_month_end).astype(float)
    weekly_base = np.array([0.90, 0.90, 0.95, 1.00, 1.10, 1.25, 1.20])

    frames = []
    for s in range(1, n_stores + 1):
        store_mult = rng.uniform(0.6, 1.6)
        for f in families:
            base = rng.uniform(80, 900) * store_mult
            trend = 1 + rng.uniform(0.0002, 0.0007) * t
            weekly = (weekly_base * rng.uniform(0.9, 1.1, 7))[dates.dayofweek]
            phase = rng.uniform(0, 2 * np.pi)
            yearly = 1 + 0.12 * np.sin(2 * np.pi * dates.dayofyear / 365.25 + phase)
            xmas = 1 + 0.25 * np.exp(-((dates.dayofyear.values - 355) / 8.0) ** 2)
            # épisodes promo : 3 à 10 jours, intensité = nb d'articles en promo
            promo_intensity = np.zeros(len(dates))
            i = 0
            while i < len(dates):
                if rng.random() < 0.03:
                    length = int(rng.integers(3, 11))
                    promo_intensity[i:i + length] = rng.integers(1, 20)
                    i += length
                i += 1
            promo_effect = 1 + 0.012 * promo_intensity
            mu = (base * trend * weekly * yearly * xmas * promo_effect
                  * (1 + 0.25 * is_hol) * (1 + 0.08 * payday))
            r = 25.0  # sur-dispersion
            sales = rng.negative_binomial(r, r / (r + mu)).astype(float)
            # ruptures : ~0.6 % de jours (1 à 3 jours de ventes nulles)
            k = 0
            while k < len(dates):
                if rng.random() < 0.006:
                    sales[k:k + int(rng.integers(1, 4))] = 0.0
                    k += 3
                k += 1
            frames.append(pd.DataFrame({
                "date": dates, "store_nbr": s, "family": f,
                "sales": sales, "onpromotion": promo_intensity,
            }))
    df = pd.concat(frames, ignore_index=True)
    return Dataset(complete_calendar(df), holidays, "synthetic")
