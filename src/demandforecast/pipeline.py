"""Pipeline de bout en bout : backtest rolling-origin -> métriques -> graphiques -> simulation stock -> artefacts.

Usage :
    python -m demandforecast.pipeline --synthetic
    python -m demandforecast.pipeline --data-dir data/raw --horizon 14 --n-folds 6
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import inventory, metrics  # noqa: E402
from .data import KEY, Dataset, load_favorita, make_synthetic  # noqa: E402
from .features import build_features, make_maps  # noqa: E402
from .models import STAT_MODELS, LGBMDirect  # noqa: E402

log = logging.getLogger("demandforecast")
ALL_MODELS = ["naive_saisonnier", "moyenne_mobile", "ets", "sarima", "prophet", "lightgbm"]


# ------------------------------------------------------------------ backtest
def make_cutoffs(last_date: pd.Timestamp, horizon: int, n_folds: int) -> list[pd.Timestamp]:
    """Origines de prévision consécutives, sans chevauchement : la période de test est contiguë."""
    first_test_day = last_date - pd.Timedelta(days=n_folds * horizon - 1)
    return [first_test_day - pd.Timedelta(days=1) + pd.Timedelta(days=k * horizon) for k in range(n_folds)]


def run_backtest(ds: Dataset, horizon: int, n_folds: int, models: list[str], maps: tuple[dict, dict]):
    df = ds.df
    feats = build_features(df, horizon, ds.holidays, *maps)
    cutoffs = make_cutoffs(df["date"].max(), horizon, n_folds)
    out, importances = [], []

    for fold, cutoff in enumerate(cutoffs):
        end = cutoff + pd.Timedelta(days=horizon)
        log.info("Fold %d/%d — origine %s", fold + 1, n_folds, cutoff.date())

        # --- modèles par série
        stat_models = [m for m in models if m in STAT_MODELS]
        for (store, family), g in df.groupby(KEY):
            hist = g[g["date"] <= cutoff]
            test = g[(g["date"] > cutoff) & (g["date"] <= end)]
            for name in stat_models:
                yhat = STAT_MODELS[name](hist["sales"].values, horizon=horizon,
                                         dates=pd.DatetimeIndex(hist["date"]), holidays=ds.holidays)
                out.append(pd.DataFrame({
                    "date": test["date"].values, "store_nbr": store, "family": family, "fold": fold,
                    "horizon_day": np.arange(1, len(test) + 1), "y": test["sales"].values,
                    "model": name, "yhat": yhat[: len(test)], "yhat_lo": np.nan, "yhat_hi": np.nan,
                }))

        # --- LightGBM global
        if "lightgbm" in models:
            train = feats[feats["date"] <= cutoff]
            test = feats[(feats["date"] > cutoff) & (feats["date"] <= end)]
            m = LGBMDirect(horizon).fit(train)
            pred = m.predict(test)
            out.append(pd.DataFrame({
                "date": test["date"].values, "store_nbr": test["store_nbr"].values,
                "family": test["family"].values, "fold": fold,
                "horizon_day": ((test["date"] - cutoff).dt.days).values, "y": test["sales"].values,
                "model": "lightgbm", **{c: pred[c].values for c in pred.columns},
            }))
            importances.append(m.importance())

    preds = pd.concat(out, ignore_index=True)
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values() if importances else None
    return preds, imp, cutoffs


# ------------------------------------------------------------------ graphiques
def plot_example(preds: pd.DataFrame, history: pd.DataFrame, path: Path, series: tuple):
    store, family = series
    g = history[(history["store_nbr"] == store) & (history["family"] == family)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    test_start = preds["date"].min()
    ax.plot(g["date"], g["sales"], color="#444", lw=1, label="Ventes observées")
    p = preds[(preds["store_nbr"] == store) & (preds["family"] == family)]
    colors = {"naive_saisonnier": "#999", "lightgbm": "#d1495b"}
    for model, c in colors.items():
        pm = p[p["model"] == model].sort_values("date")
        if pm.empty:
            continue
        ax.plot(pm["date"], pm["yhat"], color=c, lw=1.6, label=model)
        if model == "lightgbm" and pm["yhat_lo"].notna().any():
            ax.fill_between(pm["date"], pm["yhat_lo"], pm["yhat_hi"], color=c, alpha=0.15, label="Intervalle 90 %")
    ax.set_xlim(test_start - pd.Timedelta(days=60), preds["date"].max())
    ax.set_title(f"Magasin {store} — {family} : prévisions sur la période de test")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_wape_by_horizon(by_h: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, g in by_h.groupby("model"):
        ax.plot(g["horizon_day"], g["WAPE"], marker="o", ms=3, label=model)
    ax.set_xlabel("Jour de l'horizon"); ax.set_ylabel("WAPE"); ax.set_title("Erreur selon l'horizon")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_wape_by_family(by_f: pd.DataFrame, path: Path):
    pivot = by_f.pivot(index="family", columns="model", values="WAPE")
    ax = pivot.plot.bar(figsize=(9, 4), width=0.8)
    ax.set_ylabel("WAPE"); ax.set_title("Erreur par famille de produits"); ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_importance(imp: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    (imp / imp.sum()).tail(15).plot.barh(ax=ax, color="#d1495b")
    ax.set_title("LightGBM — importance des variables (gain, part relative)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ rapport
def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_summary(path: Path, ds: Dataset, args, cutoffs, overall, interval, sim, by_family) -> None:
    note = ""
    if ds.source == "synthetic":
        note = ("> ⚠️ **Rapport généré sur données SYNTHÉTIQUES** (test du pipeline). "
                "Ne pas citer ces chiffres comme résultats réels : relancer sur Favorita.\n\n")
    best = overall.sort_values("WAPE").iloc[0]
    base = overall[overall["model"] == "naive_saisonnier"]
    gain = ""
    if not base.empty and best["model"] != "naive_saisonnier":
        gain = f" soit **{(1 - best['WAPE'] / base.iloc[0]['WAPE']) * 100:.0f} %** de mieux que le naïf saisonnier."
    txt = f"""# Note de synthèse — prévision de la demande

{note}**Données** : {ds.source} — {ds.df['store_nbr'].nunique()} magasins x {ds.df['family'].nunique()} familles,
{ds.df['date'].min().date()} → {ds.df['date'].max().date()}.
**Validation** : rolling-origin, {args.n_folds} origines successives, horizon {args.horizon} jours
(test : {(cutoffs[0] + pd.Timedelta(days=1)).date()} → {ds.df['date'].max().date()}).

## Comparaison des modèles
{_md(overall.sort_values('WAPE'))}

Meilleur modèle : **{best['model']}** (WAPE {best['WAPE']:.3f}){gain}
Biais > 0 : sur-prévision (risque de surstock) ; biais < 0 : sous-prévision (risque de rupture).

## Intervalles de prévision (LightGBM, quantiles 5 % - 95 %)
{_md(interval)}

Une couverture inférieure à la cible nominale de 90 % signifie des intervalles trop étroits :
à prendre en compte avant de s'en servir pour dimensionner un stock.

## Impact sur le stock (simulation base-stock, délai {args.lead_time} j, niveau de service {args.service_level:.0%})
{_md(sim)}

`fill_rate` = part de la demande servie ; `avg_inventory_days` = stock moyen en rayon (hors commandes en transit) exprimé en jours de demande.
Le meilleur modèle est celui qui offre le meilleur taux de service **à stock moyen comparable**, ou moins de stock à taux de service comparable.

## Erreur par famille de produits
{_md(by_family.pivot(index='family', columns='model', values='WAPE').reset_index())}

## Limites
- Les ventes observées censurent la demande réelle (ruptures) : le modèle apprend des ventes, pas de la demande.
- La simulation de stock est simplifiée (pas de lot minimal, pas de coût, délai fixe) et le stock de sécurité est calibré en
  échantillon sur la période de test : les taux de service sont légèrement optimistes.
- Promotions futures supposées connues (`onpromotion`) ; en production, ce plan promo doit être fourni par le métier.
"""
    path.write_text(txt, encoding="utf-8")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--synthetic", action="store_true", help="données synthétiques (test du pipeline)")
    ap.add_argument("--n-stores", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--lead-time", type=int, default=3, help="délai de réapprovisionnement (jours)")
    ap.add_argument("--service-level", type=float, default=0.95)
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    models = [m.strip() for m in args.models.split(",")]
    ds = make_synthetic() if args.synthetic else load_favorita(args.data_dir, n_stores=args.n_stores)
    reports, figs = Path(args.reports_dir), Path(args.reports_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True); Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    log.info("Données : %s, %d séries, %s → %s", ds.source, ds.df.groupby(KEY).ngroups,
             ds.df["date"].min().date(), ds.df["date"].max().date())

    maps = make_maps(ds.df)
    preds, imp, cutoffs = run_backtest(ds, args.horizon, args.n_folds, models, maps)
    preds.to_csv(reports / "predictions.csv", index=False)

    # --- métriques
    scales = (ds.df[ds.df["date"] <= cutoffs[0]].groupby(KEY)["sales"].apply(lambda s: metrics.seasonal_scale(s.values)))
    overall = metrics.score_table(preds, scales)
    by_h = metrics.score_table(preds, scales, by=["horizon_day"])
    by_f = metrics.score_table(preds, scales, by=["family"])
    overall.to_csv(reports / "metrics_by_model.csv", index=False)
    by_h.to_csv(reports / "metrics_by_horizon.csv", index=False)
    by_f.to_csv(reports / "metrics_by_family.csv", index=False)

    lgb = preds[preds["model"] == "lightgbm"]
    interval = pd.DataFrame()
    if not lgb.empty:
        interval = pd.DataFrame([{
            "Couverture 90 % (cible 0.90)": metrics.coverage(lgb["y"], lgb["yhat_lo"], lgb["yhat_hi"]),
            "Pinball q5": metrics.pinball(lgb["y"], lgb["yhat_lo"], 0.05),
            "Pinball q95": metrics.pinball(lgb["y"], lgb["yhat_hi"], 0.95),
        }])

    # --- simulation de stock
    sim = pd.DataFrame([inventory.simulate_policy(preds, m, args.lead_time, args.service_level)
                        for m in preds["model"].unique()])
    sim.to_csv(reports / "inventory_simulation.csv", index=False)

    # --- graphiques
    first = preds[KEY].drop_duplicates().iloc[0]
    plot_example(preds, ds.df, figs / "exemple_prevision.png", (first["store_nbr"], first["family"]))
    plot_wape_by_horizon(by_h, figs / "erreur_par_horizon.png")
    plot_wape_by_family(by_f, figs / "erreur_par_famille.png")
    if imp is not None:
        plot_importance(imp, figs / "importance_variables.png")

    write_summary(reports / "summary.md", ds, args, cutoffs, overall, interval, sim, by_f)

    # --- artefact de service : modèle final entraîné sur tout l'historique
    if "lightgbm" in models:
        feats = build_features(ds.df, args.horizon, ds.holidays, *maps)
        final = LGBMDirect(args.horizon).fit(feats)
        errors = lgb.assign(error=lgb["yhat"] - lgb["y"])[KEY + ["date", "error"]]
        history = ds.df[ds.df["date"] >= ds.df["date"].max() - pd.Timedelta(days=500)]
        joblib.dump({
            "model": final, "horizon": args.horizon, "store_map": maps[0], "family_map": maps[1],
            "holidays": ds.holidays, "history": history, "backtest_errors": errors, "source": ds.source,
        }, Path(args.models_dir) / "artifacts.joblib")
        log.info("Artefact sauvegardé.")

    print("\n" + overall.sort_values("WAPE").to_string(index=False, float_format="%.3f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
