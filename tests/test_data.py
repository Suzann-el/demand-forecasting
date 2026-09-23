"""Le chargeur Favorita est testé sur de faux fichiers au format exact de la compétition Kaggle."""
import pandas as pd

from demandforecast.data import KEY, complete_calendar, load_favorita, load_holidays, make_synthetic


def _write_fake_favorita(d):
    syn = make_synthetic(n_stores=3, families=["GROCERY I", "BEVERAGES", "AUTOMOTIVE"],
                         start="2015-06-01", end="2015-12-31").df
    syn = syn[syn["date"] != "2015-12-25"]  # Favorita n'a pas de ligne le 25/12
    syn = syn.assign(id=range(len(syn)))[["id", "date", "store_nbr", "family", "sales", "onpromotion"]]
    syn.to_csv(d / "train.csv", index=False)
    pd.DataFrame([
        ["2015-08-10", "Holiday", "National", "Ecuador", "Primer Grito", False],
        ["2015-10-09", "Holiday", "National", "Ecuador", "Independencia de Guayaquil", True],   # transféré : pas férié ce jour-là
        ["2015-10-12", "Transfer", "National", "Ecuador", "Traslado Independencia", False],    # jour effectivement chômé
        ["2015-11-02", "Holiday", "Local", "Quito", "Fundacion", False],                       # local : ignoré
        ["2015-09-05", "Work Day", "National", "Ecuador", "Recupero", False],                  # jour travaillé : ignoré
    ], columns=["date", "type", "locale", "locale_name", "description", "transferred"]).to_csv(d / "holidays_events.csv", index=False)


def test_holidays_filtering(tmp_path):
    _write_fake_favorita(tmp_path)
    hol = load_holidays(tmp_path / "holidays_events.csv")
    assert list(hol.strftime("%Y-%m-%d")) == ["2015-08-10", "2015-10-12"]


def test_load_favorita_subset_and_calendar(tmp_path):
    _write_fake_favorita(tmp_path)
    ds = load_favorita(tmp_path, n_stores=2, families=["GROCERY I", "BEVERAGES"], start="2015-07-01")
    assert ds.df["store_nbr"].nunique() == 2 and set(ds.df["family"]) == {"GROCERY I", "BEVERAGES"}
    assert ds.df["date"].min() == pd.Timestamp("2015-07-01")
    # 25/12 réintroduit à 0 : calendrier complet sans trou pour chaque série
    counts = ds.df.groupby(KEY)["date"].agg(["count", "min", "max"])
    assert (counts["count"] == (counts["max"] - counts["min"]).dt.days + 1).all()
    assert (ds.df[ds.df["date"] == "2015-12-25"]["sales"] == 0).all()


def test_complete_calendar_fills_zero():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "store_nbr": 1, "family": "A",
                       "sales": [5.0, 7.0], "onpromotion": [0.0, 1.0]})
    out = complete_calendar(df)
    assert len(out) == 3 and out.loc[1, "sales"] == 0
