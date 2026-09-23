import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from demandforecast.data import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def ds():
    return make_synthetic(n_stores=2, families=["GROCERY I", "DAIRY"], start="2016-01-01", end="2017-03-31")


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Exécute le pipeline complet (petit) une fois pour tous les tests d'intégration."""
    from demandforecast import pipeline

    d = tmp_path_factory.mktemp("run")
    pipeline.main(["--synthetic", "--n-folds", "2", "--horizon", "14",
                   "--models", "naive_saisonnier,lightgbm",
                   "--reports-dir", str(d / "reports"), "--models-dir", str(d / "models")])
    return d
