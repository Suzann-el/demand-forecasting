import numpy as np
import pytest

from demandforecast import metrics


def test_wape_and_bias():
    y, yhat = np.array([10, 20, 30]), np.array([12, 18, 30])
    assert metrics.wape(y, yhat) == pytest.approx(4 / 60)
    assert metrics.bias(y, yhat) == pytest.approx(0.0)
    assert metrics.bias(y, y + 6) == pytest.approx(18 / 60)  # sur-prévision => biais > 0


def test_wape_handles_all_zero():
    assert np.isfinite(metrics.wape([0, 0], [0, 0]))


def test_mase_scale_is_seasonal_naive_mae():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8], float)
    assert metrics.seasonal_scale(y, m=7) == pytest.approx(1.0)


def test_pinball_and_coverage():
    y = np.array([10.0, 10.0])
    assert metrics.pinball(y, np.array([8.0, 12.0]), 0.9) == pytest.approx((0.9 * 2 + 0.1 * 2) / 2)
    assert metrics.coverage(y, [9, 11], [11, 12]) == 0.5
