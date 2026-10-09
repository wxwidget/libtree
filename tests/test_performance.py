"""Optional broad performance guard; benchmark reports measure relative speed."""
import time
import numpy as np
from libtree import GBDTRegressor


def test_representative_workload():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(10000, 10)).astype(np.float32)
    y = x[:, 0] * 2 + x[:, 1] ** 2
    model = GBDTRegressor(n_estimators=50)
    start = time.perf_counter()
    model.fit(x, y)
    duration = time.perf_counter() - start
    prediction = model.predict(x)
    assert np.mean((prediction - y) ** 2) < .2
    # Deliberately broad: catches catastrophic regressions, not scheduler noise.
    assert duration < 30, f"10k-row fit exceeded 30 seconds: {duration:.3f}s"
