"""Linux RSS plateau check; native ASan/LSan provides precise leak diagnostics."""
import gc
from pathlib import Path
import numpy as np
from libtree import GBDTRegressor


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux RSS measurement unavailable")


def test_repeated_python_ownership_plateaus():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 5)).astype(np.float32)
    y = x[:, 0] + x[:, 1]
    rss = []
    for cycle in range(300):
        model = GBDTRegressor(n_estimators=10)
        model.fit(x, y)
        assert np.isfinite(model.predict(x)).all()
        # Alternate explicit close and Python finalization.
        if cycle % 2:
            model.close()
        del model
        if cycle in (99, 199, 299):
            gc.collect()
            rss.append(rss_bytes())
    assert rss[-1] - rss[0] < 16 * 1024 * 1024, rss
