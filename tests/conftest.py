import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images

BASELINE_DIR = Path(__file__).parent / "baseline_images"


def pytest_addoption(parser):
    parser.addoption(
        "--visual-result-dir",
        action="store",
        default=None,
        help="Directory for visual test result images (defaults to a temp directory).",
    )


@pytest.fixture
def assert_plot(request):
    result_dir = request.config.getoption("--visual-result-dir")
    if result_dir is None:
        result_dir = tempfile.mkdtemp(prefix="plotnine-beeswarm-results-")
    result_dir = Path(result_dir)

    def compare(plot, name):
        result = result_dir / f"{name}.png"
        baseline = BASELINE_DIR / f"{name}.png"
        result.parent.mkdir(parents=True, exist_ok=True)
        plot.save(result, verbose=False)
        if os.environ.get("P9BEESWARM_GENERATE_BASELINES") == "1":
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result, baseline)
            plt.close("all")
            return
        if not baseline.exists():
            pytest.fail(f"Missing baseline image: {baseline}")
        error = compare_images(str(baseline), str(result), tol=2)
        plt.close("all")
        assert error is None, error

    return compare
