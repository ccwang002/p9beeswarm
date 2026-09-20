"""Shared fixtures for comparing this package against the real R packages."""

from typing import Any

import pytest


@pytest.fixture(scope="session")
def r_vipor() -> Any:
    """Return ``(vipor R package, rpy2.robjects)``, skipping if unavailable."""
    robjects = pytest.importorskip("rpy2.robjects")
    packages = pytest.importorskip("rpy2.robjects.packages")
    try:
        return packages.importr("vipor"), robjects
    except packages.PackageNotInstalledError as error:
        pytest.skip(f"R vipor package is unavailable: {error}")
