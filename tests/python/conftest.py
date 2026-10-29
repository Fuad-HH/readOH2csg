from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture(scope="session")
def assets():
    return ASSETS


@pytest.fixture(scope="session")
def degas2_mesh():
    return ASSETS / "simple_degas2_case" / "tagged-dg2mesh.osh"
