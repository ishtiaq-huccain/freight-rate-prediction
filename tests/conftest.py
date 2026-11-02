from __future__ import annotations

import pytest

from freight_rate.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
