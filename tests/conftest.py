import pytest
from icu_stepdown.config import load_config


@pytest.fixture()
def config():
    return load_config("configs/default.yaml")

