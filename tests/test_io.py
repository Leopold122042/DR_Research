from pathlib import Path

import numpy as np
import pytest

from dr_research.config import load_config
from dr_research.io import load_scenario, save_scenario, scenario_fingerprint
from dr_research.scenario import generate_scenario

CONFIG = Path(__file__).resolve().parents[1]/"configs"/"experiment.toml"


@pytest.fixture
def scenario():
    return generate_scenario(load_config(CONFIG), "S1", 1024)


def test_scenario_roundtrip_and_exclusive_creation(tmp_path, scenario):
    path = tmp_path/"scenario.npz"
    digest = save_scenario(path, scenario)
    loaded = load_scenario(path)
    assert digest == scenario_fingerprint(loaded)
    assert loaded.network.pump_models == scenario.network.pump_models
    np.testing.assert_array_equal(loaded.heat_demand_kw, scenario.heat_demand_kw)
    with pytest.raises(FileExistsError):
        save_scenario(path, scenario)


def test_tampered_scenario_is_rejected(tmp_path, scenario):
    path = tmp_path/"original.npz"
    save_scenario(path, scenario)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["scenario.price_yuan_per_kwh"][0] += 0.01
    np.savez_compressed(tmp_path/"tampered.npz", **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        load_scenario(tmp_path/"tampered.npz")

