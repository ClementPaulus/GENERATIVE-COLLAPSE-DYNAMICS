import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# Ecological States mapped to the 0-1 continuous kernel space.
# biomass: 1.0 (carrying capacity) to EPSILON (barren)
# diversity: 1.0 (climax community) to EPSILON (monoculture/empty)
# connectivity: 1.0 (rich food web) to EPSILON (fragmented/isolated)
# energy_flux: 1.0 (optimal flow) to EPSILON (stagnant/frozen)
# resilience: 1.0 (rapid rebound) to EPSILON (brittle)
ECOLOGICAL_STATES = {
    "CLIMAX_FOREST": {
        "biomass": 0.95,
        "diversity": 0.95,
        "connectivity": 0.90,
        "energy_flux": 0.85,
        "resilience": 0.80,
    },
    "CORAL_REEF": {"biomass": 0.85, "diversity": 1.0, "connectivity": 1.0, "energy_flux": 0.90, "resilience": 0.60},
    "GRASSLAND": {"biomass": 0.70, "diversity": 0.70, "connectivity": 0.80, "energy_flux": 0.75, "resilience": 0.95},
    "TUNDRA": {"biomass": 0.40, "diversity": 0.50, "connectivity": 0.60, "energy_flux": 0.30, "resilience": 0.50},
    "DESERT": {"biomass": 0.20, "diversity": 0.30, "connectivity": 0.40, "energy_flux": 0.20, "resilience": 0.40},
    "AGRICULTURE": {"biomass": 0.90, "diversity": 0.10, "connectivity": 0.20, "energy_flux": 0.80, "resilience": 0.10},
    "KEYSTONE_LOSS": {
        "biomass": 0.80,
        "diversity": 0.70,
        "connectivity": EPSILON,
        "energy_flux": 0.50,
        "resilience": 0.30,
    },
    "TROPHIC_CASCADE": {
        "biomass": 0.40,
        "diversity": 0.30,
        "connectivity": EPSILON,
        "energy_flux": EPSILON,
        "resilience": EPSILON,
    },
    "ALGAL_BLOOM": {
        "biomass": 1.0,
        "diversity": EPSILON,
        "connectivity": EPSILON,
        "energy_flux": 0.40,
        "resilience": EPSILON,
    },
    "MASS_EXTINCTION": {
        "biomass": EPSILON,
        "diversity": EPSILON,
        "connectivity": EPSILON,
        "energy_flux": EPSILON,
        "resilience": EPSILON,
    },
    "PIONEER_STAGE": {
        "biomass": 0.10,
        "diversity": 0.20,
        "connectivity": 0.10,
        "energy_flux": 0.90,
        "resilience": 0.90,
    },
    "URBAN_GRID": {"biomass": 0.05, "diversity": 0.10, "connectivity": 0.05, "energy_flux": 0.10, "resilience": 0.80},
}


def analyze_ecology_state(name: str):
    data = ECOLOGICAL_STATES[name]
    c = np.array([data["biomass"], data["diversity"], data["connectivity"], data["energy_flux"], data["resilience"]])
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(5) / 5
    return compute_kernel_outputs(c, w)
