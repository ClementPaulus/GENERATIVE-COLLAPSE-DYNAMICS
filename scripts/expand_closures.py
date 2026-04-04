
ecology_kernel = """import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# Ecological States mapped to the 0-1 continuous kernel space.
# biomass: 1.0 (carrying capacity) to EPSILON (barren)
# diversity: 1.0 (climax community) to EPSILON (monoculture/empty)
# connectivity: 1.0 (rich food web) to EPSILON (fragmented/isolated)
# energy_flux: 1.0 (optimal flow) to EPSILON (stagnant/frozen)
# resilience: 1.0 (rapid rebound) to EPSILON (brittle)
ECOLOGICAL_STATES = {
    "CLIMAX_FOREST": {"biomass": 0.95, "diversity": 0.95, "connectivity": 0.90, "energy_flux": 0.85, "resilience": 0.80},
    "CORAL_REEF": {"biomass": 0.85, "diversity": 1.0, "connectivity": 1.0, "energy_flux": 0.90, "resilience": 0.60},
    "GRASSLAND": {"biomass": 0.70, "diversity": 0.70, "connectivity": 0.80, "energy_flux": 0.75, "resilience": 0.95},
    "TUNDRA": {"biomass": 0.40, "diversity": 0.50, "connectivity": 0.60, "energy_flux": 0.30, "resilience": 0.50},
    "DESERT": {"biomass": 0.20, "diversity": 0.30, "connectivity": 0.40, "energy_flux": 0.20, "resilience": 0.40},
    "AGRICULTURE": {"biomass": 0.90, "diversity": 0.10, "connectivity": 0.20, "energy_flux": 0.80, "resilience": 0.10},
    "KEYSTONE_LOSS": {"biomass": 0.80, "diversity": 0.70, "connectivity": EPSILON, "energy_flux": 0.50, "resilience": 0.30},
    "TROPHIC_CASCADE": {"biomass": 0.40, "diversity": 0.30, "connectivity": EPSILON, "energy_flux": EPSILON, "resilience": EPSILON},
    "ALGAL_BLOOM": {"biomass": 1.0, "diversity": EPSILON, "connectivity": EPSILON, "energy_flux": 0.40, "resilience": EPSILON},
    "MASS_EXTINCTION": {"biomass": EPSILON, "diversity": EPSILON, "connectivity": EPSILON, "energy_flux": EPSILON, "resilience": EPSILON},
    "PIONEER_STAGE": {"biomass": 0.10, "diversity": 0.20, "connectivity": 0.10, "energy_flux": 0.90, "resilience": 0.90},
    "URBAN_GRID": {"biomass": 0.05, "diversity": 0.10, "connectivity": 0.05, "energy_flux": 0.10, "resilience": 0.80},
}

def analyze_ecology_state(name: str):
    data = ECOLOGICAL_STATES[name]
    c = np.array([
        data["biomass"], 
        data["diversity"], 
        data["connectivity"],
        data["energy_flux"],
        data["resilience"]
    ])
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(5) / 5
    return compute_kernel_outputs(c, w)
"""

with open("closures/ecology/ecology_kernel.py", "w") as f:
    f.write(ecology_kernel)


linguistics_kernel = """import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# Linguistic States mapped to the 0-1 continuous kernel space.
# phonology: 1.0 (stable sound laws) to EPSILON (rapid unprincipled drift/loss)
# semantics: 1.0 (stable meaning) to EPSILON (bleached/lost)
# syntax: 1.0 (complex grammar) to EPSILON (parataxis/lost)
# pragmatics: 1.0 (high contextual grounding) to EPSILON (isolated strings)
# transmission: 1.0 (full L1 acquisition) to EPSILON (no native speakers)
LINGUISTIC_STATES = {
    "STABLE_L1": {"phonology": 0.95, "semantics": 0.95, "syntax": 0.95, "pragmatics": 0.95, "transmission": 1.0},
    "PHONOLOGICAL_SHIFT": {"phonology": 0.30, "semantics": 0.90, "syntax": 0.95, "pragmatics": 0.90, "transmission": 1.0},
    "SEMANTIC_BLEACHING": {"phonology": 0.90, "semantics": 0.20, "syntax": 0.90, "pragmatics": 0.40, "transmission": 1.0},
    "PIDGIN": {"phonology": 0.60, "semantics": 0.80, "syntax": EPSILON, "pragmatics": 0.80, "transmission": EPSILON},
    "CREOLE": {"phonology": 0.70, "semantics": 0.80, "syntax": 0.80, "pragmatics": 0.80, "transmission": 1.0},
    "MORIBUND": {"phonology": 0.40, "semantics": 0.50, "syntax": 0.30, "pragmatics": 0.20, "transmission": EPSILON},
    "DEAD_LANGUAGE": {"phonology": EPSILON, "semantics": EPSILON, "syntax": EPSILON, "pragmatics": EPSILON, "transmission": EPSILON},
    "LITURGICAL_DEAD": {"phonology": 0.80, "semantics": 0.40, "syntax": 0.80, "pragmatics": EPSILON, "transmission": EPSILON},
    "JARGON": {"phonology": 0.90, "semantics": 0.90, "syntax": EPSILON, "pragmatics": 1.0, "transmission": 0.50},
    "MACHINE_TRANSLATED": {"phonology": 0.95, "semantics": 0.90, "syntax": 0.80, "pragmatics": EPSILON, "transmission": EPSILON},
    "ISOLATE_CONTACT": {"phonology": 0.50, "semantics": 0.50, "syntax": 0.50, "pragmatics": 0.80, "transmission": 0.80},
    "ACROLECT": {"phonology": 0.90, "semantics": 0.80, "syntax": 0.90, "pragmatics": 0.60, "transmission": 0.50},
}

def analyze_language_state(name: str):
    data = LINGUISTIC_STATES[name]
    c = np.array([
        data["phonology"], 
        data["semantics"], 
        data["syntax"],
        data["pragmatics"],
        data["transmission"]
    ])
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(5) / 5
    return compute_kernel_outputs(c, w)
"""

with open("closures/dynamic_semiotics/linguistic_evolution_kernel.py", "w") as f:
    f.write(linguistics_kernel)
