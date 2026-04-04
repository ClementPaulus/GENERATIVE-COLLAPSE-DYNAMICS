import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# Information-Theoretic Entities mapped to the 0-1 continuous kernel space.
# halting_prob: 1.0 (always halts) to EPSILON (undecidable)
# circuit_depth: 1.0 (constant O(1)) to EPSILON (exponential/infinite)
# kolmogorov: 1.0 (low complexity/compressible) to EPSILON (random/incompressible)
# oracle_access: 1.0 (no oracle needed) to EPSILON (requires uncomputable oracle)
# determinism: 1.0 (purely deterministic) to EPSILON (purely non-deterministic)
COMPLEXITY_CLASSES = {
    "O1": {"halting_prob": 1.0, "circuit_depth": 1.0, "kolmogorov": 0.9, "oracle_access": 1.0, "determinism": 1.0},
    "P": {"halting_prob": 1.0, "circuit_depth": 0.8, "kolmogorov": 0.8, "oracle_access": 1.0, "determinism": 1.0},
    "NP": {"halting_prob": 1.0, "circuit_depth": 0.3, "kolmogorov": 0.3, "oracle_access": 1.0, "determinism": EPSILON},
    "co-NP": {"halting_prob": 1.0, "circuit_depth": 0.3, "kolmogorov": 0.3, "oracle_access": 1.0, "determinism": 0.1},
    "BPP": {"halting_prob": 0.9, "circuit_depth": 0.7, "kolmogorov": 0.5, "oracle_access": 1.0, "determinism": 0.5},
    "BQP": {"halting_prob": 0.9, "circuit_depth": 0.6, "kolmogorov": 0.4, "oracle_access": 1.0, "determinism": 0.4},
    "PSPACE": {
        "halting_prob": 1.0,
        "circuit_depth": EPSILON,
        "kolmogorov": 0.2,
        "oracle_access": 1.0,
        "determinism": 1.0,
    },
    "EXPTIME": {
        "halting_prob": 1.0,
        "circuit_depth": EPSILON,
        "kolmogorov": 0.1,
        "oracle_access": 1.0,
        "determinism": 1.0,
    },
    "RE": {
        "halting_prob": EPSILON,
        "circuit_depth": EPSILON,
        "kolmogorov": EPSILON,
        "oracle_access": 1.0,
        "determinism": 1.0,
    },
    "co-RE": {
        "halting_prob": EPSILON,
        "circuit_depth": EPSILON,
        "kolmogorov": EPSILON,
        "oracle_access": 1.0,
        "determinism": 1.0,
    },
    "HALT": {
        "halting_prob": EPSILON,
        "circuit_depth": EPSILON,
        "kolmogorov": EPSILON,
        "oracle_access": EPSILON,
        "determinism": 1.0,
    },
    "R": {"halting_prob": 1.0, "circuit_depth": EPSILON, "kolmogorov": 0.5, "oracle_access": 1.0, "determinism": 1.0},
}


def analyze_complexity_class(name: str):
    data = COMPLEXITY_CLASSES[name]
    c = np.array(
        [data["halting_prob"], data["circuit_depth"], data["kolmogorov"], data["oracle_access"], data["determinism"]]
    )
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(5) / 5
    return compute_kernel_outputs(c, w)
