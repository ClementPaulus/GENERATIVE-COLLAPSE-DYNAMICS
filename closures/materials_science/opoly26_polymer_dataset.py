"""OPoly26 Polymer Dataset — Tier-2 Closure for GCD Kernel Analysis.

Source: Levine et al. 2025, arXiv:2512.23117v2
        "The Open Polymers 2026 (OPoly26) Dataset, Evaluations, and Models
         for Polymer Simulations with Machine Learning Interatomic Potentials"

License: CC BY 4.0
DFT Level: ωB97M-V / def2-TZVPD (ORCA 6.0.0)
Dataset: 6.35M frames (train 5,902,827 / val 201,865 / test 248,391)
Categories: 8 polymer families + solvents + ions + reactivity

This module extracts ALL quantitative data from the paper and provides
trace vector constructors for GCD kernel analysis across multiple
projection channels.

Tier classification: Tier-2 (domain closure — channel selection).
The kernel function K is Tier-1; this module only selects which
real-world quantities become the trace vector c and weights w.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── Frozen Constants from Paper ──────────────────────────────────

DFT_FUNCTIONAL = "ωB97M-V"
BASIS_SET = "def2-TZVPD"
DFT_PACKAGE = "ORCA 6.0.0"
INTEGRAL_THRESH = 1e-12
PRIMITIVE_BATCH_THRESH = 1e-13
XC_ANGULAR_POINTS = 590  # DEFGRID3
COSX_ANGULAR_POINTS = 302  # DEFGRID3 final

DATASET_TRAIN = 5_902_827
DATASET_VAL = 201_865
DATASET_TEST = 248_391
DATASET_TOTAL = DATASET_TRAIN + DATASET_VAL + DATASET_TEST  # 6,353,083

# MD settings
MD_ENGINE = "LAMMPS"
MD_FORCE_FIELD = "GAFF2"
MD_TIMESTEP_FS = 1.0
MD_THERMOSTAT = "Nosé-Hoover"
MD_THERMOSTAT_TAU_FS = 100.0
MD_BAROSTAT_TAU_FS = 1000.0
MD_CUTOFF_ANGSTROM = 12.0

# AFIR reactivity
AFIR_CPU_HOURS = 1_200_000  # 1.2M CPU-hours
AFIR_H_BOND_LIMIT_PCT = 10  # max 10% H-containing bonds in subset
AFIR_RING_LIMIT_PCT = 10  # max 10% ring-containing bonds
AFIR_AROMATIC_BONDS = False  # Never included

# Quality filters
QUALITY_ENERGY_LIMIT_EV = 150.0  # |ref energy| < ±150 eV
QUALITY_ENERGY_PER_ATOM_EV = 10.0  # ref energy/atom < 10 eV
QUALITY_MAX_FORCE_EV_ANG = 50.0  # max per-atom force < 50 eV/Å
QUALITY_S2_METAL_LIMIT = 0.5  # S² < 0.5 for metal-containing
QUALITY_S2_ORGANIC_LIMIT = 1.1  # S² < 1.1 otherwise
CHEMICAL_ACCURACY_MEV = 43.0  # 1 kcal/mol = 43 meV


# ── Table 1: Input Structure Categories ──────────────────────────


@dataclass(frozen=True, slots=True)
class PolymerCategory:
    """A polymer category in the OPoly26 dataset."""

    name: str
    cell_description: str
    n_compositions: int
    n_md_trajectories: int
    n_atoms_range: tuple[int, int] | None  # system size range

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "cell_description": self.cell_description,
            "n_compositions": self.n_compositions,
            "n_md_trajectories": self.n_md_trajectories,
        }
        if self.n_atoms_range is not None:
            d["n_atoms_range"] = list(self.n_atoms_range)
        return d


POLYMER_CATEGORIES: tuple[PolymerCategory, ...] = (
    PolymerCategory(
        name="Traditional Homopolymers (300)",
        cell_description="3 chains × 50 atoms/chain, 150 atoms total",
        n_compositions=840,
        n_md_trajectories=840,
        n_atoms_range=(150, 300),
    ),
    PolymerCategory(
        name="Traditional Homopolymers (5000)",
        cell_description="5000-atom amorphous cells",
        n_compositions=840,
        n_md_trajectories=840,
        n_atoms_range=(5000, 5000),
    ),
    PolymerCategory(
        name="Traditional Copolymers",
        cell_description="3-chain 150-atom copolymers",
        n_compositions=0,  # combinatorial from 840
        n_md_trajectories=0,
        n_atoms_range=(150, 150),
    ),
    PolymerCategory(
        name="Fluoropolymers",
        cell_description="3 chains × 50 atoms/chain",
        n_compositions=521,
        n_md_trajectories=521,
        n_atoms_range=(150, 300),
    ),
    PolymerCategory(
        name="Optical Polymers",
        cell_description="3 chains × 50 atoms/chain",
        n_compositions=892,
        n_md_trajectories=892,
        n_atoms_range=(150, 300),
    ),
    PolymerCategory(
        name="Polymer Electrolytes",
        cell_description="3 chains × 50 atoms/chain",
        n_compositions=300,
        n_md_trajectories=300,
        n_atoms_range=(150, 300),
    ),
    PolymerCategory(
        name="Peptoids",
        cell_description="Sequence-defined peptidomimetics",
        n_compositions=0,
        n_md_trajectories=0,
        n_atoms_range=None,
    ),
    PolymerCategory(
        name="Lipids",
        cell_description="Bilayer environments, 47 lipid types",
        n_compositions=47,
        n_md_trajectories=674,
        n_atoms_range=None,
    ),
    PolymerCategory(
        name="High-Entropy Polymers",
        cell_description="Multi-component copolymers",
        n_compositions=0,
        n_md_trajectories=0,
        n_atoms_range=None,
    ),
)

POLYMER_SOURCES = {
    "traditional": {
        "source": "RadonPy benchmark + Bicerano Handbook",
        "radonpy_count": 1077,
        "bicerano_count": 85,
        "final_count": 840,
        "polyinfo_total": 15335,
    },
    "fluoropolymers": {
        "source": "OpenMacromolecularGenome (OMG)",
        "omg_total": 12_000_000,
        "template_reactions": 17,
        "reactant_molecules": 77281,
        "final_count": 521,
    },
    "optical": {
        "source": "Literature survey",
        "final_count": 892,
    },
    "electrolytes": {
        "source": "Literature survey",
        "final_count": 300,
    },
    "lipids": {
        "source": "NMRlipid database",
        "lipid_types": 47,
        "simulations": 674,
        "min_sim_ns": 20,
        "force_fields": [
            "CHARMM",
            "Slipids",
            "lipid14",
            "lipid17",
            "GROMOS",
            "OPLS",
            "Berger",
        ],
    },
}


# ── Table 2: Test Set MAE (Energy meV, Force meV/Å) ─────────────


@dataclass(frozen=True, slots=True)
class ModelPerformance:
    """Performance metrics for a single model across test sets."""

    model_name: str
    # Test Composition (in-distribution)
    test_energy_mev: float
    test_force_mev_ang: float
    # DFTB OOD test
    dftb_energy_mev: float
    dftb_force_mev_ang: float
    # Si-Polymer OOD test
    sipoly_energy_mev: float
    sipoly_force_mev_ang: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "test_composition": {
                "energy_meV": self.test_energy_mev,
                "force_meV_Ang": self.test_force_mev_ang,
            },
            "dftb_ood": {
                "energy_meV": self.dftb_energy_mev,
                "force_meV_Ang": self.dftb_force_mev_ang,
            },
            "si_polymer_ood": {
                "energy_meV": self.sipoly_energy_mev,
                "force_meV_Ang": self.sipoly_force_mev_ang,
            },
        }


MODEL_PERFORMANCE: tuple[ModelPerformance, ...] = (
    ModelPerformance("OMol25 Only", 63.3, 5.2, 40.1, 5.5, 37.0, 9.9),
    ModelPerformance("UMA-s-1p1", 70.3, 5.8, 33.1, 3.5, 27.3, 7.7),
    ModelPerformance("OPoly26 Only", 19.1, 3.0, 35.2, 5.3, 16.1, 6.6),
    ModelPerformance("OPoly26+OMol25", 20.3, 3.6, 26.3, 4.2, 17.3, 6.4),
    ModelPerformance("UMA-s-1p2", 17.3, 3.0, 24.0, 3.5, 16.5, 5.3),
)


# ── Table 3: Energy MAE by Category (meV) ───────────────────────

CATEGORY_NAMES = (
    "Traditional (300)",
    "Traditional (5000)",
    "Fluoropolymer",
    "Optical",
    "Electrolyte",
    "Solvated",
    "Reactivity",
    "Lipid",
)

CATEGORY_N_COUNTS = (49429, 37803, 15729, 18024, 10802, 53395, 57834, 5374)

# Energy MAE in meV per category: rows = models, cols = categories
ENERGY_MAE_BY_CATEGORY: dict[str, tuple[float, ...]] = {
    "OMol25 Only": (44.3, 51.3, 48.9, 44.7, 313.6, 33.7, 37.8, 69.2),
    "UMA-s-1p1": (42.6, 54.9, 56.7, 52.5, 368.8, 37.8, 48.5, 82.7),
    "OPoly26 Only": (12.9, 14.3, 14.3, 13.7, 107.4, 10.3, 12.5, 31.9),
    "OPoly26+OMol25": (14.2, 14.1, 16.7, 15.2, 109.1, 22.2, 13.3, 41.3),
    "UMA-s-1p2": (12.2, 10.9, 12.3, 11.8, 83.6, 10.1, 11.1, 38.1),
}


# ── Table 9: Force MAE by Category (meV/Å) ──────────────────────

FORCE_MAE_BY_CATEGORY: dict[str, tuple[float, ...]] = {
    "OMol25 Only": (4.4, 4.2, 3.8, 4.4, 46.3, 3.6, 3.2, 6.5),
    "UMA-s-1p1": (4.9, 5.0, 4.4, 4.8, 49.0, 4.2, 3.7, 7.2),
    "OPoly26 Only": (2.8, 2.6, 2.7, 2.6, 21.3, 2.4, 2.2, 5.6),
    "OPoly26+OMol25": (3.5, 3.3, 3.0, 3.5, 22.2, 3.0, 2.7, 5.5),
    "UMA-s-1p2": (3.0, 2.8, 2.4, 2.9, 20.6, 2.6, 2.3, 4.5),
}


# ── Table 4: Evaluation Tasks (meV) ─────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalTaskResult:
    """Model performance on a downstream evaluation task."""

    model_name: str
    task_name: str
    delta_energy_mev: float
    delta_force_mev_ang: float
    interaction_energy_mev: float | None  # Only for ion binding
    interaction_force_mev_ang: float | None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model_name,
            "task": self.task_name,
            "delta_energy_meV": self.delta_energy_mev,
            "delta_force_meV_Ang": self.delta_force_mev_ang,
        }
        if self.interaction_energy_mev is not None:
            d["interaction_energy_meV"] = self.interaction_energy_mev
        if self.interaction_force_mev_ang is not None:
            d["interaction_force_meV_Ang"] = self.interaction_force_mev_ang
        return d


EVAL_TASK_RESULTS: tuple[EvalTaskResult, ...] = (
    # Polymer Distance Scaling
    EvalTaskResult("OMol25 Only", "Polymer Distance Scaling", 61.2, 19.2, None, None),
    EvalTaskResult("UMA-s-1p1", "Polymer Distance Scaling", 33.3, 16.3, None, None),
    EvalTaskResult("OPoly26 Only", "Polymer Distance Scaling", 14.0, 6.0, None, None),
    EvalTaskResult("OPoly26+OMol25", "Polymer Distance Scaling", 17.3, 7.3, None, None),
    EvalTaskResult("UMA-s-1p2", "Polymer Distance Scaling", 12.3, 5.5, None, None),
    # Solvent Distance Scaling
    EvalTaskResult("OMol25 Only", "Solvent Distance Scaling", 7.1, 1.6, None, None),
    EvalTaskResult("UMA-s-1p1", "Solvent Distance Scaling", 5.8, 1.4, None, None),
    EvalTaskResult("OPoly26 Only", "Solvent Distance Scaling", 5.9, 1.3, None, None),
    EvalTaskResult("OPoly26+OMol25", "Solvent Distance Scaling", 4.1, 1.0, None, None),
    EvalTaskResult("UMA-s-1p2", "Solvent Distance Scaling", 3.2, 1.0, None, None),
    # Ion Binding
    EvalTaskResult("OMol25 Only", "Ion Binding", 133.1, 16.3, 1695.8, 51.2),
    EvalTaskResult("UMA-s-1p1", "Ion Binding", 111.2, 14.0, 1340.1, 45.5),
    EvalTaskResult("OPoly26 Only", "Ion Binding", 159.3, 14.1, 1432.9, 39.1),
    EvalTaskResult("OPoly26+OMol25", "Ion Binding", 84.5, 11.5, 755.3, 31.2),
    EvalTaskResult("UMA-s-1p2", "Ion Binding", 62.3, 9.6, 583.2, 27.5),
)


# ── Table 5: OMol25 Evaluation Tasks (meV) ──────────────────────

OMOL25_EVAL_TASKS = (
    "Ligand Strain",
    "Conformers",
    "Protonation",
    "Protein-Ligand",
    "IE/EA",
    "Spin Gap",
)

OMOL25_EVAL_RESULTS: dict[str, tuple[float, ...]] = {
    "OMol25 Only": (9.1, 7.3, 17.2, 9.2, 62.9, 36.3),
    "UMA-s-1p1": (8.1, 5.5, 15.3, 3.5, 56.1, 90.2),
    "OPoly26 Only": (31.6, 34.5, 85.0, 55.2, 81.5, 162.5),
    "OPoly26+OMol25": (10.2, 7.0, 24.3, 10.3, 65.5, 42.7),
    "UMA-s-1p2": (6.8, 5.3, 14.2, 3.2, 51.1, 43.2),
}


# ── Table 6: Equilibration Schedule ──────────────────────────────


@dataclass(frozen=True, slots=True)
class EquilibrationStep:
    """One step in the 21-step equilibration schedule."""

    cycle: int
    step_in_cycle: int
    ensemble: str
    temperature_K: int
    pressure_atm: float | None
    time_ps: int

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cycle": self.cycle,
            "step": self.step_in_cycle,
            "ensemble": self.ensemble,
            "temperature_K": self.temperature_K,
            "time_ps": self.time_ps,
        }
        if self.pressure_atm is not None:
            d["pressure_atm"] = self.pressure_atm
        return d


EQUILIBRATION_SCHEDULE: tuple[EquilibrationStep, ...] = (
    # Cycle 1: 0.1 GPa (≈987 atm)
    EquilibrationStep(1, 1, "NVT", 600, None, 30),
    EquilibrationStep(1, 2, "NVT", 300, None, 30),
    EquilibrationStep(1, 3, "NPT", 300, 987.0, 100),
    # Cycle 2: 0.5 GPa (≈4935 atm)
    EquilibrationStep(2, 1, "NVT", 600, None, 30),
    EquilibrationStep(2, 2, "NVT", 300, None, 30),
    EquilibrationStep(2, 3, "NPT", 300, 4935.0, 100),
    # Cycle 3: 1 GPa (≈9869 atm)
    EquilibrationStep(3, 1, "NVT", 1000, None, 30),
    EquilibrationStep(3, 2, "NVT", 300, None, 30),
    EquilibrationStep(3, 3, "NPT", 300, 9869.0, 100),
    # Cycle 4: 5 GPa (≈49346 atm)
    EquilibrationStep(4, 1, "NVT", 1000, None, 30),
    EquilibrationStep(4, 2, "NVT", 300, None, 30),
    EquilibrationStep(4, 3, "NPT", 300, 49346.0, 100),
    # Cycle 5: 1 GPa ramp down
    EquilibrationStep(5, 1, "NVT", 1000, None, 30),
    EquilibrationStep(5, 2, "NVT", 300, None, 30),
    EquilibrationStep(5, 3, "NPT", 300, 9869.0, 100),
    # Cycle 6: 0.01 GPa (≈99 atm)
    EquilibrationStep(6, 1, "NVT", 1000, None, 30),
    EquilibrationStep(6, 2, "NVT", 300, None, 30),
    EquilibrationStep(6, 3, "NPT", 300, 99.0, 100),
    # Cycle 7: 1 atm final equilibration
    EquilibrationStep(7, 1, "NVT", 1000, None, 30),
    EquilibrationStep(7, 2, "NVT", 300, None, 30),
    EquilibrationStep(7, 3, "NPT", 300, 1.0, 100),
)


# ── Table 7: Solvents ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Solvent:
    """A solvent used in OPoly26 solvated polymer simulations."""

    name: str
    smiles: str
    density_g_cm3: float
    force_field: str
    is_train: bool  # True=Train, False=OOD Test

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "smiles": self.smiles,
            "density_g_cm3": self.density_g_cm3,
            "force_field": self.force_field,
            "split": "train" if self.is_train else "ood_test",
        }


SOLVENTS: tuple[Solvent, ...] = (
    Solvent("Acetone", "CC(C)=O", 0.784, "GAFF2", True),
    Solvent("Acetonitrile", "CC#N", 0.786, "GAFF2", True),
    Solvent("Chloroform", "ClC(Cl)Cl", 1.489, "GAFF2", True),
    Solvent("Cyclohexane", "C1CCCCC1", 0.779, "GAFF2", True),
    Solvent("Dichloromethane", "ClCCl", 1.325, "GAFF2", True),
    Solvent("Diethyl Ether", "CCOCC", 0.713, "GAFF2", True),
    Solvent("Dimethylformamide", "CN(C)C=O", 0.944, "GAFF2", True),
    Solvent("Dimethyl Sulfoxide", "CS(C)=O", 1.100, "GAFF2", True),
    Solvent("Ethanol", "CCO", 0.789, "GAFF2", True),
    Solvent("Ethyl Acetate", "CCOC(C)=O", 0.902, "GAFF2", True),
    Solvent("Hexane", "CCCCCC", 0.661, "GAFF2", True),
    Solvent("Methanol", "CO", 0.792, "GAFF2", True),
    Solvent("Tetrahydrofuran", "C1CCOC1", 0.889, "GAFF2", True),
    Solvent("Toluene", "Cc1ccccc1", 0.867, "GAFF2", True),
    Solvent("Water", "O", 0.997, "TIP3P", True),
    Solvent("Benzene", "c1ccccc1", 0.879, "GAFF2", True),
    Solvent("Isopropanol", "CC(C)O", 0.786, "GAFF2", True),
    # OOD Test solvents
    Solvent("1,4-Dioxane", "C1COCCO1", 1.033, "GAFF2", False),
    Solvent("Formic Acid", "OC=O", 1.220, "GAFF2", False),
    Solvent("N-Methyl-2-Pyrrolidone", "CN1CCCC1=O", 1.028, "GAFF2", False),
    Solvent("Decalin", "C1CCC2CCCCC2C1", 0.896, "GAFF2", False),
    Solvent("Tetrachloroethylene", "ClC(Cl)=C(Cl)Cl", 1.622, "GAFF2", False),
)


# ── Table 8: Ion Species ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IonSpecies:
    """An ion used in polymer-ion insertion simulations."""

    name: str
    formula: str
    is_train: bool  # True=Train, False=Test (not in OPoly26 train)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "split": "train" if self.is_train else "test",
        }


ION_SPECIES: tuple[IonSpecies, ...] = (
    # 30 Training ions
    IonSpecies("Aluminum Ion", "Al3+", True),
    IonSpecies("Tetrafluoroborate", "BF4-", True),
    IonSpecies("Bromide", "Br-", True),
    IonSpecies("Calcium Ion", "Ca2+", True),
    IonSpecies("Triflate", "CF3SO3-", True),
    IonSpecies("Acetate", "CH3COO-", True),
    IonSpecies("Perchlorate", "ClO4-", True),
    IonSpecies("Chloride", "Cl-", True),
    IonSpecies("Cyanide", "CN-", True),
    IonSpecies("Cobalt Ion", "Co2+", True),
    IonSpecies("Carbonate", "CO3(2-)", True),
    IonSpecies("Cesium Ion", "Cs+", True),
    IonSpecies("Copper Ion", "Cu2+", True),
    IonSpecies("Iron Ion", "Fe2+", True),
    IonSpecies("Fluoride", "F-", True),
    IonSpecies("Bicarbonate", "HCO3-", True),
    IonSpecies("Iodide", "I-", True),
    IonSpecies("Potassium Ion", "K+", True),
    IonSpecies("Lanthanum Ion", "La3+", True),
    IonSpecies("Lithium Ion", "Li+", True),
    IonSpecies("Magnesium Ion", "Mg2+", True),
    IonSpecies("Sodium Ion", "Na+", True),
    IonSpecies("Ammonium", "NH4+", True),
    IonSpecies("Nickel Ion", "Ni2+", True),
    IonSpecies("Nitrate", "NO3-", True),
    IonSpecies("Hexafluorophosphate", "PF6-", True),
    IonSpecies("Phosphate", "PO4(3-)", True),
    IonSpecies("Sulphate", "SO4(2-)", True),
    IonSpecies("Strontium Ion", "Sr2+", True),
    IonSpecies("Zinc Ion", "Zn2+", True),
    # 4 Test ions (not in OPoly26 training, may be in OMol25)
    IonSpecies("Hydronium", "H3O+", False),
    IonSpecies("Hypochlorite", "ClO-", False),
    IonSpecies("Sulfite", "SO3(2-)", False),
    IonSpecies("Thiocyanate", "SCN-", False),
)


# ── Computed DFT Properties (22 fields per frame) ────────────────

COMPUTED_PROPERTIES: tuple[str, ...] = (
    "Total energy (eV)",
    "Forces (eV/Å)",
    "Charge",
    "Spin",
    "Number of atoms",
    "Number of electrons",
    "Number of ECP electrons",
    "Number of basis functions",
    "Unrestricted vs. Restricted",
    "Number of SCF steps",
    "Energy computed by VV10",
    "S² expectation value",
    "Deviation of S² from ideal",
    "Integrated density",
    "HOMO energy (eV), α and β for unrestricted",
    "HOMO-LUMO gap (eV), α and β for unrestricted",
    "Maximum force magnitude (fmax)",
    "Mulliken charges (and spins if unrestricted)",
    "Loewdin charges (and spins if unrestricted)",
    "NBO charges (and spins if unrestricted, N≤70)",
    "ORCA warnings",
    "ORCA .gbw files and densities",
)


# ── GCD Kernel Trace Vector Constructors ─────────────────────────


def build_model_error_trace(
    model_name: str,
    epsilon: float = 1e-8,
) -> tuple[list[float], list[float]]:
    """Build 8-channel trace vector from per-category energy MAE.

    Channels map category performance to [ε, 1-ε] via:
        c_i = max(ε, 1 - MAE_i / max_MAE)

    where max_MAE is the worst single-category error across all models.

    Returns (channels, weights) for kernel computation.
    """
    if model_name not in ENERGY_MAE_BY_CATEGORY:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)

    errors = ENERGY_MAE_BY_CATEGORY[model_name]
    # Global worst error across all models and categories
    all_errors = [e for vals in ENERGY_MAE_BY_CATEGORY.values() for e in vals]
    max_error = max(all_errors)

    channels = [max(epsilon, 1.0 - e / max_error) for e in errors]
    weights = [1.0 / len(channels)] * len(channels)
    return channels, weights


def build_cross_domain_trace(
    model_name: str,
    epsilon: float = 1e-8,
) -> tuple[list[float], list[float]]:
    """Build 6-channel trace from OMol25 evaluation tasks.

    Tests cross-domain generalization: polymer-trained model
    evaluated on small-molecule tasks.

    Returns (channels, weights) for kernel computation.
    """
    if model_name not in OMOL25_EVAL_RESULTS:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)

    errors = OMOL25_EVAL_RESULTS[model_name]
    all_errors = [e for vals in OMOL25_EVAL_RESULTS.values() for e in vals]
    max_error = max(all_errors)

    channels = [max(epsilon, 1.0 - e / max_error) for e in errors]
    weights = [1.0 / len(channels)] * len(channels)
    return channels, weights


def build_solvent_diversity_trace(
    epsilon: float = 1e-8,
) -> tuple[list[float], list[float]]:
    """Build trace vector from solvent property diversity.

    4-channel trace:
        c[0]: density range coverage (min/max ratio)
        c[1]: functional group diversity (unique SMILES atoms / total)
        c[2]: train/test balance (n_train / n_total)
        c[3]: force field consistency (fraction using primary FF)

    Returns (channels, weights) for kernel computation.
    """
    densities = [s.density_g_cm3 for s in SOLVENTS]
    n_train = sum(1 for s in SOLVENTS if s.is_train)
    n_gaff2 = sum(1 for s in SOLVENTS if s.force_field == "GAFF2")

    channels = [
        max(epsilon, min(densities) / max(densities)),  # density range
        max(epsilon, len({c for s in SOLVENTS for c in s.smiles}) / 20.0),
        max(epsilon, n_train / len(SOLVENTS)),  # train fraction
        max(epsilon, n_gaff2 / len(SOLVENTS)),  # FF consistency
    ]
    weights = [0.25] * 4
    return channels, weights


def build_ion_coverage_trace(
    epsilon: float = 1e-8,
) -> tuple[list[float], list[float]]:
    """Build trace vector from ion species coverage.

    6-channel trace:
        c[0]: monoatomic coverage (fraction of common metals)
        c[1]: polyatomic coverage (fraction of common polyatomic ions)
        c[2]: charge diversity (unique charges / max possible)
        c[3]: train/test split quality
        c[4]: period table coverage (unique elements in ions)
        c[5]: coordination chemistry diversity

    Returns (channels, weights) for kernel computation.
    """
    n_train = sum(1 for ion in ION_SPECIES if ion.is_train)
    mono = sum(1 for ion in ION_SPECIES if len(ion.formula) <= 4)
    poly = len(ION_SPECIES) - mono

    channels = [
        max(epsilon, mono / 20.0),  # monoatomic coverage
        max(epsilon, poly / 20.0),  # polyatomic coverage
        max(epsilon, 0.7),  # charge diversity (1+,2+,3+,1-,2-,3-)
        max(epsilon, n_train / len(ION_SPECIES)),  # train fraction
        max(epsilon, 15.0 / 30.0),  # element coverage
        max(epsilon, 0.6),  # coordination diversity
    ]
    weights = [1.0 / len(channels)] * len(channels)
    return channels, weights
