#!/usr/bin/env python3
"""Generate high-resolution diagnostic charts across all entity-bearing closures.

Produces 8 publication-quality PNGs into images/:
  1. heterogeneity_gap_landscape.png   — Δ = F − IC across all domains (strip chart)
  2. regime_mosaic.png                 — Regime classification heatmap (entities × domains)
  3. channel_radar_gallery.png         — Per-domain radar overlays (strongest vs weakest)
  4. geometric_slaughter_map.png       — Channel-level IC contribution (who kills IC?)
  5. fidelity_integrity_scatter.png    — F vs IC scatter with IC = F diagonal (all domains)
  6. sensitivity_stability_comparison.png — Sensitivity analysis results across 4 deepened closures
  7. omega_distribution_violin.png     — Drift distribution by domain (violin plots)
  8. cross_domain_kernel_heatmap.png   — Full kernel output heatmap (F, ω, S, C, κ, IC) × entities

Usage:
    python scripts/generate_closure_diagnostics.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "images"
OUT_DIR.mkdir(exist_ok=True)

from umcp.kernel_optimized import compute_kernel_outputs

# ── Shared styling (matches generate_figures.py) ───────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STABLE_COLOR = "#2ca02c"
WATCH_COLOR = "#ff7f0e"
COLLAPSE_COLOR = "#d62728"
CRITICAL_COLOR = "#8B0000"
ACCENT_BLUE = "#1f77b4"
ACCENT_PURPLE = "#9467bd"

REGIME_COLORS = {
    "STABLE": STABLE_COLOR,
    "WATCH": WATCH_COLOR,
    "COLLAPSE": COLLAPSE_COLOR,
}

DOMAIN_COLORS = {
    "Standard Model": "#e6194b",
    "Evolution": "#3cb44b",
    "Semiotics": "#4363d8",
    "Awareness": "#f58231",
    "Consciousness": "#911eb4",
    "Atomic Physics": "#42d4f4",
}


# ── Data loading ────────────────────────────────────────────────────────────
@dataclass
class EntityRecord:
    """One entity processed through the kernel."""

    name: str
    domain: str
    trace: np.ndarray
    n_channels: int
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    delta: float  # heterogeneity gap = F - IC
    regime: str


def load_domain(domain_name: str, entities: list[tuple[str, list[float]]]) -> list[EntityRecord]:
    """Run kernel on a list of (name, trace) pairs."""
    eps = 1e-8
    records = []
    for name, trace_list in entities:
        c = np.clip(np.array(trace_list, dtype=np.float64), eps, 1.0 - eps)
        n = len(c)
        w = np.full(n, 1.0 / n, dtype=np.float64)
        out = compute_kernel_outputs(c, w)
        records.append(
            EntityRecord(
                name=name,
                domain=domain_name,
                trace=c,
                n_channels=n,
                F=out["F"],
                omega=out["omega"],
                S=out["S"],
                C=out["C"],
                kappa=out["kappa"],
                IC=out["IC"],
                delta=out["F"] - out["IC"],
                regime=out["regime"],
            )
        )
    return records


def gather_all_entities() -> list[EntityRecord]:
    """Load entities from all 6 entity-bearing closures."""
    all_records: list[EntityRecord] = []

    # 1. Standard Model (fundamental particles)
    from closures.standard_model.subatomic_kernel import FUNDAMENTAL_PARTICLES

    sm_entities = []
    for p in FUNDAMENTAL_PARTICLES:
        mass_log = np.log10(max(p.mass_GeV, 1e-12)) / 6.0
        mass_log = max(0.0, min(1.0, (mass_log + 12) / 18))
        trace = [
            mass_log,
            p.spin / 2.0,
            abs(p.charge_e),
            min(p.color_dof / 3.0, 1.0),
            abs(p.weak_T3) * 2.0,
            float(p.is_fermion),
            min(p.generation / 3.0, 1.0) if p.generation > 0 else 0.0,
            min(abs(p.hypercharge_Y), 1.0),
        ]
        sm_entities.append((p.name, trace))
    all_records.extend(load_domain("Standard Model", sm_entities))

    # 2. Evolution
    from closures.evolution.evolution_kernel import ORGANISMS

    evo_entities = [
        (
            org.name,
            [
                org.genetic_diversity,
                org.morphological_fitness,
                org.reproductive_success,
                org.metabolic_efficiency,
                org.immune_competence,
                org.environmental_breadth,
                org.behavioral_complexity,
                org.lineage_persistence,
            ],
        )
        for org in ORGANISMS
    ]
    all_records.extend(load_domain("Evolution", evo_entities))

    # 3. Semiotics
    from closures.dynamic_semiotics.semiotic_kernel import SIGN_SYSTEMS

    sem_entities = [
        (
            s.name,
            [
                s.sign_repertoire,
                s.interpretant_depth,
                s.ground_stability,
                s.translation_fidelity,
                s.semiotic_density,
                s.indexical_coupling,
                s.iconic_persistence,
                s.symbolic_recursion,
            ],
        )
        for s in SIGN_SYSTEMS
    ]
    all_records.extend(load_domain("Semiotics", sem_entities))

    # 4. Awareness-Cognition
    from closures.awareness_cognition.awareness_kernel import ORGANISM_CATALOG

    awc_entities = [(org.name, list(org.channels)) for org in ORGANISM_CATALOG]
    all_records.extend(load_domain("Awareness", awc_entities))

    # 5. Consciousness-Coherence
    from closures.consciousness_coherence.coherence_kernel import COHERENCE_CATALOG

    con_entities = [
        (
            s.name,
            [
                s.harmonic_ratio,
                s.recursive_depth,
                s.return_fidelity,
                s.spectral_coherence,
                s.phase_stability,
                s.information_density,
                s.temporal_persistence,
                s.cross_scale_coupling,
            ],
        )
        for s in COHERENCE_CATALOG
    ]
    all_records.extend(load_domain("Consciousness", con_entities))

    # 6. Atomic Physics (sample: first 30 elements for readability)
    from closures.atomic_physics.periodic_kernel import _normalize_element
    from closures.materials_science.element_database import ELEMENTS

    atom_entities = []
    for el in ELEMENTS[:30]:
        try:
            c, _w, _labels = _normalize_element(el)
            atom_entities.append((el.symbol, list(c)))
        except Exception:
            pass
    all_records.extend(load_domain("Atomic Physics", atom_entities))

    return all_records


# ── Chart 1: Heterogeneity Gap Landscape ────────────────────────────────────
def chart_heterogeneity_gap(records: list[EntityRecord]) -> None:
    """Strip chart: Δ = F − IC for every entity, grouped by domain."""
    fig, ax = plt.subplots(figsize=(14, 7))

    domains = list(DOMAIN_COLORS.keys())
    domain_data: dict[str, list[EntityRecord]] = {d: [] for d in domains}
    for r in records:
        if r.domain in domain_data:
            domain_data[r.domain].append(r)

    y_positions = []
    y_labels = []
    colors = []
    deltas = []
    fidelities = []
    regimes = []

    y = 0
    for domain in domains:
        ents = sorted(domain_data[domain], key=lambda x: x.delta, reverse=True)
        for e in ents:
            y_positions.append(y)
            y_labels.append(e.name)
            colors.append(DOMAIN_COLORS[domain])
            deltas.append(e.delta)
            fidelities.append(e.F)
            regimes.append(e.regime)
            y += 1
        y += 1  # gap between domains

    deltas_arr = np.array(deltas)
    fid_arr = np.array(fidelities)

    # Main horizontal bars
    bars = ax.barh(y_positions, deltas_arr, color=colors, alpha=0.8, height=0.7, edgecolor="white", linewidth=0.3)

    # Regime markers on right edge
    for i, (yp, d, reg) in enumerate(zip(y_positions, deltas, regimes)):
        marker_color = REGIME_COLORS.get(reg, "#999999")
        ax.plot(d + 0.005, yp, "o", color=marker_color, markersize=4, zorder=5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel("Heterogeneity Gap  Δ = F − IC", fontsize=13)
    ax.set_title("Heterogeneity Gap Landscape — All Entity-Bearing Closures", fontsize=15, fontweight="bold")

    # Domain legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=d) for d, c in DOMAIN_COLORS.items()]
    legend_elements.extend(
        [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=STABLE_COLOR, markersize=8, label="Stable"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=WATCH_COLOR, markersize=8, label="Watch"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLLAPSE_COLOR, markersize=8, label="Collapse"),
        ]
    )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2)
    ax.invert_yaxis()

    fig.tight_layout()
    path = OUT_DIR / "heterogeneity_gap_landscape.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name} ({len(records)} entities)")


# ── Chart 2: F vs IC Scatter ────────────────────────────────────────────────
def chart_fidelity_integrity_scatter(records: list[EntityRecord]) -> None:
    """F vs IC scatter with IC = F diagonal and regime coloring."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # IC = F line (integrity bound)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="IC = F (integrity bound)")

    # Regime gate lines
    ax.axvline(x=0.90, color=STABLE_COLOR, alpha=0.3, linestyle=":", linewidth=1)
    ax.axhline(y=0.30, color=CRITICAL_COLOR, alpha=0.3, linestyle=":", linewidth=1, label="IC critical = 0.30")

    for r in records:
        color = DOMAIN_COLORS.get(r.domain, "#999999")
        edge = REGIME_COLORS.get(r.regime, "#999999")
        ax.scatter(r.F, r.IC, c=color, edgecolors=edge, s=60, alpha=0.85, linewidths=1.5, zorder=4)

    # Annotate extreme points
    max_gap = max(records, key=lambda x: x.delta)
    min_gap = min(records, key=lambda x: x.delta)
    ax.annotate(
        f"{max_gap.name}\nΔ={max_gap.delta:.3f}",
        xy=(max_gap.F, max_gap.IC),
        xytext=(max_gap.F + 0.05, max_gap.IC + 0.05),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax.annotate(
        f"{min_gap.name}\nΔ={min_gap.delta:.4f}",
        xy=(min_gap.F, min_gap.IC),
        xytext=(min_gap.F - 0.15, min_gap.IC - 0.08),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    ax.set_xlabel("Fidelity  F", fontsize=13)
    ax.set_ylabel("Integrity Composite  IC = exp(κ)", fontsize=13)
    ax.set_title("Fidelity vs Integrity — IC ≤ F Everywhere", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=d) for d, c in DOMAIN_COLORS.items()]
    handles.append(plt.Line2D([0], [0], linestyle="--", color="k", alpha=0.4, label="IC = F bound"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "fidelity_integrity_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Chart 3: Kernel Heatmap ─────────────────────────────────────────────────
def chart_kernel_heatmap(records: list[EntityRecord]) -> None:
    """Heatmap of (F, ω, S, C, κ, IC) across all entities, grouped by domain."""
    fields = ["F", "omega", "S", "C", "IC", "delta"]
    field_labels = ["F", "ω", "S", "C", "IC", "Δ"]

    # Sort by domain then by F within domain
    domains = list(DOMAIN_COLORS.keys())
    sorted_records = []
    domain_boundaries = []
    for domain in domains:
        group = sorted([r for r in records if r.domain == domain], key=lambda x: -x.F)
        domain_boundaries.append((len(sorted_records), domain))
        sorted_records.extend(group)

    n = len(sorted_records)
    matrix = np.zeros((n, len(fields)))
    names = []
    for i, r in enumerate(sorted_records):
        matrix[i] = [r.F, r.omega, r.S, r.C, r.IC, r.delta]
        names.append(r.name)

    fig, ax = plt.subplots(figsize=(8, max(16, n * 0.18)))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(field_labels)))
    ax.set_xticklabels(field_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=5)

    # Domain separation lines
    for start, domain in domain_boundaries:
        if start > 0:
            ax.axhline(y=start - 0.5, color="white", linewidth=2)
        ax.text(
            len(fields) + 0.3,
            start + 2,
            domain,
            fontsize=8,
            fontweight="bold",
            color=DOMAIN_COLORS.get(domain, "k"),
            va="top",
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.12)
    cbar.set_label("Value [0, 1]", fontsize=10)

    ax.set_title("Cross-Domain Kernel Output Heatmap", fontsize=15, fontweight="bold")

    fig.tight_layout()
    path = OUT_DIR / "cross_domain_kernel_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name} ({n} entities × {len(fields)} fields)")


# ── Chart 4: Omega Distribution Violins ─────────────────────────────────────
def chart_omega_violins(records: list[EntityRecord]) -> None:
    """Violin plots of drift ω by domain."""
    domains = list(DOMAIN_COLORS.keys())
    data = []
    labels = []
    colors_list = []
    for d in domains:
        vals = [r.omega for r in records if r.domain == d]
        if vals:
            data.append(vals)
            labels.append(d.replace(" ", "\n"))
            colors_list.append(DOMAIN_COLORS[d])

    fig, ax = plt.subplots(figsize=(12, 6))

    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True, widths=0.75)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")

    # Overlay individual points with jitter
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            c=colors_list[i],
            s=15,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.3,
            zorder=5,
        )

    # Regime boundaries
    ax.axhline(y=0.038, color=STABLE_COLOR, linestyle="--", alpha=0.5, linewidth=1, label="Stable threshold (ω=0.038)")
    ax.axhline(
        y=0.30, color=COLLAPSE_COLOR, linestyle="--", alpha=0.5, linewidth=1, label="Collapse threshold (ω=0.30)"
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Drift  ω = 1 − F", fontsize=13)
    ax.set_title("Drift Distribution by Domain", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "omega_distribution_violin.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Chart 5: Geometric Slaughter Map ────────────────────────────────────────
def chart_geometric_slaughter(records: list[EntityRecord]) -> None:
    """Show per-channel contribution to IC loss — which channels kill integrity?"""
    # Group by domain, compute per-channel ln(c_i) contribution
    domains = list(DOMAIN_COLORS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        domain_records = [r for r in records if r.domain == domain]
        if not domain_records:
            continue

        # All entities in a domain may have different channel counts (e.g., atomic physics)
        # Use the most common channel count
        channel_counts = [r.n_channels for r in domain_records]
        target_n = max(set(channel_counts), key=channel_counts.count)
        domain_records = [r for r in domain_records if r.n_channels == target_n]

        n_channels = target_n
        # Compute per-channel log contributions
        log_contributions = np.zeros((len(domain_records), n_channels))
        for i, r in enumerate(domain_records):
            eps = 1e-8
            c_clamped = np.clip(r.trace, eps, 1.0 - eps)
            log_contributions[i] = np.log(c_clamped) / n_channels  # weighted by 1/n

        # Mean and std across entities
        mean_contrib = log_contributions.mean(axis=0)
        std_contrib = log_contributions.std(axis=0)

        # Get channel labels
        channel_labels = _get_channel_labels(domain, n_channels)

        x = np.arange(n_channels)
        bars = ax.bar(x, mean_contrib, yerr=std_contrib, color=DOMAIN_COLORS[domain], alpha=0.8, capsize=3)

        # Color the most damaging channel darker
        worst = np.argmin(mean_contrib)
        bars[worst].set_color(CRITICAL_COLOR)
        bars[worst].set_alpha(1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(channel_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean ln(cᵢ)/n", fontsize=9)
        ax.set_title(f"{domain}", fontsize=12, fontweight="bold", color=DOMAIN_COLORS[domain])
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle(
        "Geometric Slaughter Map — Per-Channel IC Contribution (most damaging = red)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    path = OUT_DIR / "geometric_slaughter_map.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _get_channel_labels(domain: str, n: int) -> list[str]:
    """Return channel labels for a domain."""
    labels_map = {
        "Standard Model": ["mass", "spin", "|Q|", "color", "T₃", "fermion", "gen", "Y"],
        "Evolution": ["genetic", "morpho", "reprod", "metabol", "immune", "environ", "behav", "lineage"],
        "Semiotics": ["repertoire", "interpret", "ground", "translate", "density", "indexical", "iconic", "recursive"],
        "Awareness": [
            "mirror",
            "metacog",
            "planning",
            "symbolic",
            "social",
            "sensory",
            "motor",
            "env_tol",
            "reprod",
            "somatic",
        ],
        "Consciousness": [
            "harmonic",
            "recursive",
            "return",
            "spectral",
            "phase",
            "info_dens",
            "temporal",
            "cross_scale",
        ],
        "Atomic Physics": ["Z_norm", "EN", "radius", "IE", "EA", "T_melt", "T_boil", "density"],
    }
    return labels_map.get(domain, [f"ch{i}" for i in range(n)])


# ── Chart 6: Regime Mosaic ──────────────────────────────────────────────────
def chart_regime_mosaic(records: list[EntityRecord]) -> None:
    """Compact regime mosaic — each cell colored by regime."""
    domains = list(DOMAIN_COLORS.keys())
    regime_to_int = {"STABLE": 0, "WATCH": 1, "COLLAPSE": 2}

    fig, axes = plt.subplots(1, len(domains), figsize=(20, 8), gridspec_kw={"width_ratios": [1] * len(domains)})

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        ents = sorted([r for r in records if r.domain == domain], key=lambda x: -x.F)
        if not ents:
            continue

        n = len(ents)
        # Single column heatmap
        regime_vals = np.array([[regime_to_int.get(e.regime, 1)] for e in ents])

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap([STABLE_COLOR, WATCH_COLOR, COLLAPSE_COLOR])
        ax.imshow(regime_vals, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

        ax.set_yticks(range(n))
        ax.set_yticklabels([e.name for e in ents], fontsize=5)
        ax.set_xticks([])
        ax.set_title(domain.replace(" ", "\n"), fontsize=10, fontweight="bold", color=DOMAIN_COLORS[domain])

        # Count regimes
        counts = {r: sum(1 for e in ents if e.regime == r) for r in ["STABLE", "WATCH", "COLLAPSE"]}
        ax.set_xlabel(
            f"S:{counts['STABLE']} W:{counts['WATCH']} C:{counts['COLLAPSE']}",
            fontsize=7,
        )

    fig.suptitle("Regime Classification Mosaic — All Entities", fontsize=15, fontweight="bold")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=STABLE_COLOR, label="Stable"),
        Patch(facecolor=WATCH_COLOR, label="Watch"),
        Patch(facecolor=COLLAPSE_COLOR, label="Collapse"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = OUT_DIR / "regime_mosaic.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Chart 7: Radar Gallery ─────────────────────────────────────────────────
def chart_radar_gallery(records: list[EntityRecord]) -> None:
    """Radar/spider charts: strongest vs weakest entity per domain."""
    domains = list(DOMAIN_COLORS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        ents = [r for r in records if r.domain == domain]
        if not ents:
            continue

        # Filter to most common channel count for comparable radar
        channel_counts = [r.n_channels for r in ents]
        target_n = max(set(channel_counts), key=channel_counts.count)
        ents = [r for r in ents if r.n_channels == target_n]
        if not ents:
            continue

        best = max(ents, key=lambda x: x.IC)
        worst = min(ents, key=lambda x: x.IC)

        n_channels = best.n_channels
        labels = _get_channel_labels(domain, n_channels)
        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        best_vals = best.trace.tolist() + [best.trace[0]]
        worst_vals = worst.trace.tolist() + [worst.trace[0]]

        ax.plot(angles, best_vals, "o-", color=STABLE_COLOR, linewidth=2, markersize=4, label=f"Best IC: {best.name}")
        ax.fill(angles, best_vals, color=STABLE_COLOR, alpha=0.15)
        ax.plot(
            angles, worst_vals, "s-", color=COLLAPSE_COLOR, linewidth=2, markersize=4, label=f"Worst IC: {worst.name}"
        )
        ax.fill(angles, worst_vals, color=COLLAPSE_COLOR, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(domain, fontsize=12, fontweight="bold", color=DOMAIN_COLORS[domain], pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    fig.suptitle(
        "Channel Radar — Highest vs Lowest Integrity Entity per Domain", fontsize=15, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    path = OUT_DIR / "channel_radar_gallery.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Chart 8: Δ vs F Phase Diagram ──────────────────────────────────────────
def chart_delta_phase_diagram(records: list[EntityRecord]) -> None:
    """Phase diagram: Δ (heterogeneity gap) vs F, with iso-IC contours."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Iso-IC contours: IC = F - Δ, so for fixed IC, Δ = F - IC
    f_range = np.linspace(0, 1, 200)
    for ic_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        delta_line = f_range - ic_val
        valid = delta_line >= 0
        ax.plot(f_range[valid], delta_line[valid], ":", color="gray", alpha=0.4, linewidth=0.8)
        # Label at the right edge
        xi = f_range[valid][-1] if np.any(valid) else 1.0
        yi = xi - ic_val
        ax.text(xi + 0.01, yi, f"IC={ic_val:.1f}", fontsize=7, color="gray", alpha=0.6, va="center")

    # Theoretic maximum: Δ = F (when IC → 0)
    ax.plot(f_range, f_range, "-", color="black", alpha=0.2, linewidth=1, label="Δ = F (IC → 0)")
    # Theoretic minimum: Δ = 0 (when IC = F, homogeneous)
    ax.axhline(y=0, color="black", alpha=0.2, linewidth=1)

    # Stable region shading
    ax.axvspan(0.90, 1.0, alpha=0.05, color=STABLE_COLOR)
    ax.text(0.95, 0.02, "F > 0.90", fontsize=8, color=STABLE_COLOR, alpha=0.5, ha="center")

    # Plot entities
    for r in records:
        color = DOMAIN_COLORS.get(r.domain, "#999999")
        edge = REGIME_COLORS.get(r.regime, "#999999")
        size = 30 + r.n_channels * 8  # bigger markers for more channels
        ax.scatter(r.F, r.delta, c=color, edgecolors=edge, s=size, alpha=0.8, linewidths=1.5, zorder=5)

    # Annotate the 5 entities with largest gaps
    top_gap = sorted(records, key=lambda x: x.delta, reverse=True)[:5]
    for r in top_gap:
        ax.annotate(
            r.name,
            xy=(r.F, r.delta),
            xytext=(r.F + 0.03, r.delta + 0.015),
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    # Annotate bottom 3
    bottom = sorted(records, key=lambda x: x.delta)[:3]
    for r in bottom:
        ax.annotate(
            r.name,
            xy=(r.F, r.delta),
            xytext=(r.F - 0.08, r.delta + 0.02),
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    ax.set_xlabel("Fidelity  F", fontsize=13)
    ax.set_ylabel("Heterogeneity Gap  Δ = F − IC", fontsize=13)
    ax.set_title("Δ–F Phase Diagram with Iso-IC Contours", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, max(r.delta for r in records) + 0.05)

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=d) for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "delta_phase_diagram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("GENERATING HIGH-RESOLUTION CLOSURE DIAGNOSTIC CHARTS")
    print("=" * 70)

    print("\nLoading entities from 6 domains...")
    records = gather_all_entities()
    print(f"  Total: {len(records)} entities across {len(set(r.domain for r in records))} domains")

    # Summary stats
    regime_counts = {}
    for r in records:
        regime_counts[r.regime] = regime_counts.get(r.regime, 0) + 1
    for reg, cnt in sorted(regime_counts.items()):
        print(f"    {reg}: {cnt}")
    print(f"  Mean Δ = {np.mean([r.delta for r in records]):.4f}")
    print(f"  Max  Δ = {max(r.delta for r in records):.4f} ({max(records, key=lambda x: x.delta).name})")
    print(f"  Min  Δ = {min(r.delta for r in records):.4f} ({min(records, key=lambda x: x.delta).name})")
    print()

    print("Generating charts...")
    chart_heterogeneity_gap(records)
    chart_fidelity_integrity_scatter(records)
    chart_kernel_heatmap(records)
    chart_omega_violins(records)
    chart_geometric_slaughter(records)
    chart_regime_mosaic(records)
    chart_radar_gallery(records)
    chart_delta_phase_diagram(records)

    print(f"\n✓ All 8 charts saved to {OUT_DIR}/")
    print("  300 DPI publication quality")


if __name__ == "__main__":
    main()
