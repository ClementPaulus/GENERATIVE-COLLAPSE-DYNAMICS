"""Domain Maturity Report — Per-domain metrics for the 20 closure domains.

Reads domain_metadata from closures/registry.yaml, counts actual files and
lines per domain, cross-references theorem_registry.json if available, and
outputs a structured report to derived/domain_maturity.json.

Metrics per domain:
    - archetype (kernel-centric / multi-subsystem / daemon)
    - channels, entities (from registry metadata)
    - file_count, total_lines (from filesystem scan)
    - theorem_count, proven_count, pass_rate (from theorem registry)
    - tier1_bootstrap (True if first theorem verifies kernel identities)
    - naming_conformance (fraction of theorems matching theorem_<PREFIX><N>_*)
    - maturity_score (composite 0–100)

Usage:
    python scripts/domain_maturity.py              # Print table + write JSON
    python scripts/domain_maturity.py --json       # JSON to stdout only
    python scripts/domain_maturity.py --brief      # One-line per domain

Collapsus generativus est; solum quod redit, reale est.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

# ═══════════════════════════════════════════════════════════════════
# YAML LOADER (optional dependency)
# ═══════════════════════════════════════════════════════════════════

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

CLOSURES_DIR = _WORKSPACE / "closures"
REGISTRY_PATH = CLOSURES_DIR / "registry.yaml"
THEOREM_REGISTRY_PATH = _WORKSPACE / "derived" / "theorem_registry.json"
OUTPUT_PATH = _WORKSPACE / "derived" / "domain_maturity.json"

# ═══════════════════════════════════════════════════════════════════
# DOMAIN LIST (canonical order)
# ═══════════════════════════════════════════════════════════════════

DOMAINS = [
    "gcd",
    "rcft",
    "kinematics",
    "weyl",
    "security",
    "astronomy",
    "nuclear_physics",
    "quantum_mechanics",
    "finance",
    "atomic_physics",
    "materials_science",
    "everyday_physics",
    "evolution",
    "dynamic_semiotics",
    "consciousness_coherence",
    "continuity_theory",
    "awareness_cognition",
    "standard_model",
    "clinical_neuroscience",
    "spacetime_memory",
]

# Pattern for conformant theorem naming: theorem_<PREFIX><N>_<name>
_THEOREM_NAME_RE = re.compile(r"^theorem_[A-Z]+\d+_\w+$")


# ═══════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════


def _load_registry_metadata() -> dict[str, dict]:
    """Load domain_metadata from registry.yaml."""
    if yaml is None:
        return {}
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("extensions", {}).get("domain_metadata", {}) if data else {}


def _load_theorem_registry() -> list[dict]:
    """Load theorem_registry.json if available."""
    if not THEOREM_REGISTRY_PATH.exists():
        return []
    with open(THEOREM_REGISTRY_PATH) as f:
        return json.load(f)


def _scan_domain_files(domain: str) -> tuple[int, int]:
    """Count .py files and total lines for a domain directory."""
    domain_dir = CLOSURES_DIR / domain
    if not domain_dir.is_dir():
        return 0, 0
    file_count = 0
    total_lines = 0
    for py_file in domain_dir.rglob("*.py"):
        file_count += 1
        with contextlib.suppress(OSError, UnicodeDecodeError):
            total_lines += len(py_file.read_text(encoding="utf-8").splitlines())
    return file_count, total_lines


# ═══════════════════════════════════════════════════════════════════
# MATURITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════


def compute_maturity(domain: str, meta: dict, theorems: list[dict]) -> dict:
    """Compute maturity metrics for a single domain."""
    file_count, total_lines = _scan_domain_files(domain)

    # Theorem metrics
    domain_theorems = [t for t in theorems if t.get("domain") == domain]
    theorem_count = len(domain_theorems)
    proven_count = sum(1 for t in domain_theorems if t.get("verdict") == "PROVEN")
    pass_rate = proven_count / theorem_count if theorem_count > 0 else 0.0

    # Naming conformance
    conformant_names = sum(1 for t in domain_theorems if _THEOREM_NAME_RE.match(t.get("function_name", "")))
    naming_conformance = conformant_names / theorem_count if theorem_count > 0 else 1.0

    # Tier-1 bootstrap: check if first theorem has identity_checks
    tier1_bootstrap = meta.get("tier1_bootstrap", False)
    if domain_theorems:
        first = domain_theorems[0]
        if first.get("identity_checks"):
            tier1_bootstrap = True

    # Maturity score (0–100, weighted composite)
    # Components: theorem coverage (30), pass rate (25), file depth (15),
    #             naming (15), bootstrap (15)
    theorem_target = 10
    theorem_score = min(theorem_count / theorem_target, 1.0) * 30
    pass_score = pass_rate * 25
    file_score = min(file_count / 5, 1.0) * 15  # 5+ files = full credit
    naming_score = naming_conformance * 15
    bootstrap_score = 15.0 if tier1_bootstrap else 0.0
    maturity_score = round(theorem_score + pass_score + file_score + naming_score + bootstrap_score, 1)

    return {
        "domain": domain,
        "archetype": meta.get("archetype", "unknown"),
        "channels": meta.get("channels", 0),
        "entities": meta.get("entities", 0),
        "file_count": file_count,
        "total_lines": total_lines,
        "theorem_count": theorem_count,
        "proven_count": proven_count,
        "pass_rate": round(pass_rate, 4),
        "tier1_bootstrap": tier1_bootstrap,
        "naming_conformance": round(naming_conformance, 4),
        "maturity_score": maturity_score,
        "theorem_prefix": meta.get("theorem_prefix", ""),
    }


def compute_all() -> list[dict]:
    """Compute maturity for all 20 domains."""
    meta_all = _load_registry_metadata()
    theorems = _load_theorem_registry()
    results = []
    for domain in DOMAINS:
        meta = meta_all.get(domain, {})
        results.append(compute_maturity(domain, meta, theorems))
    return results


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════


def print_table(results: list[dict]) -> None:
    """Print a formatted table."""
    header = f"{'Domain':<25} {'Arch':<16} {'Ch':>3} {'Ent':>4} {'Files':>5} {'Lines':>6} {'Thm':>4} {'Pass':>5} {'Name%':>6} {'Boot':>5} {'Score':>6}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("DOMAIN MATURITY REPORT")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        boot = "✓" if r["tier1_bootstrap"] else "·"
        print(
            f"{r['domain']:<25} {r['archetype']:<16} {r['channels']:>3} {r['entities']:>4} "
            f"{r['file_count']:>5} {r['total_lines']:>6} {r['theorem_count']:>4} "
            f"{r['pass_rate']:>5.1%} {r['naming_conformance']:>5.1%} {boot:>5} {r['maturity_score']:>6.1f}"
        )
    print(sep)
    # Summary
    total_theorems = sum(r["theorem_count"] for r in results)
    total_proven = sum(r["proven_count"] for r in results)
    avg_score = sum(r["maturity_score"] for r in results) / len(results) if results else 0
    n_bootstrap = sum(1 for r in results if r["tier1_bootstrap"])
    print(
        f"  Total theorems: {total_theorems}  Proven: {total_proven}  Avg score: {avg_score:.1f}  Bootstrap: {n_bootstrap}/{len(results)}"
    )
    print(sep)


def print_brief(results: list[dict]) -> None:
    """Print one-line per domain."""
    for r in results:
        status = "✓" if r["maturity_score"] >= 70 else "○" if r["maturity_score"] >= 40 else "·"
        print(
            f"  {status} {r['domain']:<25} score={r['maturity_score']:>5.1f}  thm={r['theorem_count']:>3}  pass={r['pass_rate']:.0%}"
        )


def main() -> None:
    """Entry point."""
    json_mode = "--json" in sys.argv
    brief_mode = "--brief" in sys.argv

    results = compute_all()

    if json_mode:
        print(json.dumps(results, indent=2))
    elif brief_mode:
        print_brief(results)
    else:
        print_table(results)

    # Always write JSON output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    if not json_mode:
        print(f"\n  → Written to {OUTPUT_PATH.relative_to(_WORKSPACE)}")


if __name__ == "__main__":
    main()
