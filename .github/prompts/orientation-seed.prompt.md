---
description: "Load GCD orientation receipts — 10 numbers that constrain classification. Use before any GCD/UMCP work."
tools: [search, read, execute]
---

# GCD Orientation Seed

> *Collapsus generativus est; solum quod redit, reale est.*

You are working in a system governed by one axiom: **collapse is generative; only what returns is real.** The kernel K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC) has 3 effective DOF (F, κ, C). These 10 numbers are the compressed derivation chains. They constrain what you can say.

## The 10 Receipts

| § | Name | Number | Constraint |
|---|------|--------|------------|
| 1 | Duality | max\|F+ω−1\| = **0.0** | Exact. Not approximate. Zero. |
| 2 | Integrity Bound | Δ(0.95, 0.001) = **0.4447** | One dead channel → massive gap |
| 3 | Geometric Slaughter | IC/F (1 dead, 8ch) = **0.1143** | 7 perfect channels cannot save IC |
| 4 | First Weld | Γ(0.682) = **0.9975** | Weld threshold at c ≈ 0.318 |
| 5 | Confinement Cliff | Neutron IC/F = **0.0089** | Color→0 kills IC 100× (phase boundary) |
| 6 | Scale Inversion | Nickel IC/F = **0.9573** | New DOF restore coherence |
| 7 | Regime Partition | Stable = **12.5%** | 87.5% of Fisher space is NOT stable |
| 8 | Equator | S+κ at c=½ = **0.0** | Four-way convergence, perfect cancellation |
| 9 | Super-Exponential | Gap shrinks **28.5×** | IC converges faster than exponential |
| 10 | Seam Monoid | Associativity error = **5.55e-17** | Exact algebraic structure |

## What These Numbers Forbid

- IC ≤ F is the **solvability condition** (c₁,₂ = F ± √(F²−IC²) needs real roots). It is not the classical inequality.
- S is **Bernoulli field entropy** of the collapse field. The classical form is its degenerate limit.
- F + ω = 1 is the **duality identity**. Receipt §1 proves it is exact to 0.0.
- Frozen parameters (ε=1e-8, p=3, α=1.0, tol=0.005) are **seam-derived**, not tuned.
- Verdicts are three-valued: CONFORMANT / NONCONFORMANT / NON_EVALUABLE.
- Tier-1 symbols (F, ω, S, C, κ, IC) are **immutable**. Import from `frozen_contract.py`.

## The Spine

Every claim: **Contract → Canon → Closures → Integrity Ledger → Stance**.
Five words: **Drift · Fidelity · Roughness · Return · Integrity**.
Stance is derived from gates. Never asserted.

## Verify

```bash
python scripts/orientation_checkpoint.py --verify
```

Lookup any symbol: `CATALOGUE.md`. After code changes: `python scripts/pre_commit_protocol.py`.
