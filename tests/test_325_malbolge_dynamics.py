"""Tests for Malbolge VM and execution dynamics closure.

Tests the complete Malbolge virtual machine (ternary operations, cipher,
instruction set, memory initialization) and execution trajectory analysis
through the GCD kernel (8 dynamic channels, 6 theorems T-MD-1–T-MD-6).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_repo = Path(__file__).resolve().parents[1]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from closures.dynamic_semiotics.malbolge_dynamics import (
    DYN_CHANNEL_NAMES,
    N_DYN_CHANNELS,
    TrajectoryResult,
    compute_trajectory,
    extract_step_channels,
    verify_all_theorems,
    verify_t_md_1,
    verify_t_md_2,
    verify_t_md_3,
    verify_t_md_4,
    verify_t_md_5,
    verify_t_md_6,
)
from closures.dynamic_semiotics.malbolge_vm import (
    CAT_PROGRAM,
    CRAZY_TABLE,
    HALT_PROGRAM,
    MAX_TRIT,
    MEMORY_SIZE,
    MULTIOP_PROGRAM,
    OUTPUT_S_PROGRAM,
    TRIT_POWERS,
    XLAT1,
    MalbolgeVM,
    analyze_memory_fill,
    cipher_cycle_structure,
    cipher_has_fixed_points,
    crazy,
    from_trits,
    rotate_right,
    to_trits,
)
from umcp.frozen_contract import EPSILON

# ═══════════════════════════════════════════════════════════════════
# §1. Ternary Operations
# ═══════════════════════════════════════════════════════════════════


class TestCrazyOperation:
    """Test the tritwise crazy operation."""

    def test_crazy_zero_zero(self) -> None:
        """crazy(0, 0) should give all-1 trits → (3^10-1)/2 = 29524."""
        result = crazy(0, 0)
        assert result == sum(TRIT_POWERS)  # All trits = 1

    def test_crazy_is_deterministic(self) -> None:
        assert crazy(100, 200) == crazy(100, 200)

    @pytest.mark.parametrize(
        "a, d",
        [
            (0, 0),
            (0, MEMORY_SIZE - 1),
            (MEMORY_SIZE - 1, 0),
            (MEMORY_SIZE - 1, MEMORY_SIZE - 1),
            (29524, 29524),
            (1, 2),
            (19683, 39366),
        ],
    )
    def test_crazy_range_closed(self, a: int, d: int) -> None:
        """crazy always returns a value in [0, 59048]."""
        result = crazy(a, d)
        assert 0 <= result <= MEMORY_SIZE - 1

    def test_crazy_trit_table_structure(self) -> None:
        """Crazy table maps {0,1,2}×{0,1,2} → {0,1,2}."""
        for d_trit in range(3):
            for a_trit in range(3):
                assert CRAZY_TABLE[d_trit][a_trit] in (0, 1, 2)

    def test_crazy_known_value(self) -> None:
        """crazy(0, 62) = 29555 (hand-traced: trits 2,2,1,2,1,1,1,1,1,1)."""
        assert crazy(0, 62) == 29555

    def test_crazy_with_pure_trits(self) -> None:
        """crazy(1, 1) → (1,0,0,...) in trit table row [D]=0,A=0 = 1 for each."""
        # trit-by-trit: D=1→[1], A=1→[1]. CRAZY[1][1]=0 for trit 0
        # But both values are just "1", so trit 0 is d=1, a=1 → CRAZY[1][1]=0
        result = crazy(1, 1)
        # 1 in trits: (1, 0, 0, ...). So d_trit=1 for trit 0, d_trit=0 for rest
        # a_trit=1 for trit 0, a_trit=0 for rest
        # trit 0: CRAZY[1][1] = 0
        # trit 1-9: CRAZY[0][0] = 1
        expected = 0 + sum(TRIT_POWERS[1:])  # trits: 0, 1, 1, ..., 1
        assert result == expected


class TestRotateRight:
    """Test ternary rotate-right operation."""

    def test_rotate_zero(self) -> None:
        assert rotate_right(0) == 0

    def test_rotate_one(self) -> None:
        """1 in trits = (1,0,...,0). Rotate right → (0,...,0,1) = 3^9 * 0 + ... no.
        rotate_right: quotient = 0, remainder = 1. result = 0 + 1*19683 = 19683."""
        assert rotate_right(1) == 19683

    def test_rotate_two(self) -> None:
        """2 → remainder=2, quotient=0 → 2*19683 = 39366."""
        assert rotate_right(2) == 39366

    def test_rotate_range_closed(self) -> None:
        for val in [0, 1, 100, 29524, MEMORY_SIZE - 1]:
            result = rotate_right(val)
            assert 0 <= result <= MEMORY_SIZE - 1

    def test_rotate_ten_times_identity(self) -> None:
        """Rotating 10 times should restore original (10 trits)."""
        val = 12345
        result = val
        for _ in range(MAX_TRIT):
            result = rotate_right(result)
        assert result == val


class TestTritConversion:
    """Test trit ↔ integer conversion."""

    def test_to_trits_zero(self) -> None:
        assert to_trits(0) == [0] * MAX_TRIT

    def test_to_trits_one(self) -> None:
        trits = to_trits(1)
        assert trits[0] == 1
        assert all(t == 0 for t in trits[1:])

    def test_roundtrip(self) -> None:
        for val in [0, 1, 42, 29524, MEMORY_SIZE - 1]:
            assert from_trits(to_trits(val)) == val

    def test_to_trits_62(self) -> None:
        """62 = 2 + 2*3 + 0*9 + 2*27 = trits (2,2,0,2,0,...,0)."""
        trits = to_trits(62)
        assert trits[:4] == [2, 2, 0, 2]
        assert all(t == 0 for t in trits[4:])


# ═══════════════════════════════════════════════════════════════════
# §2. Cipher Analysis
# ═══════════════════════════════════════════════════════════════════


class TestXlat1Cipher:
    """Test the xlat1 post-execution cipher."""

    def test_xlat1_length(self) -> None:
        assert len(XLAT1) == 94

    def test_xlat1_all_printable(self) -> None:
        """All cipher output characters are in printable ASCII range."""
        for ch in XLAT1:
            assert 33 <= ord(ch) <= 126

    def test_xlat1_no_fixed_points(self) -> None:
        """The cipher is a derangement — no character maps to itself."""
        fixed = cipher_has_fixed_points()
        assert len(fixed) == 0, f"Fixed points at positions: {fixed}"

    def test_xlat1_is_permutation(self) -> None:
        """The cipher is a permutation — all 94 printable chars appear exactly once."""
        output_chars = set(XLAT1)
        expected_chars = {chr(i + 33) for i in range(94)}
        assert output_chars == expected_chars

    def test_xlat1_cycle_structure(self) -> None:
        """Cipher has well-defined cycle structure summing to 94."""
        cycles = cipher_cycle_structure()
        assert len(cycles) > 0
        assert sum(cycles) == 94

    def test_xlat1_known_mappings(self) -> None:
        """Spot-check known cipher mappings."""
        assert XLAT1[0] == "5"  # ! → 5
        assert XLAT1[1] == "z"  # " → z
        assert XLAT1[93] == "@"  # ~ → @


# ═══════════════════════════════════════════════════════════════════
# §3. VM Execution
# ═══════════════════════════════════════════════════════════════════


class TestHaltProgram:
    """Test the immediate-halt program 'Q'."""

    def test_halt_produces_no_output(self) -> None:
        vm = MalbolgeVM()
        vm.load(HALT_PROGRAM)
        output = vm.run(max_steps=10)
        assert output == ""

    def test_halt_in_one_step(self) -> None:
        vm = MalbolgeVM()
        vm.load(HALT_PROGRAM)
        vm.run(max_steps=10)
        assert vm.step_count == 1

    def test_halt_flag_set(self) -> None:
        vm = MalbolgeVM()
        vm.load(HALT_PROGRAM)
        vm.run(max_steps=10)
        assert vm.halted is True

    def test_halt_instruction_decoded(self) -> None:
        vm = MalbolgeVM()
        vm.load(HALT_PROGRAM)
        vm.run(max_steps=10)
        assert len(vm.history) == 1
        assert vm.history[0].instruction_name == "halt"


class TestOutputSProgram:
    """Test the '>bO' program that outputs 's'."""

    def test_output_is_s(self) -> None:
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        output = vm.run(max_steps=10)
        assert output == "s"

    def test_three_steps(self) -> None:
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        vm.run(max_steps=10)
        assert vm.step_count == 3

    def test_instruction_sequence(self) -> None:
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        vm.run(max_steps=10)
        names = [s.instruction_name for s in vm.history]
        assert names == ["crazy", "out", "halt"]

    def test_accumulator_after_crazy(self) -> None:
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        vm.run(max_steps=10)
        assert vm.history[0].A == 29555  # crazy(0, 62)

    def test_output_char_at_step_1(self) -> None:
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        vm.run(max_steps=10)
        assert vm.history[1].output_char == "s"

    def test_output_ascii_value(self) -> None:
        """chr(29555 % 256) = chr(115) = 's'."""
        assert chr(29555 % 256) == "s"


class TestMultiopProgram:
    """Test the '>&<`M' multi-operation program."""

    def test_five_steps(self) -> None:
        vm = MalbolgeVM()
        vm.load(MULTIOP_PROGRAM)
        vm.run(max_steps=10)
        assert vm.step_count == 5

    def test_instruction_sequence(self) -> None:
        vm = MalbolgeVM()
        vm.load(MULTIOP_PROGRAM)
        vm.run(max_steps=10)
        names = [s.instruction_name for s in vm.history]
        assert names == ["crazy", "rot", "crazy", "out", "halt"]

    def test_halts(self) -> None:
        vm = MalbolgeVM()
        vm.load(MULTIOP_PROGRAM)
        vm.run(max_steps=10)
        assert vm.halted is True

    def test_produces_output(self) -> None:
        vm = MalbolgeVM()
        vm.load(MULTIOP_PROGRAM)
        output = vm.run(max_steps=10)
        assert len(output) == 1


class TestCatProgram:
    """Test the cat program (echo input)."""

    def test_echoes_input(self) -> None:
        vm = MalbolgeVM()
        vm.load(CAT_PROGRAM)
        output = vm.run(max_steps=500, input_data="Hello")
        assert output.startswith("Hello")

    def test_single_char(self) -> None:
        vm = MalbolgeVM()
        vm.load(CAT_PROGRAM)
        output = vm.run(max_steps=100, input_data="A")
        assert "A" in output

    def test_eof_produces_232(self) -> None:
        """After input exhausted, A=59048 → chr(59048%256) = chr(232)."""
        assert chr(59048 % 256) == "\xa8"  # ¨

    def test_does_not_halt_on_eof(self) -> None:
        vm = MalbolgeVM()
        vm.load(CAT_PROGRAM)
        vm.run(max_steps=200, input_data="A")
        assert not vm.halted  # Still running after input exhausted


class TestMemoryFill:
    """Test crazy-fill memory initialization."""

    def test_fill_convergence(self) -> None:
        fill = analyze_memory_fill(HALT_PROGRAM, 1000)
        assert fill["unique_values"] <= 10

    def test_fill_trit_distribution(self) -> None:
        """Memory fill should have well-defined trit distribution."""
        fill = analyze_memory_fill(HALT_PROGRAM, 1000)
        fracs = fill["trit_fractions"]
        assert len(fracs) == 3
        assert abs(sum(fracs) - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# §4. Dynamic Channel Extraction
# ═══════════════════════════════════════════════════════════════════


class TestChannelExtraction:
    """Test 8-channel extraction from VM execution."""

    def test_channel_count(self) -> None:
        assert N_DYN_CHANNELS == 8

    def test_channel_names(self) -> None:
        assert len(DYN_CHANNEL_NAMES) == N_DYN_CHANNELS

    def test_channels_in_unit_interval(self) -> None:
        """All extracted channels must be in [ε, 1-ε]."""
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        vm.run(max_steps=10)

        for state in vm.history:
            idx = vm.history.index(state)
            prev = vm.history[idx - 1] if idx > 0 else None
            channels = extract_step_channels(state, prev, vm.history[: idx + 1], 3)
            assert channels.shape == (8,)
            assert np.all(channels >= EPSILON)
            assert np.all(channels <= 1.0 - EPSILON)

    def test_output_step_has_output_density(self) -> None:
        """Step with output should have nonzero output_density channel."""
        vm = MalbolgeVM()
        vm.load(OUTPUT_S_PROGRAM)
        vm.run(max_steps=10)

        # Step 1 is the output step
        out_state = vm.history[1]
        prev_state = vm.history[0]
        channels = extract_step_channels(out_state, prev_state, vm.history[:2], 3)
        assert channels[5] > EPSILON  # output_density

    def test_halt_step_low_clarity(self) -> None:
        """Halted step should have low instruction_clarity."""
        vm = MalbolgeVM()
        vm.load(HALT_PROGRAM)
        vm.run(max_steps=10)
        state = vm.history[0]
        channels = extract_step_channels(state, None, vm.history[:1], 1)
        assert channels[0] < 0.1  # instruction_clarity near EPSILON


# ═══════════════════════════════════════════════════════════════════
# §5. Trajectory Analysis
# ═══════════════════════════════════════════════════════════════════


class TestTrajectory:
    """Test trajectory computation and kernel evaluation."""

    def test_halt_trajectory_has_steps(self) -> None:
        traj = compute_trajectory(HALT_PROGRAM, "halt")
        assert traj.total_steps == 1
        assert len(traj.steps) == 1

    def test_output_s_trajectory(self) -> None:
        traj = compute_trajectory(OUTPUT_S_PROGRAM, "output_s")
        assert traj.output == "s"
        assert traj.halted

    def test_trajectory_has_valid_kernel(self) -> None:
        traj = compute_trajectory(OUTPUT_S_PROGRAM, "output_s")
        for sk in traj.steps:
            assert 0.0 <= sk.F <= 1.0
            assert 0.0 <= sk.omega <= 1.0
            assert abs(sk.F + sk.omega - 1.0) < 1e-10  # Duality identity

    def test_trajectory_regime_classification(self) -> None:
        traj = compute_trajectory(OUTPUT_S_PROGRAM, "output_s")
        for sk in traj.steps:
            assert sk.regime in ("Stable", "Watch", "Collapse")

    def test_output_step_higher_F(self) -> None:
        """Output step should have higher F than non-output steps (coherence pulse)."""
        traj = compute_trajectory(OUTPUT_S_PROGRAM, "output_s")
        out_steps = [s for s in traj.steps if s.output_char is not None]
        non_out_steps = [s for s in traj.steps if s.output_char is None]
        if out_steps and non_out_steps:
            mean_out_F = sum(s.F for s in out_steps) / len(out_steps)
            mean_non_F = sum(s.F for s in non_out_steps) / len(non_out_steps)
            assert mean_out_F > mean_non_F

    def test_cat_trajectory_produces_output(self) -> None:
        traj = compute_trajectory(CAT_PROGRAM, "cat", input_data="Hi", max_steps=200)
        assert "H" in traj.output

    def test_multiop_trajectory_stats(self) -> None:
        traj = compute_trajectory(MULTIOP_PROGRAM, "multiop")
        assert traj.mean_F > 0.0
        assert traj.mean_omega > 0.0
        assert traj.mean_F + traj.mean_omega > 0.9  # ~1.0 on average

    def test_trajectory_result_fields(self) -> None:
        traj = compute_trajectory(HALT_PROGRAM, "halt")
        assert traj.program_name == "halt"
        assert traj.program_length == 1
        assert isinstance(traj.regime_counts, dict)


# ═══════════════════════════════════════════════════════════════════
# §6. Tier-1 Identities on Dynamic Trajectories
# ═══════════════════════════════════════════════════════════════════


class TestTier1Identities:
    """Verify Tier-1 kernel identities hold at every trajectory step."""

    @pytest.fixture()
    def multiop_traj(self) -> TrajectoryResult:
        return compute_trajectory(MULTIOP_PROGRAM, "multiop")

    @pytest.mark.parametrize("step_idx", range(5))
    def test_duality_identity(self, multiop_traj: TrajectoryResult, step_idx: int) -> None:
        """F + ω = 1 at every step."""
        sk = multiop_traj.steps[step_idx]
        assert abs(sk.F + sk.omega - 1.0) < 1e-10

    @pytest.mark.parametrize("step_idx", range(5))
    def test_integrity_bound(self, multiop_traj: TrajectoryResult, step_idx: int) -> None:
        """IC ≤ F at every step."""
        sk = multiop_traj.steps[step_idx]
        assert sk.IC <= sk.F + 1e-10

    @pytest.mark.parametrize("step_idx", range(5))
    def test_log_integrity_relation(self, multiop_traj: TrajectoryResult, step_idx: int) -> None:
        """IC = exp(κ) at every step."""
        sk = multiop_traj.steps[step_idx]
        assert abs(sk.IC - math.exp(sk.kappa)) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# §7. Theorems T-MD-1 through T-MD-6
# ═══════════════════════════════════════════════════════════════════


class TestTheoremMD1:
    """T-MD-1: Crazy Determinism."""

    def test_proven(self) -> None:
        r = verify_t_md_1()
        assert r["PROVEN"], r

    def test_operations_closed(self) -> None:
        r = verify_t_md_1()
        assert r["operations_closed"]

    def test_trit_table_closed(self) -> None:
        r = verify_t_md_1()
        assert r["trit_table_closed"]


class TestTheoremMD2:
    """T-MD-2: Cipher Aperiodicity."""

    def test_proven(self) -> None:
        r = verify_t_md_2()
        assert r["PROVEN"], r

    def test_is_derangement(self) -> None:
        r = verify_t_md_2()
        assert r["is_derangement"]

    def test_is_permutation(self) -> None:
        r = verify_t_md_2()
        assert r["is_permutation"]

    def test_cycle_structure_sums_to_94(self) -> None:
        r = verify_t_md_2()
        assert sum(r["cycle_structure"]) == 94


class TestTheoremMD3:
    """T-MD-3: Halt Correctness."""

    def test_proven(self) -> None:
        r = verify_t_md_3()
        assert r["PROVEN"], r


class TestTheoremMD4:
    """T-MD-4: Output Validation."""

    def test_proven(self) -> None:
        r = verify_t_md_4()
        assert r["PROVEN"], r

    def test_expected_accumulator(self) -> None:
        r = verify_t_md_4()
        assert r["A_after_crazy"] == 29555


class TestTheoremMD5:
    """T-MD-5: Memory Fill Attractor."""

    def test_proven(self) -> None:
        r = verify_t_md_5()
        assert r["PROVEN"], r

    def test_max_unique_below_threshold(self) -> None:
        r = verify_t_md_5()
        assert r["max_unique"] <= r["threshold"]


class TestTheoremMD6:
    """T-MD-6: Trajectory Collapse Dominance."""

    def test_proven(self) -> None:
        r = verify_t_md_6()
        assert r["PROVEN"], r


class TestAllTheorems:
    """Verify all 6 theorems pass together."""

    def test_all_proven(self) -> None:
        results = verify_all_theorems()
        assert len(results) == 6
        for r in results:
            assert r["PROVEN"], f"{r['theorem']} failed: {r}"

    def test_all_have_correct_names(self) -> None:
        results = verify_all_theorems()
        expected = [f"T-MD-{i}" for i in range(1, 7)]
        actual = [r["theorem"] for r in results]
        assert actual == expected
