"""Malbolge Virtual Machine — Complete Faithful Implementation.

Tier-0 implementation of the Malbolge virtual machine as specified by
Ben Olmstead (1998), following reference interpreter conventions.

The VM implements:
  - Ternary (base-3) arithmetic with 10-trit words (range 0–59048)
  - Self-encrypting memory (xlat1 cipher applied after each instruction)
  - The "crazy" tritwise operation (from specification table)
  - Instruction decode: (C + mem[C]) % 94
  - Memory initialization: fill with crazy(mem[i-1], mem[i-2])

This is a Tier-0 implementation (protocol machinery). The ternary operations
are exact. The cipher table is the canonical one from the Esolang wiki,
matching the reference interpreter (94 characters, verified derangement).

*Numquam binarius; tertia via semper patet.*
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Constants ────────────────────────────────────────────────────────

MEMORY_SIZE = 59049  # 3^10
MAX_TRIT = 10  # 10 ternary digits per word
TRIT_POWERS = tuple(3**i for i in range(MAX_TRIT))  # (1, 3, 9, ..., 19683)

# ── Crazy Operation ──────────────────────────────────────────────────
#
# The tritwise "crazy" operation, from the specification:
#
#          A (accumulator trit)
#          0    1    2
# [D]  0:  1    0    0
#      1:  1    0    2
#      2:  2    2    1
#
# Applied trit-by-trit to both 10-trit words.

CRAZY_TABLE = (
    (1, 0, 0),  # [D] trit = 0, A trit = 0/1/2
    (1, 0, 2),  # [D] trit = 1
    (2, 2, 1),  # [D] trit = 2
)


def crazy(a: int, d: int) -> int:
    """Tritwise 'crazy' operation on two ternary numbers."""
    result = 0
    for i in range(MAX_TRIT):
        a_trit = a % 3
        d_trit = d % 3
        result += CRAZY_TABLE[d_trit][a_trit] * TRIT_POWERS[i]
        a //= 3
        d //= 3
    return result


def rotate_right(val: int) -> int:
    """Rotate ternary number right by one trit (10 trits wide)."""
    quotient, remainder = divmod(val, 3)
    return quotient + remainder * TRIT_POWERS[MAX_TRIT - 1]


# ── Cipher Table (xlat1) ────────────────────────────────────────────
#
# Post-execution encryption from the reference interpreter.
# Maps printable ASCII 33–126 (94 chars) to replacement values.
# Source: Esolang wiki (verified against reference interpreter).
#
# ORIGINAL:   !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRS...
# TRANSLATED: 5z]&gqtyfr$(we4{WP)H-Zn,[%\3dL+Q;>U!pJS72FhOA1CB6v...
#
# The cipher is a DERANGEMENT (no fixed points): every executed
# instruction is necessarily mutated. This guarantees permanent drift.

XLAT1 = "5z]&gqtyfr$(we4{WP)H-Zn,[%\\3dL+Q;>U!pJS72FhOA1CB6v^=I_0/8|jsb9m<.TVac`uY*MK'X~xDl}REokN:#?G\"i@"
assert len(XLAT1) == 94, f"xlat1 must be 94 chars, got {len(XLAT1)}"

# ── Instruction Set ──────────────────────────────────────────────────
#
# (C + mem[C]) % 94 → instruction code
# Using reference interpreter conventions (input/output swapped from spec):

INSTR_JMP = 4  # C = [D]
INSTR_OUT = 5  # print(A % 256)
INSTR_IN = 23  # A = input (59048 on EOF)
INSTR_ROT = 39  # A = [D] = rotate_right([D])
INSTR_MOVD = 40  # D = [D]
INSTR_CRAZY = 62  # A = [D] = crazy(A, [D])
INSTR_NOP = 68  # no operation
INSTR_HALT = 81  # halt

VALID_INSTRUCTIONS = frozenset(
    {INSTR_JMP, INSTR_OUT, INSTR_IN, INSTR_ROT, INSTR_MOVD, INSTR_CRAZY, INSTR_NOP, INSTR_HALT}
)

INSTR_NAMES = {
    INSTR_JMP: "jmp",
    INSTR_OUT: "out",
    INSTR_IN: "in",
    INSTR_ROT: "rot",
    INSTR_MOVD: "movd",
    INSTR_CRAZY: "crazy",
    INSTR_NOP: "nop",
    INSTR_HALT: "halt",
}


# ── Known Programs ───────────────────────────────────────────────────

# Immediate halt: Q at position 0 → (81+0)%94 = 81 = HALT
HALT_PROGRAM = "Q"

# Output 's' then halt: crazy + output + halt
# Position 0: '>' (62) → (62+0)%94 = 62 → CRAZY
# Position 1: 'b' (98) → (98+1)%94 = 5 → OUT
# Position 2: 'O' (79) → (79+2)%94 = 81 → HALT
OUTPUT_S_PROGRAM = ">bO"

# Multi-operation: crazy + rotate + crazy + output + halt
# Position 0: '>' → CRAZY
# Position 1: '&' (38) → (38+1)%94 = 39 → ROT
# Position 2: '<' (60) → (60+2)%94 = 62 → CRAZY
# Position 3: '`' (96) → (96+3)%94 = 5 → OUT
# Position 4: 'M' (77) → (77+4)%94 = 81 → HALT
MULTIOP_PROGRAM = ">&<`M"

# Cat program (echoes input, does not stop on EOF)
# From Esolang wiki — the simplest known Malbolge cat program.
CAT_PROGRAM = '(=BA#9"=<;:3y7x54-21q/p-,+*)"!h%B0/.~P<<:(8&66#"!~}|{zyxwvugJ%'


# ── VM State ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VMState:
    """Snapshot of VM state at a single execution step."""

    step: int
    A: int
    C: int
    D: int
    instruction: int  # decoded instruction code (-1 if halted)
    instruction_name: str
    output_char: str | None  # character output this step, if any
    halted: bool
    mem_C_before: int  # memory[C] before cipher
    mem_C_after: int  # memory[C] after cipher


# ── Virtual Machine ──────────────────────────────────────────────────


@dataclass
class MalbolgeVM:
    """Complete Malbolge virtual machine.

    Implements the full specification: ternary memory, crazy operation,
    self-encrypting cipher, instruction decode, and memory initialization.
    """

    memory: list[int] = field(default_factory=lambda: [0] * MEMORY_SIZE)
    A: int = 0  # accumulator
    C: int = 0  # code pointer
    D: int = 0  # data pointer
    step_count: int = 0
    halted: bool = False
    output: list[str] = field(default_factory=list)
    input_buffer: list[int] = field(default_factory=list)
    history: list[VMState] = field(default_factory=list)
    program_length: int = 0

    def load(self, program: str) -> None:
        """Load a Malbolge program into memory."""
        self.memory = [0] * MEMORY_SIZE
        self.A = 0
        self.C = 0
        self.D = 0
        self.step_count = 0
        self.halted = False
        self.output = []
        self.history = []

        pos = 0
        for ch in program:
            if ch in " \t\n\r":
                continue
            if pos >= MEMORY_SIZE:
                break
            val = ord(ch)
            if val < 33 or val > 126:
                continue
            self.memory[pos] = val
            pos += 1

        self.program_length = pos

        # Fill remaining memory with crazy operation
        for i in range(pos, MEMORY_SIZE):
            prev1 = self.memory[i - 1] if i >= 1 else 0
            prev2 = self.memory[i - 2] if i >= 2 else 0
            self.memory[i] = crazy(prev1, prev2)

    def step(self) -> VMState:
        """Execute one instruction cycle. Returns state snapshot."""
        if self.halted:
            return VMState(
                step=self.step_count,
                A=self.A,
                C=self.C,
                D=self.D,
                instruction=-1,
                instruction_name="halted",
                output_char=None,
                halted=True,
                mem_C_before=0,
                mem_C_after=0,
            )

        val = self.memory[self.C]

        # If value not in printable range, halt
        if val < 33 or val > 126:
            self.halted = True
            state = VMState(
                step=self.step_count,
                A=self.A,
                C=self.C,
                D=self.D,
                instruction=-1,
                instruction_name="halt_oob",
                output_char=None,
                halted=True,
                mem_C_before=val,
                mem_C_after=val,
            )
            self.history.append(state)
            return state

        # Decode instruction
        instr = (val + self.C) % 94
        instr_name = INSTR_NAMES.get(instr, "nop")
        output_char = None

        # Execute
        if instr == INSTR_JMP:
            self.C = self.memory[self.D]
        elif instr == INSTR_OUT:
            ch = chr(self.A % 256)
            self.output.append(ch)
            output_char = ch
        elif instr == INSTR_IN:
            if self.input_buffer:
                self.A = self.input_buffer.pop(0)
            else:
                self.A = 59048  # EOF
        elif instr == INSTR_ROT:
            self.memory[self.D] = rotate_right(self.memory[self.D])
            self.A = self.memory[self.D]
        elif instr == INSTR_MOVD:
            self.D = self.memory[self.D]
        elif instr == INSTR_CRAZY:
            self.memory[self.D] = crazy(self.A, self.memory[self.D])
            self.A = self.memory[self.D]
        elif instr == INSTR_HALT:
            self.halted = True
            state = VMState(
                step=self.step_count,
                A=self.A,
                C=self.C,
                D=self.D,
                instruction=instr,
                instruction_name="halt",
                output_char=None,
                halted=True,
                mem_C_before=val,
                mem_C_after=val,
            )
            self.history.append(state)
            self.step_count += 1
            return state
        # else: NOP

        # Encrypt memory[C] using xlat1 cipher
        mem_C_before = self.memory[self.C]
        if 33 <= self.memory[self.C] <= 126:
            self.memory[self.C] = ord(XLAT1[self.memory[self.C] - 33])
        mem_C_after = self.memory[self.C]

        state = VMState(
            step=self.step_count,
            A=self.A,
            C=self.C,
            D=self.D,
            instruction=instr,
            instruction_name=instr_name,
            output_char=output_char,
            halted=False,
            mem_C_before=mem_C_before,
            mem_C_after=mem_C_after,
        )
        self.history.append(state)
        self.step_count += 1

        # Advance pointers
        self.C = (self.C + 1) % MEMORY_SIZE
        self.D = (self.D + 1) % MEMORY_SIZE

        return state

    def run(self, max_steps: int = 100000, input_data: str = "") -> str:
        """Run the VM until halt or max_steps."""
        self.input_buffer = [ord(c) for c in input_data]
        while not self.halted and self.step_count < max_steps:
            self.step()
        return "".join(self.output)

    def get_output(self) -> str:
        """Return accumulated output as string."""
        return "".join(self.output)


# ── Ternary Utilities ────────────────────────────────────────────────


def to_trits(val: int) -> list[int]:
    """Convert value to list of 10 trits (LSB first)."""
    trits = []
    for _ in range(MAX_TRIT):
        trits.append(val % 3)
        val //= 3
    return trits


def from_trits(trits: list[int]) -> int:
    """Convert list of trits (LSB first) to value."""
    result = 0
    for i, t in enumerate(trits):
        result += t * TRIT_POWERS[i]
    return result


# ── Analysis Utilities ───────────────────────────────────────────────


def cipher_has_fixed_points() -> list[int]:
    """Check if xlat1 has any fixed points. Returns list of fixed-point indices."""
    fixed = []
    for i in range(94):
        original_char = chr(i + 33)
        translated_char = XLAT1[i]
        if original_char == translated_char:
            fixed.append(i)
    return fixed


def cipher_cycle_structure() -> list[int]:
    """Compute cycle lengths of the xlat1 permutation.

    Returns sorted list of cycle lengths. If xlat1 is not a permutation
    (has repeated output chars), returns empty list.
    """
    # Check if it's a permutation first
    output_chars = set(XLAT1)
    expected_chars = {chr(i + 33) for i in range(94)}
    if output_chars != expected_chars:
        return []  # Not a permutation

    visited = [False] * 94
    cycles = []
    for start in range(94):
        if visited[start]:
            continue
        cycle_len = 0
        pos = start
        while not visited[pos]:
            visited[pos] = True
            cycle_len += 1
            # Follow: char at position pos maps to xlat1[pos]
            # Find which position xlat1[pos] corresponds to
            next_char = XLAT1[pos]
            pos = ord(next_char) - 33
        cycles.append(cycle_len)
    return sorted(cycles)


def analyze_memory_fill(program: str, n_cells: int = 1000) -> dict:
    """Analyze the crazy-fill memory initialization pattern."""
    vm = MalbolgeVM()
    vm.load(program)

    beyond_program = vm.memory[vm.program_length : vm.program_length + n_cells]
    unique_vals = len(set(beyond_program))

    # Trit distribution
    trit_counts = [0, 0, 0]
    for val in beyond_program:
        for t in to_trits(val):
            trit_counts[t] += 1

    total_trits = len(beyond_program) * MAX_TRIT
    trit_fractions = [c / total_trits for c in trit_counts]

    return {
        "n_cells": n_cells,
        "unique_values": unique_vals,
        "trit_fractions": trit_fractions,
    }


def main() -> None:
    """Demonstrate the Malbolge VM with known programs."""
    print("=" * 70)
    print("MALBOLGE VIRTUAL MACHINE — TERNARY COMPUTATION ENGINE")
    print("=" * 70)

    # Cipher analysis
    fixed = cipher_has_fixed_points()
    print(f"\nCipher fixed points: {len(fixed)} (derangement: {len(fixed) == 0})")
    cycles = cipher_cycle_structure()
    if cycles:
        print(f"Cipher cycle structure: {cycles} (sum={sum(cycles)})")
    else:
        print("Cipher is NOT a permutation (information-destroying)")

    # Run programs
    programs = [
        ("HALT", HALT_PROGRAM),
        ("OUTPUT_S", OUTPUT_S_PROGRAM),
        ("MULTIOP", MULTIOP_PROGRAM),
    ]

    for name, prog in programs:
        vm = MalbolgeVM()
        vm.load(prog)
        output = vm.run(max_steps=100)
        print(f"\n{name} ({len(prog)} chars): {vm.step_count} steps, output={output!r}, halted={vm.halted}")
        for s in vm.history[:10]:
            print(
                f"  step {s.step}: {s.instruction_name:6s} A={s.A:>6d} "
                f"C={s.C:>3d} D={s.D:>3d} "
                f"{'→ ' + repr(s.output_char) if s.output_char else ''}"
            )

    # Memory fill analysis
    fill = analyze_memory_fill(HALT_PROGRAM, 1000)
    print(f"\nMemory fill (beyond program): {fill['unique_values']} unique values in {fill['n_cells']} cells")
    print(f"Trit distribution: {fill['trit_fractions']}")


if __name__ == "__main__":
    main()
