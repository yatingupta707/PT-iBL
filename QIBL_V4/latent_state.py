"""
components/latent_state.py  v4
───────────────────────────────
v4: Full N-option support.

CHANGES FROM v3:
  - BasicQuantumState: now works for any N options.
    q becomes a softmax vector over N blended values.
    phi becomes a length-N array of per-option phases.
  - ClassicalState: same N-option extension.
  - update_phase(new_phi, option_idx): updates ONLY the chosen option's phase.
    Previously updated a single shared phase. Now each option has its own.
  - n_options is set at construction from env.get_options() (via pipeline).

N-OPTION QUANTUM STATE:
────────────────────────
  |ψ⟩ = [√q_0·e^(iφ_0),  √q_1·e^(iφ_1),  ...,  √q_{N-1}·e^(iφ_{N-1})]  ∈ ℂ^N

  q_k = softmax(β_q · V̄)[k]       preference toward option k
  φ_k = persisted per-option phase  updated only when option k is chosen

  For N=2 this reduces to the v3 formula:
    q_0 = 1 - q_1 = sigmoid(-β_q · ΔV) ← comes from softmax
    φ_0 = 0 (safe option phase, usually not updated unless safe is chosen)
    φ_1 = phi     ← same as v3's single phase

  For N>2: Each option independently holds a preference strength q_k and
  an accumulated phase φ_k. The interference term in context_transform.py
  uses the top-2 options' q and φ values to maintain a meaningful single
  interference metric for logging.

OPTION ORDERING CONTRACT (unchanged from v3):
  Use positional indexing. Index 0 = first option (safe-analog),
  index 1 = second (risky-analog for 2-option), etc.
  Do NOT match on option name strings.

PHASE UPDATE (v4):
  Only the CHOSEN option's phase evolves each trial:
      φ_chosen ← (φ_chosen + α_φ · PE)  mod 2π
  Other options' phases are unchanged. This means:
  - Phases carry information about individual option histories
  - An unchosen option's phase remains from its last selection
  - This is more psychologically plausible than a single global phase
"""

import numpy as np
from typing import Any, List, Optional
from .base import BaseLatentState, LatentStateData, MemoryEstimate


def _softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax: exp(β·v) / Σ exp(β·v)."""
    v = beta * values
    v = v - v.max()   # shift for numerical stability
    e = np.exp(v)
    return e / e.sum()


def _build_amplitudes(q_vec: np.ndarray, phi_vec: np.ndarray) -> np.ndarray:
    """
    Build N-option complex amplitude vector:
        ψ_k = √q_k · e^(iφ_k)
    Normalisation: Σ |ψ_k|² = Σ q_k = 1 ✓ (q_vec sums to 1 by softmax)
    """
    return np.array([
        np.sqrt(max(q_vec[k], 0.0)) * np.exp(1j * phi_vec[k])
        for k in range(len(q_vec))
    ], dtype=complex)


class BasicQuantumState(BaseLatentState):
    """
    N-dimensional quantum amplitude state:
        |ψ⟩ = [√q_0·e^(iφ_0), ..., √q_{N-1}·e^(iφ_{N-1})]

    q_vec  is recomputed each trial from IBL blended values via softmax.
    phi_vec persists across trials; only the chosen option's phase evolves.

    Parameters
    ----------
    n_options : number of options (set by pipeline from env.get_options())
    beta_q    : softmax temperature for q mapping  [learnable]
    phi_init  : initial phase for all options (radians)
    """

    def __init__(self,
                 n_options: int   = 2,
                 beta_q:    float = 1.0,
                 phi_init:  float = 0.0):
        self.n_options = int(n_options)
        self.beta_q    = float(beta_q)
        self._phi_vec  = np.full(n_options, float(phi_init))
        self._q_vec    = np.full(n_options, 1.0 / n_options)
        self._amps     = _build_amplitudes(self._q_vec, self._phi_vec)

    def build(self, memory_estimate: MemoryEstimate, config: Any = None) -> LatentStateData:
        """
        Derive q_vec from blended values via softmax; use persisted phi_vec.

        q_k = softmax(β_q · V̄)[k]   for each option k

        Option ordering follows the key order of blended_values (from env.get_options()).
        """
        keys   = list(memory_estimate.blended_values.keys())
        n      = len(keys)

        if n != self.n_options:
            # Adapt if environment changed options (e.g., reset with different env)
            self._resize(n)

        v_vec       = np.array([memory_estimate.blended_values[k] for k in keys])
        self._q_vec = _softmax(v_vec, self.beta_q)
        self._amps  = _build_amplitudes(self._q_vec, self._phi_vec)

        return self._make_state_data()

    def get_state(self) -> LatentStateData:
        return self._make_state_data()

    def update_phase(self, new_phi: float, option_idx: int = 1) -> None:
        """
        Update phase for ONE option (the chosen one).
        Other options' phases remain unchanged.
        """
        idx = int(option_idx) % self.n_options
        self._phi_vec[idx] = float(new_phi % (2.0 * np.pi))
        self._amps = _build_amplitudes(self._q_vec, self._phi_vec)

    def reset(self, phi_init: float = 0.0) -> None:
        self._phi_vec = np.full(self.n_options, float(phi_init))
        self._q_vec   = np.full(self.n_options, 1.0 / self.n_options)
        self._amps    = _build_amplitudes(self._q_vec, self._phi_vec)

    # ── Private ───────────────────────────────────────────────────────────────

    def _resize(self, n: int) -> None:
        """Resize internal state if n_options changes."""
        self.n_options = n
        self._phi_vec  = np.full(n, 0.0)
        self._q_vec    = np.full(n, 1.0 / n)
        self._amps     = _build_amplitudes(self._q_vec, self._phi_vec)

    def _make_state_data(self) -> LatentStateData:
        """
        Build LatentStateData with both N-option arrays and 2-option compat scalars.

        q   scalar = q of option-1 (risky-analog, index 1) for 2-option compat
        phi scalar = phi of option-1 for 2-option compat
        For N>2: q = max(q_vec), phi = phi of argmax option
        """
        if self.n_options == 2:
            q_scalar   = float(self._q_vec[1])
            phi_scalar = float(self._phi_vec[1])
        else:
            top_idx    = int(self._q_vec.argmax())
            q_scalar   = float(self._q_vec[top_idx])
            phi_scalar = float(self._phi_vec[top_idx])

        return LatentStateData(
            amplitudes = self._amps.copy(),
            q          = q_scalar,
            phi        = phi_scalar,
            q_vec      = self._q_vec.copy(),
            phi_vec    = self._phi_vec.copy(),
        )

    def __repr__(self) -> str:
        return (f"BasicQuantumState(n={self.n_options}, β_q={self.beta_q}, "
                f"q={np.round(self._q_vec,3)}, φ={np.round(self._phi_vec,3)})")


class ClassicalState(BaseLatentState):
    """
    ABLATION: N-dimensional classical probability state — no phases, no interference.

    P(k) = q_k = softmax(β · V̄)[k]   for each option k
    No phase, so P(k) = |ψ_k|² = q_k exactly (Born rule collapses to classical).

    Use as comparison baseline to test whether quantum phases add predictive power.
    """

    def __init__(self, n_options: int = 2, beta: float = 1.0):
        self.n_options = int(n_options)
        self.beta      = float(beta)
        self._q_vec    = np.full(n_options, 1.0 / n_options)

    def build(self, memory_estimate: MemoryEstimate, config: Any = None) -> LatentStateData:
        keys       = list(memory_estimate.blended_values.keys())
        n          = len(keys)
        if n != self.n_options:
            self.n_options = n
            self._q_vec    = np.full(n, 1.0 / n)

        v_vec      = np.array([memory_estimate.blended_values[k] for k in keys])
        self._q_vec = _softmax(v_vec, self.beta)

        # Real amplitudes (no phase)
        amps = np.array([complex(np.sqrt(max(q, 0.0))) for q in self._q_vec])

        q_scalar   = float(self._q_vec[1]) if n == 2 else float(self._q_vec.max())
        phi_scalar = 0.0

        return LatentStateData(
            amplitudes = amps,
            q          = q_scalar,
            phi        = phi_scalar,
            q_vec      = self._q_vec.copy(),
            phi_vec    = np.zeros(n),
        )

    def get_state(self) -> LatentStateData:
        amps = np.array([complex(np.sqrt(max(q, 0.0))) for q in self._q_vec])
        q_scalar = float(self._q_vec[1]) if self.n_options == 2 else float(self._q_vec.max())
        return LatentStateData(
            amplitudes = amps, q = q_scalar, phi = 0.0,
            q_vec      = self._q_vec.copy(), phi_vec = np.zeros(self.n_options),
        )

    def update_phase(self, new_phi: float, option_idx: int = 1) -> None:
        pass   # no phase in classical state

    def reset(self) -> None:
        self._q_vec = np.full(self.n_options, 1.0 / self.n_options)
