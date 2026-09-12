"""
components/memory.py  v4
─────────────────────────
v4: Redo IBL implementation with 3 learnable parameters.

WHAT CHANGED FROM v3:
  v3: BasicIBL used exponential decay γ^(t-t_j) — a simplification.
  v4: IBL uses the correct ACT-R power-law decay (t-t_j)^(-d), plus
      Gaussian retrieval noise σ_s on log-activations.
      pInertia moved to BornRulePolicy (it is a choice modifier, not memory).

THREE IBL PARAMETERS:
─────────────────────
1. decay_d (d) — power-law forgetting exponent  [LEARNABLE]
   Controls how fast memories lose influence over time.
   Per-instance activation: a_j = (t - t_j)^(-d)

   d → 0 : all past instances equally weighted (perfect memory)
   d = 0.5: ACT-R default; moderate forgetting
   d → ∞ : only the most recent instance matters

   NOTE: This replaces v3's exponential γ. The power-law is empirically
   better supported (Anderson 1990, Rubin & Wenzel 1996). The parameter
   is no longer called 'decay' but 'decay_d' to avoid confusion.

2. noise_sigma (σ_s) — Gaussian retrieval noise  [LEARNABLE]
   Models the stochasticity of memory retrieval. Applied as log-normal
   noise to per-instance activations:
       a_j_noisy = a_j · exp(ε_j)   where ε_j ~ N(0, σ_s²)

   σ_s = 0   : deterministic retrieval (no noise)
   σ_s = 0.25: ACT-R default; moderate variability
   σ_s → ∞   : random retrieval (uniform over past instances)

   Log-normal formulation keeps activations positive and is equivalent
   to Gaussian noise on log-activations (log-activation = ln(a_j) + ε_j).

3. pInertia (p_inertia) — response repetition probability
   NOT in this file. Lives in components/choice_policy.py → BornRulePolicy.
   pInertia is a choice-level modifier, not a memory process.

PT BEFORE BLENDING (unchanged from v3):
─────────────────────────────────────────
   storage:  raw_outcome → valuation.evaluate() → pt_value → store(pt_value)
   retrieval: blend(stored pt_values) → V̄_k  [already in utility space]

   This means V̄_k = Σ w_j · v(x_j),  NOT  v(Σ w_j · x_j).
   These differ because PT is nonlinear. This is the correct implementation.
   To flip the order, store raw_outcome and apply PT after retrieval.

BOTH pt_value AND raw_outcome are stored per instance for auditability.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base import BaseMemory, MemoryEstimate


@dataclass
class _Instance:
    """One memory trace. Stores both PT value (for blending) and raw outcome (for audit)."""
    pt_value:    float
    raw_outcome: float
    time:        int
    option:      str


class IBL(BaseMemory):
    """
    Instance-Based Learning with power-law decay and retrieval noise.

    Replaces v3 BasicIBL (exponential decay) with the correct ACT-R formulation.

    Blending equations:
        a_j      = (t − t_j)^(−d)                   power-law activation
        a_j_nsy  = a_j · exp(N(0, σ_s²))            log-normal noise
        w_j      = a_j_nsy / Σ a_j_nsy              normalised weight
        V̄_k      = Σ_j w_j · v_j                    blended PT value
        σ²_k     = Σ_j w_j · (v_j − V̄_k)²          uncertainty

    Parameters
    ----------
    decay_d      : d — power-law forgetting exponent    ∈ (0, ∞)  [learnable]
    noise_sigma  : σ_s — Gaussian noise on log-activations ∈ [0, ∞)  [learnable]
    default_val  : seed value for cold-start instances
    n_init       : number of seed instances per option
    seed         : RNG seed for noise reproducibility
    """

    def __init__(self,
                 decay_d:     float = 0.5,
                 noise_sigma: float = 0.25,
                 default_val: float = 0.0,
                 n_init:      int   = 3,
                 seed:        Optional[int] = None):
        if decay_d <= 0:
            raise ValueError(f"decay_d must be > 0. Got {decay_d}")
        if noise_sigma < 0:
            raise ValueError(f"noise_sigma must be ≥ 0. Got {noise_sigma}")

        self.decay_d     = float(decay_d)
        self.noise_sigma = float(noise_sigma)
        self.default_val = float(default_val)
        self.n_init      = int(n_init)
        self.rng         = np.random.default_rng(seed)
        self._store:     Dict[str, List[_Instance]] = {}

    # ── Private ───────────────────────────────────────────────────────────────

    def _seed_option(self, option: str) -> None:
        """Cold-start: seed with n_init neutral instances at negative times."""
        if option not in self._store:
            self._store[option] = [
                _Instance(pt_value=self.default_val, raw_outcome=self.default_val,
                          time=-(i + 1), option=option)
                for i in range(self.n_init)
            ]

    def _compute_weights(self, times: np.ndarray, current_time: int) -> np.ndarray:
        """
        Compute power-law activations with log-normal noise, then normalise.

        a_j      = max(t - t_j, ε)^(-d)         [power-law; guard against dt=0]
        a_j_nsy  = a_j · exp(N(0, σ_s²))        [log-normal noise]
        w_j      = a_j_nsy / Σ a_j_nsy          [normalised]

        Fallback: if all activations are ≤ 0 after noise (pathological case),
        assign weight 1 to the most recent instance.
        """
        dt = np.maximum(current_time - times, 1e-9)   # avoid dt = 0
        activations = dt ** (-self.decay_d)             # power-law

        # Log-normal noise: multiply by exp(N(0, σ_s²))
        if self.noise_sigma > 0:
            noise       = self.rng.normal(0.0, self.noise_sigma, len(activations))
            activations = activations * np.exp(noise)

        activations = np.maximum(activations, 0.0)      # ensure non-negative
        total       = activations.sum()

        if total == 0.0:
            # Pathological: fall back to most-recent instance
            w          = np.zeros(len(activations))
            w[times.argmax()] = 1.0
        else:
            w = activations / total

        return w

    # ── Protocol ──────────────────────────────────────────────────────────────

    def retrieve(self, options: List[str], current_time: int) -> MemoryEstimate:
        """
        Compute noise-weighted blended PT value and uncertainty for each option.
        Each call draws fresh noise → retrieval is stochastic when noise_sigma > 0.
        """
        for opt in options:
            self._seed_option(opt)

        blended, uncertainties, n_instances = {}, {}, {}

        for opt in options:
            instances = self._store[opt]
            values    = np.array([ins.pt_value for ins in instances], dtype=float)
            times     = np.array([ins.time     for ins in instances], dtype=float)
            w         = self._compute_weights(times, current_time)

            V        = float(np.dot(w, values))
            variance = float(np.dot(w, (values - V) ** 2))

            blended[opt]       = V
            uncertainties[opt] = variance
            n_instances[opt]   = len(instances)

        return MemoryEstimate(blended_values=blended,
                              uncertainties=uncertainties,
                              n_instances=n_instances)

    def store(self, option: str, pt_value: float, raw_outcome: float, time: int) -> None:
        """
        Store experience.
        pt_value    → used for blending (already in PT/utility space)
        raw_outcome → stored for audit, accessible via get_raw_outcomes()
        """
        self._seed_option(option)
        self._store[option].append(
            _Instance(pt_value=float(pt_value), raw_outcome=float(raw_outcome),
                      time=time, option=option)
        )

    def reset(self) -> None:
        self._store = {}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_raw_outcomes(self, option: str) -> List[float]:
        """Return raw (non-PT) outcomes stored for an option (t ≥ 0 only)."""
        return [ins.raw_outcome for ins in self._store.get(option, [])
                if ins.time >= 0]

    def get_activation_profile(self, option: str, current_time: int) -> Dict:
        """Return per-instance activation profile for inspection/debugging."""
        self._seed_option(option)
        instances   = self._store[option]
        times       = np.array([ins.time for ins in instances], dtype=float)
        dt          = np.maximum(current_time - times, 1e-9)
        activations = dt ** (-self.decay_d)
        return {
            'times':       times.tolist(),
            'dt':          dt.tolist(),
            'activations': activations.tolist(),
            'values':      [ins.pt_value for ins in instances],
        }

    def __repr__(self) -> str:
        return (f"IBL(decay_d={self.decay_d}, "
                f"noise_sigma={self.noise_sigma}, "
                f"n_init={self.n_init})")


# Keep v3 name as alias for scripts that import BasicIBL
BasicIBL = IBL
