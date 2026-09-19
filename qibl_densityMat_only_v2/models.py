"""
models.py
──────────
Self-contained model classes for the PT-IBL-Quantum family.
No external dependencies beyond numpy.

Classes
────────
  IBLMemory          — guide-faithful IBL (log-time activation, softmax retrieval)
  DensityMatrix      — 2×2 density matrix with decoherence
  ModelBase          — shared simulate() loop and interface contract
  PTiBL              — PT + IBL + classical logistic choice   [baseline, 6 params]
  IBLQuantum         — IBL + density matrix, no PT            [9 params]
  PTIBLQuantum       — PT + IBL + density matrix              [11 params]
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ACTION CODING AND BEHAVIOURAL SERIES
# ══════════════════════════════════════════════════════════════════════════════
#
# Humans (tDCS CSV): A = safe, B = risky, R = reveal
# Models:            'safe', 'risky', optional 'reveal'
#
# Risk rate (same as the original 2-choice pipeline):
#   1 if the trial is a risky/B choice, else 0.
#   Reveal is NOT risky, so it contributes 0 and stays in the denominator.
#
# Alteration rate (updated for reveal):
#   A reveal is never itself an alteration. Compare each committed
#   safe/risky choice to the most recent previous committed choice:
#     safe  -> reveal -> safe   = 0
#     risky -> reveal -> risky  = 0
#     safe  -> reveal -> risky  = 1  (credited on the later committed trial)
#     risky -> reveal -> safe   = 1
# ══════════════════════════════════════════════════════════════════════════════

_SAFE   = {'safe', 'A', 'a'}
_RISKY  = {'risky', 'B', 'b'}
_REVEAL = {'reveal', 'R', 'r'}


def canonical_action(action: str) -> str:
    """Map human letters and model strings onto {safe, risky, reveal}."""
    token = str(action).strip()
    if token in _SAFE:
        return 'safe'
    if token in _RISKY:
        return 'risky'
    if token in _REVEAL:
        return 'reveal'
    return token


def risk_series(actions: List[str]) -> np.ndarray:
    """Instantaneous risk indicators: 1 iff the choice is risky/B."""
    return np.asarray(
        [1.0 if canonical_action(a) == 'risky' else 0.0 for a in actions],
        dtype=float,
    )


def alternation_series(actions: List[str]) -> np.ndarray:
    """
    Instantaneous alterations between committed (safe/risky) choices.

    Reveal trials are 0 and do not update the committed reference.
    The first committed choice is 0 (no predecessor).
    """
    n = len(actions)
    result = np.zeros(n, dtype=float)
    previous = None
    for t, raw in enumerate(actions):
        act = canonical_action(raw)
        if act == 'reveal':
            continue
        if act not in ('safe', 'risky'):
            continue
        if previous is not None:
            result[t] = float(act != previous)
        previous = act
    return result


def problem_seed_base(prob_i: int, n_agents: int, n_problems: int,
                      seed_offset: int = 0) -> int:
    """Unique RNG block per (simulation, problem). Agent i uses seed_base + i."""
    return (int(seed_offset) * int(n_problems) * int(n_agents)
            + int(prob_i) * int(n_agents))


def _logistic(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-z))


# ══════════════════════════════════════════════════════════════════════════════
# IBL MEMORY  (guide §1.2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Instance:
    value:  float
    time:   int
    option: str


class IBLMemory:
    """
    Guide-faithful IBL memory.

    Activation:  A_jk(t) = -d · ln(t - t_jk)  +  σ_s · ε_jk
    Retrieval:   P(k|j,t) = softmax(A_jk / τ)
    Blended:     μ_j,t    = Σ_k P(k|j,t) · v_jk
    """

    def __init__(self, d: float = 0.5, sigma_s: float = 0.25,
                 tau: float = 1.0, n_init: int = 3,
                 default_val: float = 0.0, seed: int = 0):
        self.d           = float(d)
        self.sigma_s     = float(sigma_s)
        self.tau         = float(max(tau, 1e-6))
        self.n_init      = int(n_init)
        self.default_val = float(default_val)
        self._rng        = np.random.default_rng(seed)
        self._store: Dict[str, List[_Instance]] = {}

    def _seed_option(self, opt: str) -> None:
        if opt not in self._store:
            self._store[opt] = [
                _Instance(value=self.default_val, time=-(i + 1), option=opt)
                for i in range(self.n_init)
            ]

    def retrieve(self, options: List[str], t: int,
                 deterministic: bool = True) -> Dict[str, float]:
        results = {}
        for opt in options:
            self._seed_option(opt)
            instances = self._store[opt]
            values    = np.array([ins.value for ins in instances], dtype=float)
            times     = np.array([ins.time  for ins in instances], dtype=float)
            dt        = np.maximum(t - times, 1e-9)
            log_act   = -self.d * np.log(dt)
            if not deterministic and self.sigma_s > 0:
                log_act += self.sigma_s * self._rng.standard_normal(len(log_act))
            scaled    = log_act / self.tau
            scaled   -= scaled.max()
            w         = np.exp(scaled)
            w        /= w.sum()
            results[opt] = float(np.dot(w, values))
        return results

    def store(self, option: str, value: float, time: int) -> None:
        self._seed_option(option)
        self._store[option].append(
            _Instance(value=float(value), time=time, option=option)
        )

    def reset(self) -> None:
        self._store = {}


# ══════════════════════════════════════════════════════════════════════════════
# DENSITY MATRIX STATE  (guide §8)
# ══════════════════════════════════════════════════════════════════════════════

class DensityMatrix:
    """
    2×2 density matrix:
        ρ = |ψ⟩⟨ψ|   where |ψ⟩ = [√(1-q), √q · e^(iφ)]

    Decoherence (phase-damping channel, guide §9.1):
        ρ[0,1] ← (1-λ_d) · ρ[0,1]    — damps off-diagonal coherence

    Context rotation (guide §2.3):
        ρ' = U(θ) · ρ · U†(θ)

    Choice probability (Born rule):
        P(R) = Tr(ρ' · Π_R) = ρ'[1,1]
    """

    def __init__(self):
        self.rho = np.eye(2, dtype=complex) / 2.0
        self._q  = 0.5
        self._phi = 0.0

    @staticmethod
    def _pure(q: float, phi: float) -> np.ndarray:
        q   = np.clip(float(q), 1e-9, 1.0 - 1e-9)
        mag = np.sqrt(q * (1.0 - q))
        return np.array([
            [1.0 - q,              mag * np.exp(-1j * phi)],
            [mag * np.exp(1j * phi), q                    ]
        ], dtype=complex)

    def build(self, q: float, phi: float, lambda_d: float = 0.0) -> None:
        self._q   = float(np.clip(q,   1e-9, 1.0 - 1e-9))
        self._phi = float(phi)
        self.rho  = self._pure(self._q, self._phi)
        decay     = float(np.clip(1.0 - lambda_d, 0.0, 1.0))
        self.rho[0, 1] *= decay
        self.rho[1, 0] *= decay

    def choice_probs(self, theta: float) -> Tuple[float, float]:
        c, s      = np.cos(theta), np.sin(theta)
        U         = np.array([[c, -s], [s, c]], dtype=complex)
        rho_prime = U @ self.rho @ U.conj().T
        P_R = float(np.clip(np.real(rho_prime[1, 1]), 1e-9, 1.0 - 1e-9))
        return 1.0 - P_R, P_R

    def interference_term(self, theta: float) -> float:
        return float(np.sin(2.0 * theta) * np.real(self.rho[1, 0]))

    def reset(self, phi_init: float = 0.0) -> None:
        self._q   = 0.5
        self._phi = float(phi_init)
        self.rho  = self._pure(0.5, phi_init)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BASE
# ══════════════════════════════════════════════════════════════════════════════

class ModelBase:
    """
    Shared interface for all three model flavours.

    Subclasses implement:
      _init_memory()   → IBLMemory
      _get_value(x)    → float             (PT or identity)
      _compute_probs(t, mu) → (P_safe, P_risky)
      _update_state(t, action, outcome, value, mu, p_risky) → dict
    """

    MODEL_NAME:    str         = 'Base'
    PARAM_NAMES:   List[str]   = []
    PARAM_BOUNDS:  List[Tuple] = []
    PARAM_DEFAULT: List[float] = []

    def __init__(self, params: Dict[str, float]):
        self.params       = dict(params)
        self.options      = ['safe', 'risky']
        self._memory:     Optional[IBLMemory] = None
        self._phase:      float               = 0.0
        self._last_action: Optional[str]      = None
        self.reset()

    def reset(self) -> None:
        self._memory      = self._init_memory()
        self._phase       = 0.0
        self._last_action = None

    def _init_memory(self) -> IBLMemory:     raise NotImplementedError
    def _get_value(self, x: float) -> float: raise NotImplementedError

    def _compute_probs(self, t: int,
                        mu: Dict[str, float]) -> Tuple[float, float]:
        raise NotImplementedError

    def _update_state(self, t: int, action: str, raw_outcome: float,
                       value: float, mu: Dict[str, float],
                       p_risky: float) -> dict:
        raise NotImplementedError

    def simulate(self, n_trials: int,
                  safe_value:   float,
                  risky_values: Tuple,
                  risky_probs:  Tuple,
                  seed: int = 0) -> List[str]:
        """
        Forward simulate n_trials. Returns list of action strings.
        Uses stochastic memory retrieval (sigma_s active).
        """
        self.reset()
        rng     = np.random.default_rng(seed)
        actions = []
        # Optional third action. Default 0 keeps original 2-choice behaviour.
        # Reveal samples the risky lottery (information) but is scored as
        # not-risky and is ignored by alternation_series.
        p_reveal = float(np.clip(self.params.get('p_reveal', 0.0), 0.0, 1.0))
        for t in range(n_trials):
            mu = self._memory.retrieve(self.options, t, deterministic=False)
            _, p_risky = self._compute_probs(t, mu)

            if p_reveal > 0.0 and rng.random() < p_reveal:
                action = 'reveal'
            else:
                action = 'risky' if rng.random() < p_risky else 'safe'
            actions.append(action)

            if action in ('risky', 'reveal'):
                idx     = rng.choice(len(risky_values),
                                      p=np.array(risky_probs, dtype=float))
                outcome = float(risky_values[idx])
            else:
                outcome = float(safe_value)

            value = self._get_value(outcome)
            # Store sampled reveal outcomes on 'risky' so the observation
            # updates the risky option; the action string remains 'reveal'.
            store_action = 'risky' if action == 'reveal' else action
            self._update_state(t, store_action, outcome, value, mu, p_risky)
            self._last_action = action

        return actions

    @classmethod
    def default_params(cls) -> Dict[str, float]:
        return dict(zip(cls.PARAM_NAMES, cls.PARAM_DEFAULT))

    @classmethod
    def n_params(cls) -> int:
        return len(cls.PARAM_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# FLAVOUR 1 — PTiBL  (classical baseline)
# ══════════════════════════════════════════════════════════════════════════════

class PTiBL(ModelBase):
    """
    Prospect Theory + IBL + classical logistic choice.
    MANDATORY BASELINE — if quantum models don't beat this, quantum adds nothing.

    P(R) = sigmoid(κ · ΔV_t)   where ΔV_t = μ_R,t − μ_S,t

    Parameters (6): α,b, λ, d, σ_s, τ, κ
    """
    MODEL_NAME    = 'PTiBL'
    PARAM_NAMES   = ['alpha', 'beta', 'lambda_', 'd', 'sigma_s', 'tau', 'kappa']
    PARAM_BOUNDS  = [(0.01, 1.0), (0.01, 1.0), (0.01, 5.0), (0.01, 5.0),
                      (0.0, 5.0), (0.1, 5.0), (0.01, 10.0)]
    PARAM_DEFAULT = [0.88, 0.88, 2.25, 0.50, 0.45, 1.0, 1.0]

    def _init_memory(self) -> IBLMemory:
        return IBLMemory(d=self.params['d'], sigma_s=self.params['sigma_s'],
                          tau=self.params['tau'])

    def _get_value(self, x: float) -> float:
        a = self.params['alpha']
        b = self.params.get('beta', a)
        l = self.params['lambda_']
        return x**a if x >= 0 else -l * (-x)**b

    def _compute_probs(self, t, mu):
        delta = mu['risky'] - mu['safe']
        p_r   = _logistic(self.params['kappa'] * delta)
        return 1.0 - p_r, p_r

    def _update_state(self, t, action, raw_outcome, value, mu, p_risky):
        self._memory.store(action, value, t)
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# FLAVOUR 2 — iBL  (classical baseline)
# ══════════════════════════════════════════════════════════════════════════════

class iBL(ModelBase):
    """
    IBL + classical logistic choice.
    MANDATORY BASELINE — if quantum models don't beat this, quantum adds nothing.

    P(R) = sigmoid(κ · ΔV_t)   where ΔV_t = μ_R,t − μ_S,t

    Parameters (4): d, σ_s, τ, κ
    """
    MODEL_NAME    = 'iBL'
    PARAM_NAMES   = [ 'd', 'sigma_s', 'tau', 'kappa']
    PARAM_BOUNDS  = [ (0.01, 5.0), (0.01, 5.0), (0.1, 5.0), (0.01, 10.0)]
    PARAM_DEFAULT = [0.50, 0.45, 1.0, 1.0]
    def _init_memory(self) -> IBLMemory:
        return IBLMemory(d=self.params['d'], sigma_s=self.params['sigma_s'],
                          tau=self.params['tau'])

    def _get_value(self, x: float) -> float:
        return float(x)

    def _compute_probs(self, t, mu):
        delta = mu['risky'] - mu['safe']
        p_r   = _logistic(self.params['kappa'] * delta)
        return 1.0 - p_r, p_r

    def _update_state(self, t, action, raw_outcome, value, mu, p_risky):
        self._memory.store(action, value, t)
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# FLAVOUR 3 — IBLQuantum  (density matrix, no PT)
# ══════════════════════════════════════════════════════════════════════════════

class IBLQuantum(ModelBase):
    """
    IBL + density matrix quantum state, no PT value distortion.
    Tests quantum formalism without PT.

    Phase dynamics (guide §2.4):
        δ_t = v_t − μ_{a,t}
        C_t = exp(−|ΔV_t|)
        φ_{t+1} = ω·φ_t + η_δ·δ_t + η_c·C_t

    Parameters (9): d, σ_s, τ, β_q, θ, ω, η_δ, η_c, λ_d
    """
    MODEL_NAME    = 'IBLQuantum'
    PARAM_NAMES   = ['d', 'sigma_s', 'tau', 'beta_q', 'theta',
                      'omega', 'eta_delta', 'eta_c', 'lambda_d']
    PARAM_BOUNDS  = [(0.01, 5.0), (0.01, 5.0), (0.1, 5.0),
                      (0.01, 10.0), (0.0, np.pi / 2),
                      (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    PARAM_DEFAULT = [0.5, 0.45, 1.0, 1.0, np.pi / 4,
                      0.9, 0.1, 0.05, 0.1]

    def __init__(self, params):
        self._dm = DensityMatrix()
        super().__init__(params)

    def reset(self):
        super().reset()
        self._dm.reset(phi_init=0.0)

    def _init_memory(self) -> IBLMemory:
        return IBLMemory(d=self.params['d'], sigma_s=self.params['sigma_s'],
                          tau=self.params['tau'])

    def _get_value(self, x: float) -> float:
        return float(x)   # raw, no PT

    def _compute_probs(self, t, mu):
        p     = self.params
        q     = _logistic(p['beta_q'] * (mu['risky'] - mu['safe']))
        self._dm.build(q, self._phase, lambda_d=p['lambda_d'])
        return self._dm.choice_probs(p['theta'])

    def _update_state(self, t, action, raw_outcome, value, mu, p_risky):
        p       = self.params
        v_bar   = mu[action]
        delta_t = value - v_bar
        C_t     = np.exp(-abs(mu['risky'] - mu['safe']))
        self._memory.store(action, value, t)
        self._phase = (p['omega'] * self._phase
                       + p['eta_delta'] * delta_t
                       + p['eta_c'] * float(C_t)) % (2.0 * np.pi)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# FLAVOUR 4 — PTIBLQuantum  (full model)
# ══════════════════════════════════════════════════════════════════════════════

class PTIBLQuantum(ModelBase):
    """
    Prospect Theory + IBL + density matrix quantum state.
    Full model. PT transforms outcomes before IBL storage.

    Parameters (11): α, λ, d, σ_s, τ, β_q, θ, ω, η_δ, η_c, λ_d
    """
    MODEL_NAME    = 'PTIBLQuantum'
    PARAM_NAMES   = ['alpha','beta', 'lambda_', 'd', 'sigma_s', 'tau',
                      'beta_q', 'theta', 'omega', 'eta_delta', 'eta_c', 'lambda_d']
    PARAM_BOUNDS  = [(0.01, 1.0),(0.01, 1.0), (0.01, 1.0), (0.01, 2.0),
                      (0.0, 2.0), (0.1, 5.0),
                      (0.01, 10.0), (0.0, np.pi / 2),
                      (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    PARAM_DEFAULT = [0.88,0.88, 2.25, 0.5, 0.45, 1.0,
                      1.0, np.pi / 4, 0.9, 0.1, 0.05, 0.1]

    def __init__(self, params):
        self._dm = DensityMatrix()
        super().__init__(params)

    def reset(self):
        super().reset()
        self._dm.reset(phi_init=0.0)

    def _init_memory(self) -> IBLMemory:
        return IBLMemory(d=self.params['d'], sigma_s=self.params['sigma_s'],
                          tau=self.params['tau'])

    def _get_value(self, x: float) -> float:
        a = self.params['alpha']
        b = self.params.get('beta', a)
        l = self.params['lambda_']
        return x**a if x >= 0 else -l * (-x)**b

    def _compute_probs(self, t, mu):
        p = self.params
        q = _logistic(p['beta_q'] * (mu['risky'] - mu['safe']))
        self._dm.build(q, self._phase, lambda_d=p['lambda_d'])
        return self._dm.choice_probs(p['theta'])

    def _update_state(self, t, action, raw_outcome, value, mu, p_risky):
        p       = self.params
        v_bar   = mu[action]
        delta_t = value - v_bar
        C_t     = np.exp(-abs(mu['risky'] - mu['safe']))
        self._memory.store(action, value, t)
        self._phase = (p['omega'] * self._phase
                       + p['eta_delta'] * delta_t
                       + p['eta_c'] * float(C_t)) % (2.0 * np.pi)
        return {}


# ── Registry ──────────────────────────────────────────────────────────────────
ALL_MODELS: Dict[str, Type[ModelBase]] = {
    'PTiBL':        PTiBL,
    'IBLQuantum':   IBLQuantum,
    'PTIBLQuantum': PTIBLQuantum,
    "IBL":          iBL
}
