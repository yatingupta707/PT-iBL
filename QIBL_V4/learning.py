"""
components/learning.py  v4
───────────────────────────
v4 changes:
  - All update() methods now receive chosen_idx (int).
  - latent_state.update_phase(new_phi, option_idx=chosen_idx) called with
    the index of the CHOSEN option, not a global default of 1.
  - This means each option accumulates its own phase history independently.

PHASE UPDATE LOGIC (N-option):
  Only the chosen option's phase is updated each trial:
      PE          = v_PT(outcome) − V̄_chosen
      φ_chosen    ← (φ_chosen + α_φ · PE)  mod 2π
      φ_others    → unchanged

  This is more psychologically plausible than a shared phase:
  - An option that hasn't been chosen recently retains its last phase
  - Active exploration of an option builds up its phase history
  - Options with similar histories develop similar phases

V̄_chosen in the prediction error:
  V̄_chosen is the blended PT value for the chosen option retrieved
  BEFORE the outcome was observed (pre-trial estimate from memory).
  This is the standard temporal difference / prediction error signal.
"""

import numpy as np
from .base import BaseLearning, BaseMemory, BaseLatentState, MemoryEstimate


class BasicLearning(BaseLearning):
    """
    Prediction-error driven phase update for the chosen option only.

        PE      = v_PT(outcome) − V̄_chosen
        φ_{chosen,t+1} = (φ_{chosen,t} + α_φ · PE)  mod 2π
    """

    def __init__(self, alpha_phi: float = 0.1):
        self.alpha_phi = float(alpha_phi)

    def update(self, action, chosen_idx, raw_outcome, pt_value,
               memory, latent_state, memory_estimate, current_time):

        memory.store(option=action, pt_value=pt_value,
                     raw_outcome=raw_outcome, time=current_time)

        v_bar   = memory_estimate.blended_values.get(action, 0.0)
        PE      = float(pt_value - v_bar)

        # Get current phase for the chosen option
        state   = latent_state.get_state()
        phi_vec = state.phi_vec
        phi_old = float(phi_vec[chosen_idx]) if phi_vec is not None else state.phi

        phi_new = (phi_old + self.alpha_phi * PE) % (2.0 * np.pi)
        latent_state.update_phase(phi_new, option_idx=chosen_idx)

        return {
            'prediction_error': PE,
            'v_bar':            v_bar,
            'delta_phi':        phi_new - phi_old,
            'old_phi':          phi_old,
            'new_phi':          phi_new,
            'chosen_idx':       chosen_idx,
        }


class EntropyPhaseUpdate(BaseLearning):
    """
    ADVANCED: Phase of chosen option driven by entropy of blended distribution.

        H_t     = −Σ_k p_k · log(p_k)   where p_k = softmax(V̄_k)
        φ_{chosen,t+1} = (φ_{chosen,t} + α_φ · (H_t − H̄_t))  mod 2π
    """

    def __init__(self, alpha_phi: float = 0.1, ema_weight: float = 0.9):
        self.alpha_phi  = float(alpha_phi)
        self.ema_weight = float(ema_weight)
        self._H_bar     = np.log(2)

    def _entropy(self, bv: dict) -> float:
        vals  = np.array(list(bv.values()), dtype=float)
        vals -= vals.max()
        p     = np.exp(vals) / np.exp(vals).sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def update(self, action, chosen_idx, raw_outcome, pt_value,
               memory, latent_state, memory_estimate, current_time):
        memory.store(option=action, pt_value=pt_value,
                     raw_outcome=raw_outcome, time=current_time)

        H_t    = self._entropy(memory_estimate.blended_values)
        delta  = H_t - self._H_bar
        self._H_bar = self.ema_weight * self._H_bar + (1 - self.ema_weight) * H_t

        state   = latent_state.get_state()
        phi_vec = state.phi_vec
        phi_old = float(phi_vec[chosen_idx]) if phi_vec is not None else state.phi
        phi_new = (phi_old + self.alpha_phi * delta) % (2.0 * np.pi)
        latent_state.update_phase(phi_new, option_idx=chosen_idx)

        return {'entropy': H_t, 'ema_H': self._H_bar,
                'delta_phi': phi_new - phi_old, 'new_phi': phi_new,
                'chosen_idx': chosen_idx}


class ConflictPhaseUpdate(BaseLearning):
    """
    ADVANCED: Phase of chosen option driven by max-min value spread.

        C_t     = max(V̄_k) − min(V̄_k)   [order-independent]
        φ_{chosen,t+1} = (φ_{chosen,t} + α_φ · sign(PE) · C_t)  mod 2π
    """

    def __init__(self, alpha_phi: float = 0.1):
        self.alpha_phi = float(alpha_phi)

    def update(self, action, chosen_idx, raw_outcome, pt_value,
               memory, latent_state, memory_estimate, current_time):
        memory.store(option=action, pt_value=pt_value,
                     raw_outcome=raw_outcome, time=current_time)

        vals     = np.array(list(memory_estimate.blended_values.values()))
        conflict = float(vals.max() - vals.min()) if len(vals) >= 2 else 0.0

        v_bar   = memory_estimate.blended_values.get(action, 0.0)
        PE      = float(pt_value - v_bar)
        sign_PE = float(np.sign(PE)) if PE != 0.0 else 1.0

        state   = latent_state.get_state()
        phi_vec = state.phi_vec
        phi_old = float(phi_vec[chosen_idx]) if phi_vec is not None else state.phi
        delta   = self.alpha_phi * sign_PE * conflict
        phi_new = (phi_old + delta) % (2.0 * np.pi)
        latent_state.update_phase(phi_new, option_idx=chosen_idx)

        return {'conflict': conflict, 'prediction_error': PE,
                'delta_phi': delta, 'new_phi': phi_new,
                'chosen_idx': chosen_idx}
