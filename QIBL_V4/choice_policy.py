"""
components/choice_policy.py  v4
────────────────────────────────
v4 changes:
  - compute_probabilities() now accepts optional context dict.
  - BornRulePolicy: add p_inertia — probability of repeating the last action.
  - SoftmaxPolicy: same context dict extension.

PINERTIA:
  pInertia is a choice-level modifier representing response repetition bias
  (status quo bias, inertia). On each trial, with probability p_inertia,
  the agent repeats the previous choice regardless of its current cognitive state.

      P_final(k) = p_inertia · I(k = last_action) + (1 - p_inertia) · P_born(k)

  where I(k = last_action) = 1 if k was chosen on the previous trial, 0 otherwise.

  p_inertia = 0.0 : no inertia (pure Born rule / softmax)
  p_inertia = 1.0 : always repeat last action
  p_inertia = 0.1 : 10% chance of inertia; 90% Born rule

  This is the THIRD IBL parameter. It is stored in configs.ibl_p_inertia and
  passed through to BornRulePolicy at pipeline construction.

  The pipeline passes last_action in context: {'last_action': action_name_or_None}
  BornRulePolicy uses it only when p_inertia > 0.

WHY pInertia is in choice_policy (not memory):
  pInertia is NOT a memory retrieval phenomenon. It operates AFTER probabilities
  are computed from memory — it is a response selection bias. Placing it here
  keeps memory and choice mechanisms properly separated.
"""

import numpy as np
from typing import List, Optional
from .base import BaseChoicePolicy, TransformedStateData, ChoiceOutput


class BornRulePolicy(BaseChoicePolicy):
    """
    Quantum measurement via Born rule, with optional pInertia.

        P_born(k) = |ψ'[k]|²

    When p_inertia > 0 and last_action is in context:
        P_final(k) = p_inertia · I(k = last_action) + (1 - p_inertia) · P_born(k)

    Parameters
    ----------
    p_inertia : probability of repeating last action  [learnable], ∈ [0, 1)
    rng_seed  : RNG seed for reproducible sampling
    """

    def __init__(self, p_inertia: float = 0.0, rng_seed: Optional[int] = None):
        if not (0.0 <= p_inertia < 1.0):
            raise ValueError(f"p_inertia must be in [0, 1). Got {p_inertia}")
        self.p_inertia = float(p_inertia)
        self.rng       = np.random.default_rng(rng_seed)

    def compute_probabilities(self,
                               transformed: TransformedStateData,
                               options:     List[str],
                               context:     Optional[dict] = None) -> ChoiceOutput:
        """
        Compute final action probabilities and sample.

        Steps:
          1. Born rule: P_born(k) = |ψ'[k]|²
          2. Apply pInertia (if > 0 and last_action available in context)
          3. Sample action
        """
        # Step 1: Born rule
        amps  = transformed.amplitudes
        p_born = np.abs(amps) ** 2
        p_born = np.maximum(p_born, 0.0)
        p_born = p_born / p_born.sum()   # renormalise

        # Step 2: pInertia blending
        final_probs = p_born.copy()
        if self.p_inertia > 0.0 and context is not None:
            last_action = context.get('last_action')
            if last_action is not None and last_action in options:
                last_idx = options.index(last_action)
                inertia_vec = np.zeros(len(options))
                inertia_vec[last_idx] = 1.0
                final_probs = ((1.0 - self.p_inertia) * p_born
                               + self.p_inertia * inertia_vec)
                final_probs = final_probs / final_probs.sum()

        # Step 3: Sample
        prob_dict = {opt: float(final_probs[i]) for i, opt in enumerate(options)}
        action    = str(self.rng.choice(options, p=final_probs))

        return ChoiceOutput(
            probabilities = prob_dict,
            action        = action,
            log_prob      = float(np.log(prob_dict[action] + 1e-12)),
        )


class SoftmaxPolicy(BaseChoicePolicy):
    """
    ABLATION: Classical softmax over amplitude magnitudes, with optional pInertia.

        P_softmax(k) = exp(|ψ'[k]|² / τ) / Σ exp(|ψ'[j]|² / τ)

    pInertia applied identically to BornRulePolicy.

    Parameters
    ----------
    temperature : τ — exploration parameter > 0
    p_inertia   : response repetition probability  [learnable]
    rng_seed    : RNG seed
    """

    def __init__(self, temperature: float = 1.0, p_inertia: float = 0.0,
                 rng_seed: Optional[int] = None):
        self.temperature = float(temperature)
        self.p_inertia   = float(p_inertia)
        self.rng         = np.random.default_rng(rng_seed)

    def compute_probabilities(self,
                               transformed: TransformedStateData,
                               options:     List[str],
                               context:     Optional[dict] = None) -> ChoiceOutput:
        logits  = np.abs(transformed.amplitudes) ** 2 / self.temperature
        logits -= logits.max()
        p_soft  = np.exp(logits) / np.exp(logits).sum()

        final_probs = p_soft.copy()
        if self.p_inertia > 0.0 and context is not None:
            last_action = context.get('last_action')
            if last_action is not None and last_action in options:
                last_idx = options.index(last_action)
                inertia_vec = np.zeros(len(options))
                inertia_vec[last_idx] = 1.0
                final_probs = ((1.0 - self.p_inertia) * p_soft
                               + self.p_inertia * inertia_vec)
                final_probs = final_probs / final_probs.sum()

        prob_dict = {opt: float(final_probs[i]) for i, opt in enumerate(options)}
        action    = str(self.rng.choice(options, p=final_probs))

        return ChoiceOutput(
            probabilities = prob_dict,
            action        = action,
            log_prob      = float(np.log(prob_dict[action] + 1e-12)),
        )
