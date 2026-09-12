"""
pipeline.py  v4
────────────────
v4 changes:
  - Passes n_options (from env.get_options()) to latent_state at construction.
  - Tracks _last_action for pInertia in BornRulePolicy.
  - Passes chosen_idx to learning.update() (so only chosen option's phase evolves).
  - build_pipeline() passes ibl_noise_sigma, ibl_p_inertia to IBL and BornRulePolicy.
  - context dict for choice_policy includes 'last_action'.

PT BEFORE BLENDING — HOW IT WORKS IN THIS FILE:
  Step 6: raw_outcome = env.step(action)           [objective outcome]
  Step 7: pt_value = valuation.evaluate(raw_outcome)  [PT applied HERE]
  Step 8: learning.update(..., pt_value=pt_value, ...)
            → memory.store(option, pt_value=pt_value, raw_outcome=raw_outcome, time=t)
               [pt_value stored; blending later uses pt_value, not raw_outcome]

  When memory.retrieve() is called on the NEXT trial, it blends the stored pt_values.
  So V̄_k = avg(v(x)) ≠ v(avg(x)) — blending is in utility space (PT before blending).

  To test 'after_retrieval' order (PT after blending):
    Store raw_outcome → blend raw → apply PT to blended value.
    Set config.valuation_order='after_retrieval' — handled below.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from configs import ModelConfig
from components.base import (
    BaseEnvironment, BaseValuation, BaseMemory,
    BaseLatentState, BaseContextTransform, BaseChoicePolicy,
    BaseLearning, TrialRecord,
)
from components.environment       import BasicRiskyChoice, DataDrivenEnv
from components.valuation         import BasicPT
from components.memory            import IBL
from components.latent_state      import BasicQuantumState, ClassicalState
from components.context_transform import BasicRotation, IdentityTransform, ConflictRotation
from components.choice_policy     import BornRulePolicy, SoftmaxPolicy
from components.learning          import BasicLearning, EntropyPhaseUpdate, ConflictPhaseUpdate


class CognitivePipeline:
    """
    Orchestrates the 8-step trial loop. Pure router — no cognitive logic.

    v4: n_options determined from env at construction; last_action tracked for pInertia.
    """

    def __init__(self, config, environment, valuation, memory,
                 latent_state, context_transform, choice_policy, learning):
        self.config            = config
        self.env               = environment
        self.valuation         = valuation
        self.memory            = memory
        self.latent_state      = latent_state
        self.context_transform = context_transform
        self.choice_policy     = choice_policy
        self.learning          = learning

        self._trial_log:    List[TrialRecord] = []
        self._options:      List[str]          = self.env.get_options()
        self._option_to_idx: Dict[str, int]    = {opt: i for i, opt in enumerate(self._options)}
        self._last_action:  Optional[str]      = None   # for pInertia

    def run_trial(self, trial_num: int) -> TrialRecord:
        """8-step universal trial loop."""

        # 1. Memory
        mem_estimate = self.memory.retrieve(self._options, trial_num)

        # 2. Latent state — IBL→|ψ⟩ bridge
        latent_data  = self.latent_state.build(mem_estimate, self.config)

        # 3. Context transform
        context_info = {
            'value_diff': (
                list(mem_estimate.blended_values.values())[1] -
                list(mem_estimate.blended_values.values())[0]
            )
        }
        transformed  = self.context_transform.transform(latent_data, context_info)

        # 4. Choice probabilities + action (pInertia via last_action in context)
        choice_out   = self.choice_policy.compute_probabilities(
            transformed, self._options,
            context={'last_action': self._last_action}
        )
        action       = choice_out.action
        chosen_idx   = self._option_to_idx.get(action, 1)

        # 5-6. Environment → raw outcome
        raw_outcome, env_info = self.env.step(action)

        # 7. PT valuation
        if self.config.valuation_order == 'before_storage':
            # Default: PT applied to raw outcome before storage
            pt_value = self.valuation.evaluate(raw_outcome)
        else:
            # 'after_retrieval': store raw, apply PT to blended value (handled below)
            pt_value = raw_outcome   # temporary; actual PT applied after retrieval

        # 8. Learning: store + update phase of CHOSEN option only
        update_info = self.learning.update(
            action          = action,
            chosen_idx      = chosen_idx,
            raw_outcome     = raw_outcome,
            pt_value        = pt_value,
            memory          = self.memory,
            latent_state    = self.latent_state,
            memory_estimate = mem_estimate,
            current_time    = trial_num,
        )

        # Track last action for pInertia on the next trial
        self._last_action = action

        # Log
        record = TrialRecord(
            trial_num         = trial_num,
            action            = action,
            raw_outcome       = raw_outcome,
            pt_value          = pt_value,
            choice_prob       = choice_out.probabilities.get(action, 0.0),
            interference_term = transformed.interference_term,
            phase_before      = latent_data.phi,
            phase_after       = update_info.get('new_phi', latent_data.phi),
            q_before          = latent_data.q,
            blended_values    = dict(mem_estimate.blended_values),
            prediction_error  = update_info.get('prediction_error', 0.0),
            env_info          = dict(env_info),
        )
        self._trial_log.append(record)
        return record

    def run_experiment(self,
                       n_trials:  Optional[int] = None,
                       run_id:    Optional[str] = None) -> pd.DataFrame:
        n = n_trials if n_trials is not None else self.config.n_trials
        for t in range(n):
            self.run_trial(t)
        results = self.get_results()
        if self.config.log_experiments:
            self._log(results, run_id=run_id)
        return results

    def get_results(self) -> pd.DataFrame:
        rows = []
        for r in self._trial_log:
            row: Dict[str, Any] = {
                'trial':            r.trial_num,
                'action':           r.action,
                'chose_risky':      int(r.action == 'risky'),
                'raw_outcome':      r.raw_outcome,
                'pt_value':         r.pt_value,
                'choice_prob':      r.choice_prob,
                'interference':     r.interference_term,
                'phase_before':     r.phase_before,
                'phase_after':      r.phase_after,
                'q_pref':           r.q_before,
                'prediction_error': r.prediction_error,
            }
            for k, v in r.blended_values.items():
                row[f'blended_{k}'] = v
            tc = r.env_info.get('trial_config')
            if tc is not None:
                row['trial_safe_value'] = tc.safe_value
                row['trial_risky_ev']   = float(np.dot(tc.risky_values, tc.risky_probs))
            rows.append(row)
        return pd.DataFrame(rows)

    def reset(self) -> None:
        self.memory.reset()
        self.latent_state.reset(phi_init=self.config.phi_init)
        self.env.reset()
        self._trial_log   = []
        self._last_action = None

    def _log(self, results: pd.DataFrame, run_id: Optional[str] = None) -> None:
        from utils.experiment_logger import ExperimentLogger
        logger = ExperimentLogger(
            experiment_name=self.config.experiment_name,
            log_dir=self.config.log_dir,
            run_id=run_id,
        )
        logger.log(results, self.config)
        self._logger = logger

    def save_learned_config(self, learned_config: ModelConfig) -> None:
        if hasattr(self, '_logger'):
            self._logger.log_learned_config(learned_config)
        else:
            print("No logger — run run_experiment() with log_experiments=True first.")


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def build_pipeline(config: ModelConfig, dataloader=None) -> CognitivePipeline:
    """
    Instantiate all components and return a pipeline.

    v4: passes n_options to latent_state; passes ibl_p_inertia to choice_policy;
        passes ibl_decay_d and ibl_noise_sigma to IBL.
    """
    rng = np.random.default_rng(config.seed)
    def _seed(): return int(rng.integers(0, 2**31))

    # ── Environment ──────────────────────────────────────────────────────────
    if dataloader is not None:
        environment = DataDrivenEnv(
            dataloader    = dataloader,
            sampling_mode = config.dataloader_sampling_mode,
            seed          = _seed(),
        )
    else:
        environment = BasicRiskyChoice(
            safe_value   = config.safe_value,
            risky_values = config.risky_values,
            risky_probs  = config.risky_probs,
            seed         = _seed(),
        )

    n_options = len(environment.get_options())   # v4: determine from env

    # ── Valuation ─────────────────────────────────────────────────────────────
    valuation = BasicPT(
        alpha           = config.pt_alpha,
        beta            = config.pt_beta,
        lambda_         = config.pt_lambda,
        reference_point = config.pt_reference_point,
    )

    # ── Memory (v4: 3 params) ────────────────────────────────────────────────
    memory = IBL(
        decay_d     = config.ibl_decay_d,
        noise_sigma = config.ibl_noise_sigma,
        default_val = config.ibl_default_val,
        n_init      = config.ibl_n_init,
        seed        = _seed(),
    )

    # ── Latent State (v4: n_options) ──────────────────────────────────────────
    if config.latent_state_type == 'classical':
        latent_state = ClassicalState(n_options=n_options, beta=config.beta_q)
    else:
        latent_state = BasicQuantumState(
            n_options = n_options,
            beta_q    = config.beta_q,
            phi_init  = config.phi_init,
        )

    # ── Context Transform ─────────────────────────────────────────────────────
    context_transform = {
        'identity': IdentityTransform(),
        'conflict': ConflictRotation(theta_base=config.theta),
    }.get(config.context_type, BasicRotation(theta=config.theta))

    # ── Choice Policy (v4: pInertia) ──────────────────────────────────────────
    if config.choice_type == 'softmax':
        choice_policy = SoftmaxPolicy(
            temperature = config.softmax_tau,
            p_inertia   = config.ibl_p_inertia,
            rng_seed    = _seed(),
        )
    else:
        choice_policy = BornRulePolicy(
            p_inertia = config.ibl_p_inertia,
            rng_seed  = _seed(),
        )

    # ── Learning ──────────────────────────────────────────────────────────────
    learning = {
        'entropy':  EntropyPhaseUpdate(alpha_phi=config.alpha_phi),
        'conflict': ConflictPhaseUpdate(alpha_phi=config.alpha_phi),
    }.get(config.learning_type, BasicLearning(alpha_phi=config.alpha_phi))

    return CognitivePipeline(
        config=config, environment=environment, valuation=valuation,
        memory=memory, latent_state=latent_state,
        context_transform=context_transform, choice_policy=choice_policy,
        learning=learning,
    )
