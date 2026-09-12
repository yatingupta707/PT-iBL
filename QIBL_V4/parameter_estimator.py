"""
utils/parameter_estimator.py  v4
──────────────────────────────────
v4 changes:
  1. 8 learnable parameters (added ibl_noise_sigma, ibl_p_inertia).
  2. fitting_mode='individual' | 'group' — controls who is the target.
  3. training_objective='mle' | 'metric' — controls how to compute loss.
  4. GroupParameterEstimator — fits one model to group-averaged targets.
  5. IndividualParameterEstimator — fits one model per participant.
  6. Metric-based fitting: forward simulation + behavioural metric matching.

FITTING MODES:
──────────────
  individual (default):
      One set of parameters per participant. Call fit(participant_df) once
      per participant. Best for individual-differences analysis.
      Returns: one ModelConfig per participant.

  group:
      Average behaviour targets across all participants, then fit ONE shared
      model. Best for characterising the "average agent" in your population.
      Returns: one ModelConfig for the whole group.

TRAINING OBJECTIVES:
─────────────────────
  mle — Teacher-forcing Negative Log-Likelihood (trial-level, default):
      At each trial t: compute P(human_action_t | model_state_t, Θ)
      Minimize: NLL(Θ) = −Σ_t log P(human_action_t | Θ)
      Fast. Captures trial-level dynamics. Good when you have full trial sequences.

  metric — Forward simulation + Behavioural Metric Matching:
      Simulate the model n_metric_sims times.
      Average metrics: risk_rate, alternation_rate, etc.
      Minimize: L(Θ) = Σ_m w_m · (model_m − human_m)²
      Slower. Captures aggregate patterns. Good for:
        - Summary-level data (no trial sequences)
        - Correcting for MLE overfitting to noisy trial sequences
        - Fitting to group-averaged targets

  Both objectives use the same optimiser (Nelder-Mead, scipy >= 1.7).

TEACHER-FORCING (mle mode):
  The estimator NEVER calls env.step(). It uses human outcomes directly:
  1. Retrieve memory built from human choices 0..t-1
  2. Build |ψ⟩ from memory
  3. Apply U(θ)
  4. P(human_action_t) = |ψ'[k]|²  — look up human's choice
  5. NLL += −log P
  6. Update memory + phase using human's action and outcome

  This works identically for fixed-payoff and CSV/random-sampling experiments.
  The environment inside the pipeline is never touched during fitting.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from copy import deepcopy

from configs import ModelConfig
from utils.behaviour_metrics import (
    compute_metrics, compute_group_metrics, metric_loss
)


LEARNABLE_PARAMS = [
    # (config_key,        default_init,  lower,        upper)
    ('pt_alpha',          0.88,          0.01,          1.00),
    ('pt_beta',           0.88,          0.01,          1.00),
    ('pt_lambda',         2.25,          1.00,         10.00),
    ('ibl_decay_d',       0.50,          0.01,          2.00),
    ('ibl_noise_sigma',   0.25,          0.00,          2.00),
    ('ibl_p_inertia',     0.00,          0.00,          0.90),
    ('beta_q',            1.00,          0.01,         10.00),
    ('theta',             np.pi / 4,     0.00,          np.pi / 2),
]

FIXED_PARAMS = ['alpha_phi', 'phi_init', 'ibl_n_init', 'pt_reference_point',
                'ibl_default_val', 'seed']


class ParameterEstimator:
    """
    Unified parameter estimator for individual fitting.

    Supports both training objectives (mle / metric) controlled by
    config.training_objective.

    Parameters
    ----------
    base_config        : ModelConfig with fixed params set. Learnable overwritten.
    learnable          : list of config keys to optimise. Default: base_config.learnable_params.
    n_restarts         : random restarts for global search (default 3).
    max_iter           : Nelder-Mead iterations per restart (default 500).
    verbose            : print progress (default True).
    """

    def __init__(self,
                 base_config: ModelConfig,
                 learnable:   Optional[List[str]] = None,
                 n_restarts:  int  = 3,
                 max_iter:    int  = 500,
                 verbose:     bool = True):
        self.base_config  = deepcopy(base_config)
        self.learnable    = learnable or list(base_config.learnable_params)
        self.n_restarts   = n_restarts
        self.max_iter     = max_iter
        self.verbose      = verbose
        self._param_defs  = [p for p in LEARNABLE_PARAMS if p[0] in self.learnable]
        self._rng         = np.random.default_rng(getattr(base_config, 'seed', 42))
        self._target_metrics: Optional[Dict[str, float]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, human_data: pd.DataFrame) -> ModelConfig:
        """
        Fit model parameters to one participant's data.

        Parameters
        ----------
        human_data : DataFrame with columns: trial, action, raw_outcome
                     (sorted by trial, 0-indexed)

        Returns
        -------
        ModelConfig with learned parameters.
        fit_nll  set for 'mle' objective.
        fit_loss set for 'metric' objective.
        """
        from scipy.optimize import minimize

        # Validate
        required = {'trial', 'action', 'raw_outcome'}
        missing  = required - set(human_data.columns)
        if missing:
            raise ValueError(f"human_data missing columns: {missing}")

        self._data = human_data.sort_values('trial').reset_index(drop=True)

        # Precompute human metrics once (for metric-based fitting)
        if self.base_config.training_objective == 'metric':
            self._target_metrics = compute_metrics(
                self._data, self.base_config.metrics_to_use
            )
            if self.verbose:
                print(f"  Target metrics: {self._target_metrics}")

        bounds = [(lo, hi) for _, _, lo, hi in self._param_defs]
        best_nll, best_loss, best_params = np.inf, np.inf, None
        errors = []

        for restart in range(self.n_restarts):
            x0 = self._initial_point(restart)
            try:
                result = minimize(
                    self._objective, x0,
                    method  = 'Nelder-Mead',
                    bounds  = bounds,
                    options = {'maxiter': self.max_iter, 'xatol': 1e-4, 'fatol': 1e-4},
                )
                if self.verbose:
                    pstr = ', '.join(f"{k}={v:.3f}"
                                     for k, v in zip([p[0] for p in self._param_defs], result.x))
                    print(f"  Restart {restart+1}/{self.n_restarts}: "
                          f"loss={result.fun:.4f}  [{pstr}]")
                if result.fun < best_nll:
                    best_nll    = result.fun
                    best_params = result.x.copy()
            except Exception as e:
                msg = f"Restart {restart+1} failed: {e}"
                errors.append(msg)
                if self.verbose:
                    print(f"  WARNING: {msg}")

        if best_params is None:
            raise RuntimeError(
                f"All {self.n_restarts} restarts failed.\nErrors: {errors}"
            )

        learned = self._vector_to_config(best_params)
        if self.base_config.training_objective == 'mle':
            learned.fit_nll = float(best_nll)
        else:
            learned.fit_loss = float(best_nll)

        if self.verbose:
            print(f"\n  Best loss: {best_nll:.4f}")
            for k, v in zip([p[0] for p in self._param_defs], best_params):
                print(f"    {k:<22} = {v:.6f}")

        return learned

    def compute_nll(self, config: ModelConfig, human_data: pd.DataFrame) -> float:
        """Compute NLL for a given config (for BIC or model comparison)."""
        self._data = human_data.sort_values('trial').reset_index(drop=True)
        x = np.array([getattr(config, p[0]) for p in self._param_defs])
        # Temporarily force mle for this computation
        old_obj = self.base_config.training_objective
        self.base_config.training_objective = 'mle'
        result = self._objective(x)
        self.base_config.training_objective = old_obj
        return result

    def parameter_summary(self) -> pd.DataFrame:
        rows = []
        for key, init, lo, hi in LEARNABLE_PARAMS:
            rows.append({'parameter': key,
                          'status':    'LEARNABLE' if key in self.learnable else 'fixed',
                          'default':   init, 'bounds': f'[{lo:.2f}, {hi:.2f}]'})
        for key in FIXED_PARAMS:
            rows.append({'parameter': key, 'status': 'FIXED',
                          'default': getattr(self.base_config, key, '—'), 'bounds': '—'})
        return pd.DataFrame(rows)

    # ── Objectives ────────────────────────────────────────────────────────────

    def _objective(self, param_vector: np.ndarray) -> float:
        if self.base_config.training_objective == 'metric':
            return self._metric_objective(param_vector)
        return self._mle_objective(param_vector)

    def _mle_objective(self, param_vector: np.ndarray) -> float:
        """
        Teacher-forcing NLL.
        Environment is NEVER called — uses human outcomes directly.
        """
        from pipeline import build_pipeline

        cfg      = self._vector_to_config(param_vector)
        pipeline = build_pipeline(cfg)
        nll      = 0.0

        for _, row in self._data.iterrows():
            t             = int(row['trial'])
            human_action  = str(row['action'])
            human_outcome = float(row['raw_outcome'])
            human_pt      = pipeline.valuation.evaluate(human_outcome)
            chosen_idx    = pipeline._option_to_idx.get(human_action, 1)

            # Forward pass — compute P(human_action); do NOT sample
            mem_est     = pipeline.memory.retrieve(pipeline._options, t)
            latent      = pipeline.latent_state.build(mem_est, cfg)
            transformed = pipeline.context_transform.transform(latent)
            choice_out  = pipeline.choice_policy.compute_probabilities(
                transformed, pipeline._options,
                context={'last_action': pipeline._last_action}
            )

            p_human = choice_out.probabilities.get(human_action, 1e-10)
            nll    -= np.log(max(p_human, 1e-10))

            # Teacher-forcing: update with human's action + outcome
            pipeline.learning.update(
                action          = human_action,
                chosen_idx      = chosen_idx,
                raw_outcome     = human_outcome,
                pt_value        = human_pt,
                memory          = pipeline.memory,
                latent_state    = pipeline.latent_state,
                memory_estimate = mem_est,
                current_time    = t,
            )
            pipeline._last_action = human_action   # track for pInertia

        return float(nll)

    def _metric_objective(self, param_vector: np.ndarray) -> float:
        """
        Forward simulation + behavioural metric matching.

        Runs n_metric_sims independent simulations of the model.
        Averages metrics across simulations to reduce sampling noise.
        Computes weighted squared error vs human target metrics.
        """
        from pipeline import build_pipeline

        cfg    = self._vector_to_config(param_vector)
        n_sims = self.base_config.n_metric_sims
        all_metrics = []

        for sim_idx in range(n_sims):
            sim_cfg      = deepcopy(cfg)
            sim_cfg.seed = sim_idx
            pipe = build_pipeline(sim_cfg)
            df   = pipe.run_experiment(n_trials=len(self._data))
            all_metrics.append(compute_metrics(df, self.base_config.metrics_to_use))

        # Average across simulations
        metric_names = list(all_metrics[0].keys())
        model_metrics = {
            name: float(np.nanmean([m[name] for m in all_metrics]))
            for name in metric_names
        }

        return metric_loss(
            model_metrics,
            self._target_metrics,
            self.base_config.metric_weights or None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _initial_point(self, restart: int) -> np.ndarray:
        if restart == 0:
            return np.array([init for _, init, _, _ in self._param_defs])
        return np.array([
            float(self._rng.uniform(lo + 1e-3, hi - 1e-3))
            for _, _, lo, hi in self._param_defs
        ])

    def _vector_to_config(self, param_vector: np.ndarray) -> ModelConfig:
        """
        CRITICAL: log_experiments always False during optimisation.
        Without this, every function evaluation creates a log folder.
        """
        cfg = deepcopy(self.base_config)
        cfg.log_experiments = False
        for (key, _, lo, hi), val in zip(self._param_defs, param_vector):
            setattr(cfg, key, float(np.clip(val, lo + 1e-9, hi - 1e-9)))
        return cfg


class GroupParameterEstimator:
    """
    Fits ONE shared model to group-averaged targets across multiple participants.

    Two modes depending on training_objective:
      'mle'    — Soft-target MLE: for each trial t, target is the fraction of
                 participants who chose 'risky'. Minimises cross-entropy with
                 soft targets. Requires full trial sequences from all participants.

      'metric' — Average metrics across participants, fit one model to the mean.
                 Only requires summary statistics per participant.

    Parameters
    ----------
    base_config : ModelConfig with config.fitting_mode='group'
    learnable   : parameters to optimise
    n_restarts  : optimiser restarts
    max_iter    : max iterations per restart
    verbose     : print progress
    """

    def __init__(self,
                 base_config: ModelConfig,
                 learnable:   Optional[List[str]] = None,
                 n_restarts:  int  = 3,
                 max_iter:    int  = 500,
                 verbose:     bool = True):
        self.base_config = deepcopy(base_config)
        self.learnable   = learnable or list(base_config.learnable_params)
        self.n_restarts  = n_restarts
        self.max_iter    = max_iter
        self.verbose     = verbose
        self._param_defs = [p for p in LEARNABLE_PARAMS if p[0] in self.learnable]
        self._rng        = np.random.default_rng(getattr(base_config, 'seed', 42))

    def fit(self, participants_data: List[pd.DataFrame]) -> ModelConfig:
        """
        Fit one model to the group average.

        Parameters
        ----------
        participants_data : list of DataFrames, one per participant.
                            Each must have: trial, action, raw_outcome.

        Returns
        -------
        One ModelConfig representing the best-fit "average agent."
        """
        from scipy.optimize import minimize

        if not participants_data:
            raise ValueError("participants_data is empty.")

        # Validate all DataFrames
        required = {'trial', 'action', 'raw_outcome'}
        for i, df in enumerate(participants_data):
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Participant {i} data missing columns: {missing}")

        if self.base_config.training_objective == 'metric':
            # Compute group-averaged metrics once
            self._group_metrics = compute_group_metrics(
                participants_data, self.base_config.metrics_to_use
            )
            if self.verbose:
                print(f"  Group target metrics ({len(participants_data)} participants):")
                for k, v in self._group_metrics.items():
                    print(f"    {k}: {v:.4f}")
        else:
            # Soft-target MLE: average trial-level choice proportions
            self._soft_targets = self._compute_soft_targets(participants_data)

        bounds     = [(lo, hi) for _, _, lo, hi in self._param_defs]
        best_loss  = np.inf
        best_params = None
        errors     = []

        for restart in range(self.n_restarts):
            x0 = self._initial_point(restart)
            try:
                result = minimize(
                    self._group_objective, x0,
                    method  = 'Nelder-Mead',
                    bounds  = bounds,
                    options = {'maxiter': self.max_iter, 'xatol': 1e-4, 'fatol': 1e-4},
                )
                if self.verbose:
                    pstr = ', '.join(f"{k}={v:.3f}"
                                     for k, v in zip([p[0] for p in self._param_defs], result.x))
                    print(f"  Restart {restart+1}/{self.n_restarts}: "
                          f"loss={result.fun:.4f}  [{pstr}]")
                if result.fun < best_loss:
                    best_loss   = result.fun
                    best_params = result.x.copy()
            except Exception as e:
                errors.append(f"Restart {restart+1}: {e}")
                if self.verbose:
                    print(f"  WARNING: {errors[-1]}")

        if best_params is None:
            raise RuntimeError(f"All restarts failed.\nErrors: {errors}")

        learned = self._vector_to_config(best_params)
        learned.fit_loss = float(best_loss)

        if self.verbose:
            print(f"\n  Best group loss: {best_loss:.4f}")

        return learned

    def _compute_soft_targets(self, dfs: List[pd.DataFrame]) -> np.ndarray:
        """
        Compute trial-level risky choice proportions across participants.
        Shape: (n_trials,) where value_t = mean(chose_risky) at trial t.
        Handles missing trials gracefully.
        """
        max_trial = max(df['trial'].max() for df in dfs)
        sums   = np.zeros(int(max_trial) + 1)
        counts = np.zeros(int(max_trial) + 1)
        for df in dfs:
            cr = (df['action'] == 'risky').astype(float).values
            t  = df['trial'].astype(int).values
            for i, ti in enumerate(t):
                if ti <= max_trial:
                    sums[ti]   += cr[i]
                    counts[ti] += 1
        mask = counts > 0
        soft = np.where(mask, sums / np.maximum(counts, 1), 0.5)
        return soft

    def _group_objective(self, param_vector: np.ndarray) -> float:
        if self.base_config.training_objective == 'metric':
            return self._group_metric_objective(param_vector)
        return self._group_mle_objective(param_vector)

    def _group_mle_objective(self, param_vector: np.ndarray) -> float:
        """
        Soft-target cross-entropy: compare model's P(risky_t) to the group
        proportion at each trial t.
        """
        from pipeline import build_pipeline

        cfg      = self._vector_to_config(param_vector)
        pipeline = build_pipeline(cfg)
        targets  = self._soft_targets
        loss     = 0.0

        for t, p_risky_human in enumerate(targets):
            mem_est     = pipeline.memory.retrieve(pipeline._options, t)
            latent      = pipeline.latent_state.build(mem_est, cfg)
            transformed = pipeline.context_transform.transform(latent)
            choice_out  = pipeline.choice_policy.compute_probabilities(
                transformed, pipeline._options,
                context={'last_action': getattr(pipeline, '_last_action', None)}
            )

            p_risky_model = choice_out.probabilities.get('risky', 0.5)
            p_risky_model = np.clip(p_risky_model, 1e-10, 1 - 1e-10)

            # Binary cross-entropy with soft target
            loss -= (p_risky_human * np.log(p_risky_model)
                     + (1 - p_risky_human) * np.log(1 - p_risky_model))

            # Use majority choice to update memory
            majority_action  = 'risky' if p_risky_human >= 0.5 else 'safe'
            majority_outcome = pipeline.env.risky_values[0] if majority_action == 'risky' \
                               else pipeline.env.safe_value
            majority_pt      = pipeline.valuation.evaluate(majority_outcome)
            chosen_idx       = pipeline._option_to_idx.get(majority_action, 1)

            pipeline.learning.update(
                action=majority_action, chosen_idx=chosen_idx,
                raw_outcome=majority_outcome, pt_value=majority_pt,
                memory=pipeline.memory, latent_state=pipeline.latent_state,
                memory_estimate=mem_est, current_time=t,
            )
            pipeline._last_action = majority_action

        return float(loss)

    def _group_metric_objective(self, param_vector: np.ndarray) -> float:
        from pipeline import build_pipeline

        cfg  = self._vector_to_config(param_vector)
        n_sims = self.base_config.n_metric_sims
        all_metrics = []

        for sim_idx in range(n_sims):
            sim_cfg      = deepcopy(cfg)
            sim_cfg.seed = sim_idx
            pipe = build_pipeline(sim_cfg)
            df   = pipe.run_experiment()
            all_metrics.append(compute_metrics(df, self.base_config.metrics_to_use))

        metric_names = list(all_metrics[0].keys())
        model_metrics = {
            name: float(np.nanmean([m[name] for m in all_metrics]))
            for name in metric_names
        }
        return metric_loss(model_metrics, self._group_metrics,
                           self.base_config.metric_weights or None)

    def _initial_point(self, restart: int) -> np.ndarray:
        if restart == 0:
            return np.array([init for _, init, _, _ in self._param_defs])
        return np.array([float(self._rng.uniform(lo + 1e-3, hi - 1e-3))
                          for _, _, lo, hi in self._param_defs])

    def _vector_to_config(self, param_vector: np.ndarray) -> ModelConfig:
        cfg = deepcopy(self.base_config)
        cfg.log_experiments = False
        for (key, _, lo, hi), val in zip(self._param_defs, param_vector):
            setattr(cfg, key, float(np.clip(val, lo + 1e-9, hi - 1e-9)))
        return cfg
