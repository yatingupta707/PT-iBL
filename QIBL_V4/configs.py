"""
configs.py  v4
───────────────
v4 changes:
  - IBL: ibl_decay_d (replaces ibl_decay), ibl_noise_sigma — both learnable
  - Choice: ibl_p_inertia — learnable (3rd IBL parameter, in choice policy)
  - Fitting: fitting_mode ('individual' | 'group')
  - Training objective: training_objective ('mle' | 'metric')
  - Metrics: metrics_to_use, metric_weights, n_metric_sims
  - valuation_order: 'before_storage' (default) | 'after_retrieval'
  - learnable_params updated to include all 8 learnable parameters
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math


@dataclass
class ModelConfig:
    """All parameters for the PT-IBL-Quantum pipeline (v4)."""

    # ── Meta ─────────────────────────────────────────────────────────────────
    seed: Optional[int] = 42

    # ── Component selections ──────────────────────────────────────────────────
    latent_state_type: str = 'quantum'
    context_type:      str = 'rotation'
    choice_type:       str = 'born_rule'
    learning_type:     str = 'prediction_error'

    # ── Environment / DataLoader ──────────────────────────────────────────────
    dataloader_type:          str = 'config'
    dataloader_sampling_mode: str = 'sequential'
    safe_value:               float       = 2.0
    risky_values:             List[float] = field(default_factory=lambda: [0.0, 4.0])
    risky_probs:              List[float] = field(default_factory=lambda: [0.5, 0.5])

    # ── Prospect Theory  [LEARNABLE] ─────────────────────────────────────────
    pt_alpha:           float = 0.88    # gain curvature
    pt_beta:            float = 0.88    # loss curvature
    pt_lambda:          float = 2.25    # loss aversion
    pt_reference_point: float = 0.0    # FIXED (set to task midpoint if needed)

    # PT application order:
    # 'before_storage'  → v(raw) → store v → blend v  (DEFAULT, v(avg) ≠ avg(v))
    # 'after_retrieval' → store raw → blend raw → v(blended)  (alternative)
    valuation_order: str = 'before_storage'

    # ── IBL Memory  [ALL LEARNABLE in v4] ────────────────────────────────────
    ibl_decay_d:     float = 0.5    # d — power-law forgetting exponent
                                     # Replaces v3 ibl_decay (exponential γ)
    ibl_noise_sigma: float = 0.25   # σ_s — Gaussian noise on log-activations
    ibl_default_val: float = 0.0    # FIXED — neutral seed value
    ibl_n_init:      int   = 3      # FIXED — seed instances per option

    # ── Choice Inertia  [LEARNABLE in v4] ────────────────────────────────────
    ibl_p_inertia: float = 0.0     # p — response repetition probability
                                    # Applied in BornRulePolicy AFTER Born rule
                                    # P_final = p·I(repeat) + (1-p)·P_born

    # ── Latent State  [LEARNABLE: beta_q] ────────────────────────────────────
    beta_q:   float = 1.0          # softmax temperature for q mapping
    phi_init: float = 0.0          # FIXED — initial phase for all options

    # ── Context Transform  [LEARNABLE: theta] ────────────────────────────────
    theta: float = math.pi / 4

    # ── Softmax (ablation only)  [FIXED] ─────────────────────────────────────
    softmax_tau: float = 1.0

    # ── Online Learning  [FIXED] ─────────────────────────────────────────────
    alpha_phi: float = 0.1         # within-task phase learning rate
                                    # NOT fitted via MLE (see theory.md §9)

    # ── Experiment ────────────────────────────────────────────────────────────
    n_trials: int = 100

    # ── Parameter estimation ──────────────────────────────────────────────────
    fitting_mode: str = 'individual'
    # 'individual' — fit one model per participant (returns one config per participant)
    # 'group'      — average participant metrics, fit one model to group average

    training_objective: str = 'mle'
    # 'mle'    — teacher-forcing negative log-likelihood (trial-by-trial)
    # 'metric' — forward simulation + behavioural metric matching

    metrics_to_use: List[str] = field(default_factory=lambda: [
        'risk_rate', 'alternation_rate', 'win_stay_rate', 'lose_shift_rate'
    ])
    # metrics used when training_objective='metric'. See utils/behaviour_metrics.py.

    metric_weights: Dict[str, float] = field(default_factory=dict)
    # Optional per-metric weights for metric-based objective. Default: all 1.0.

    n_metric_sims: int = 10
    # Number of forward simulation runs to average for metric-based fitting.
    # Higher = less noisy estimates; lower = faster fitting.

    learnable_params: List[str] = field(default_factory=lambda: [
        # PT (3 params)
        'pt_alpha',
        'pt_beta',
        'pt_lambda',
        # IBL (2 memory params + 1 choice param)
        'ibl_decay_d',
        'ibl_noise_sigma',
        'ibl_p_inertia',
        # Latent state
        'beta_q',
        # Context
        'theta',
    ])
    # 8 learnable parameters total. Remove any entry to fix that parameter.

    fit_nll: Optional[float] = None     # set by ParameterEstimator after fitting
    fit_loss: Optional[float] = None    # set for metric-based fitting

    # ── Logging ───────────────────────────────────────────────────────────────
    log_experiments: bool = True
    log_dir:         str  = 'logs'
    experiment_name: str  = 'experiment'

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, list)
                    else dict(v) if isinstance(v, dict) else v)
                for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        return (
            f"ModelConfig v4(\n"
            f"  [PT]      α={self.pt_alpha:.3f} β={self.pt_beta:.3f} λ={self.pt_lambda:.3f}\n"
            f"  [IBL]     d={self.ibl_decay_d:.3f} σ_s={self.ibl_noise_sigma:.3f} "
            f"p_in={self.ibl_p_inertia:.3f}\n"
            f"  [State]   β_q={self.beta_q:.3f}  θ={self.theta:.3f}  "
            f"α_φ={self.alpha_phi} [fixed]\n"
            f"  [Fitting] mode={self.fitting_mode}  obj={self.training_objective}\n"
            f"  [Arch]    {self.latent_state_type}/{self.context_type}/"
            f"{self.choice_type}/{self.learning_type}\n"
            f")"
        )
