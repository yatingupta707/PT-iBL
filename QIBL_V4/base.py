"""
components/base.py  v4
──────────────────────
v4 changes:
  - LatentStateData: q_vec and phi_vec added for N-option support.
    q (float) and phi (float) remain for backward compat and 2-option logging.
  - BaseLatentState.update_phase(): now takes option_idx so the learning
    component can update only the CHOSEN option's phase.
  - BaseLearning.update(): now receives chosen_idx (int) so it can tell the
    latent state which option's phase to evolve.
  - BaseChoicePolicy.compute_probabilities(): receives optional context dict
    so BornRulePolicy can apply pInertia using last_action from context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Shared data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrialConfig:
    """Per-trial environment configuration (DataDrivenEnv)."""
    safe_value:   float
    risky_values: List[float]
    risky_probs:  List[float]
    meta:         Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEstimate:
    """
    Output of the Memory component.
    Keys in blended_values are ordered to match env.get_options().
    Index 0 = first option (safe-analog), index 1 = second (risky-analog), etc.
    """
    blended_values: Dict[str, float]
    uncertainties:  Dict[str, float]
    n_instances:    Dict[str, int]


@dataclass
class LatentStateData:
    """
    Output of the LatentState component.

    2-option backward compat fields:
      amplitudes : complex ndarray shape (n_options,)
      q          : preference toward option-1 (float) — or max preference for N>2
      phi        : phase of option-1 (float) — or phase of max-preference option

    N-option extension fields (None when n_options == 2):
      q_vec      : ndarray shape (n_options,) — softmax preference over all options
      phi_vec    : ndarray shape (n_options,) — phase per option
    """
    amplitudes: np.ndarray
    q:          float
    phi:        float
    q_vec:      Optional[np.ndarray] = None
    phi_vec:    Optional[np.ndarray] = None

    @property
    def n_options(self) -> int:
        return len(self.amplitudes)


@dataclass
class TransformedStateData:
    """Output of the ContextTransform component."""
    amplitudes:        np.ndarray
    theta:             float
    interference_term: float   # sin(2θ)·√(q(1-q))·cos(φ) for 2 options; pairwise avg for N>2


@dataclass
class ChoiceOutput:
    """Output of the ChoicePolicy component."""
    probabilities: Dict[str, float]
    action:        str
    log_prob:      float


@dataclass
class TrialRecord:
    """Complete record of one trial."""
    trial_num:         int
    action:            str
    raw_outcome:       float
    pt_value:          float
    choice_prob:       float
    interference_term: float
    phase_before:      float
    phase_after:       float
    q_before:          float
    blended_values:    Dict[str, float]
    prediction_error:  float
    env_info:          Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Abstract base classes
# ═══════════════════════════════════════════════════════════════════

class BaseEnvironment(ABC):
    """External task. Knows NOTHING about cognition."""

    @abstractmethod
    def step(self, action: str) -> Tuple[float, dict]: ...

    @abstractmethod
    def get_options(self) -> List[str]: ...

    @abstractmethod
    def reset(self) -> None: ...


class BaseValuation(ABC):
    """Transforms objective outcomes into subjective PT values."""

    @abstractmethod
    def evaluate(self, outcome: float) -> float: ...

    @abstractmethod
    def get_params(self) -> dict: ...


class BaseMemory(ABC):
    """Stores and retrieves past experiences."""

    @abstractmethod
    def retrieve(self, options: List[str], current_time: int) -> MemoryEstimate: ...

    @abstractmethod
    def store(self, option: str, pt_value: float, raw_outcome: float, time: int) -> None:
        """
        Store an experience.
        pt_value    → used for blending (learning in utility/PT space)
        raw_outcome → stored for audit/analysis
        """

    @abstractmethod
    def reset(self) -> None: ...


class BaseLatentState(ABC):
    """Maintains internal cognitive state between trials."""

    @abstractmethod
    def build(self, memory_estimate: MemoryEstimate, config: Any) -> LatentStateData:
        """
        IBL → |ψ⟩ bridge.

        OPTION ORDERING CONTRACT:
          memory_estimate.blended_values keys are ordered to match env.get_options().
          Index 0 = first option (safe-analog), index 1 = risky-analog, etc.
          Do NOT match on option name strings.
        """

    @abstractmethod
    def get_state(self) -> LatentStateData: ...

    @abstractmethod
    def update_phase(self, new_phi: float, option_idx: int = 1) -> None:
        """
        Update phase for ONE option.

        v4 change: option_idx specifies which option's phase to update.
        Default = 1 (risky, for 2-option backward compat).
        The learning component passes the index of the CHOSEN option.
        """


class BaseContextTransform(ABC):
    """Transforms latent state before choice measurement."""

    @abstractmethod
    def transform(self, state: LatentStateData,
                  context: Optional[dict] = None) -> TransformedStateData: ...

    @abstractmethod
    def get_theta(self) -> float: ...


class BaseChoicePolicy(ABC):
    """Converts transformed state into action probabilities."""

    @abstractmethod
    def compute_probabilities(self,
                               transformed: TransformedStateData,
                               options:     List[str],
                               context:     Optional[dict] = None) -> ChoiceOutput:
        """
        v4 change: context dict may include 'last_action' for pInertia.
        """


class BaseLearning(ABC):
    """Updates memory and phase after outcome is observed."""

    @abstractmethod
    def update(self,
               action:          str,
               chosen_idx:      int,
               raw_outcome:     float,
               pt_value:        float,
               memory:          BaseMemory,
               latent_state:    BaseLatentState,
               memory_estimate: MemoryEstimate,
               current_time:    int) -> dict:
        """
        v4 change: chosen_idx (int) is the index of the chosen option in
        env.get_options(). Used to update only the chosen option's phase.
        """
