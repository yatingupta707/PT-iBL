"""
utils/behaviour_metrics.py
───────────────────────────
Behavioural metrics for comparing model output against human data.

Used in two ways:
  1. INSPECTION: Compute metrics from results DataFrame to characterise behaviour.
  2. TRAINING (metric-based fitting): Compare model metrics to human metrics
     as the objective function instead of trial-by-trial log-likelihood.

AVAILABLE METRICS:
  risk_rate          — proportion of risky choices
  alternation_rate   — proportion of trials where choice differs from previous
  reveal_rate        — proportion of 'reveal'/'sample' choices (sampling paradigms)
  win_stay_rate      — P(repeat choice | last outcome ≥ 0 after PT)
  lose_shift_rate    — P(switch choice | last outcome < 0 after PT)
  switch_after_loss  — alias for lose_shift_rate

These metrics capture AGGREGATE PATTERNS in behaviour rather than trial-level
probabilities. Using them as training targets (training_objective='metric') is
appropriate when:
  - You have summary-level data (just aggregate stats, not trial sequences)
  - You want to match population-level behaviour patterns
  - MLE is unstable due to extreme choices (all risky / all safe)

METRIC-BASED FITTING:
  The parameter estimator runs the model in FORWARD SIMULATION mode (not
  teacher-forcing): the model makes its own choices and learns from its own
  outcomes. This produces model metrics that can be compared to human metrics.

  Objective: minimize  Σ_m w_m · (model_m − human_m)²
  where m runs over the metrics in config.metrics_to_use
  and w_m are optional weights (default all 1.0).

INDIVIDUAL vs GROUP:
  For individual fitting: compute metrics per participant, fit to that individual.
  For group fitting: average metrics across participants, fit one model to the mean.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ── Core metric functions ─────────────────────────────────────────────────────

def risk_rate(df: pd.DataFrame, risky_option: str = 'risky') -> float:
    """Proportion of trials where the risky option was chosen."""
    if 'chose_risky' in df.columns:
        return float(df['chose_risky'].mean())
    if 'action' in df.columns:
        return float((df['action'] == risky_option).mean())
    raise ValueError("DataFrame must have 'chose_risky' or 'action' column.")


def alternation_rate(df: pd.DataFrame) -> float:
    """
    Proportion of trials where the choice DIFFERS from the previous trial.

    First trial has no predecessor → excluded from calculation.
    Range: [0, 1]. Higher = more switching.
    """
    if 'chose_risky' in df.columns:
        choices = df['chose_risky'].values
    elif 'action' in df.columns:
        # Convert to binary (risky=1, safe/other=0)
        choices = (df['action'] == 'risky').astype(int).values
    else:
        raise ValueError("DataFrame must have 'chose_risky' or 'action' column.")

    if len(choices) < 2:
        return 0.0
    return float(np.mean(choices[1:] != choices[:-1]))


def reveal_rate(df: pd.DataFrame, reveal_option: str = 'reveal') -> float:
    """
    Proportion of 'reveal' or 'sample' choices.
    For sampling paradigms where the agent can sample outcomes before committing.
    Returns 0.0 if the option is not present in the data.
    """
    if 'action' not in df.columns:
        return 0.0
    return float((df['action'] == reveal_option).mean())


def win_stay_rate(df: pd.DataFrame, win_threshold: float = 0.0) -> float:
    """
    P(same choice on trial t | outcome on trial t-1 ≥ win_threshold).
    'Win' defined as pt_value ≥ win_threshold (in PT-value space).

    Uses pt_value column if available, else raw_outcome.
    """
    if len(df) < 2:
        return np.nan

    value_col = 'pt_value' if 'pt_value' in df.columns else 'raw_outcome'
    chose_risky = df['chose_risky'].values if 'chose_risky' in df.columns \
                  else (df['action'] == 'risky').astype(int).values
    outcomes    = df[value_col].values

    wins   = outcomes[:-1] >= win_threshold   # was last trial a win?
    stayed = (chose_risky[1:] == chose_risky[:-1])  # same choice?

    if wins.sum() == 0:
        return np.nan
    return float(stayed[wins].mean())


def lose_shift_rate(df: pd.DataFrame, win_threshold: float = 0.0) -> float:
    """
    P(switch choice on trial t | outcome on trial t-1 < win_threshold).
    Complement of win-stay: measures whether losses trigger switching.
    """
    if len(df) < 2:
        return np.nan

    value_col   = 'pt_value' if 'pt_value' in df.columns else 'raw_outcome'
    chose_risky = df['chose_risky'].values if 'chose_risky' in df.columns \
                  else (df['action'] == 'risky').astype(int).values
    outcomes    = df[value_col].values

    losses   = outcomes[:-1] < win_threshold
    switched = (chose_risky[1:] != chose_risky[:-1])

    if losses.sum() == 0:
        return np.nan
    return float(switched[losses].mean())


# Alias
switch_after_loss = lose_shift_rate


# ── Registry and batch computation ───────────────────────────────────────────

METRIC_REGISTRY: Dict[str, callable] = {
    'risk_rate':         risk_rate,
    'alternation_rate':  alternation_rate,
    'reveal_rate':       reveal_rate,
    'win_stay_rate':     win_stay_rate,
    'lose_shift_rate':   lose_shift_rate,
    'switch_after_loss': switch_after_loss,
}


def compute_metrics(df: pd.DataFrame,
                    metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute all requested metrics from a results DataFrame.

    Parameters
    ----------
    df      : results DataFrame from pipeline.run_experiment()
    metrics : list of metric names to compute. Default: all in METRIC_REGISTRY.

    Returns
    -------
    dict {metric_name: value}. NaN values are included.
    """
    if metrics is None:
        metrics = list(METRIC_REGISTRY.keys())

    result = {}
    for name in metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY)}")
        try:
            result[name] = float(METRIC_REGISTRY[name](df))
        except Exception as e:
            result[name] = np.nan
    return result


def compute_group_metrics(dfs: List[pd.DataFrame],
                          metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute metrics for each participant and return the mean across participants.
    NaN values are ignored in the mean.

    Parameters
    ----------
    dfs     : list of DataFrames, one per participant
    metrics : metric names to compute

    Returns
    -------
    dict {metric_name: group_mean}
    """
    if not dfs:
        raise ValueError("dfs list is empty.")

    per_participant = [compute_metrics(df, metrics) for df in dfs]
    metric_names   = list(per_participant[0].keys())

    return {
        name: float(np.nanmean([m[name] for m in per_participant]))
        for name in metric_names
    }


def metric_loss(model_metrics:  Dict[str, float],
                target_metrics: Dict[str, float],
                weights:        Optional[Dict[str, float]] = None) -> float:
    """
    Weighted squared loss between model and target metrics.

        L = Σ_m w_m · (model_m − target_m)²

    NaN values are skipped. Returns 0 if no shared metrics.

    Parameters
    ----------
    model_metrics  : metrics computed from model forward simulation
    target_metrics : metrics from human data (or group average)
    weights        : per-metric weights (default: all 1.0)
    """
    loss = 0.0
    n    = 0
    for name, target_val in target_metrics.items():
        model_val = model_metrics.get(name, np.nan)
        if np.isnan(target_val) or np.isnan(model_val):
            continue
        w     = (weights or {}).get(name, 1.0)
        loss += w * (model_val - target_val) ** 2
        n    += 1
    return float(loss)


def metrics_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted single-row DataFrame of all metrics."""
    m = compute_metrics(df)
    return pd.DataFrame([m])
