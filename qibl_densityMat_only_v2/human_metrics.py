"""
human_metrics.py
─────────────────
Human behavioural time series for the tDCS/load dataset.

Each participant contributes one 50-trial sequence of A/B/R choices.
A = safe, B = risky, R = reveal.

Aggregation (same as the original 2-choice pipeline):
  1. Instantaneous 0/1 series per person (risk, alteration)
  2. Mean across people at each trial
  3. Cumulative mean: cumsum(inst) / trial_index

This is equivalent to averaging per-person running means, because mean
and cumsum commute. We recompute from raw `choices`, not from the CSV
`cum_*` columns, so reveal-aware alteration is applied uniformly.

Splits
  estimation / load_0  — pooled tDCS_0_load_0 + tDCS_1_load_0
  competition / load_1 — pooled tDCS_0_load_1 + tDCS_1_load_1
  conditions           — each tDCS × load cell separately
"""

from __future__ import annotations

import ast
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models import alternation_series, risk_series

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.abspath(
    os.path.join(_HERE, '..', 'tdcs_load_60_final_comb.csv')
)

CONDITION_NAMES = [
    'tDCS_0_load_0',
    'tDCS_1_load_0',
    'tDCS_0_load_1',
    'tDCS_1_load_1',
]


def condition_problem_split(condition: Optional[str]) -> str:
    """load_0 is scored on the estimation problems; load_1 on competition."""
    if not condition:
        return 'est'
    return 'est' if str(condition).endswith('load_0') else 'comp'


def _parse_list(raw) -> list:
    """Parse a CSV cell that stores a Python/JSON list, including trailing nan."""
    if isinstance(raw, list):
        values = raw
    elif not isinstance(raw, str):
        values = [raw]
    else:
        text = raw.strip()
        try:
            values = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            values = json.loads(
                text.replace('nan', 'null').replace("'", '"')
            )
    cleaned = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, float) and np.isnan(item):
            continue
        if isinstance(item, str) and item.lower() in ('nan', 'none', 'null', ''):
            continue
        cleaned.append(item)
    return cleaned


def _parse_choice_list(raw) -> List[str]:
    return [str(item).strip() for item in _parse_list(raw)]


def cumulative_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    return np.cumsum(x) / np.arange(1, len(x) + 1, dtype=float)


def _stack_mean(series_list: List[np.ndarray], n_trials: int) -> np.ndarray:
    if not series_list:
        return np.zeros(n_trials, dtype=float)
    mat = np.full((len(series_list), n_trials), np.nan, dtype=float)
    for i, series in enumerate(series_list):
        n = min(len(series), n_trials)
        mat[i, :n] = np.asarray(series, dtype=float)[:n]
    with np.errstate(all='ignore'):
        out = np.nanmean(mat, axis=0)
    return np.where(np.isnan(out), 0.0, out)


def _pack(risk_inst: np.ndarray, alt_inst: np.ndarray,
          n_people: int) -> dict:
    return {
        'n_people': int(n_people),
        'n_trials': int(len(risk_inst)),
        'risk_inst': risk_inst,
        'alternate_inst': alt_inst,
        'risk': cumulative_mean(risk_inst),
        'alternate': cumulative_mean(alt_inst),
    }


def load_human_data(csv_path: Optional[str] = None) -> dict:
    path = csv_path or os.environ.get('PTIBL_HUMAN_CSV', _DEFAULT_CSV)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Human CSV not found: {path}. '
            'Expected tdcs_load_60_final_comb.csv next to the project folder.'
        )

    df = pd.read_csv(path)
    if 'choices' not in df.columns or 'condition' not in df.columns:
        raise ValueError('CSV must have columns: condition, choices')

    people = []
    for _, row in df.iterrows():
        actions = _parse_choice_list(row['choices'])
        if not actions:
            continue
        people.append({
            'condition': str(row['condition']).strip(),
            'pid': row.get('pid'),
            'actions': actions,
            'risk': risk_series(actions),
            'alternate': alternation_series(actions),
        })

    if not people:
        raise ValueError(f'No valid choice sequences in {path}')

    # Designed experiment length is 50. A couple of rows are 51/55 because
    # of padding; use the modal length so late trials are not all-NaN.
    n_trials = Counter(len(p['actions']) for p in people).most_common(1)[0][0]

    def subset(predicate) -> dict:
        rows = [p for p in people if predicate(p)]
        if not rows:
            z = np.zeros(n_trials, dtype=float)
            return _pack(z, z, 0)
        return _pack(
            _stack_mean([p['risk'] for p in rows], n_trials),
            _stack_mean([p['alternate'] for p in rows], n_trials),
            len(rows),
        )

    result = {
        'csv_path': os.path.abspath(path),
        'n_trials': n_trials,
        'est':  subset(lambda p: p['condition'].endswith('load_0')),
        'comp': subset(lambda p: p['condition'].endswith('load_1')),
        'conditions': {},
    }
    for condition in CONDITION_NAMES:
        result['conditions'][condition] = subset(
            lambda p, c=condition: p['condition'] == c
        )
    # Include any extra condition names that appear in the file.
    for condition in sorted({p['condition'] for p in people}):
        if condition not in result['conditions']:
            result['conditions'][condition] = subset(
                lambda p, c=condition: p['condition'] == c
            )
    return result


_combined_human = load_human_data()

N_TRIALS = int(_combined_human['n_trials'])

human_r_ts_est = _combined_human['est']['risk']
human_a_ts_est = _combined_human['est']['alternate']
human_r_inst_est = _combined_human['est']['risk_inst']
human_a_inst_est = _combined_human['est']['alternate_inst']

human_r_ts_comp = _combined_human['comp']['risk']
human_a_ts_comp = _combined_human['comp']['alternate']
human_r_inst_comp = _combined_human['comp']['risk_inst']
human_a_inst_comp = _combined_human['comp']['alternate_inst']

human_condition_series = _combined_human['conditions']


def condition_n_trials(condition=None) -> int:
    if not condition:
        return int(len(human_r_ts_est))
    return int(len(human_condition_series[condition]['risk']))
