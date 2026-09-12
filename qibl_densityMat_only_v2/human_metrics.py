"""
human_metrics.py
─────────────────
Human behavioural benchmark time series.
All four arrays are shape (100,) — cumulative means up to each trial,
averaged across all participants and all problems in each set.

R-rate: cumulative proportion of risky choices
A-rate: cumulative proportion of alternation events (choice ≠ previous choice)

REPLACE human_a_ts_est, human_r_ts_comp, human_a_ts_comp
with your actual data from the original human_metrc.py file.
"""

import ast
import csv
import os
import numpy as np

# ── Estimation set (training) R-rate ──────────────────────────────────────────────

human_r_ts_est = np.array([
    0.5175, 0.5025, 0.5200, 0.476667, 0.491667, 0.4625, 0.4950, 0.455833, 0.469167, 0.485833,
    0.4575, 0.454167, 0.451667, 0.450833, 0.461667, 0.4525, 0.436667, 0.423333, 0.4475, 0.433333,
    0.4325, 0.441667, 0.4325, 0.4525, 0.4375, 0.436667, 0.410833, 0.4325, 0.410833, 0.410833,
    0.3950, 0.4100, 0.3850, 0.4000, 0.380833, 0.405833, 0.4000, 0.406667, 0.380833, 0.385833,
    0.3700, 0.4025, 0.375833, 0.390833, 0.3925, 0.4025, 0.375833, 0.385833, 0.383333, 0.3825,
    0.3700, 0.3750, 0.365833, 0.3750, 0.375833, 0.386667, 0.379167, 0.3825, 0.363333, 0.369167,
    0.373333, 0.3775, 0.360833, 0.374167, 0.366667, 0.361667, 0.353333, 0.364167, 0.365833, 0.3575,
    0.3675, 0.360833, 0.360833, 0.356667, 0.346667, 0.3500, 0.3475, 0.360833, 0.3550, 0.366667,
    0.360833, 0.3700, 0.365833, 0.3600, 0.351667, 0.350833, 0.356667, 0.358333, 0.354167, 0.3550,
    0.3500, 0.349167, 0.363333, 0.349167, 0.361667, 0.356667, 0.360833, 0.359167, 0.349167, 0.336667
])

# ── A-rate for estimation set ────────────────────────────────────

human_a_ts_est = np.array([
    0.0,0.856667,0.554167,0.426667,0.361667,0.2975,0.275833,0.259167,0.2450,0.233333,
    0.221667,0.206667,0.190833,0.179167,0.175833,0.160833,0.175833,0.158333,0.154167,0.140833,
    0.154167,0.1475,0.144167,0.14,0.121667,0.130833,0.119167,0.121667,0.128333,0.12,
    0.104167,0.118333,0.118333,0.12,0.125,0.1175,0.115833,0.123333,0.115833,0.113333,
    0.1125,0.120833,0.111667,0.095,0.118333,0.113333,0.116667,0.123333,0.109167,0.1125,
    0.120833,0.135,0.105833,0.115833,0.1025,0.095833,0.110833,0.101667,0.110833,0.114167,
    0.1125,0.109167,0.101667,0.108333,0.1075,0.115,0.108333,0.100833,0.116667,0.108333,
    0.101667,0.11,0.101667,0.1025,0.098333,0.11,0.099167,0.101667,0.094167,0.093333,
    0.1025,0.1025,0.1025,0.094167,0.093333,0.089167,0.090833,0.095,0.105833,0.084167,
    0.095,0.099167,0.084167,0.094167,0.094167,0.086667,0.094167,0.078333,0.085,0.084167
])
# ── R-rate for competition set ───────────────────────────────────
human_r_ts_comp = np.array([
    0.49,0.50,0.48,0.49,0.48,0.46,0.45,0.44,0.44,0.43,0.45,0.43,0.43,0.43,0.43,0.43,
    0.42,0.41,0.40,0.40,0.42,0.41,0.41,0.41,0.41,0.40,0.40,0.39,0.40,0.39,0.38,0.38,
    0.39,0.38,0.38,0.38,0.38,0.39,0.38,0.38,0.37,0.37,0.37,0.37,0.37,0.37,0.38,0.37,
    0.38,0.36,0.37,0.37,0.37,0.38,0.36,0.37,0.37,0.36,0.35,0.36,0.35,0.36,0.35,0.36,
    0.35,0.35,0.35,0.37,0.37,0.36,0.35,0.34,0.34,0.35,0.34,0.33,0.34,0.36,0.36,0.35,
    0.35,0.35,0.35,0.36,0.35,0.36,0.34,0.36,0.35,0.35,0.35,0.35,0.34,0.34,0.34,0.35,
    0.34,0.35,0.36,0.35
])


# ── A-rate for competition set ───────────────────────────────────
human_a_ts_comp = np.array([
    0.0,0.82,0.53,0.42,0.37,0.30,0.26,0.24,0.23,0.21,0.21,0.20,0.18,0.17,0.17,0.16,
    0.17,0.15,0.17,0.13,0.13,0.13,0.13,0.13,0.13,0.14,0.13,0.13,0.13,0.11,0.12,0.12,
    0.11,0.12,0.11,0.12,0.11,0.11,0.11,0.11,0.11,0.10,0.11,0.10,0.10,0.10,0.11,0.10,
    0.10,0.10,0.10,0.11,0.08,0.10,0.10,0.10,0.11,0.10,0.10,0.10,0.09,0.10,0.10,0.10,
    0.09,0.08,0.09,0.09,0.09,0.09,0.08,0.09,0.08,0.10,0.09,0.08,0.08,0.08,0.08,0.08,
    0.08,0.07,0.08,0.08,0.07,0.08,0.08,0.07,0.07,0.08,0.08,0.08,0.08,0.08,0.08,0.09,
    0.08,0.08,0.08,0.08
])


def _series_mean(rows, key, n_trials):
    values = np.full((len(rows), n_trials), np.nan, dtype=float)
    for row_i, row in enumerate(rows):
        series = np.asarray(ast.literal_eval(row[key]), dtype=float)
        values[row_i, :len(series)] = series[:n_trials]
    return np.nanmean(values, axis=0)


def _load_combined_human_data():
    """Load three-choice human rates from the combined behavioural CSV."""
    data_path = os.environ.get(
        'PTIBL_HUMAN_DATA_FILE',
        os.path.join(os.path.dirname(__file__), '..', 'tdcs_load_60_final_comb.csv'),
    )
    if not os.path.exists(data_path):
        return None

    with open(data_path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    groups = {
        'est': [row for row in rows if 'load_0' in row['condition']],
        'comp': [row for row in rows if 'load_1' in row['condition']],
    }
    result = {}
    for split, split_rows in groups.items():
        if not split_rows:
            continue
        split_trials = max(
            len(ast.literal_eval(row['risk_series'])) for row in split_rows
        )
        risk = _series_mean(split_rows, 'risk_series', split_trials)
        reveal = _series_mean(split_rows, 'reveal_series', split_trials)
        alternate = _series_mean(split_rows, 'alt_series', split_trials)
        result[split] = {
            'risk': np.cumsum(risk) / np.arange(1, split_trials + 1),
            'reveal': np.cumsum(reveal) / np.arange(1, split_trials + 1),
            'alternate': np.cumsum(alternate) / np.arange(1, split_trials + 1),
        }
    result['conditions'] = {}
    for condition in sorted({row['condition'] for row in rows}):
        condition_rows = [row for row in rows if row['condition'] == condition]
        condition_trials = max(
            len(ast.literal_eval(row['risk_series'])) for row in condition_rows
        )
        risk = _series_mean(condition_rows, 'risk_series', condition_trials)
        alternate = _series_mean(condition_rows, 'alt_series', condition_trials)
        result['conditions'][condition] = {
            'risk': np.cumsum(risk) / np.arange(1, condition_trials + 1),
            'alternate': np.cumsum(alternate) / np.arange(1, condition_trials + 1),
        }
    return result


_combined_human = _load_combined_human_data()
if _combined_human:
    human_r_ts_est = _combined_human['est']['risk']
    human_reveal_ts_est = _combined_human['est']['reveal']
    human_a_ts_est = _combined_human['est']['alternate']
    human_r_ts_comp = _combined_human['comp']['risk']
    human_reveal_ts_comp = _combined_human['comp']['reveal']
    human_a_ts_comp = _combined_human['comp']['alternate']
else:
    human_reveal_ts_est = np.zeros_like(human_r_ts_est)
    human_reveal_ts_comp = np.zeros_like(human_r_ts_comp)

human_condition_series = (
    _combined_human.get('conditions', {}) if _combined_human else {}
)
