"""
evaluate.py
══════════════════════════════════════════════════════════════════════════════
Loads fitted parameters from a training run and evaluates all models
on the estimation (load_0) and competition (load_1) problem sets.

Saves inside each model's subdirectory:
  eval_train_r_ts.npy   — cumulative R-rate on estimation set
  eval_train_a_ts.npy   — cumulative A-rate on estimation set
  eval_test_r_ts.npy    — cumulative R-rate on competition set
  eval_test_a_ts.npy    — cumulative A-rate on competition set
  eval_summary.json     — MSD and correlation vs human benchmarks

Human comparisons:
  train  vs pooled load_0 humans
  test   vs pooled load_1 humans
  plus per-condition MSD/corr (same model series overlaid on that cell)

Results are averaged over --n-sims independent simulations.
No post-hoc rescaling: model and human series use the same 0/1 coding,
the same reveal-aware alteration rule, and the same cumsum/t transform.

USAGE
------
  python evaluate.py --run-dir runs/exp1_15epochs --n-sims 5 --n-agents 20
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import (
    ALL_MODELS, alternation_series, problem_seed_base, risk_series,
)
from human_metrics import (
    N_TRIALS, CONDITION_NAMES, condition_problem_split,
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp,
    human_condition_series,
)

COLS = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']


def _complete_params(model_class, params: dict) -> dict:
    full = model_class.default_params()
    full.update(params)
    return full


def _run_one_problem(model_class, params, row, n_agents, seed_base=0):
    risky_values = (float(row['val_high']), float(row['val_low']))
    risky_probs  = (float(row['p_high']), 1.0 - float(row['p_high']))
    safe_value   = float(row['val_safe'])

    R = np.zeros((n_agents, N_TRIALS))
    A = np.zeros((n_agents, N_TRIALS))

    for ai in range(n_agents):
        model   = model_class(params)
        actions = model.simulate(N_TRIALS, safe_value,
                                  risky_values, risky_probs,
                                  seed=seed_base + ai)
        R[ai] = risk_series(actions)
        A[ai] = alternation_series(actions)

    return R.mean(axis=0), A.mean(axis=0)


def eval_ts_once(dataset, model_class, params, n_agents, seed_offset=0):
    """One forward pass over entire dataset. Returns (r_cum, a_cum)."""
    P     = len(dataset)
    r_acc = np.zeros(N_TRIALS)
    a_acc = np.zeros(N_TRIALS)

    for prob_i, (_, row) in enumerate(dataset.iterrows()):
        seed_base = problem_seed_base(prob_i, n_agents, P, seed_offset)
        r_row, a_row = _run_one_problem(model_class, params, row,
                                         n_agents, seed_base)
        r_acc += r_row
        a_acc += a_row

    r_inst = r_acc / P
    a_inst = a_acc / P
    div    = np.arange(1, N_TRIALS + 1, dtype=float)
    return np.cumsum(r_inst) / div, np.cumsum(a_inst) / div


def eval_ts_multi(dataset, model_class, params, n_agents, n_sims):
    """Average over n_sims independent simulations. Returns (r_cum, a_cum)."""
    r_sum = np.zeros(N_TRIALS)
    a_sum = np.zeros(N_TRIALS)

    for sim in range(n_sims):
        r, a = eval_ts_once(dataset, model_class, params, n_agents,
                             seed_offset=sim)
        r_sum += r
        a_sum += a

    return r_sum / n_sims, a_sum / n_sims


def _metrics(model_ts, human_ts):
    n = min(len(model_ts), len(human_ts))
    if n < 2:
        return {'msd': None, 'corr': None}
    m = np.asarray(model_ts[:n], dtype=float)
    h = np.asarray(human_ts[:n], dtype=float)
    msd = float(np.mean((m - h) ** 2))
    if float(np.std(m)) < 1e-12 or float(np.std(h)) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(m, h)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    return {'msd': round(msd, 6), 'corr': round(corr, 4)}


def evaluate_model(model_name, model_dir, est_data, comp_data,
                    n_sims, n_agents) -> dict:
    params_path = os.path.join(model_dir, 'best_params.json')
    if not os.path.exists(params_path):
        print(f"  [{model_name}] SKIP — best_params.json not found")
        return {}

    with open(params_path) as f:
        saved = json.load(f)

    model_class = ALL_MODELS[model_name]
    params      = _complete_params(model_class, saved['params'])

    print(f"\n  [{model_name}] Evaluating  "
          f"(n_sims={n_sims}, n_agents={n_agents}, n_trials={N_TRIALS})")
    print(f"    params: {params}")

    t0 = time.time()
    r_train, a_train = eval_ts_multi(est_data,  model_class, params,
                                      n_agents, n_sims)
    r_test,  a_test  = eval_ts_multi(comp_data, model_class, params,
                                      n_agents, n_sims)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    np.save(os.path.join(model_dir, 'eval_train_r_ts.npy'), r_train)
    np.save(os.path.join(model_dir, 'eval_train_a_ts.npy'), a_train)
    np.save(os.path.join(model_dir, 'eval_test_r_ts.npy'),  r_test)
    np.save(os.path.join(model_dir, 'eval_test_a_ts.npy'),  a_test)

    summary = {
        'model':     model_name,
        'n_sims':    n_sims,
        'n_agents':  n_agents,
        'n_trials':  N_TRIALS,
        'elapsed_s': round(elapsed, 1),
        'params':    params,
        'train': {
            'r_rate': _metrics(r_train, human_r_ts_est),
            'a_rate': _metrics(a_train, human_a_ts_est),
        },
        'test': {
            'r_rate': _metrics(r_test, human_r_ts_comp),
            'a_rate': _metrics(a_test, human_a_ts_comp),
        },
        'conditions': {},
        'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    for condition in CONDITION_NAMES:
        split = condition_problem_split(condition)
        r_ts = r_train if split == 'est' else r_test
        a_ts = a_train if split == 'est' else a_test
        human = human_condition_series[condition]
        summary['conditions'][condition] = {
            'split': split,
            'n_people': human['n_people'],
            'r_rate': _metrics(r_ts, human['risk']),
            'a_rate': _metrics(a_ts, human['alternate']),
        }

    with open(os.path.join(model_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    for split, label in [('train', 'Est (load_0)'), ('test', 'Cmp (load_1)')]:
        sr = summary[split]['r_rate']
        sa = summary[split]['a_rate']
        print(f"    {label} -> R: MSD={sr['msd']:.5f}  corr={sr['corr']:.3f}  |  "
              f"A: MSD={sa['msd']:.5f}  corr={sa['corr']:.3f}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on estimation and competition sets.'
    )
    parser.add_argument('--run-dir',   required=True,
                         help='Path to the training run directory')
    parser.add_argument('--data-dir',  default='data',
                         help='Directory containing the .dat files')
    parser.add_argument('--n-sims',    type=int, default=5,
                         help='Independent simulations to average over')
    parser.add_argument('--n-agents',  type=int, default=20,
                         help='Agents per problem per simulation')
    parser.add_argument('--models',    nargs='*', default=None,
                         help='Models to evaluate (default: all found in run-dir)')
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: run directory not found: {args.run_dir}")
        sys.exit(1)

    est_data  = pd.read_csv(os.path.join(args.data_dir, '60estimationset.dat'),
                              sep=r'\s+', header=None, names=COLS)
    comp_data = pd.read_csv(os.path.join(args.data_dir, '60competitionset.dat'),
                              sep=r'\s+', header=None, names=COLS)
    print(f"  Est  set: {len(est_data)} problems  (load_0, n_trials={N_TRIALS})")
    print(f"  Comp set: {len(comp_data)} problems  (load_1, n_trials={N_TRIALS})")

    if args.models:
        model_names = args.models
    else:
        model_names = [
            m for m in sorted(os.listdir(args.run_dir))
            if os.path.isfile(os.path.join(args.run_dir, m, 'best_params.json'))
        ]
        if not model_names:
            print("No trained models found (missing best_params.json). "
                  "Run training first.")
            sys.exit(1)

    print(f"  Evaluating: {model_names}")
    print(f"  n_sims={args.n_sims}  n_agents={args.n_agents}\n")

    all_summaries = {}
    for model_name in model_names:
        if model_name not in ALL_MODELS:
            print(f"  [{model_name}] SKIP — not a known model")
            continue
        model_dir = os.path.join(args.run_dir, model_name)
        summary   = evaluate_model(model_name, model_dir,
                                    est_data, comp_data,
                                    args.n_sims, args.n_agents)
        if summary:
            all_summaries[model_name] = summary

    combined_path = os.path.join(args.run_dir, 'eval_all_summary.json')
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY -- {os.path.basename(args.run_dir)}")
    print(f"{'-'*70}")
    hdr = f"  {'Model':<16} {'Set':<8} {'R-MSD':>8} {'R-corr':>7} {'A-MSD':>8} {'A-corr':>7}"
    print(hdr)
    print(f"{'-'*70}")
    for name, s in all_summaries.items():
        for split, label in [('train', 'Train'), ('test', 'Test')]:
            sr, sa = s[split]['r_rate'], s[split]['a_rate']
            print(f"  {name:<16} {label:<8} "
                  f"{sr['msd']:>8.5f} {sr['corr']:>7.3f} "
                  f"{sa['msd']:>8.5f} {sa['corr']:>7.3f}")
    print(f"{'='*70}")
    print(f"  Saved: {combined_path}")
    print(f"  Next step: python plot_results.py --run-dir {args.run_dir} --no-show")


if __name__ == '__main__':
    main()
