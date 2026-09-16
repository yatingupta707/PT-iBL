"""
evaluate.py
══════════════════════════════════════════════════════════════════════════════
Loads fitted parameters from a training run and evaluates all models
on both the estimation (train) and competition (test) sets.

Saves inside each model's subdirectory:
  eval_train_r_ts.npy   — cumulative R-rate on estimation set
  eval_train_a_ts.npy   — cumulative A-rate on estimation set
  eval_test_r_ts.npy    — cumulative R-rate on competition set
  eval_test_a_ts.npy    — cumulative A-rate on competition set
  eval_summary.json     — MSD and correlation vs human benchmarks

Results are averaged over --n-sims independent simulations to reduce
Monte Carlo variance (more sims → smoother, slower).

USAGE
──────
  python evaluate.py \\
      --run-dir runs/exp1_15epochs \\
      --data-dir data \\
      --n-sims 5 \\
      --n-agents 20
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import ALL_MODELS
from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp,
)

COLS     = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']
N_TRIALS = 100


# ── Core simulation (same logic as training script) ───────────────────────────

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
        R[ai]   = np.array([1.0 if a == 'risky' else 0.0 for a in actions])
        A[ai, 0] = 0.0
        for t in range(1, N_TRIALS):
            A[ai, t] = float(actions[t] != actions[t - 1])

    return R.mean(axis=0), A.mean(axis=0)


def eval_ts_once(dataset, model_class, params, n_agents, seed_offset=0):
    """One forward pass over entire dataset. Returns (r_cum, a_cum)."""
    P     = len(dataset)
    r_acc = np.zeros(N_TRIALS)
    a_acc = np.zeros(N_TRIALS)

    for prob_i, (_, row) in enumerate(dataset.iterrows()):
        seed_base = seed_offset * P * n_agents + prob_i * n_agents
        r_row, a_row = _run_one_problem(model_class, params, row,
                                         n_agents, seed_base)
        r_acc += r_row
        a_acc += a_row

    r_inst = r_acc / P
    a_inst = a_acc / P
    div    = np.arange(1, N_TRIALS + 1, dtype=float)
    return np.cumsum(r_inst) / div, np.cumsum(a_inst) / div


def eval_ts_multi(dataset, model_class, params, n_agents, n_sims):
    """
    Average over n_sims independent simulations to reduce MC variance.
    Returns (r_cum, a_cum) each shape (N_TRIALS,).
    """
    r_sum = np.zeros(N_TRIALS)
    a_sum = np.zeros(N_TRIALS)

    for sim in range(n_sims):
        r, a    = eval_ts_once(dataset, model_class, params, n_agents,
                                seed_offset=sim)
        r_sum  += r
        a_sum  += a

    return r_sum / n_sims, a_sum / n_sims


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(model_ts, human_ts):
    msd  = float(np.mean((model_ts - human_ts) ** 2))
    corr = float(np.corrcoef(model_ts, human_ts)[0, 1])
    return {'msd': round(msd, 6), 'corr': round(corr, 4)}


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model_name, model_dir, est_data, comp_data,
                    n_sims, n_agents) -> dict:
    """Load params and evaluate on both sets. Save .npy and summary JSON."""

    params_path = os.path.join(model_dir, 'best_params.json')
    if not os.path.exists(params_path):
        print(f"  [{model_name}] SKIP — best_params.json not found")
        return {}

    with open(params_path) as f:
        saved = json.load(f)

    params      = saved['params']
    model_class = ALL_MODELS[model_name]

    print(f"\n  [{model_name}] Evaluating  "
          f"(n_sims={n_sims}, n_agents={n_agents})")
    print(f"    params: {params}")

    t0 = time.time()

    # Estimation set (train)
    r_train, a_train = eval_ts_multi(est_data,  model_class, params,
                                      n_agents, n_sims)
    # Competition set (test)
    r_test,  a_test  = eval_ts_multi(comp_data, model_class, params,
                                      n_agents, n_sims)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Save time series arrays
    np.save(os.path.join(model_dir, 'eval_train_r_ts.npy'), r_train)
    np.save(os.path.join(model_dir, 'eval_train_a_ts.npy'), a_train)
    np.save(os.path.join(model_dir, 'eval_test_r_ts.npy'),  r_test)
    np.save(os.path.join(model_dir, 'eval_test_a_ts.npy'),  a_test)

    # Metrics vs human
    summary = {
        'model':     model_name,
        'n_sims':    n_sims,
        'n_agents':  n_agents,
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
        'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(model_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print metrics
    for split, label in [('train', 'Est (train)'), ('test', 'Cmp (test)')]:
        sr = summary[split]['r_rate']
        sa = summary[split]['a_rate']
        print(f"    {label} → R: MSD={sr['msd']:.5f}  corr={sr['corr']:.3f}  |  "
              f"A: MSD={sa['msd']:.5f}  corr={sa['corr']:.3f}")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on estimation and competition sets.'
    )
    parser.add_argument('--run-dir',   required=True,
                         help='Path to the training run directory '
                              '(e.g. runs/exp1_15epochs)')
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

    # Load datasets
    est_data  = pd.read_csv(os.path.join(args.data_dir, '60estimationset.dat'),
                              sep=r'\s+', header=None, names=COLS)
    comp_data = pd.read_csv(os.path.join(args.data_dir, '60competitionset.dat'),
                              sep=r'\s+', header=None, names=COLS)
    print(f"  Est  set: {len(est_data)} problems")
    print(f"  Comp set: {len(comp_data)} problems")

    # Discover models to evaluate
    if args.models:
        model_names = args.models
    else:
        # Auto-detect: any subdirectory with best_params.json
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

    # Save combined summary
    combined_path = os.path.join(args.run_dir, 'eval_all_summary.json')
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    # Print comparison table
    print(f"\n{'═'*70}")
    print(f"  EVALUATION SUMMARY — {os.path.basename(args.run_dir)}")
    print(f"{'─'*70}")
    hdr = f"  {'Model':<16} {'Set':<8} {'R-MSD':>8} {'R-corr':>7} {'A-MSD':>8} {'A-corr':>7}"
    print(hdr)
    print(f"{'─'*70}")
    for name, s in all_summaries.items():
        for split, label in [('train', 'Train'), ('test', 'Test')]:
            sr, sa = s[split]['r_rate'], s[split]['a_rate']
            print(f"  {name:<16} {label:<8} "
                  f"{sr['msd']:>8.5f} {sr['corr']:>7.3f} "
                  f"{sa['msd']:>8.5f} {sa['corr']:>7.3f}")
    print(f"{'═'*70}")
    print(f"  Saved: {combined_path}")
    print(f"  Next step: python plot_results.py --run-dir {args.run_dir}")


if __name__ == '__main__':
    main()
