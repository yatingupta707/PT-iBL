"""
evaluate.py
══════════════════════════════════════════════════════════════════════════════
Loads fitted parameters from a training run and evaluates all models
on both the estimation (train) and competition (test) sets.

Saves inside each model's subdirectory:
  eval_train_r_ts.npy    — cumulative R-rate on estimation set (plots)
  eval_train_a_ts.npy    — cumulative A-rate on estimation set (plots)
  eval_train_r_inst.npy  — instantaneous R-rate
  eval_train_r_win.npy   — trailing-window R-rate (training scoring space)
  eval_test_*.npy        — same for the competition set
  eval_summary.json      — MSD and correlation vs human in cum / win / inst spaces

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

from models import ALL_MODELS, alternation_series, problem_seed_base
from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp,
    human_reveal_ts_est, human_reveal_ts_comp,
    human_r_inst_est, human_a_inst_est, human_reveal_inst_est,
    human_r_inst_comp, human_a_inst_comp, human_reveal_inst_comp,
    human_condition_series, CONDITION_NAMES,
    condition_problem_split, condition_n_trials,
)

COLS     = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']
N_TRIALS = len(human_r_ts_est)


def windowed_mean(x, window=5):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        out[t] = x[lo:t + 1].mean()
    return out


def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    div = np.arange(1, len(x) + 1, dtype=float)
    return np.cumsum(x) / div


# ── Core simulation (same logic as training script) ───────────────────────────

def _run_one_problem(model_class, params, row, n_agents, seed_base=0, n_trials=None):
    n_trials = int(n_trials if n_trials is not None else N_TRIALS)
    risky_values = (float(row['val_high']), float(row['val_low']))
    risky_probs  = (float(row['p_high']), 1.0 - float(row['p_high']))
    safe_value   = float(row['val_safe'])

    R = np.zeros((n_agents, n_trials))
    A = np.zeros((n_agents, n_trials))
    V = np.zeros((n_agents, n_trials))

    for ai in range(n_agents):
        model   = model_class(params)
        actions = model.simulate(n_trials, safe_value,
                                  risky_values, risky_probs,
                                  seed=seed_base + ai)
        R[ai]   = np.array([1.0 if a == 'risky' else 0.0 for a in actions])
        A[ai] = alternation_series(actions)
        V[ai] = np.array([1.0 if a == 'reveal' else 0.0 for a in actions])

    return R.mean(axis=0), A.mean(axis=0), V.mean(axis=0)


def eval_ts_once(dataset, model_class, params, n_agents, seed_offset=0, n_trials=None):
    """One forward pass. Returns instantaneous R/A/reveal (not cumulative)."""
    n_trials = int(n_trials if n_trials is not None else N_TRIALS)
    P     = len(dataset)
    r_acc = np.zeros(n_trials)
    a_acc = np.zeros(n_trials)
    v_acc = np.zeros(n_trials)

    for prob_i, (_, row) in enumerate(dataset.iterrows()):
        seed_base = problem_seed_base(prob_i, n_agents, P, seed_offset)
        r_row, a_row, v_row = _run_one_problem(model_class, params, row,
                            n_agents, seed_base, n_trials=n_trials)
        r_acc += r_row
        a_acc += a_row
        v_acc += v_row

    return r_acc / P, a_acc / P, v_acc / P


def eval_ts_multi(dataset, model_class, params, n_agents, n_sims, window=5,
                  n_trials=None):
    """
    Average over n_sims independent simulations.
    Returns dict with inst / win / cum arrays for R, A, reveal.
    """
    n_trials = int(n_trials if n_trials is not None else N_TRIALS)
    r_sum = np.zeros(n_trials)
    a_sum = np.zeros(n_trials)
    v_sum = np.zeros(n_trials)

    for sim in range(n_sims):
        r, a, v = eval_ts_once(dataset, model_class, params, n_agents,
                                seed_offset=sim, n_trials=n_trials)
        r_sum  += r
        a_sum  += a
        v_sum  += v

    r_inst = r_sum / n_sims
    a_inst = a_sum / n_sims
    v_inst = v_sum / n_sims
    return {
        'r_inst': r_inst, 'a_inst': a_inst, 'v_inst': v_inst,
        'r_win':  windowed_mean(r_inst, window),
        'a_win':  windowed_mean(a_inst, window),
        'v_win':  windowed_mean(v_inst, window),
        'r_cum':  cumulative_mean(r_inst),
        'a_cum':  cumulative_mean(a_inst),
        'v_cum':  cumulative_mean(v_inst),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(model_ts, human_ts):
    a = np.asarray(model_ts, dtype=float)
    b = np.asarray(human_ts, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return {'msd': None, 'corr': None}
    a, b = a[:n], b[:n]
    msd  = float(np.mean((a - b) ** 2))
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        c = 0.0
    else:
        c = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(c):
            c = 0.0
    return {'msd': round(msd, 6), 'corr': round(c, 4)}


def _humans_pooled(split):
    if split == 'train':
        return {
            'r_cum': human_r_ts_est, 'a_cum': human_a_ts_est, 'v_cum': human_reveal_ts_est,
            'r_inst': human_r_inst_est, 'a_inst': human_a_inst_est, 'v_inst': human_reveal_inst_est,
        }
    return {
        'r_cum': human_r_ts_comp, 'a_cum': human_a_ts_comp, 'v_cum': human_reveal_ts_comp,
        'r_inst': human_r_inst_comp, 'a_inst': human_a_inst_comp, 'v_inst': human_reveal_inst_comp,
    }


def _humans_condition(condition):
    s = human_condition_series[condition]
    z = np.zeros_like(s['risk_inst'])
    return {
        'r_cum': s['risk'],
        'a_cum': s['alternate'],
        'v_cum': s.get('reveal', z),
        'r_inst': s['risk_inst'],
        'a_inst': s['alternate_inst'],
        'v_inst': s.get('reveal_inst', z),
    }


def _space_metrics(bundle, split=None, humans=None, window=5):
    if humans is None:
        humans = _humans_pooled(split)
    h_r_win = windowed_mean(humans['r_inst'], window)
    h_a_win = windowed_mean(humans['a_inst'], window)
    h_v_win = windowed_mean(humans['v_inst'], window)
    return {
        'r_rate': _metrics(bundle['r_cum'], humans['r_cum']),
        'a_rate': _metrics(bundle['a_cum'], humans['a_cum']),
        'reveal_rate': _metrics(bundle['v_cum'], humans['v_cum']),
        'windowed': {
            'r_rate': _metrics(bundle['r_win'], h_r_win),
            'a_rate': _metrics(bundle['a_win'], h_a_win),
            'reveal_rate': _metrics(bundle['v_win'], h_v_win),
        },
        'instantaneous': {
            'r_rate': _metrics(bundle['r_inst'], humans['r_inst']),
            'a_rate': _metrics(bundle['a_inst'], humans['a_inst']),
            'reveal_rate': _metrics(bundle['v_inst'], humans['v_inst']),
        },
    }


def _save_split_npy(model_dir, split, bundle):
    np.save(os.path.join(model_dir, f'eval_{split}_r_ts.npy'), bundle['r_cum'])
    np.save(os.path.join(model_dir, f'eval_{split}_a_ts.npy'), bundle['a_cum'])
    np.save(os.path.join(model_dir, f'eval_{split}_r_inst.npy'), bundle['r_inst'])
    np.save(os.path.join(model_dir, f'eval_{split}_a_inst.npy'), bundle['a_inst'])
    np.save(os.path.join(model_dir, f'eval_{split}_r_win.npy'), bundle['r_win'])
    np.save(os.path.join(model_dir, f'eval_{split}_a_win.npy'), bundle['a_win'])


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model_name, model_dir, est_data, comp_data,
                    n_sims, n_agents, window=5, condition=None) -> dict:
    """Load params and evaluate. Save .npy and summary JSON."""

    params_path = os.path.join(model_dir, 'best_params.json')
    if not os.path.exists(params_path):
        print(f"  [{model_name}] SKIP -- best_params.json not found")
        return {}

    with open(params_path) as f:
        saved = json.load(f)

    params      = saved['params']
    model_class = ALL_MODELS[model_name]
    tag = f'{condition}/{model_name}' if condition else model_name

    print(f"\n  [{tag}] Evaluating  "
          f"(n_sims={n_sims}, n_agents={n_agents})")
    print(f"    params: {params}")

    t0 = time.time()

    if condition:
        split = condition_problem_split(condition)
        dataset = est_data if split == 'est' else comp_data
        n_trials = condition_n_trials(condition)
        train_b = eval_ts_multi(dataset, model_class, params, n_agents, n_sims,
                                 window, n_trials=n_trials)
        _save_split_npy(model_dir, 'train', train_b)
        humans = _humans_condition(condition)
        train_metrics = _space_metrics(train_b, humans=humans, window=window)
        summary = {
            'model':     model_name,
            'condition': condition,
            'fit_split': split,
            'n_trials':  n_trials,
            'n_sims':    n_sims,
            'n_agents':  n_agents,
            'window':    window,
            'elapsed_s': round(time.time() - t0, 1),
            'params':    params,
            'train':     train_metrics,
            'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        print(f"    Done in {summary['elapsed_s']}s")
        sr = train_metrics['r_rate']
        sw = train_metrics['windowed']['r_rate']
        print(f"    {condition}  R-corr  cum={sr['corr']:+.3f}  "
              f"win={sw['corr']:+.3f}  R-MSD cum={sr['msd']:.5f}")
    else:
        train_b = eval_ts_multi(est_data, model_class, params, n_agents, n_sims, window)
        test_b  = eval_ts_multi(comp_data, model_class, params, n_agents, n_sims, window)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        _save_split_npy(model_dir, 'train', train_b)
        _save_split_npy(model_dir, 'test', test_b)
        summary = {
            'model':     model_name,
            'n_sims':    n_sims,
            'n_agents':  n_agents,
            'window':    window,
            'elapsed_s': round(elapsed, 1),
            'params':    params,
            'train':     _space_metrics(train_b, split='train', window=window),
            'test':      _space_metrics(test_b, split='test', window=window),
            'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        for split, label in [('train', 'Est (train)'), ('test', 'Cmp (test)')]:
            sr = summary[split]['r_rate']
            sw = summary[split]['windowed']['r_rate']
            si = summary[split]['instantaneous']['r_rate']
            print(f"    {label}  R-corr  cum={sr['corr']:+.3f}  "
                  f"win={sw['corr']:+.3f}  inst={si['corr']:+.3f}  "
                  f"R-MSD cum={sr['msd']:.5f}")

    with open(os.path.join(model_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
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
    parser.add_argument('--window',    type=int, default=5,
                         help='Trailing window used for windowed metrics (default 5)')
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

    def _condition_runs():
        found = []
        for cond in CONDITION_NAMES:
            cond_dir = os.path.join(args.run_dir, cond)
            if not os.path.isdir(cond_dir):
                continue
            models = [
                m for m in sorted(os.listdir(cond_dir))
                if os.path.isfile(os.path.join(cond_dir, m, 'best_params.json'))
            ]
            if args.models:
                models = [m for m in models if m in args.models]
            if models:
                found.append((cond, models))
        return found

    condition_runs = _condition_runs()
    all_summaries = {}

    if condition_runs:
        print(f"  Per-condition fits: {[c for c, _ in condition_runs]}")
        print(f"  n_sims={args.n_sims}  n_agents={args.n_agents}\n")
        for cond, models in condition_runs:
            all_summaries[cond] = {}
            for model_name in models:
                if model_name not in ALL_MODELS:
                    continue
                model_dir = os.path.join(args.run_dir, cond, model_name)
                summary = evaluate_model(
                    model_name, model_dir, est_data, comp_data,
                    args.n_sims, args.n_agents, window=args.window,
                    condition=cond,
                )
                if summary:
                    all_summaries[cond][model_name] = summary
    else:
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

        for model_name in model_names:
            if model_name not in ALL_MODELS:
                print(f"  [{model_name}] SKIP -- not a known model")
                continue
            model_dir = os.path.join(args.run_dir, model_name)
            summary   = evaluate_model(model_name, model_dir,
                                        est_data, comp_data,
                                        args.n_sims, args.n_agents,
                                        window=args.window)
            if summary:
                all_summaries[model_name] = summary

    combined_path = os.path.join(args.run_dir, 'eval_all_summary.json')
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY -- {os.path.basename(args.run_dir)}")
    print(f"{'-'*70}")
    if condition_runs:
        hdr = f"  {'Condition':<18} {'Model':<16} {'R-MSD':>8} {'R-corr':>7} {'A-MSD':>8} {'A-corr':>7}"
        print(hdr)
        print(f"{'-'*70}")
        for cond, models in all_summaries.items():
            for name, s in models.items():
                sr, sa = s['train']['r_rate'], s['train']['a_rate']
                print(f"  {cond:<18} {name:<16} "
                      f"{sr['msd']:>8.5f} {sr['corr']:>7.3f} "
                      f"{sa['msd']:>8.5f} {sa['corr']:>7.3f}")
    else:
        hdr = f"  {'Model':<16} {'Set':<8} {'R-MSD':>8} {'R-corr':>7} {'A-MSD':>8} {'A-corr':>7}"
        print(hdr)
        print(f"{'-'*70}")
        for name, s in all_summaries.items():
            for split, label in [('train', 'Train'), ('test', 'Test')]:
                if split not in s:
                    continue
                sr, sa = s[split]['r_rate'], s[split]['a_rate']
                print(f"  {name:<16} {label:<8} "
                      f"{sr['msd']:>8.5f} {sr['corr']:>7.3f} "
                      f"{sa['msd']:>8.5f} {sa['corr']:>7.3f}")
    print(f"{'='*70}")
    print(f"  Saved: {combined_path}")
    print(f"  Next step: python plot_results.py --run-dir {args.run_dir}")


if __name__ == '__main__':
    main()
