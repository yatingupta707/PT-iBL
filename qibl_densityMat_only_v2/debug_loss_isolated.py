"""
debug_loss_isolated.py
----------------------
One-shot diagnostic for a fitted run, using instantaneous rates (r_inst / a_inst),
not full-history cumulative.

Checks:
  1. p_reveal in best_params.json (reveal is scored as R=0)
  2. Whether model r_inst can rise-then-fall (or is structurally monotonic)
  3. Loss breakdown on cumulative vs windowed vs instantaneous scoring

USAGE
  python debug_loss_isolated.py --run-dir runs/iteration_2 --n-agents 15 --n-sims 2
"""

import argparse, ast, csv, json, os, sys, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import ALL_MODELS, alternation_series

COLS = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']


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


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def msd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    return float(np.mean((a[:n] - b[:n]) ** 2))


def _series_mean(rows, key, n_trials):
    values = np.full((len(rows), n_trials), np.nan, dtype=float)
    for row_i, row in enumerate(rows):
        series = np.asarray(ast.literal_eval(row[key]), dtype=float)
        values[row_i, :len(series)] = series[:n_trials]
    return np.nanmean(values, axis=0)


def load_human_instantaneous(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    groups = {
        'est':  [row for row in rows if 'load_0' in row['condition']],
        'comp': [row for row in rows if 'load_1' in row['condition']],
    }
    out = {'splits': {}, 'conditions': {}}
    for split, split_rows in groups.items():
        n_trials = max(len(ast.literal_eval(row['risk_series'])) for row in split_rows)
        out['splits'][split] = {
            'risk':     _series_mean(split_rows, 'risk_series', n_trials),
            'alt':      _series_mean(split_rows, 'alt_series', n_trials),
            'reveal':   _series_mean(split_rows, 'reveal_series', n_trials),
            'n_people': len(split_rows),
            'n_trials': n_trials,
        }
    for condition in sorted({row['condition'] for row in rows}):
        condition_rows = [row for row in rows if row['condition'] == condition]
        n_trials = max(len(ast.literal_eval(row['risk_series'])) for row in condition_rows)
        out['conditions'][condition] = {
            'risk':     _series_mean(condition_rows, 'risk_series', n_trials),
            'alt':      _series_mean(condition_rows, 'alt_series', n_trials),
            'reveal':   _series_mean(condition_rows, 'reveal_series', n_trials),
            'n_people': len(condition_rows),
            'n_trials': n_trials,
        }
    return out


def turning_points(x, smooth=5, min_delta=0.015):
    """Sign changes in the smoothed first difference that exceed min_delta."""
    w = windowed_mean(x, window=smooth)
    d = np.diff(w)
    signs = np.sign(d)
    signs[signs == 0] = np.nan
    # forward-fill zeros so tiny flats don't count as turns
    last = 0.0
    filled = []
    for s in signs:
        if np.isnan(s):
            filled.append(last)
        else:
            last = s
            filled.append(s)
    filled = np.asarray(filled)
    turns = []
    for t in range(1, len(filled)):
        if filled[t] != 0 and filled[t - 1] != 0 and filled[t] != filled[t - 1]:
            # require the move after the turn to be large enough
            span = w[min(t + smooth, len(w) - 1)] - w[t]
            if abs(span) >= min_delta:
                kind = 'peak' if filled[t] < 0 else 'trough'
                turns.append((int(t + 1), kind, float(w[t])))
    return w, turns


def block_means(x, edges=(0, 10, 20, 30, 40, 50)):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], min(edges[i + 1], n)
        if lo >= n:
            break
        out[f't{lo + 1}-{hi}'] = float(np.mean(x[lo:hi]))
    return out


def shape_label(turns, blocks):
    keys = list(blocks)
    vals = [blocks[k] for k in keys]
    if not vals:
        return 'empty'
    trend = vals[-1] - vals[0]
    if not turns:
        if abs(trend) < 0.02:
            return 'flat'
        return 'monotonic-down' if trend < 0 else 'monotonic-up'
    kinds = [k for _, k, _ in turns]
    if 'peak' in kinds and 'trough' in kinds:
        return 'non-monotonic (peak+trough)'
    if kinds[0] == 'peak':
        return 'rise-then-fall'
    return 'fall-then-rise'


def run_one_problem(model_class, params, row, n_trials, n_agents, seed_base):
    risky_values = (float(row['val_high']), float(row['val_low']))
    risky_probs  = (float(row['p_high']), 1.0 - float(row['p_high']))
    safe_value   = float(row['val_safe'])
    R = np.zeros((n_agents, n_trials))
    A = np.zeros((n_agents, n_trials))
    V = np.zeros((n_agents, n_trials))
    Rx = np.zeros((n_agents, n_trials))  # risk-exposure: risky OR reveal
    for ai in range(n_agents):
        model = model_class(params)
        actions = model.simulate(n_trials, safe_value, risky_values, risky_probs,
                                 seed=seed_base + ai)
        R[ai]  = np.array([1.0 if a == 'risky'  else 0.0 for a in actions])
        V[ai]  = np.array([1.0 if a == 'reveal' else 0.0 for a in actions])
        Rx[ai] = np.array([1.0 if a in ('risky', 'reveal') else 0.0 for a in actions])
        A[ai]  = alternation_series(actions)
    return R.mean(0), A.mean(0), V.mean(0), Rx.mean(0)


def eval_inst(dataset, model_class, params, n_trials, n_agents, n_sims):
    r_sum = np.zeros(n_trials)
    a_sum = np.zeros(n_trials)
    v_sum = np.zeros(n_trials)
    x_sum = np.zeros(n_trials)
    P = len(dataset)
    for sim in range(n_sims):
        r_acc = np.zeros(n_trials)
        a_acc = np.zeros(n_trials)
        v_acc = np.zeros(n_trials)
        x_acc = np.zeros(n_trials)
        for prob_i, (_, row) in enumerate(dataset.iterrows()):
            seed_base = sim * P * n_agents + prob_i * n_agents
            r, a, v, x = run_one_problem(model_class, params, row, n_trials,
                                         n_agents, seed_base)
            r_acc += r; a_acc += a; v_acc += v; x_acc += x
        r_sum += r_acc / P
        a_sum += a_acc / P
        v_sum += v_acc / P
        x_sum += x_acc / P
    return r_sum / n_sims, a_sum / n_sims, v_sum / n_sims, x_sum / n_sims


def fmt_blocks(blocks):
    return '  '.join(f'{k}={v:.3f}' for k, v in blocks.items())


def fmt_turns(turns):
    if not turns:
        return 'none'
    return ', '.join(f'{kind}@{t} ({val:.3f})' for t, kind, val in turns)


def score_pair(model_ts, human_ts, label):
    return {
        'label': label,
        'msd':   round(msd(model_ts, human_ts), 6),
        'corr':  round(corr(model_ts, human_ts), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/iteration_2')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--human-csv', default=None)
    parser.add_argument('--n-agents', type=int, default=15)
    parser.add_argument('--n-sims', type=int, default=2)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--models', nargs='*', default=None)
    parser.add_argument('--shape-weight', type=float, default=0.002)
    parser.add_argument('--r-weight', type=float, default=0.5)
    args = parser.parse_args()

    human_csv = args.human_csv or os.path.join(_HERE, '..', 'tdcs_load_60_final_comb.csv')
    human = load_human_instantaneous(human_csv)
    n_trials = human['splits']['est']['n_trials']

    est  = pd.read_csv(os.path.join(args.data_dir, '60estimationset.dat'),
                       sep=r'\s+', header=None, names=COLS)
    print(f'Run dir     : {args.run_dir}')
    print(f'Human CSV   : {human_csv}')
    print(f'N_TRIALS    : {n_trials}   est problems: {len(est)}')
    print(f'MC budget   : n_agents={args.n_agents}  n_sims={args.n_sims}')
    print()

    # ── Human instantaneous shape (the target) ──────────────────────────────
    print('=' * 78)
    print('  HUMAN instantaneous R-rate (from raw risk_series)')
    print('=' * 78)
    for split, payload in human['splits'].items():
        r = payload['risk']
        v = payload['reveal']
        w, turns = turning_points(r, smooth=args.window)
        blocks = block_means(r)
        print(f'  [{split:4s}] n={payload["n_people"]:3d}  meanR={r.mean():.3f}  '
              f'meanReveal={v.mean():.3f}  R+reveal={r.mean()+v.mean():.3f}')
        print(f'         blocks: {fmt_blocks(blocks)}')
        print(f'         turns : {fmt_turns(turns)}   -> {shape_label(turns, blocks)}')
    print()
    for cond, payload in human['conditions'].items():
        r = payload['risk']
        v = payload['reveal']
        w, turns = turning_points(r, smooth=args.window)
        blocks = block_means(r)
        print(f'  [{cond}] n={payload["n_people"]:3d}  meanR={r.mean():.3f}  '
              f'meanReveal={v.mean():.3f}')
        print(f'         blocks: {fmt_blocks(blocks)}')
        print(f'         turns : {fmt_turns(turns)}   -> {shape_label(turns, blocks)}')
    print()

    if args.models:
        model_names = args.models
    else:
        model_names = [
            m for m in sorted(os.listdir(args.run_dir))
            if os.path.isfile(os.path.join(args.run_dir, m, 'best_params.json'))
        ]

    h_r = human['splits']['est']['risk']
    h_a = human['splits']['est']['alt']
    h_v = human['splits']['est']['reveal']
    h_r_win = windowed_mean(h_r, args.window)
    h_a_win = windowed_mean(h_a, args.window)
    h_r_cum = cumulative_mean(h_r)
    h_a_cum = cumulative_mean(h_a)
    h_v_cum = cumulative_mean(h_v)

    all_rows = []
    for model_name in model_names:
        params_path = os.path.join(args.run_dir, model_name, 'best_params.json')
        with open(params_path) as f:
            saved = json.load(f)
        params = saved['params']
        p_reveal = float(params.get('p_reveal', 0.0))
        print('=' * 78)
        print(f'  {model_name}   p_reveal={p_reveal:.4f}   '
              f'({"SIZEABLE - reveal understates R-rate" if p_reveal >= 0.05 else "negligible"})')
        print('=' * 78)

        t0 = time.time()
        r_inst, a_inst, v_inst, x_inst = eval_inst(
            est, ALL_MODELS[model_name], params, n_trials, args.n_agents, args.n_sims
        )
        elapsed = time.time() - t0
        r_win = windowed_mean(r_inst, args.window)
        a_win = windowed_mean(a_inst, args.window)
        r_cum = cumulative_mean(r_inst)
        a_cum = cumulative_mean(a_inst)
        v_cum = cumulative_mean(v_inst)

        w, turns = turning_points(r_inst, smooth=args.window)
        blocks = block_means(r_inst)
        print(f'  simulated in {elapsed:.1f}s')
        print(f'  mean r_inst={r_inst.mean():.3f}  mean reveal={v_inst.mean():.3f}  '
              f'mean risk-exposure (R+reveal)={x_inst.mean():.3f}')
        print(f'  p_reveal param={p_reveal:.3f} vs realised reveal rate={v_inst.mean():.3f}')
        print(f'  r_inst blocks: {fmt_blocks(blocks)}')
        print(f'  r_inst turns : {fmt_turns(turns)}   -> {shape_label(turns, blocks)}')
        print(f'  early(1-10)={blocks.get("t1-10", float("nan")):.3f}  '
              f'late({list(blocks)[-1]})={list(blocks.values())[-1]:.3f}  '
              f'delta={list(blocks.values())[-1] - blocks.get("t1-10", list(blocks.values())[0]):+.3f}')

        scores = {
            'cum':  score_pair(r_cum,  h_r_cum, 'cumulative R'),
            'win':  score_pair(r_win,  h_r_win, f'window-{args.window} R'),
            'inst': score_pair(r_inst, h_r,     'instantaneous R'),
            'cum_a':  score_pair(a_cum,  h_a_cum, 'cumulative A'),
            'win_a':  score_pair(a_win,  h_a_win, f'window-{args.window} A'),
            'inst_a': score_pair(a_inst, h_a,     'instantaneous A'),
        }
        print()
        print(f'  {"scoring":<22} {"R-MSD":>8} {"R-corr":>8} {"A-MSD":>8} {"A-corr":>8}  '
              f'{"msd_term":>9} {"shape":>7} {"loss@sw":>8}')
        for key, rkey, akey in [
            ('cumulative', 'cum', 'cum_a'),
            (f'window-{args.window}', 'win', 'win_a'),
            ('instantaneous', 'inst', 'inst_a'),
        ]:
            rs, as_ = scores[rkey], scores[akey]
            msd_term = args.r_weight * rs['msd'] + (1.0 - args.r_weight) * as_['msd']
            shape_term = (args.r_weight * (1.0 - rs['corr'])
                          + (1.0 - args.r_weight) * (1.0 - as_['corr']))
            loss = msd_term + args.shape_weight * shape_term
            shape_frac = (args.shape_weight * shape_term) / max(loss, 1e-12)
            print(f'  {key:<22} {rs["msd"]:8.5f} {rs["corr"]:+8.3f} '
                  f'{as_["msd"]:8.5f} {as_["corr"]:+8.3f}  '
                  f'{msd_term:9.5f} {shape_term:7.3f} {loss:8.5f}  '
                  f'(shape {100 * shape_frac:.1f}% of loss @ sw={args.shape_weight})')

        # what-if shape weights on the WINDOWED scores (the proposed next loss)
        print()
        print('  Windowed-loss if shape_weight were:')
        rs, as_ = scores['win'], scores['win_a']
        msd_term = args.r_weight * rs['msd'] + (1.0 - args.r_weight) * as_['msd']
        shape_term = (args.r_weight * (1.0 - rs['corr'])
                      + (1.0 - args.r_weight) * (1.0 - as_['corr']))
        for sw in (0.002, 0.02, 0.1, 0.5, 1.0):
            loss = msd_term + sw * shape_term
            frac = (sw * shape_term) / max(loss, 1e-12)
            print(f'    sw={sw:<5}  loss={loss:.5f}  msd_term={msd_term:.5f}  '
                  f'shape_contrib={sw * shape_term:.5f}  ({100 * frac:.1f}%)  '
                  f'R-corr={rs["corr"]:+.3f}')

        out_dir = os.path.join(args.run_dir, model_name)
        np.save(os.path.join(out_dir, 'debug_r_inst.npy'), r_inst)
        np.save(os.path.join(out_dir, 'debug_a_inst.npy'), a_inst)
        np.save(os.path.join(out_dir, 'debug_v_inst.npy'), v_inst)
        np.save(os.path.join(out_dir, 'debug_r_win.npy'),  r_win)
        row = {
            'model': model_name,
            'p_reveal': p_reveal,
            'realised_reveal': round(float(v_inst.mean()), 4),
            'mean_r_inst': round(float(r_inst.mean()), 4),
            'mean_risk_exposure': round(float(x_inst.mean()), 4),
            'shape': shape_label(turns, blocks),
            'n_turns': len(turns),
            'r_corr_cum': scores['cum']['corr'],
            'r_corr_win': scores['win']['corr'],
            'r_corr_inst': scores['inst']['corr'],
            'a_corr_win': scores['win_a']['corr'],
            'blocks': blocks,
            'turns': [{'trial': t, 'kind': k, 'value': v} for t, k, v in turns],
        }
        all_rows.append(row)
        print()

    summary_path = os.path.join(args.run_dir, 'debug_inst_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f'Saved {summary_path}')
    print()
    print('=' * 78)
    print('  CAPACITY CHECK (can r_inst rise-then-fall, or only settle?)')
    print('=' * 78)
    for row in all_rows:
        print(f'  {row["model"]:<16} {row["shape"]:<28} '
              f'turns={row["n_turns"]}  '
              f'R-corr inst={row["r_corr_inst"]:+.3f}  win={row["r_corr_win"]:+.3f}  '
              f'cum={row["r_corr_cum"]:+.3f}')


if __name__ == '__main__':
    main()
