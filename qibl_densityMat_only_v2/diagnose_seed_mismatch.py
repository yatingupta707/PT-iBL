"""
Re-score iteration_4 best_params under training vs evaluate seeding.

Training eval_bundle currently uses seed_base=0 for EVERY problem.
evaluate.py uses unique seeds per (sim, problem).

This script reports R-corr in windowed and cumulative spaces for both,
plus a unique-seed / n_agents=10 control so we can separate seeding
from the 10-vs-20 agent count.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import ALL_MODELS, problem_seed_base
from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_inst_est, human_a_inst_est,
)
from evaluate import (
    COLS, N_TRIALS, _run_one_problem, windowed_mean, cumulative_mean, _metrics,
)


def corr(a, b):
    m = _metrics(a, b)
    return m['corr']


def msd(a, b):
    return _metrics(a, b)['msd']


def eval_once(dataset, model_class, params, n_agents, seed_offset=0, shared_seed=False):
    P = len(dataset)
    r_acc = np.zeros(N_TRIALS)
    a_acc = np.zeros(N_TRIALS)
    v_acc = np.zeros(N_TRIALS)
    for prob_i, (_, row) in enumerate(dataset.iterrows()):
        seed_base = (0 if shared_seed
                     else problem_seed_base(prob_i, n_agents, P, seed_offset))
        r_row, a_row, v_row = _run_one_problem(model_class, params, row, n_agents, seed_base)
        r_acc += r_row
        a_acc += a_row
        v_acc += v_row
    return r_acc / P, a_acc / P, v_acc / P


def eval_multi(dataset, model_class, params, n_agents, n_sims, shared_seed=False):
    r_sum = np.zeros(N_TRIALS)
    a_sum = np.zeros(N_TRIALS)
    v_sum = np.zeros(N_TRIALS)
    for sim in range(n_sims):
        r, a, v = eval_once(
            dataset, model_class, params, n_agents,
            seed_offset=sim, shared_seed=shared_seed,
        )
        r_sum += r
        a_sum += a
        v_sum += v
    r_inst = r_sum / n_sims
    a_inst = a_sum / n_sims
    v_inst = v_sum / n_sims
    return {
        'r_inst': r_inst, 'a_inst': a_inst, 'v_inst': v_inst,
        'r_win': windowed_mean(r_inst, 5),
        'a_win': windowed_mean(a_inst, 5),
        'r_cum': cumulative_mean(r_inst),
        'a_cum': cumulative_mean(a_inst),
    }


def summarize(bundle, tag):
    h_r_win = windowed_mean(human_r_inst_est, 5)
    n = len(bundle['r_inst'])
    early = float(bundle['r_inst'][:min(10, n)].mean())
    late = float(bundle['r_inst'][-min(10, n):].mean())
    return {
        'tag': tag,
        'r_corr_win': corr(bundle['r_win'], h_r_win),
        'r_corr_cum': corr(bundle['r_cum'], human_r_ts_est),
        'a_corr_cum': corr(bundle['a_cum'], human_a_ts_est),
        'r_msd_cum': msd(bundle['r_cum'], human_r_ts_est),
        'mean_r': round(float(bundle['r_inst'].mean()), 4),
        'mean_reveal': round(float(bundle['v_inst'].mean()), 4),
        'early_r': round(early, 4),
        'late_r': round(late, 4),
        'delta_r': round(late - early, 4),
        'r_cum0': round(float(bundle['r_cum'][0]), 4),
        'r_cum_end': round(float(bundle['r_cum'][-1]), 4),
    }


def load_eval_npy(model_dir):
    r_cum = np.load(os.path.join(model_dir, 'eval_train_r_ts.npy'))
    a_cum = np.load(os.path.join(model_dir, 'eval_train_a_ts.npy'))
    r_inst = np.load(os.path.join(model_dir, 'eval_train_r_inst.npy'))
    r_win = np.load(os.path.join(model_dir, 'eval_train_r_win.npy'))
    # reveal not saved as npy in all runs; derive nothing
    dummy_v = np.zeros_like(r_inst)
    dummy_a = np.zeros_like(r_inst)
    return {
        'r_inst': r_inst, 'a_inst': dummy_a, 'v_inst': dummy_v,
        'r_win': r_win, 'a_win': dummy_a,
        'r_cum': r_cum, 'a_cum': a_cum,
    }


def main():
    run_dir = os.path.join(_HERE, 'runs', 'iteration_4')
    data_dir = os.path.join(_HERE, 'data')
    est = pd.read_csv(os.path.join(data_dir, '60estimationset.dat'),
                      sep=r'\s+', header=None, names=COLS)
    print(f'N_TRIALS={N_TRIALS}  problems={len(est)}')
    print(f'human_r_cum start={human_r_ts_est[0]:.4f} end={human_r_ts_est[-1]:.4f}')
    print(f'human_r_inst early={human_r_inst_est[:10].mean():.4f} '
          f'late={human_r_inst_est[-10:].mean():.4f}')
    print()

    protocols = [
        ('TRAIN shared-seed n=10 sims=1', dict(n_agents=10, n_sims=1, shared_seed=True)),
        ('UNIQUE n=10 sims=1', dict(n_agents=10, n_sims=1, shared_seed=False)),
        ('UNIQUE n=20 sims=1', dict(n_agents=20, n_sims=1, shared_seed=False)),
    ]

    all_out = {}
    models = ['IBL', 'PTiBL', 'IBLQuantum', 'PTIBLQuantum']
    for name in models:
        model_dir = os.path.join(run_dir, name)
        with open(os.path.join(model_dir, 'best_params.json')) as f:
            saved = json.load(f)
        params = saved['params']
        cls = ALL_MODELS[name]
        print('=' * 78)
        print(f'{name}  saved train_r_corr={saved.get("train_r_corr")}  '
              f'train_r_corr_cum={saved.get("train_r_corr_cum")}  '
              f'early={saved.get("train_early_r")} late={saved.get("train_late_r")}  '
              f'p_reveal={params.get("p_reveal")}')
        rows = []
        for tag, kw in protocols:
            t0 = time.time()
            bundle = eval_multi(est, cls, params, **kw)
            row = summarize(bundle, tag)
            row['elapsed_s'] = round(time.time() - t0, 1)
            rows.append(row)
            print(f'  {tag:<32}  Rcorr_win={row["r_corr_win"]:+.3f}  '
                  f'Rcorr_cum={row["r_corr_cum"]:+.3f}  '
                  f'early={row["early_r"]:.3f} late={row["late_r"]:.3f}  '
                  f'delta={row["delta_r"]:+.3f}  '
                  f'cum {row["r_cum0"]:.3f}->{row["r_cum_end"]:.3f}  '
                  f'({row["elapsed_s"]}s)')

        saved_bundle = load_eval_npy(model_dir)
        row = summarize(saved_bundle, 'EVAL npy n=20 sims=5')
        # mean_reveal unknown from npy
        row['mean_reveal'] = None
        rows.append(row)
        print(f'  {"EVAL npy n=20 sims=5":<32}  Rcorr_win={row["r_corr_win"]:+.3f}  '
              f'Rcorr_cum={row["r_corr_cum"]:+.3f}  '
              f'early={row["early_r"]:.3f} late={row["late_r"]:.3f}  '
              f'delta={row["delta_r"]:+.3f}  '
              f'cum {row["r_cum0"]:.3f}->{row["r_cum_end"]:.3f}')
        all_out[name] = {
            'saved': {
                'train_r_corr': saved.get('train_r_corr'),
                'train_r_corr_cum': saved.get('train_r_corr_cum'),
                'train_early_r': saved.get('train_early_r'),
                'train_late_r': saved.get('train_late_r'),
                'p_reveal': params.get('p_reveal'),
            },
            'rescored': rows,
        }
        print()

    out_path = os.path.join(run_dir, 'seed_protocol_rescore.json')
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f'Saved {out_path}')
    plot_ibl_iblq(run_dir, est, all_out)


def plot_ibl_iblq(run_dir, est, all_out):
    """Shared-seed vs unique-seed cumulative R for the two qualitative cases."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('iteration_4 params: shared-seed training vs unique-seed evaluate',
                 fontsize=13, fontweight='bold', y=1.02)
    trials = np.arange(1, N_TRIALS + 1)
    for ax, name in zip(axes, ['IBL', 'IBLQuantum']):
        model_dir = os.path.join(run_dir, name)
        with open(os.path.join(model_dir, 'best_params.json')) as f:
            params = json.load(f)['params']
        cls = ALL_MODELS[name]
        shared = eval_multi(est, cls, params, n_agents=10, n_sims=1, shared_seed=True)
        unique = eval_multi(est, cls, params, n_agents=10, n_sims=1, shared_seed=False)
        eval_cum = np.load(os.path.join(model_dir, 'eval_train_r_ts.npy'))
        ax.plot(trials, human_r_ts_est[:N_TRIALS], color='black', lw=2.2, label='Human (pooled load_0)')
        ax.plot(trials, shared['r_cum'], color='#e09c40', lw=1.8, ls='--',
                label=f"TRAIN shared-seed (R-corr={corr(shared['r_cum'], human_r_ts_est):+.2f})")
        ax.plot(trials, unique['r_cum'], color='#e05252', lw=1.8, ls='--',
                label=f"UNIQUE n=10 (R-corr={corr(unique['r_cum'], human_r_ts_est):+.2f})")
        ax.plot(trials[:len(eval_cum)], eval_cum, color='#4c8edb', lw=1.4, ls=':',
                label=f"EVAL 20×5 (R-corr={corr(eval_cum, human_r_ts_est):+.2f})")
        ax.set_title(name)
        ax.set_xlabel('Trial')
        ax.set_xlim(1, N_TRIALS)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.legend(fontsize=8, loc='best')
        ax.xaxis.set_major_locator(MultipleLocator(10))
    axes[0].set_ylabel('Cumulative R-rate')
    out = os.path.join(run_dir, 'seed_protocol_ibl_vs_iblq.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
