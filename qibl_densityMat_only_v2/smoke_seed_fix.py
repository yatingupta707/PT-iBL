"""Smoke-check: training eval_bundle now matches evaluate unique-seed protocol."""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from models import ALL_MODELS
from human_metrics import human_r_ts_est
from evaluate import COLS, eval_ts_once
from density_ibl_quantum_v3 import eval_bundle, corr

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    data_dir = os.path.join(_HERE, 'data')
    run_dir = os.path.join(_HERE, 'runs', 'iteration_4')
    est = pd.read_csv(os.path.join(data_dir, '60estimationset.dat'),
                      sep=r'\s+', header=None, names=COLS)

    print('--- eval_bundle vs evaluate.eval_ts_once (unique seeds, n_agents=10) ---')
    for name in ['IBL', 'IBLQuantum']:
        with open(os.path.join(run_dir, name, 'best_params.json')) as f:
            params = json.load(f)['params']
        cls = ALL_MODELS[name]
        train_b = eval_bundle(est, cls, params, n_agents=10, seed_offset=0)
        ev_r, _, _ = eval_ts_once(est, cls, params, n_agents=10, seed_offset=0)
        max_diff = float(np.max(np.abs(train_b['r_inst'] - ev_r)))
        rc = corr(train_b['r_cum'], human_r_ts_est)
        print(f'  {name}: max|r_inst train-eval|={max_diff:.2e}  '
              f'eval_bundle R-corr_cum={rc:+.3f}  '
              f'early={train_b["r_inst"][:10].mean():.3f} '
              f'late={train_b["r_inst"][-10:].mean():.3f} '
              f'cum0={train_b["r_cum"][0]:.3f}')

    print('--- IBLQuantum unique-seed with p_reveal forced to 0 ---')
    with open(os.path.join(run_dir, 'IBLQuantum', 'best_params.json')) as f:
        qparams = dict(json.load(f)['params'])
    q0 = dict(qparams)
    q0['p_reveal'] = 0.0
    b0 = eval_bundle(est, ALL_MODELS['IBLQuantum'], q0, n_agents=10, seed_offset=0)
    print(f'  p_reveal=0  R-corr_cum={corr(b0["r_cum"], human_r_ts_est):+.3f}  '
          f'early={b0["r_inst"][:10].mean():.3f} late={b0["r_inst"][-10:].mean():.3f}  '
          f'cum {b0["r_cum"][0]:.3f}->{b0["r_cum"][-1]:.3f}  '
          f'mean_reveal={b0["v_inst"].mean():.3f}')

    with open(os.path.join(run_dir, 'IBL', 'best_params.json')) as f:
        iparams = dict(json.load(f)['params'])
    i0 = dict(iparams)
    i0['p_reveal'] = 0.0
    bi0 = eval_bundle(est, ALL_MODELS['IBL'], i0, n_agents=10, seed_offset=0)
    print(f'  IBL p_reveal=0  R-corr_cum={corr(bi0["r_cum"], human_r_ts_est):+.3f}  '
          f'early={bi0["r_inst"][:10].mean():.3f} late={bi0["r_inst"][-10:].mean():.3f}  '
          f'cum {bi0["r_cum"][0]:.3f}->{bi0["r_cum"][-1]:.3f}')

    # Comparison figure (shared seed reconstructed with seed_base=0 all problems)
    from density_ibl_quantum_v3 import _run_one_problem

    def shared_cum(cls, params, n_agents=10):
        P = len(est)
        r_acc = np.zeros(len(human_r_ts_est))
        for _, row in est.iterrows():
            r_row, _, _ = _run_one_problem(cls, params, row, n_agents, seed_base=0)
            r_acc += r_row
        r_inst = r_acc / P
        div = np.arange(1, len(r_inst) + 1, dtype=float)
        return np.cumsum(r_inst) / div

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('iteration_4 params: shared-seed training vs unique-seed evaluate',
                 fontsize=13, fontweight='bold', y=1.02)
    n = len(human_r_ts_est)
    trials = np.arange(1, n + 1)
    for ax, name in zip(axes, ['IBL', 'IBLQuantum']):
        with open(os.path.join(run_dir, name, 'best_params.json')) as f:
            params = json.load(f)['params']
        cls = ALL_MODELS[name]
        sh = shared_cum(cls, params)
        un = eval_bundle(est, cls, params, n_agents=10, seed_offset=0)['r_cum']
        ev = np.load(os.path.join(run_dir, name, 'eval_train_r_ts.npy'))
        ax.plot(trials, human_r_ts_est, color='black', lw=2.2, label='Human (pooled load_0)')
        ax.plot(trials, sh, color='#e09c40', lw=1.8, ls='--',
                label=f'TRAIN shared-seed (R-corr={corr(sh, human_r_ts_est):+.2f})')
        ax.plot(trials, un, color='#e05252', lw=1.8, ls='--',
                label=f'UNIQUE n=10 (R-corr={corr(un, human_r_ts_est):+.2f})')
        ax.plot(trials[:len(ev)], ev, color='#4c8edb', lw=1.4, ls=':',
                label=f'EVAL 20x5 (R-corr={corr(ev, human_r_ts_est):+.2f})')
        ax.set_title(name)
        ax.set_xlabel('Trial')
        ax.set_xlim(1, n)
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
