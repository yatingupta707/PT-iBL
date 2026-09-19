"""
plot_results.py
══════════════════════════════════════════════════════════════════════════════
Loads evaluated behavioural time series and plots them against human data.

Primary output: one two-panel figure per tDCS/load condition (4 figures,
8 graphs). Each figure overlays humans (black) with all four models.

  plot_tDCS_0_load_0.png
  plot_tDCS_1_load_0.png   — models use estimation-set (load_0) simulations
  plot_tDCS_0_load_1.png
  plot_tDCS_1_load_1.png   — models use competition-set (load_1) simulations

Also writes the original pooled overlays:
  plot_estimation_set.png
  plot_competition_set.png

Style matches the original pipeline: cumulative R-rate | cumulative A-rate,
dashed coloured model lines, no rescaling.

USAGE
------
  python plot_results.py --run-dir runs/exp1_15epochs --no-show
"""

import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp,
    human_condition_series, CONDITION_NAMES, condition_problem_split,
)

MODEL_COLOURS = {
    "IBL":          "#c4c804",
    'PTiBL':        '#e05252',
    'IBLQuantum':   '#4c8edb',
    'PTIBLQuantum': '#4cb87a',
}
DEFAULT_COLOURS = ['#e05252', '#4c8edb', '#4cb87a',
                    '#e09c40', '#9b59b6', '#34a49e']


def _colour(name: str, idx: int) -> str:
    return MODEL_COLOURS.get(name, DEFAULT_COLOURS[idx % len(DEFAULT_COLOURS)])


def _make_figure(title: str,
                  human_r: np.ndarray, human_a: np.ndarray,
                  models: list,
                  output_path: str,
                  show: bool = True) -> None:
    N      = len(human_r)
    trials = np.arange(1, N + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    for ax, human_ts, ylabel, metric in [
        (axes[0], human_r, 'Cumulative R-rate', 'R-rate'),
        (axes[1], human_a, 'Cumulative A-rate', 'A-rate'),
    ]:
        ax.plot(trials, human_ts, color='black', lw=2.2,
                 linestyle='-', label='Human', zorder=10)

        for idx, (mname, r_ts, a_ts) in enumerate(models):
            ts = r_ts if metric == 'R-rate' else a_ts
            if ts is None:
                continue
            ts = np.asarray(ts, dtype=float)[:N]
            if len(ts) != N:
                continue
            ax.plot(trials, ts,
                     color     = _colour(mname, idx),
                     lw        = 1.6,
                     linestyle = '--',
                     alpha     = 0.85,
                     label     = mname)

        ax.set_xlabel('Trial', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(1, N)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _load_model_ts(model_dir: str, split: str):
    r_path = os.path.join(model_dir, f'eval_{split}_r_ts.npy')
    a_path = os.path.join(model_dir, f'eval_{split}_a_ts.npy')
    if not (os.path.exists(r_path) and os.path.exists(a_path)):
        return None, None
    return np.load(r_path), np.load(a_path)


def main():
    parser = argparse.ArgumentParser(
        description='Plot model behavioural time series vs human data.'
    )
    parser.add_argument('--run-dir',    required=True,
                         help='Training run directory')
    parser.add_argument('--output-dir', default=None,
                         help='Where to save plots (default: inside run-dir)')
    parser.add_argument('--models',     nargs='*', default=None,
                         help='Models to include (default: all found)')
    parser.add_argument('--no-show',    action='store_true',
                         help='Save plots without displaying them')
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: run directory not found: {args.run_dir}")
        sys.exit(1)

    output_dir = args.output_dir or args.run_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.models:
        model_names = args.models
    else:
        model_names = [
            d for d in sorted(os.listdir(args.run_dir))
            if os.path.isdir(os.path.join(args.run_dir, d))
            and os.path.exists(os.path.join(args.run_dir, d, 'best_params.json'))
        ]

    if not model_names:
        print("No evaluated models found. Run evaluate.py first.")
        sys.exit(1)

    print(f"  Models found: {model_names}")

    def load_split(split: str) -> list:
        entries = []
        for name in model_names:
            model_dir = os.path.join(args.run_dir, name)
            r_ts, a_ts = _load_model_ts(model_dir, split)
            if r_ts is None:
                print(f"  [{name}] No eval_{split}_*.npy — skipped")
            else:
                entries.append((name, r_ts, a_ts))
        return entries

    train_models = load_split('train')
    test_models  = load_split('test')

    run_label = os.path.basename(args.run_dir)
    show      = not args.no_show

    if train_models:
        _make_figure(
            title       = f'Estimation set (load_0, pooled)  —  {run_label}',
            human_r     = human_r_ts_est,
            human_a     = human_a_ts_est,
            models      = train_models,
            output_path = os.path.join(output_dir, 'plot_estimation_set.png'),
            show        = show,
        )
    else:
        print("  No training-set time series available — skipping est plot.")

    if test_models:
        _make_figure(
            title       = f'Competition set (load_1, pooled)  —  {run_label}',
            human_r     = human_r_ts_comp,
            human_a     = human_a_ts_comp,
            models      = test_models,
            output_path = os.path.join(output_dir, 'plot_competition_set.png'),
            show        = show,
        )
    else:
        print("  No test-set time series available — skipping comp plot.")

    if not human_condition_series:
        print("  No per-condition human series — skipping tDCS plots.")
    else:
        for condition in CONDITION_NAMES:
            if condition not in human_condition_series:
                continue
            human_series = human_condition_series[condition]
            split = condition_problem_split(condition)
            condition_models = train_models if split == 'est' else test_models
            if not condition_models:
                print(f"  No {split} model series — skip {condition}")
                continue
            n_people = human_series.get('n_people', '?')
            _make_figure(
                title       = f'{condition}  (n={n_people})  —  {run_label}',
                human_r     = human_series['risk'],
                human_a     = human_series['alternate'],
                models      = condition_models,
                output_path = os.path.join(output_dir, f'plot_{condition}.png'),
                show        = show,
            )

    print(f"\n  Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
