"""
plot_results.py
══════════════════════════════════════════════════════════════════════════════
Loads evaluated behavioural time series and plots them against human data.

Produces two figures (saved as PNG inside the run directory):
  plot_estimation_set.png   — R-rate and A-rate on training set
  plot_competition_set.png  — R-rate and A-rate on test set

Each plot shows:
  • Human data   — black solid line
  • Each model   — distinct coloured line

USAGE
──────
  python plot_results.py --run-dir runs/exp1_15epochs
  python plot_results.py --run-dir runs/exp1_15epochs --output-dir my_plots
  python plot_results.py --run-dir runs/exp1_15epochs --no-show   # save only
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
    human_condition_series,
)

# ── Colour palette (one per model) ────────────────────────────────────────────
MODEL_COLOURS = {
    "IBL":          "#c4c804",   # yellowish
    'PTiBL':        '#e05252',   # red
    'IBLQuantum':   '#4c8edb',   # blue
    'PTIBLQuantum': '#4cb87a',   # green
}
DEFAULT_COLOURS = ['#e05252', '#4c8edb', '#4cb87a',
                    '#e09c40', '#9b59b6', '#34a49e']


def _colour(name: str, idx: int) -> str:
    return MODEL_COLOURS.get(name, DEFAULT_COLOURS[idx % len(DEFAULT_COLOURS)])


# ── Single figure: 1 row × 2 cols (R-rate | A-rate) ──────────────────────────

def _make_figure(title: str,
                  human_r: np.ndarray, human_a: np.ndarray,
                  models: list,              # list of (name, r_ts, a_ts, reveal_ts)
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

        for idx, (mname, r_ts, a_ts, reveal_ts) in enumerate(models):
            ts = r_ts if metric == 'R-rate' else a_ts
            if ts is None:
                continue
            ts = np.asarray(ts)[:N]
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


# ── Load model time series ────────────────────────────────────────────────────

def _load_model_ts(model_dir: str, split: str):
    """
    Load npy time series for one model and one set split.
    Returns (r_ts, a_ts, reveal_ts) or (None, None, None) if files are missing.
    """
    r_path = os.path.join(model_dir, f'eval_{split}_r_ts.npy')
    a_path = os.path.join(model_dir, f'eval_{split}_a_ts.npy')
    reveal_path = os.path.join(model_dir, f'eval_{split}_reveal_ts.npy')
    if not (os.path.exists(r_path) and os.path.exists(a_path)):
        return None, None, None
    reveal_ts = np.load(reveal_path) if os.path.exists(reveal_path) else None
    return np.load(r_path), np.load(a_path), reveal_ts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot model behavioural time series vs human data.'
    )
    parser.add_argument('--run-dir',    required=True,
                         help='Training run directory (e.g. runs/exp1_15epochs)')
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

    # Auto-detect model subdirectories
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

    # ── Load time series for both splits ─────────────────────────────────────
    def load_split(split: str) -> list:
        """Returns list of (model_name, r_ts, a_ts, reveal_ts)."""
        entries = []
        for name in model_names:
            model_dir = os.path.join(args.run_dir, name)
            r_ts, a_ts, reveal_ts = _load_model_ts(model_dir, split)
            if r_ts is None:
                print(f"  [{name}] No eval_{split}_*.npy — skipped")
            else:
                entries.append((name, r_ts, a_ts, reveal_ts))
        return entries

    train_models = load_split('train')
    test_models  = load_split('test')

    run_label = os.path.basename(args.run_dir)
    show      = not args.no_show

    # ── Estimation set plot ───────────────────────────────────────────────────
    if train_models:
        _make_figure(
            title       = f'Estimation set (training)  —  {run_label}',
            human_r     = human_r_ts_est,
            human_a     = human_a_ts_est,
            models      = train_models,
            output_path = os.path.join(output_dir, 'plot_estimation_set.png'),
            show        = show,
        )
    else:
        print("  No training-set time series available — skipping est plot.")

    # ── Competition set plot ──────────────────────────────────────────────────
    if test_models:
        _make_figure(
            title       = f'Competition set (test)  —  {run_label}',
            human_r     = human_r_ts_comp,
            human_a     = human_a_ts_comp,
            models      = test_models,
            output_path = os.path.join(output_dir, 'plot_competition_set.png'),
            show        = show,
        )
    else:
        print("  No test-set time series available — skipping comp plot.")

    # Produce one two-panel figure for each exact tDCS/load condition.
    for condition, human_series in sorted(human_condition_series.items()):
        split = 'train' if condition.endswith('load_0') else 'test'
        condition_models = train_models if split == 'train' else test_models
        if not condition_models:
            continue
        _make_figure(
            title       = f'{condition}  —  {run_label}',
            human_r     = human_series['risk'],
            human_a     = human_series['alternate'],
            models      = condition_models,
            output_path = os.path.join(output_dir, f'plot_{condition}.png'),
            show        = show,
        )

    print(f"\n  Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
