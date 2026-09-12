"""
density_ibl_quantum_v2.py
══════════════════════════════════════════════════════════════════════════════
Training script for PT-IBL-Quantum models.
Uses Differential Evolution (+ optional Nelder-Mead polish) to minimise
the MSD between model and human cumulative R-rate / A-rate time series.

Inspired by ibl_and_pt_re_parallel.py.

USAGE
------
  python density_ibl_quantum_v3.py
      (saves to runs/iteration_2 by default)

  python density_ibl_quantum_v3.py \\
      --run-name iteration_2 --n-epochs 100 --n-agents 10 --pop-size 15 \\
      --optimizer de --shape-weight 0.002 \\
      --warm-start runs/iteration1_50epochs \\
      --models IBL PTiBL IBLQuantum PTIBLQuantum

RESUME AFTER CRASH (checkpoint system)
  Re-run the SAME command. Already-completed models are skipped.

LOGS
------
  runs/{run_name}/
    WHAT_WE_DID.md         <- notes for this run
    checkpoint.json
    PTiBL/
      best_params.json     <- final fitted parameters + MSD / corr scores
      loss_history.json    <- per-generation loss breakdown
    IBLQuantum/  ...
    PTIBLQuantum/ ...
"""

import multiprocessing

# Must be at top level before any multiprocessing import
# (matches the template's pattern)
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

# ── Path setup so module can be run from any directory ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import (PTiBL, IBLQuantum, PTIBLQuantum, iBL, ALL_MODELS,
                    alternation_series)
from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp, human_reveal_ts_est,
)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONFIG  (inherited by worker processes via env vars)
# ══════════════════════════════════════════════════════════════════════════════

COLS      = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']
N_TRIALS  = len(human_r_ts_est)
N_AGENTS  = int(os.environ.get('PTIBL_N_AGENTS', '5'))
R_WEIGHT  = float(os.environ.get('PTIBL_R_WEIGHT', '0.5'))
SHAPE_WEIGHT = float(os.environ.get('PTIBL_SHAPE_WEIGHT', '0.002'))
_DATA_DIR = os.environ.get('PTIBL_DATA_DIR', 'data')


def _load_dataset(data_dir: str):
    est  = pd.read_csv(os.path.join(data_dir, '60estimationset.dat'),
                        sep=r'\s+', header=None, names=COLS)
    comp = pd.read_csv(os.path.join(data_dir, '60competitionset.dat'),
                        sep=r'\s+', header=None, names=COLS)
    return est, comp


try:
    _EST, _COMP = _load_dataset(_DATA_DIR)
except Exception:
    _EST = _COMP = None


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_problem(model_class, params, row, n_agents: int, seed_base: int = 0):
    """
    Simulate n_agents independent runs on one problem row.
    Returns R-array (n_agents, N_TRIALS) and A-array (n_agents, N_TRIALS).
    """
    risky_values = (float(row['val_high']), float(row['val_low']))
    risky_probs  = (float(row['p_high']), 1.0 - float(row['p_high']))
    safe_value   = float(row['val_safe'])

    R = np.zeros((n_agents, N_TRIALS))
    A = np.zeros((n_agents, N_TRIALS))
    V = np.zeros((n_agents, N_TRIALS))

    for ai in range(n_agents):
        model   = model_class(params)
        actions = model.simulate(N_TRIALS, safe_value,
                                  risky_values, risky_probs,
                                  seed=seed_base + ai)
        R[ai]   = np.array([1.0 if a == 'risky' else 0.0 for a in actions])
        A[ai] = alternation_series(actions)
        V[ai] = np.array([1.0 if a == 'reveal' else 0.0 for a in actions])

    return R.mean(axis=0), A.mean(axis=0), V.mean(axis=0)


def eval_ts(dataset: pd.DataFrame, model_class, params: dict,
             n_agents: int = 5) -> tuple:
    """
    Compute cumulative R-rate and A-rate time series averaged over all
    problems in the dataset.

    Returns
    -------
    (r_cum, a_cum) — each shape (N_TRIALS,)
        cumulative mean up to each trial, averaged over problems and agents
    """
    P     = len(dataset)
    r_acc = np.zeros(N_TRIALS)
    a_acc = np.zeros(N_TRIALS)
    v_acc = np.zeros(N_TRIALS)

    for _, row in dataset.iterrows():
        r_row, a_row, v_row = _run_one_problem(model_class, params, row, n_agents)
        r_acc += r_row
        a_acc += a_row
        v_acc += v_row

    r_inst = r_acc / P
    a_inst = a_acc / P
    v_inst = v_acc / P

    # Cumulative means (to match the human benchmark format)
    div   = np.arange(1, N_TRIALS + 1, dtype=float)
    r_cum = np.cumsum(r_inst) / div
    a_cum = np.cumsum(a_inst) / div
    v_cum = np.cumsum(v_inst) / div
    return r_cum, a_cum, v_cum


def msd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0 if either series is (near) constant."""
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _score_ts(r_ts, a_ts, v_ts) -> dict:
    """
    Combined loss: MSD (level) + shape penalty (1 − corr) on R and A curves.

    Reveal stays in the MSD term only: p_reveal is a constant coin-flip, so
    its correlation with a time-varying human series is not identifiable.
    """
    msd_r  = msd(r_ts, human_r_ts_est)
    msd_a  = msd(a_ts, human_a_ts_est)
    msd_v  = msd(v_ts, human_reveal_ts_est)
    corr_r = corr(r_ts, human_r_ts_est)
    corr_a = corr(a_ts, human_a_ts_est)
    msd_term   = R_WEIGHT * msd_r + (1.0 - R_WEIGHT) * (msd_a + msd_v) / 2.0
    shape_term = R_WEIGHT * (1.0 - corr_r) + (1.0 - R_WEIGHT) * (1.0 - corr_a)
    return {
        'total':      msd_term + SHAPE_WEIGHT * shape_term,
        'msd_term':   msd_term,
        'shape_term': shape_term,
        'msd_r':      msd_r,
        'msd_a':      msd_a,
        'msd_v':      msd_v,
        'corr_r':     corr_r,
        'corr_a':     corr_a,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL OBJECTIVE FUNCTIONS  (must be picklable for DE workers=-1)
# ══════════════════════════════════════════════════════════════════════════════

def _build_params(model_class, x: np.ndarray) -> dict:
    return {
        name: float(np.clip(val, lo + 1e-9, hi - 1e-9))
        for name, val, (lo, hi)
        in zip(model_class.PARAM_NAMES, x, model_class.PARAM_BOUNDS)
    }


def _params_to_x(model_class, params: dict) -> np.ndarray:
    """Clip a saved param dict into the current bound box (for warm-start)."""
    defaults = dict(zip(model_class.PARAM_NAMES, model_class.PARAM_DEFAULT))
    x = []
    for name, (lo, hi) in zip(model_class.PARAM_NAMES, model_class.PARAM_BOUNDS):
        val = float(params[name]) if name in params else float(defaults[name])
        x.append(float(np.clip(val, lo + 1e-9, hi - 1e-9)))
    return np.asarray(x, dtype=float)


def _init_population(model_class, popsize: int, warm_x, rng: np.random.Generator):
    """Latin-hypercube population with the previous best (and jittered copies) injected."""
    bounds = np.asarray(model_class.PARAM_BOUNDS, dtype=float)
    n = bounds.shape[0]
    m = max(int(popsize) * n, n + 1)
    lo, hi = bounds[:, 0], bounds[:, 1]
    init = np.empty((m, n))
    for j in range(n):
        perm = rng.permutation(m)
        init[:, j] = lo[j] + (perm + rng.random(m)) / m * (hi[j] - lo[j])
        init[:, j] = np.clip(init[:, j], lo[j] + 1e-9, hi[j] - 1e-9)
    if warm_x is not None:
        warm_x = np.clip(np.asarray(warm_x, dtype=float), lo + 1e-9, hi - 1e-9)
        init[0] = warm_x
        n_jitter = min(max(4, n), m - 1)
        scale = 0.08 * (hi - lo)
        for i in range(1, n_jitter + 1):
            init[i] = np.clip(warm_x + rng.normal(0.0, 1.0, n) * scale,
                              lo + 1e-9, hi - 1e-9)
    return init


def _load_warm_params(warm_dir, model_name):
    if not warm_dir:
        return None
    path = os.path.join(warm_dir, model_name, 'best_params.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get('params')


def _obj(x: np.ndarray, model_class) -> float:
    """MSD + curve-shape objective used by all model-specific functions."""
    if _EST is None:
        raise RuntimeError("Dataset not loaded. Check data directory.")
    params = _build_params(model_class, x)
    r_ts, a_ts, v_ts = eval_ts(_EST, model_class, params, N_AGENTS)
    return _score_ts(r_ts, a_ts, v_ts)['total']


# These three top-level functions are what DE workers actually call.
def _obj_ptibl(x):      return _obj(x, PTiBL)
def _obj_ibl(x):        return _obj(x, iBL)
def _obj_iblquantum(x): return _obj(x, IBLQuantum)
def _obj_ptiblq(x):     return _obj(x, PTIBLQuantum)


_OBJ_FN = {
    'PTiBL':        _obj_ptibl,
    'IBLQuantum':   _obj_iblquantum,
    'PTIBLQuantum': _obj_ptiblq,
    'IBL':          _obj_ibl
}


# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model_name: str, model_dir: str,
                n_epochs: int, optimizer: str,
                use_parallel: bool, warm_dir=None) -> dict:
    """
    Fit one model. Returns the result dict with best params and scores.

    Saves inside model_dir:
      best_params.json   — fitted parameters + MSD / correlation scores
      loss_history.json  — per-generation breakdown
    """
    model_class = ALL_MODELS[model_name]
    obj_fn      = _OBJ_FN[model_name]
    bounds      = model_class.PARAM_BOUNDS
    os.makedirs(model_dir, exist_ok=True)

    loss_history = {'de_generations': [], 'nm_polish': None}
    _gen = [0]

    # ── DE callback ────────────────────────────────────────────────────────
    def callback(xk, convergence=None):
        _gen[0] += 1
        params = _build_params(model_class, xk)
        r_ts, a_ts, v_ts = eval_ts(_EST, model_class, params, N_AGENTS)
        sc = _score_ts(r_ts, a_ts, v_ts)

        entry = {
            'gen':        _gen[0],
            'total':      round(sc['total'], 6),
            'msd_term':   round(sc['msd_term'], 6),
            'shape_term': round(sc['shape_term'], 6),
            'total_msd':  round(sc['msd_term'], 6),
            'r_msd':      round(sc['msd_r'], 6),
            'a_msd':      round(sc['msd_a'], 6),
            'reveal_msd': round(sc['msd_v'], 6),
            'r_corr':     round(sc['corr_r'], 4),
            'a_corr':     round(sc['corr_a'], 4),
            'params':     {k: round(v, 6) for k, v in params.items()},
        }
        loss_history['de_generations'].append(entry)

        with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f, indent=2)

        print(f"[{model_name}] Gen {_gen[0]:3d} | "
              f"loss={sc['total']:.5f}  r_msd={sc['msd_r']:.5f}  "
              f"r_corr={sc['corr_r']:+.3f}  a_corr={sc['corr_a']:+.3f}")
        return False

    print(f"\n{'='*65}")
    print(f"  Training {model_name}  |  DE maxiter={n_epochs}  "
          f"optimizer={optimizer}  parallel={use_parallel}")
    print(f"{'='*65}")

    workers  = -1 if use_parallel else 1
    pop_size = int(os.environ.get('POP_SIZE', '10'))

    de_kwargs = dict(
        maxiter  = n_epochs,
        popsize  = pop_size,
        updating = 'deferred',
        workers  = workers,
        tol      = 1e-6,
        seed     = 42,
        callback = callback,
        polish   = False,
    )

    warm_params = _load_warm_params(warm_dir, model_name)
    if warm_params is not None:
        warm_x = _params_to_x(model_class, warm_params)
        rng_init = np.random.default_rng(42)
        de_kwargs['init'] = _init_population(model_class, pop_size, warm_x, rng_init)
        print(f"[{model_name}] Warm-start from {warm_dir} "
              f"({len(warm_params)} params injected into DE population)")
    else:
        if warm_dir:
            print(f"[{model_name}] No warm-start file found in {warm_dir}; "
                  f"using latin hypercube init")

    res_de = differential_evolution(obj_fn, bounds, **de_kwargs)

    best_x    = res_de.x
    best_loss = float(res_de.fun)

    # ── Optional Nelder-Mead polish ─────────────────────────────────────────
    if optimizer == 'de+nm':
        print(f"\n[{model_name}] Polishing with Nelder-Mead …")
        res_nm = minimize(
            obj_fn, best_x,
            method  = 'Nelder-Mead',
            options = {'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-6},
        )
        improved = res_nm.fun < best_loss
        if improved:
            best_x    = res_nm.x
            best_loss = float(res_nm.fun)

        params_nm = _build_params(model_class, res_nm.x)
        r_nm, a_nm, v_nm = eval_ts(_EST, model_class, params_nm, N_AGENTS)
        sc_nm = _score_ts(r_nm, a_nm, v_nm)
        loss_history['nm_polish'] = {
            'total':      round(float(res_nm.fun), 6),
            'msd_term':   round(sc_nm['msd_term'], 6),
            'shape_term': round(sc_nm['shape_term'], 6),
            'total_msd':  round(sc_nm['msd_term'], 6),
            'r_msd':      round(sc_nm['msd_r'], 6),
            'a_msd':      round(sc_nm['msd_a'], 6),
            'reveal_msd': round(sc_nm['msd_v'], 6),
            'r_corr':     round(sc_nm['corr_r'], 4),
            'a_corr':     round(sc_nm['corr_a'], 4),
            'params':     {k: round(v, 6) for k, v in params_nm.items()},
            'improved_over_de': bool(improved),
        }
        print(f"[{model_name}] NM {'improved' if improved else 'did not improve'}  "
              f"best_loss={best_loss:.5f}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    best_params = _build_params(model_class, best_x)
    r_final, a_final, v_final = eval_ts(_EST, model_class, best_params, N_AGENTS)
    sc_final = _score_ts(r_final, a_final, v_final)

    result = {
        'model':            model_name,
        'optimizer':        optimizer,
        'n_epochs':         n_epochs,
        'n_agents':         N_AGENTS,
        'r_weight':         R_WEIGHT,
        'shape_weight':     SHAPE_WEIGHT,
        'warm_start':       warm_dir,
        'train_total_loss': round(best_loss, 6),
        'train_total_msd':  round(sc_final['msd_term'], 6),
        'train_shape_term': round(sc_final['shape_term'], 6),
        'train_r_msd':      round(sc_final['msd_r'], 6),
        'train_a_msd':      round(sc_final['msd_a'], 6),
        'train_reveal_msd': round(sc_final['msd_v'], 6),
        'train_r_corr':     round(sc_final['corr_r'], 4),
        'train_a_corr':     round(sc_final['corr_a'], 4),
        'params':           {k: round(v, 6) for k, v in best_params.items()},
        'fitted_at':        time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)

    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{model_name}] Done  →  loss={best_loss:.5f}  "
          f"r_msd={sc_final['msd_r']:.5f}  r_corr={sc_final['corr_r']:+.3f}  "
          f"a_corr={sc_final['corr_a']:+.3f}")
    print(f"[{model_name}] Params: {best_params}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint(ckpt_path: str) -> dict:
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {'completed_models': []}


def _write_run_notes(run_dir: str, args) -> None:
    """Write WHAT_WE_DID.md into the run folder (does not overwrite if present)."""
    path = os.path.join(run_dir, 'WHAT_WE_DID.md')
    if os.path.exists(path):
        return
    text = f"""# {args.run_name}

Notes for this training run. Written when the run directory was created.

## Goal

Fit model **R-rate and A-rate curves** to the human curves, not just drive
MSD down. Keep the same Monte Carlo budget (`n_agents={args.n_agents}`) so a
better fit means better **parameters**, not extra averaging.

`iteration1_50epochs` had low MSD but **negative R-rate correlation**
(about -0.66 to -0.87): the curves sat near the right average level while
trending the wrong way. This run targets that.

## What changed versus iteration 1

### 1. Independent IBL memory noise per agent

`IBLMemory` always used `seed=0`, so the 10 agents did not have independent
activation noise. Each agent now gets its own memory RNG, split from the
simulate seed (memory stream and choice/outcome stream are separate).

### 2. PTIBLQuantum bounds aligned with PTiBL

iteration 1 pinned several parameters to walls that were too tight:

- `lambda_` was capped at 1.0 (PTiBL allows 5.0; PT default is 2.25)
- `d` max was 2.0 and `sigma_s` max was 2.0 (now 5.0, same as PTiBL)
- `tau` floor lowered from 0.1 to 0.05 so it is not glued to the bound

### 3. Warm-start from iteration 1

`--warm-start {args.warm_start}` injects each model's `best_params.json`
into the DE population, plus jittered copies, then fills the rest with a
latin hypercube. Search continues from a known point instead of starting
from scratch.

### 4. Shape term in the loss (curve matching)

```
loss = MSD_term + shape_weight * [r_weight * (1 - R-corr)
                                  + (1 - r_weight) * (1 - A-corr)]
```

- `shape_weight = {args.shape_weight}`
- `r_weight = {args.r_weight}`
- Reveal stays in the MSD term only (`p_reveal` is still a constant coin
  flip, so its correlation with a time-varying human series is not useful)

Watch **R-corr** in the training log. The target is positive and rising.

### 5. Search settings (not more agents)

| setting        | iteration 1 | this run |
|----------------|-------------|----------|
| n_agents       | 10          | {args.n_agents} |
| pop_size       | 10          | {args.pop_size} |
| n_epochs       | 50          | {args.n_epochs} |
| optimizer      | de+nm       | {args.optimizer} |
| DE tol         | 1e-4        | 1e-6 |
| shape_weight   | 0 (MSD only)| {args.shape_weight} |
| warm_start     | none        | {args.warm_start} |

`pop_size` is the DE population multiplier (search budget), not agents.
Nelder-Mead is off by default here because on iteration 1 it barely moved
the parameters (noise, not a real improvement).

## What did not change

- `n_agents` stayed at {args.n_agents} on purpose
- `evaluate.py` output (`eval_*.npy`, `eval_summary.json`)
- `plot_results.py` output structure:
  - `plot_estimation_set.png`
  - `plot_competition_set.png`
  - `plot_tDCS_0_load_0.png`
  - `plot_tDCS_0_load_1.png`
  - `plot_tDCS_1_load_0.png`
  - `plot_tDCS_1_load_1.png`
  All models on the same axes, two graphs per figure (R-rate | A-rate).

## How to evaluate and plot

```
python evaluate.py --run-dir {run_dir} --n-sims 5 --n-agents 20
python plot_results.py --run-dir {run_dir} --no-show
```

## Models

{', '.join(args.models)}
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  Wrote notes: {path}")


def _save_checkpoint(ckpt_path: str, ckpt: dict) -> None:
    ckpt['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(ckpt_path, 'w') as f:
        json.dump(ckpt, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train PT-IBL-Quantum model family on estimation set.'
    )
    parser.add_argument('--run-name',   default='iteration_2',
                         help='Run folder name under --runs-dir (default: iteration_2)')
    parser.add_argument('--n-epochs',   type=int, default=100,
                         help='DE maxiter (number of generations)')
    parser.add_argument('--n-agents',   type=int, default=10,
                         help='Agents per problem per evaluation (MC noise budget)')
    parser.add_argument('--pop-size',   type=int, default=15,
                         help='DE population multiplier (actual pop = pop-size * n_params). '
                              'This is search budget, not Monte Carlo agents.')
    parser.add_argument('--r-weight',   type=float, default=0.5,
                         help='Weight on R-rate vs A/reveal (0-1)')
    parser.add_argument('--shape-weight', type=float, default=0.002,
                         help='Weight on (1 - corr) so the optimizer matches curve '
                              'shape, not only MSD level. 0 disables the shape term.')
    parser.add_argument('--warm-start', default='runs/iteration1_50epochs',
                         help='Previous run directory whose best_params.json seed the '
                              'DE population (e.g. runs/iteration1_50epochs)')
    parser.add_argument('--optimizer',  choices=['de', 'de+nm'], default='de',
                         help='de = DE only; de+nm = DE + Nelder-Mead polish')
    parser.add_argument('--models',     nargs='+',
                         default=['PTiBL', 'IBLQuantum', 'PTIBLQuantum','IBL'],
                         help='Which models to train')
    parser.add_argument('--data-dir',   default='data',
                         help='Directory containing 60estimationset.dat etc.')
    parser.add_argument('--runs-dir',   default='runs',
                         help='Root directory for run outputs')
    parser.add_argument('--no-parallel', action='store_true',
                         help='Disable DE workers=-1 (use single process)')
    args = parser.parse_args()

    # ── Validate model names ────────────────────────────────────────────────
    for m in args.models:
        if m not in ALL_MODELS:
            parser.error(f"Unknown model '{m}'. Valid: {list(ALL_MODELS)}")

    # ── Set module-level config AND env vars (inherited by DE workers) ───────
    global N_AGENTS, R_WEIGHT, SHAPE_WEIGHT, _EST, _COMP, _DATA_DIR
    N_AGENTS     = args.n_agents
    R_WEIGHT     = args.r_weight
    SHAPE_WEIGHT = args.shape_weight
    _DATA_DIR    = args.data_dir
    os.environ['PTIBL_N_AGENTS']      = str(args.n_agents)
    os.environ['PTIBL_R_WEIGHT']      = str(args.r_weight)
    os.environ['PTIBL_SHAPE_WEIGHT']  = str(args.shape_weight)
    os.environ['PTIBL_DATA_DIR']      = args.data_dir
    os.environ['POP_SIZE']            = str(args.pop_size)

    _EST, _COMP = _load_dataset(args.data_dir)
    print(f"  Loaded estimation set : {len(_EST)} problems")
    print(f"  Loaded competition set: {len(_COMP)} problems")
    print(f"  n_agents={args.n_agents}  pop_size={args.pop_size}  "
          f"shape_weight={args.shape_weight}  warm_start={args.warm_start}")

    # ── Create run directory (folder name is exactly --run-name) ─────────────
    run_label = args.run_name
    run_dir   = os.path.join(args.runs_dir, run_label)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run directory: {run_dir}")
    _write_run_notes(run_dir, args)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(run_dir, 'checkpoint.json')
    ckpt      = _load_checkpoint(ckpt_path)
    completed = set(ckpt.get('completed_models', []))

    if not ckpt.get('run_name'):
        ckpt.update({
            'run_name':  args.run_name,
            'n_epochs':  args.n_epochs,
            'optimizer': args.optimizer,
            'n_agents':     args.n_agents,
            'pop_size':     args.pop_size,
            'r_weight':     args.r_weight,
            'shape_weight': args.shape_weight,
            'warm_start':   args.warm_start,
            'models':       args.models,
            'n_epochs':  args.n_epochs,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        _save_checkpoint(ckpt_path, ckpt)

    if completed:
        print(f"  Resuming: already completed → {sorted(completed)}")

    # ── Train each model ─────────────────────────────────────────────────────
    all_results = {}
    for model_name in args.models:
        if model_name in completed:
            print(f"\n  Skipping {model_name} (checkpoint: already done)")
            # Load existing result for summary
            bp = os.path.join(run_dir, model_name, 'best_params.json')
            if os.path.exists(bp):
                with open(bp) as f:
                    all_results[model_name] = json.load(f)
            continue

        model_dir = os.path.join(run_dir, model_name)
        result    = train_model(
            model_name   = model_name,
            model_dir    = model_dir,
            n_epochs     = args.n_epochs,
            optimizer    = args.optimizer,
            use_parallel = not args.no_parallel,
            warm_dir     = args.warm_start,
        )
        all_results[model_name] = result

        # Update checkpoint atomically after each successful model
        ckpt['completed_models'].append(model_name)
        ckpt[f'{model_name}_completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        _save_checkpoint(ckpt_path, ckpt)
        print(f"  ✓ Checkpoint updated: {model_name} saved")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  TRAINING COMPLETE — {run_label}")
    print(f"{'─'*65}")
    print(f"  {'Model':<16}  {'loss':>10}  {'R-MSD':>8}  {'R-corr':>8}  {'A-corr':>8}  k")
    print(f"{'─'*65}")
    for name, res in all_results.items():
        k = ALL_MODELS[name].n_params()
        rcorr = res.get('train_r_corr', float('nan'))
        acorr = res.get('train_a_corr', float('nan'))
        loss  = res.get('train_total_loss', res.get('train_total_msd', float('nan')))
        print(f"  {name:<16}  {loss:>10.5f}  "
              f"{res['train_r_msd']:>8.5f}  {rcorr:>+8.3f}  {acorr:>+8.3f}  {k}")
    print(f"{'═'*65}")
    print(f"  Results saved → {run_dir}")
    print(f"  Next step: python evaluate.py --run-dir {run_dir}")


if __name__ == '__main__':
    main()
