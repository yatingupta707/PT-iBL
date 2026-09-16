"""
density_ibl_quantum_v2.py
══════════════════════════════════════════════════════════════════════════════
Training script for PT-IBL-Quantum models.
Uses Differential Evolution (+ optional Nelder-Mead polish) to minimise
the MSD between model and human cumulative R-rate / A-rate time series.

Inspired by ibl_and_pt_re_parallel.py.

USAGE
──────
  python density_ibl_quantum_v2.py \\
      --run-name exp1 --n-epochs 15 --n-agents 5 \\
      --optimizer de+nm --models PTiBL IBLQuantum PTIBLQuantum

RESUME AFTER CRASH (checkpoint system)
  Re-run the SAME command. Already-completed models are skipped.

LOGS
──────
  runs/{run_name}_{n_epochs}epochs/
    checkpoint.json
    PTiBL/
      best_params.json     ← final fitted parameters + MSD scores
      loss_history.json    ← per-generation R-MSD, A-MSD, total-MSD
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

from models import PTiBL, IBLQuantum, PTIBLQuantum,iBL, ALL_MODELS
from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp,
)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONFIG  (inherited by worker processes via env vars)
# ══════════════════════════════════════════════════════════════════════════════

COLS      = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']
N_TRIALS  = 100
N_AGENTS  = int(os.environ.get('PTIBL_N_AGENTS', '5'))
R_WEIGHT  = float(os.environ.get('PTIBL_R_WEIGHT', '0.5'))
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

    for _, row in dataset.iterrows():
        r_row, a_row = _run_one_problem(model_class, params, row, n_agents)
        r_acc += r_row
        a_acc += a_row

    r_inst = r_acc / P
    a_inst = a_acc / P

    # Cumulative means (to match the human benchmark format)
    div   = np.arange(1, N_TRIALS + 1, dtype=float)
    r_cum = np.cumsum(r_inst) / div
    a_cum = np.cumsum(a_inst) / div
    return r_cum, a_cum


def msd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL OBJECTIVE FUNCTIONS  (must be picklable for DE workers=-1)
# ══════════════════════════════════════════════════════════════════════════════

def _build_params(model_class, x: np.ndarray) -> dict:
    return {
        name: float(np.clip(val, lo + 1e-9, hi - 1e-9))
        for name, val, (lo, hi)
        in zip(model_class.PARAM_NAMES, x, model_class.PARAM_BOUNDS)
    }


def _obj(x: np.ndarray, model_class) -> float:
    """Generic MSD objective used by all three model-specific functions."""
    if _EST is None:
        raise RuntimeError("Dataset not loaded. Check data directory.")
    params  = _build_params(model_class, x)
    r_ts, a_ts = eval_ts(_EST, model_class, params, N_AGENTS)
    msd_r   = msd(r_ts, human_r_ts_est)
    msd_a   = msd(a_ts, human_a_ts_est)
    return R_WEIGHT * msd_r + (1.0 - R_WEIGHT) * msd_a


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
                use_parallel: bool) -> dict:
    """
    Fit one model. Returns the result dict with best params and scores.

    Saves inside model_dir:
      best_params.json   — fitted parameters + MSD scores
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
        params   = _build_params(model_class, xk)
        r_ts, a_ts = eval_ts(_EST, model_class, params, N_AGENTS)
        msd_r    = msd(r_ts, human_r_ts_est)
        msd_a    = msd(a_ts, human_a_ts_est)
        total    = R_WEIGHT * msd_r + (1.0 - R_WEIGHT) * msd_a

        entry = {
            'gen':        _gen[0],
            'total_msd':  round(total,  6),
            'r_msd':      round(msd_r,  6),
            'a_msd':      round(msd_a,  6),
            'params':     {k: round(v, 6) for k, v in params.items()},
        }
        loss_history['de_generations'].append(entry)

        # Save incremental loss history so a crash can be audited
        with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f, indent=2)

        print(f"[{model_name}] Gen {_gen[0]:3d} | "
              f"total={total:.5f}  r={msd_r:.5f}  a={msd_a:.5f}")
        return False   # returning True would stop DE early

    print(f"\n{'='*65}")
    print(f"  Training {model_name}  |  DE maxiter={n_epochs}  "
          f"optimizer={optimizer}  parallel={use_parallel}")
    print(f"{'='*65}")

    workers = -1 if use_parallel else 1
    pop_size  = int(os.environ.get('POP_SIZE', '10'))

    res_de = differential_evolution(
        obj_fn, bounds,
        maxiter   = n_epochs,
        popsize   = pop_size,
        updating  = 'deferred',
        workers   = workers,
        tol       = 1e-4,
        seed      = 42,
        callback  = callback,
        polish    = False,        # we do our own polish below
    )

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
        r_nm, a_nm = eval_ts(_EST, model_class, params_nm, N_AGENTS)
        loss_history['nm_polish'] = {
            'total_msd': round(float(res_nm.fun), 6),
            'r_msd':     round(msd(r_nm, human_r_ts_est), 6),
            'a_msd':     round(msd(a_nm, human_a_ts_est), 6),
            'params':    {k: round(v, 6) for k, v in params_nm.items()},
            'improved_over_de': bool(improved),
        }
        print(f"[{model_name}] NM {'improved' if improved else 'did not improve'}  "
              f"best_total={best_loss:.5f}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    best_params = _build_params(model_class, best_x)
    r_final, a_final = eval_ts(_EST, model_class, best_params, N_AGENTS)
    msd_r_final = msd(r_final, human_r_ts_est)
    msd_a_final = msd(a_final, human_a_ts_est)

    result = {
        'model':           model_name,
        'optimizer':       optimizer,
        'n_epochs':        n_epochs,
        'n_agents':        N_AGENTS,
        'r_weight':        R_WEIGHT,
        'train_total_msd': round(best_loss, 6),
        'train_r_msd':     round(msd_r_final, 6),
        'train_a_msd':     round(msd_a_final, 6),
        'params':          {k: round(v, 6) for k, v in best_params.items()},
        'fitted_at':       time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # Save loss history (final)
    with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)

    # Save best params
    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{model_name}] Done  →  total MSD={best_loss:.5f}  "
          f"r={msd_r_final:.5f}  a={msd_a_final:.5f}")
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
    parser.add_argument('--run-name',   required=True,
                         help='Name for this run (e.g. exp1)')
    parser.add_argument('--n-epochs',   type=int, default=15,
                         help='DE maxiter (number of generations)')
    parser.add_argument('--n-agents',   type=int, default=5,
                         help='Agents per problem per evaluation (MC noise budget)')    
    parser.add_argument('--pop-size',   type=int, default=10,
                         help='Population size for DE (number of agents)')
    parser.add_argument('--r-weight',   type=float, default=0.5,
                         help='Weight on R-rate MSD vs A-rate MSD (0–1)')
    parser.add_argument('--optimizer',  choices=['de', 'de+nm'], default='de+nm',
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
    global N_AGENTS, R_WEIGHT, _EST, _COMP, _DATA_DIR
    N_AGENTS  = args.n_agents
    R_WEIGHT  = args.r_weight
    _DATA_DIR = args.data_dir
    pop_size = args.pop_size
    os.environ['PTIBL_N_AGENTS']  = str(args.n_agents)
    os.environ['PTIBL_R_WEIGHT']  = str(args.r_weight)
    os.environ['PTIBL_DATA_DIR']  = args.data_dir
    os.environ['POP_SIZE']  = str(args.pop_size)

    _EST, _COMP = _load_dataset(args.data_dir)
    print(f"  Loaded estimation set : {len(_EST)} problems")
    print(f"  Loaded competition set: {len(_COMP)} problems")

    # ── Create run directory ─────────────────────────────────────────────────
    run_label = f"{args.run_name}_{args.n_epochs}epochs"
    run_dir   = os.path.join(args.runs_dir, run_label)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run directory: {run_dir}")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(run_dir, 'checkpoint.json')
    ckpt      = _load_checkpoint(ckpt_path)
    completed = set(ckpt.get('completed_models', []))

    if not ckpt.get('run_name'):
        ckpt.update({
            'run_name':  args.run_name,
            'n_epochs':  args.n_epochs,
            'optimizer': args.optimizer,
            'n_agents':  args.n_agents,
            'pop_size':  args.pop_size,
            'r_weight':  args.r_weight,
            'models':    args.models,
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
    print(f"  {'Model':<16}  {'total MSD':>10}  {'R-MSD':>8}  {'A-MSD':>8}  k")
    print(f"{'─'*65}")
    for name, res in all_results.items():
        k = ALL_MODELS[name].n_params()
        print(f"  {name:<16}  {res['train_total_msd']:>10.5f}  "
              f"{res['train_r_msd']:>8.5f}  {res['train_a_msd']:>8.5f}  {k}")
    print(f"{'═'*65}")
    print(f"  Results saved → {run_dir}")
    print(f"  Next step: python evaluate.py --run-dir {run_dir}")


if __name__ == '__main__':
    main()
