"""
density_ibl_quantum_v3.py
══════════════════════════════════════════════════════════════════════════════
Training script for PT-IBL-Quantum models on the tDCS/load human series.

Uses Differential Evolution (+ optional Nelder-Mead polish) to minimise
the MSD between model and human cumulative R-rate / A-rate time series.

This is the original fitting objective (not a post-hoc plot rescale):
    loss = r_weight * MSD(R_cum) + (1 - r_weight) * MSD(A_cum)

Human targets come from tdcs_load_60_final_comb.csv (50 trials, A/B/R).
Default target is pooled load_0. Use --fit-condition to fit one cell.

USAGE
------
  python density_ibl_quantum_v3.py --run-name tdcs_fit --n-epochs 50 --n-agents 10

  python density_ibl_quantum_v3.py --run-name tdcs_tdcs0_load0 \\
      --fit-condition tDCS_0_load_0 --n-epochs 50
"""

import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import (
    PTiBL, IBLQuantum, PTIBLQuantum, iBL, ALL_MODELS,
    alternation_series, problem_seed_base, risk_series,
)
from human_metrics import (
    N_TRIALS as HUMAN_N_TRIALS,
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp,
    human_condition_series, CONDITION_NAMES,
    condition_problem_split,
)

COLS     = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']
N_TRIALS = int(os.environ.get('PTIBL_N_TRIALS', str(HUMAN_N_TRIALS)))
N_AGENTS = int(os.environ.get('PTIBL_N_AGENTS', '5'))
R_WEIGHT = float(os.environ.get('PTIBL_R_WEIGHT', '0.5'))
FIT_CONDITION = os.environ.get('PTIBL_FIT_CONDITION', '').strip()
FIT_SPLIT = os.environ.get('PTIBL_FIT_SPLIT', 'est').strip() or 'est'
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


def _active_human():
    """Human cumulative R/A series for the current training target."""
    cond = FIT_CONDITION
    if cond:
        if cond not in human_condition_series:
            raise RuntimeError(
                f"Unknown --fit-condition {cond!r}. Valid: {sorted(human_condition_series)}"
            )
        s = human_condition_series[cond]
        return s['risk'], s['alternate']
    return human_r_ts_est, human_a_ts_est


def _active_dataset():
    if _EST is None or _COMP is None:
        raise RuntimeError("Dataset not loaded. Check data directory.")
    split = FIT_SPLIT
    if FIT_CONDITION:
        split = condition_problem_split(FIT_CONDITION)
    return _EST if split == 'est' else _COMP


def _run_one_problem(model_class, params, row, n_agents: int, seed_base: int = 0):
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


def eval_ts(dataset: pd.DataFrame, model_class, params: dict,
             n_agents: int = 5, seed_offset: int = 0) -> tuple:
    """
    Cumulative R-rate and A-rate averaged over problems and agents.
    Seeding matches evaluate.py (unique RNG block per problem).
    """
    P     = len(dataset)
    r_acc = np.zeros(N_TRIALS)
    a_acc = np.zeros(N_TRIALS)

    for prob_i, (_, row) in enumerate(dataset.iterrows()):
        seed_base = problem_seed_base(prob_i, n_agents, P, seed_offset)
        r_row, a_row = _run_one_problem(model_class, params, row, n_agents,
                                         seed_base)
        r_acc += r_row
        a_acc += a_row

    r_inst = r_acc / P
    a_inst = a_acc / P
    div    = np.arange(1, N_TRIALS + 1, dtype=float)
    return np.cumsum(r_inst) / div, np.cumsum(a_inst) / div


def msd(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.mean((np.asarray(a[:n]) - np.asarray(b[:n])) ** 2))


def _build_params(model_class, x: np.ndarray) -> dict:
    return {
        name: float(np.clip(val, lo + 1e-9, hi - 1e-9))
        for name, val, (lo, hi)
        in zip(model_class.PARAM_NAMES, x, model_class.PARAM_BOUNDS)
    }


def _params_to_x(model_class, params: dict) -> np.ndarray:
    defaults = dict(zip(model_class.PARAM_NAMES, model_class.PARAM_DEFAULT))
    x = []
    for name, (lo, hi) in zip(model_class.PARAM_NAMES, model_class.PARAM_BOUNDS):
        val = float(params[name]) if name in params else float(defaults[name])
        x.append(float(np.clip(val, lo + 1e-9, hi - 1e-9)))
    return np.asarray(x, dtype=float)


def _init_population(model_class, popsize: int, warm_x, rng: np.random.Generator):
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
    """Original MSD objective: match cumulative R-rate and A-rate."""
    params = _build_params(model_class, x)
    r_ts, a_ts = eval_ts(_active_dataset(), model_class, params, N_AGENTS)
    h_r, h_a = _active_human()
    return R_WEIGHT * msd(r_ts, h_r) + (1.0 - R_WEIGHT) * msd(a_ts, h_a)


def _obj_ptibl(x):      return _obj(x, PTiBL)
def _obj_ibl(x):        return _obj(x, iBL)
def _obj_iblquantum(x): return _obj(x, IBLQuantum)
def _obj_ptiblq(x):     return _obj(x, PTIBLQuantum)

_OBJ_FN = {
    'PTiBL':        _obj_ptibl,
    'IBLQuantum':   _obj_iblquantum,
    'PTIBLQuantum': _obj_ptiblq,
    'IBL':          _obj_ibl,
}


def train_model(model_name: str, model_dir: str,
                n_epochs: int, optimizer: str,
                use_parallel: bool, warm_dir=None) -> dict:
    model_class = ALL_MODELS[model_name]
    obj_fn      = _OBJ_FN[model_name]
    bounds      = model_class.PARAM_BOUNDS
    os.makedirs(model_dir, exist_ok=True)

    loss_history = {'de_generations': [], 'nm_polish': None}
    _gen = [0]
    h_r, h_a = _active_human()

    def callback(xk, convergence=None):
        _gen[0] += 1
        params   = _build_params(model_class, xk)
        r_ts, a_ts = eval_ts(_active_dataset(), model_class, params, N_AGENTS)
        msd_r    = msd(r_ts, h_r)
        msd_a    = msd(a_ts, h_a)
        total    = R_WEIGHT * msd_r + (1.0 - R_WEIGHT) * msd_a

        entry = {
            'gen':        _gen[0],
            'total_msd':  round(total,  6),
            'r_msd':      round(msd_r,  6),
            'a_msd':      round(msd_a,  6),
            'params':     {k: round(v, 6) for k, v in params.items()},
        }
        loss_history['de_generations'].append(entry)

        with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f, indent=2)

        print(f"[{model_name}] Gen {_gen[0]:3d} | "
              f"total={total:.5f}  r={msd_r:.5f}  a={msd_a:.5f}")
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
        tol      = 1e-4,
        seed     = 42,
        callback = callback,
        polish   = False,
    )
    warm_params = _load_warm_params(warm_dir, model_name)
    if warm_params is not None:
        warm_x = _params_to_x(model_class, warm_params)
        de_kwargs['init'] = _init_population(
            model_class, pop_size, warm_x, np.random.default_rng(42))
        print(f"[{model_name}] Warm-start from {warm_dir}")
    elif warm_dir:
        print(f"[{model_name}] No warm-start file in {warm_dir}; latin hypercube")

    res_de = differential_evolution(obj_fn, bounds, **de_kwargs)

    best_x    = res_de.x
    best_loss = float(res_de.fun)

    if optimizer == 'de+nm':
        print(f"\n[{model_name}] Polishing with Nelder-Mead ...")
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
        r_nm, a_nm = eval_ts(_active_dataset(), model_class, params_nm, N_AGENTS)
        loss_history['nm_polish'] = {
            'total_msd': round(float(res_nm.fun), 6),
            'r_msd':     round(msd(r_nm, h_r), 6),
            'a_msd':     round(msd(a_nm, h_a), 6),
            'params':    {k: round(v, 6) for k, v in params_nm.items()},
            'improved_over_de': bool(improved),
        }
        print(f"[{model_name}] NM {'improved' if improved else 'did not improve'}  "
              f"best_total={best_loss:.5f}")

    best_params = _build_params(model_class, best_x)
    r_final, a_final = eval_ts(_active_dataset(), model_class, best_params, N_AGENTS)
    msd_r_final = msd(r_final, h_r)
    msd_a_final = msd(a_final, h_a)

    result = {
        'model':           model_name,
        'optimizer':       optimizer,
        'n_epochs':        n_epochs,
        'n_agents':        N_AGENTS,
        'n_trials':        N_TRIALS,
        'r_weight':        R_WEIGHT,
        'fit_condition':   FIT_CONDITION or None,
        'train_total_msd': round(best_loss, 6),
        'train_r_msd':     round(msd_r_final, 6),
        'train_a_msd':     round(msd_a_final, 6),
        'params':          {k: round(v, 6) for k, v in best_params.items()},
        'fitted_at':       time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)
    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{model_name}] Done  ->  total MSD={best_loss:.5f}  "
          f"r={msd_r_final:.5f}  a={msd_a_final:.5f}")
    print(f"[{model_name}] Params: {best_params}")
    return result


def _load_checkpoint(ckpt_path: str) -> dict:
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {'completed_models': []}


def _save_checkpoint(ckpt_path: str, ckpt: dict) -> None:
    ckpt['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(ckpt_path, 'w') as f:
        json.dump(ckpt, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train PT-IBL-Quantum model family on tDCS human series.'
    )
    parser.add_argument('--run-name',   required=True,
                         help='Name for this run (e.g. tdcs_fit)')
    parser.add_argument('--n-epochs',   type=int, default=15,
                         help='DE maxiter (number of generations)')
    parser.add_argument('--n-agents',   type=int, default=5,
                         help='Agents per problem per evaluation')
    parser.add_argument('--pop-size',   type=int, default=10,
                         help='DE population multiplier')
    parser.add_argument('--r-weight',   type=float, default=0.5,
                         help='Weight on R-rate MSD vs A-rate MSD (0-1)')
    parser.add_argument('--optimizer',  choices=['de', 'de+nm'], default='de',
                         help='de = DE only; de+nm = DE + Nelder-Mead polish')
    parser.add_argument('--warm-start', default='qiblruns/exp2_30epochs',
                         help='Previous run whose best_params.json seed DE. '
                              'Use "none" for a fresh latin hypercube.')
    parser.add_argument('--models',     nargs='+',
                         default=['PTiBL', 'IBLQuantum', 'PTIBLQuantum', 'IBL'],
                         help='Which models to train')
    parser.add_argument('--data-dir',   default='data',
                         help='Directory containing 60estimationset.dat etc.')
    parser.add_argument('--runs-dir',   default='runs',
                         help='Root directory for run outputs')
    parser.add_argument('--fit-condition', default='',
                         help='Fit one human cell (e.g. tDCS_0_load_0). '
                              'Default empty = pooled load_0.')
    parser.add_argument('--no-parallel', action='store_true',
                         help='Disable DE workers=-1 (use single process)')
    args = parser.parse_args()

    for m in args.models:
        if m not in ALL_MODELS:
            parser.error(f"Unknown model '{m}'. Valid: {list(ALL_MODELS)}")
    if args.fit_condition and args.fit_condition not in human_condition_series:
        parser.error(
            f"Unknown --fit-condition {args.fit_condition!r}. "
            f"Valid: {CONDITION_NAMES}"
        )
    if args.warm_start and str(args.warm_start).lower() in ('none', 'off', '-', ''):
        args.warm_start = None

    global N_AGENTS, R_WEIGHT, _EST, _COMP, _DATA_DIR, FIT_CONDITION, FIT_SPLIT, N_TRIALS
    N_AGENTS      = args.n_agents
    R_WEIGHT      = args.r_weight
    _DATA_DIR     = args.data_dir
    FIT_CONDITION = (args.fit_condition or '').strip()
    FIT_SPLIT     = condition_problem_split(FIT_CONDITION) if FIT_CONDITION else 'est'
    N_TRIALS      = HUMAN_N_TRIALS
    os.environ['PTIBL_N_AGENTS']      = str(args.n_agents)
    os.environ['PTIBL_R_WEIGHT']      = str(args.r_weight)
    os.environ['PTIBL_DATA_DIR']      = args.data_dir
    os.environ['PTIBL_FIT_CONDITION'] = FIT_CONDITION
    os.environ['PTIBL_FIT_SPLIT']     = FIT_SPLIT
    os.environ['PTIBL_N_TRIALS']      = str(N_TRIALS)
    os.environ['POP_SIZE']            = str(args.pop_size)

    _EST, _COMP = _load_dataset(args.data_dir)
    print(f"  Loaded estimation set : {len(_EST)} problems")
    print(f"  Loaded competition set: {len(_COMP)} problems")
    print(f"  n_trials={N_TRIALS}  fit_condition={FIT_CONDITION or 'pooled load_0'}  "
          f"fit_split={FIT_SPLIT}  warm_start={args.warm_start}")

    run_label = f"{args.run_name}_{args.n_epochs}epochs"
    run_dir   = os.path.join(args.runs_dir, run_label)
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run directory: {run_dir}")

    ckpt_path = os.path.join(run_dir, 'checkpoint.json')
    ckpt      = _load_checkpoint(ckpt_path)
    completed = set(ckpt.get('completed_models', []))

    if not ckpt.get('run_name'):
        ckpt.update({
            'run_name':       args.run_name,
            'n_epochs':       args.n_epochs,
            'optimizer':      args.optimizer,
            'n_agents':       args.n_agents,
            'pop_size':       args.pop_size,
            'r_weight':       args.r_weight,
            'fit_condition':  FIT_CONDITION or None,
            'warm_start':     args.warm_start,
            'models':         args.models,
            'started_at':     time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        _save_checkpoint(ckpt_path, ckpt)

    if completed:
        print(f"  Resuming: already completed -> {sorted(completed)}")

    all_results = {}
    for model_name in args.models:
        if model_name in completed:
            print(f"\n  Skipping {model_name} (checkpoint: already done)")
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
        ckpt['completed_models'].append(model_name)
        ckpt[f'{model_name}_completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        _save_checkpoint(ckpt_path, ckpt)
        print(f"  Checkpoint updated: {model_name} saved")

    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE -- {run_label}")
    print(f"{'-'*65}")
    print(f"  {'Model':<16}  {'total MSD':>10}  {'R-MSD':>8}  {'A-MSD':>8}  k")
    print(f"{'-'*65}")
    for name, res in all_results.items():
        k = ALL_MODELS[name].n_params()
        print(f"  {name:<16}  {res['train_total_msd']:>10.5f}  "
              f"{res['train_r_msd']:>8.5f}  {res['train_a_msd']:>8.5f}  {k}")
    print(f"{'='*65}")
    print(f"  Results saved -> {run_dir}")
    print(f"  Next step: python evaluate.py --run-dir {run_dir}")


if __name__ == '__main__':
    main()
