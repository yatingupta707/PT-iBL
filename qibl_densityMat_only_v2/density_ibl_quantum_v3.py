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
      (saves to runs/iteration_5 by default)

  python density_ibl_quantum_v3.py \\
      --run-name iteration_5 --n-epochs 100 --n-agents 10 --pop-size 15 \\
      --optimizer de --shape-weight 0.1 --score-window 5 --cum-shape-weight 0.15 \\
      --freeze-p-reveal --warm-start runs/iteration_4 \\
      --models IBL PTiBL IBLQuantum PTIBLQuantum

  Fit tDCS_0 humans only (needed if you judge plot_tDCS_0_*.png):
      python density_ibl_quantum_v3.py --run-name iteration_5_tdcs0 \\
          --fit-condition tDCS_0_load_0 --freeze-p-reveal \\
          --warm-start runs/iteration_4

  Nested test (PTIBLQuantum with theta=0):
      python density_ibl_quantum_v3.py --run-name iteration_5_theta0 \\
          --models PTIBLQuantum --freeze-theta --freeze-p-reveal \\
          --warm-start runs/iteration_4

  Debug with few agents (inspect learning, not a full fit):
      python density_ibl_quantum_v3.py --debug --n-epochs 8 --models IBL

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
                    alternation_series, problem_seed_base)
from human_metrics import (
    human_r_ts_est, human_a_ts_est,
    human_r_ts_comp, human_a_ts_comp, human_reveal_ts_est,
    human_r_inst_est, human_a_inst_est, human_reveal_inst_est,
    human_condition_series,
)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONFIG  (inherited by worker processes via env vars)
# ══════════════════════════════════════════════════════════════════════════════

COLS      = ['id', 'val_high', 'p_high', 'val_low', 'val_safe', 'sure', 'd1', 'mode']
N_TRIALS  = len(human_r_ts_est)
N_AGENTS  = int(os.environ.get('PTIBL_N_AGENTS', '5'))
R_WEIGHT  = float(os.environ.get('PTIBL_R_WEIGHT', '0.5'))
SHAPE_WEIGHT = float(os.environ.get('PTIBL_SHAPE_WEIGHT', '0.1'))
SCORE_WINDOW = int(os.environ.get('PTIBL_SCORE_WINDOW', '5'))
CUM_SHAPE_WEIGHT = float(os.environ.get('PTIBL_CUM_SHAPE_WEIGHT', '0.15'))
DEBUG_MODE = os.environ.get('PTIBL_DEBUG', '0') == '1'
FREEZE_P_REVEAL = os.environ.get('PTIBL_FREEZE_P_REVEAL', '1') == '1'
FREEZE_THETA = os.environ.get('PTIBL_FREEZE_THETA', '0') == '1'
FIT_CONDITION = os.environ.get('PTIBL_FIT_CONDITION', '').strip()
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


def windowed_mean(x, window=5):
    """Trailing-window average. window<=0 or window>=len(x) -> full-history cummean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x
    if window is None or int(window) <= 0:
        div = np.arange(1, n + 1, dtype=float)
        return np.cumsum(x) / div
    w = int(window)
    out = np.empty(n, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        out[t] = x[lo:t + 1].mean()
    return out


def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    div = np.arange(1, len(x) + 1, dtype=float)
    return np.cumsum(x) / div


def _smooth_ts(inst):
    """Apply the training scoring transform to an instantaneous series."""
    return windowed_mean(inst, SCORE_WINDOW)


def eval_bundle(dataset: pd.DataFrame, model_class, params: dict,
                n_agents: int = 5, seed_offset: int = 0) -> dict:
    """
    One simulation pass. Returns instantaneous, windowed, and cumulative rates.

    Instantaneous r_inst[t] = P(action=='risky' | trial t), averaged over
    problems and agents. Reveal counts as 0 (same coding as human risk_series).
    Windowed series are what the training loss primarily scores.
    Cumulative series are what evaluate.py / plot_results.py show.

    Seeding matches evaluate.py: unique RNG block per (seed_offset, problem).
    Do not reuse seed_base=0 across problems — that overfits  n_agents  streams.
    """
    P     = len(dataset)
    r_acc = np.zeros(N_TRIALS)
    a_acc = np.zeros(N_TRIALS)
    v_acc = np.zeros(N_TRIALS)

    for prob_i, (_, row) in enumerate(dataset.iterrows()):
        seed_base = problem_seed_base(prob_i, n_agents, P, seed_offset)
        r_row, a_row, v_row = _run_one_problem(
            model_class, params, row, n_agents, seed_base)
        r_acc += r_row
        a_acc += a_row
        v_acc += v_row

    r_inst = r_acc / P
    a_inst = a_acc / P
    v_inst = v_acc / P
    return {
        'r_inst': r_inst, 'a_inst': a_inst, 'v_inst': v_inst,
        'r_win':  _smooth_ts(r_inst),
        'a_win':  _smooth_ts(a_inst),
        'v_win':  _smooth_ts(v_inst),
        'r_cum':  cumulative_mean(r_inst),
        'a_cum':  cumulative_mean(a_inst),
        'v_cum':  cumulative_mean(v_inst),
    }


def eval_ts(dataset: pd.DataFrame, model_class, params: dict,
             n_agents: int = 5) -> tuple:
    """Windowed (training) R/A/reveal series. See eval_bundle for all three spaces."""
    b = eval_bundle(dataset, model_class, params, n_agents)
    return b['r_win'], b['a_win'], b['v_win']


def msd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean((a[:n] - b[:n]) ** 2))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0 if either series is (near) constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _active_human():
    """Human R/A/reveal series for the current training target.

    Default is pooled load_0 (estimation set). --fit-condition uses one
    tDCS/load slice so tDCS_0 plots are a fair overlay, not a pooled fit
    judged on a low-R subgroup.
    """
    cond = FIT_CONDITION
    if cond:
        if cond not in human_condition_series:
            raise RuntimeError(
                f"Unknown --fit-condition {cond!r}. "
                f"Valid: {sorted(human_condition_series)}"
            )
        s = human_condition_series[cond]
        z = np.zeros_like(s['risk_inst'])
        return {
            'r_inst': s['risk_inst'],
            'a_inst': s['alternate_inst'],
            'v_inst': s.get('reveal_inst', z),
            'r_cum':  s['risk'],
            'a_cum':  s['alternate'],
            'v_cum':  s.get('reveal', z),
        }
    return {
        'r_inst': human_r_inst_est,
        'a_inst': human_a_inst_est,
        'v_inst': human_reveal_inst_est,
        'r_cum':  human_r_ts_est,
        'a_cum':  human_a_ts_est,
        'v_cum':  human_reveal_ts_est,
    }


def _human_score_targets():
    """Human R/A/reveal series under the same transform used for the model."""
    h = _active_human()
    return (
        _smooth_ts(h['r_inst']),
        _smooth_ts(h['a_inst']),
        _smooth_ts(h['v_inst']),
    )


def _score_bundle(bundle: dict) -> dict:
    """
    Combined loss on the windowed series, plus an optional cumulative-shape
    term so the published cumulative plots cannot invert slope.

    Windowed (primary, SCORE_WINDOW):
        loss = MSD_term + shape_weight * [r_weight*(1-Rcorr) + (1-r_weight)*(1-Acorr)]
    Cumulative (plots / evaluate.py):
        extra = cum_shape_weight * (1 - Rcorr_cum)
    Reveal stays in the MSD term only: p_reveal is a constant coin-flip.
    """
    r_ts, a_ts, v_ts = bundle['r_win'], bundle['a_win'], bundle['v_win']
    h_r, h_a, h_v = _human_score_targets()
    msd_r  = msd(r_ts, h_r)
    msd_a  = msd(a_ts, h_a)
    msd_v  = msd(v_ts, h_v)
    corr_r = corr(r_ts, h_r)
    corr_a = corr(a_ts, h_a)
    msd_term   = R_WEIGHT * msd_r + (1.0 - R_WEIGHT) * (msd_a + msd_v) / 2.0
    shape_term = R_WEIGHT * (1.0 - corr_r) + (1.0 - R_WEIGHT) * (1.0 - corr_a)
    corr_r_cum = corr(bundle['r_cum'], _active_human()['r_cum'])
    corr_a_cum = corr(bundle['a_cum'], _active_human()['a_cum'])
    cum_term   = CUM_SHAPE_WEIGHT * (1.0 - corr_r_cum)
    r_inst     = bundle['r_inst']
    n          = len(r_inst)
    early_n    = min(10, n)
    late_n     = min(10, n)
    return {
        'total':      msd_term + SHAPE_WEIGHT * shape_term + cum_term,
        'msd_term':   msd_term,
        'shape_term': shape_term,
        'cum_term':   cum_term,
        'msd_r':      msd_r,
        'msd_a':      msd_a,
        'msd_v':      msd_v,
        'corr_r':     corr_r,
        'corr_a':     corr_a,
        'corr_r_cum': corr_r_cum,
        'corr_a_cum': corr_a_cum,
        'mean_r_inst': float(r_inst.mean()),
        'mean_reveal': float(bundle['v_inst'].mean()),
        'early_r':    float(r_inst[:early_n].mean()),
        'late_r':     float(r_inst[-late_n:].mean()),
    }


def _score_ts(r_ts, a_ts, v_ts) -> dict:
    """Backward-compatible wrapper: treat arguments as already-windowed series."""
    bundle = {
        'r_win': r_ts, 'a_win': a_ts, 'v_win': v_ts,
        'r_inst': r_ts, 'a_inst': a_ts, 'v_inst': v_ts,
        'r_cum': cumulative_mean(r_ts),
        'a_cum': cumulative_mean(a_ts),
        'v_cum': cumulative_mean(v_ts),
    }
    return _score_bundle(bundle)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL OBJECTIVE FUNCTIONS  (must be picklable for DE workers=-1)
# ══════════════════════════════════════════════════════════════════════════════

def _effective_bounds(model_class):
    """PARAM_BOUNDS with optional frozen coordinates collapsed to a tiny interval."""
    bounds = [tuple(b) for b in model_class.PARAM_BOUNDS]
    names = list(model_class.PARAM_NAMES)
    if FREEZE_P_REVEAL and 'p_reveal' in names:
        bounds[names.index('p_reveal')] = (0.0, 1e-6)
    if FREEZE_THETA and 'theta' in names:
        bounds[names.index('theta')] = (0.0, 1e-6)
    return bounds


def _build_params(model_class, x: np.ndarray) -> dict:
    params = {}
    for name, val, (lo, hi) in zip(
            model_class.PARAM_NAMES, x, _effective_bounds(model_class)):
        if hi <= lo + 1e-8:
            params[name] = float(lo)
        else:
            params[name] = float(np.clip(val, lo + 1e-9, hi - 1e-9))
    if FREEZE_P_REVEAL and 'p_reveal' in params:
        params['p_reveal'] = 0.0
    if FREEZE_THETA and 'theta' in params:
        params['theta'] = 0.0
    return params


def _params_to_x(model_class, params: dict) -> np.ndarray:
    """Clip a saved param dict into the current bound box (for warm-start)."""
    defaults = dict(zip(model_class.PARAM_NAMES, model_class.PARAM_DEFAULT))
    bounds = _effective_bounds(model_class)
    x = []
    for name, (lo, hi) in zip(model_class.PARAM_NAMES, bounds):
        val = float(params[name]) if name in params else float(defaults[name])
        if name == 'p_reveal' and FREEZE_P_REVEAL:
            val = 0.0
        if name == 'theta' and FREEZE_THETA:
            val = 0.0
        x.append(float(np.clip(val, lo + 1e-9, hi - 1e-9)))
    return np.asarray(x, dtype=float)


def _init_population(model_class, popsize: int, warm_x, rng: np.random.Generator):
    """Latin-hypercube population with the previous best (and jittered copies) injected."""
    bounds = np.asarray(_effective_bounds(model_class), dtype=float)
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
    """Windowed MSD/shape + optional cumulative-slope objective."""
    if _EST is None:
        raise RuntimeError("Dataset not loaded. Check data directory.")
    params = _build_params(model_class, x)
    bundle = eval_bundle(_EST, model_class, params, N_AGENTS)
    return _score_bundle(bundle)['total']


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


def _dump_debug_trace(model_dir, model_class, params, gen):
    """Write a 3-agent trial-by-trial trace on the first estimation problem."""
    row = _EST.iloc[0]
    risky_values = (float(row['val_high']), float(row['val_low']))
    risky_probs  = (float(row['p_high']), 1.0 - float(row['p_high']))
    safe_value   = float(row['val_safe'])
    payload = {
        'gen': gen,
        'problem': {k: (float(row[k]) if k != 'id' else str(row[k]))
                    for k in ['id', 'val_high', 'p_high', 'val_low', 'val_safe']},
        'agents': [],
    }
    n_trace = 3
    for ai in range(n_trace):
        model = model_class(params)
        actions, traces = model.simulate(
            N_TRIALS, safe_value, risky_values, risky_probs,
            seed=ai, return_trace=True,
        )
        r = np.array([t['chose_risky'] for t in traces])
        payload['agents'].append({
            'agent': ai,
            'n_risky': int(r.sum()),
            'n_reveal': int(sum(t['chose_reveal'] for t in traces)),
            'mean_p_risky': float(np.mean([t['p_risky'] for t in traces])),
            'mean_outcome': float(np.mean([t['outcome'] for t in traces])),
            'p_risky_early': float(np.mean([t['p_risky'] for t in traces[:10]])),
            'p_risky_late': float(np.mean([t['p_risky'] for t in traces[-10:]])),
            'trials': traces[: min(15, len(traces))],
        })
    path = os.path.join(model_dir, f'debug_agent_trace_gen{gen:03d}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[{model_class.MODEL_NAME}] wrote {path}")


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
    bounds      = _effective_bounds(model_class)
    os.makedirs(model_dir, exist_ok=True)

    loss_history = {'de_generations': [], 'nm_polish': None,
                    'optimizer_note': 'scipy DE is derivative-free; no gradients exist. '
                                      'Logged instead: loss terms, windowed/cumulative R-corr, '
                                      'early/late r_inst, param L2 step, p_reveal.'}
    _gen = [0]
    _prev_x = [None]

    # ── DE callback ────────────────────────────────────────────────────────
    def callback(xk, convergence=None):
        _gen[0] += 1
        params = _build_params(model_class, xk)
        bundle = eval_bundle(_EST, model_class, params, N_AGENTS)
        sc = _score_bundle(bundle)

        step = None
        if _prev_x[0] is not None:
            step = float(np.linalg.norm(np.asarray(xk) - _prev_x[0]))
        _prev_x[0] = np.asarray(xk, dtype=float).copy()

        entry = {
            'gen':         _gen[0],
            'total':       round(sc['total'], 6),
            'msd_term':    round(sc['msd_term'], 6),
            'shape_term':  round(sc['shape_term'], 6),
            'cum_term':    round(sc['cum_term'], 6),
            'total_msd':   round(sc['msd_term'], 6),
            'r_msd':       round(sc['msd_r'], 6),
            'a_msd':       round(sc['msd_a'], 6),
            'reveal_msd':  round(sc['msd_v'], 6),
            'r_corr':      round(sc['corr_r'], 4),
            'a_corr':      round(sc['corr_a'], 4),
            'r_corr_cum':  round(sc['corr_r_cum'], 4),
            'a_corr_cum':  round(sc['corr_a_cum'], 4),
            'mean_r_inst': round(sc['mean_r_inst'], 4),
            'mean_reveal': round(sc['mean_reveal'], 4),
            'early_r':     round(sc['early_r'], 4),
            'late_r':      round(sc['late_r'], 4),
            'param_l2_step': None if step is None else round(step, 6),
            'p_reveal':    round(float(params.get('p_reveal', 0.0)), 4),
            'params':      {k: round(v, 6) for k, v in params.items()},
        }
        loss_history['de_generations'].append(entry)

        with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f, indent=2)

        print(f"[{model_name}] Gen {_gen[0]:3d} | "
              f"loss={sc['total']:.5f}  msd={sc['msd_term']:.5f}  "
              f"shape={sc['shape_term']:.3f}  cum={sc['cum_term']:.3f}  "
              f"r_corr_win={sc['corr_r']:+.3f}  r_corr_cum={sc['corr_r_cum']:+.3f}  "
              f"earlyR={sc['early_r']:.3f} lateR={sc['late_r']:.3f}")

        if DEBUG_MODE and _EST is not None and _gen[0] <= 3:
            _dump_debug_trace(model_dir, model_class, params, _gen[0])
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
        bundle_nm = eval_bundle(_EST, model_class, params_nm, N_AGENTS)
        sc_nm = _score_bundle(bundle_nm)
        loss_history['nm_polish'] = {
            'total':      round(float(res_nm.fun), 6),
            'msd_term':   round(sc_nm['msd_term'], 6),
            'shape_term': round(sc_nm['shape_term'], 6),
            'cum_term':   round(sc_nm['cum_term'], 6),
            'total_msd':  round(sc_nm['msd_term'], 6),
            'r_msd':      round(sc_nm['msd_r'], 6),
            'a_msd':      round(sc_nm['msd_a'], 6),
            'reveal_msd': round(sc_nm['msd_v'], 6),
            'r_corr':     round(sc_nm['corr_r'], 4),
            'a_corr':     round(sc_nm['corr_a'], 4),
            'r_corr_cum': round(sc_nm['corr_r_cum'], 4),
            'params':     {k: round(v, 6) for k, v in params_nm.items()},
            'improved_over_de': bool(improved),
        }
        print(f"[{model_name}] NM {'improved' if improved else 'did not improve'}  "
              f"best_loss={best_loss:.5f}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    best_params = _build_params(model_class, best_x)
    bundle_final = eval_bundle(_EST, model_class, best_params, N_AGENTS)
    sc_final = _score_bundle(bundle_final)

    result = {
        'model':            model_name,
        'optimizer':        optimizer,
        'n_epochs':         n_epochs,
        'n_agents':         N_AGENTS,
        'r_weight':         R_WEIGHT,
        'shape_weight':     SHAPE_WEIGHT,
        'score_window':     SCORE_WINDOW,
        'cum_shape_weight': CUM_SHAPE_WEIGHT,
        'freeze_p_reveal':  FREEZE_P_REVEAL,
        'freeze_theta':     FREEZE_THETA,
        'fit_condition':    FIT_CONDITION or None,
        'warm_start':       warm_dir,
        'train_total_loss': round(best_loss, 6),
        'train_total_msd':  round(sc_final['msd_term'], 6),
        'train_shape_term': round(sc_final['shape_term'], 6),
        'train_cum_term':   round(sc_final['cum_term'], 6),
        'train_r_msd':      round(sc_final['msd_r'], 6),
        'train_a_msd':      round(sc_final['msd_a'], 6),
        'train_reveal_msd': round(sc_final['msd_v'], 6),
        'train_r_corr':     round(sc_final['corr_r'], 4),
        'train_a_corr':     round(sc_final['corr_a'], 4),
        'train_r_corr_cum': round(sc_final['corr_r_cum'], 4),
        'train_a_corr_cum': round(sc_final['corr_a_cum'], 4),
        'train_early_r':    round(sc_final['early_r'], 4),
        'train_late_r':     round(sc_final['late_r'], 4),
        'params':           {k: round(v, 6) for k, v in best_params.items()},
        'fitted_at':        time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    with open(os.path.join(model_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)

    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{model_name}] Done  ->  loss={best_loss:.5f}  "
          f"r_msd={sc_final['msd_r']:.5f}  r_corr_win={sc_final['corr_r']:+.3f}  "
          f"r_corr_cum={sc_final['corr_r_cum']:+.3f}  "
          f"earlyR={sc_final['early_r']:.3f} lateR={sc_final['late_r']:.3f}")
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

Make **published cumulative** R/A curves match humans. iteration_4's
`train_r_corr_cum ≈ +0.90` was an artefact: training reused `seed_base=0`
on every problem (10 RNG streams copied 60 times). `evaluate.py` uses a
unique seed per problem, so IBL / PTiBL / PTIBLQuantum flipped to
**−0.80 to −0.86**. IBLQuantum stayed positive because it actually learns.

## What changed versus iteration 4

### 1. Training seeds now match evaluate.py

`eval_bundle` uses `problem_seed_base(prob_i, n_agents, n_problems, seed_offset)`
— the same formula as `evaluate.py`. Loss weights are **unchanged**.

### 2. `p_reveal` frozen at 0 (default)

Coin-flip reveal is scored as R=0 and cannot create a declining slope.
iteration_4 fitted `p_reveal ≈ 0.18–0.21`, which biased the level and,
under shared seeds, the fake first-trial R=0.70. Disable with
`--no-freeze-p-reveal`.

### 3. Optional `--fit-condition tDCS_0_load_0`

Pooled load_0 humans (mean R ≈ 0.42) cannot match tDCS_0_load_0
(mean R ≈ 0.34). Judge pooled fits on `plot_estimation_set.png`.
Use `--fit-condition` only if tDCS_0 figures are the deliverable.

## Search settings

| setting           | iteration 4 | this run |
|-------------------|-------------|----------|
| problem seeds     | shared 0    | unique (evaluate.py) |
| freeze_p_reveal   | no          | {getattr(args, 'freeze_p_reveal', True)} |
| freeze_theta      | no          | {getattr(args, 'freeze_theta', False)} |
| fit_condition     | pooled load_0 | {getattr(args, 'fit_condition', None) or 'pooled load_0'} |
| score_window      | 5           | {args.score_window} |
| shape_weight      | 0.1         | {args.shape_weight} |
| cum_shape_weight  | 0.15        | {args.cum_shape_weight} |
| n_agents          | 10          | {args.n_agents} |
| pop_size          | 15          | {args.pop_size} |
| n_epochs          | 100         | {args.n_epochs} |
| warm_start        | iteration_3 | {args.warm_start} |

## What did not change

- Default plot files are still cumulative overlays (R|A, all models)
- Reveal is still R=0 in the R-rate (same as human `risk_series`)
- Do not raise n_agents as the main lever; DE has no gradients

## How to evaluate, plot, debug

```
python evaluate.py --run-dir {run_dir} --n-sims 5 --n-agents 20
python plot_results.py --run-dir {run_dir} --no-show
python plot_results.py --run-dir {run_dir} --rate win --no-show
python debug_loss_isolated.py --run-dir {run_dir} --n-agents 3 --n-sims 1
```

Judge pooled fits on `plot_estimation_set.png`, not on `plot_tDCS_0_*`.

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
    parser.add_argument('--run-name',   default='iteration_5',
                         help='Run folder name under --runs-dir (default: iteration_5)')
    parser.add_argument('--n-epochs',   type=int, default=100,
                         help='DE maxiter (number of generations)')
    parser.add_argument('--n-agents',   type=int, default=10,
                         help='Agents per problem per evaluation (MC noise budget)')
    parser.add_argument('--pop-size',   type=int, default=15,
                         help='DE population multiplier (actual pop = pop-size * n_params). '
                              'This is search budget, not Monte Carlo agents.')
    parser.add_argument('--r-weight',   type=float, default=0.5,
                         help='Weight on R-rate vs A/reveal (0-1)')
    parser.add_argument('--shape-weight', type=float, default=0.1,
                         help='Weight on (1 - corr) so the optimizer matches curve '
                              'shape, not only MSD level. 0 disables the shape term.')
    parser.add_argument('--score-window', type=int, default=5,
                         help='Trailing-window length used for the training loss. '
                              '5 = last 5 trials; 0 = full-history cumulative (old behaviour).')
    parser.add_argument('--cum-shape-weight', type=float, default=0.15,
                         help='Extra penalty on (1 - cumulative R-corr) so published '
                              'cumulative plots cannot invert slope. 0 disables it.')
    parser.add_argument('--warm-start', default='runs/iteration_4',
                         help='Previous run directory whose best_params.json seed the '
                              'DE population. Use "none" for a fresh Latin hypercube.')
    parser.add_argument('--freeze-p-reveal', dest='freeze_p_reveal',
                         action='store_true',
                         help='Force p_reveal=0 (default). Coin-flip reveal biases R-rate.')
    parser.add_argument('--no-freeze-p-reveal', dest='freeze_p_reveal',
                         action='store_false',
                         help='Allow DE to fit p_reveal in [0, 1].')
    parser.set_defaults(freeze_p_reveal=True)
    parser.add_argument('--freeze-theta', action='store_true',
                         help='Force quantum theta=0 (nested test: density rotation off).')
    parser.add_argument('--fit-condition', default='',
                         help='Human target condition (e.g. tDCS_0_load_0). '
                              'Default empty = pooled load_0 estimation set.')
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
    parser.add_argument('--debug', action='store_true',
                         help='Few-agent debug fit: n_agents=3, dump trial traces, '
                              'single-process. Intended for inspecting learning, not a full run.')
    args = parser.parse_args()

    # ── Validate model names ────────────────────────────────────────────────
    for m in args.models:
        if m not in ALL_MODELS:
            parser.error(f"Unknown model '{m}'. Valid: {list(ALL_MODELS)}")

    if args.debug:
        args.n_agents = min(args.n_agents, 3)
        args.no_parallel = True
        print('  DEBUG MODE: n_agents=%d, single-process, agent traces on gens 1-3'
              % args.n_agents)

    if args.warm_start and str(args.warm_start).lower() in ('none', 'off', '-', ''):
        args.warm_start = None

    if args.fit_condition and args.fit_condition not in human_condition_series:
        parser.error(
            f"Unknown --fit-condition {args.fit_condition!r}. "
            f"Valid: {sorted(human_condition_series)}"
        )

    # ── Set module-level config AND env vars (inherited by DE workers) ───────
    global N_AGENTS, R_WEIGHT, SHAPE_WEIGHT, SCORE_WINDOW, CUM_SHAPE_WEIGHT
    global DEBUG_MODE, FREEZE_P_REVEAL, FREEZE_THETA, FIT_CONDITION
    global _EST, _COMP, _DATA_DIR
    N_AGENTS     = args.n_agents
    R_WEIGHT     = args.r_weight
    SHAPE_WEIGHT = args.shape_weight
    SCORE_WINDOW = args.score_window
    CUM_SHAPE_WEIGHT = args.cum_shape_weight
    DEBUG_MODE   = bool(args.debug)
    FREEZE_P_REVEAL = bool(args.freeze_p_reveal)
    FREEZE_THETA = bool(args.freeze_theta)
    FIT_CONDITION = (args.fit_condition or '').strip()
    _DATA_DIR    = args.data_dir
    os.environ['PTIBL_N_AGENTS']      = str(args.n_agents)
    os.environ['PTIBL_R_WEIGHT']      = str(args.r_weight)
    os.environ['PTIBL_SHAPE_WEIGHT']  = str(args.shape_weight)
    os.environ['PTIBL_SCORE_WINDOW']  = str(args.score_window)
    os.environ['PTIBL_CUM_SHAPE_WEIGHT'] = str(args.cum_shape_weight)
    os.environ['PTIBL_DEBUG']         = '1' if args.debug else '0'
    os.environ['PTIBL_FREEZE_P_REVEAL'] = '1' if args.freeze_p_reveal else '0'
    os.environ['PTIBL_FREEZE_THETA']  = '1' if args.freeze_theta else '0'
    os.environ['PTIBL_FIT_CONDITION'] = FIT_CONDITION
    os.environ['PTIBL_DATA_DIR']      = args.data_dir
    os.environ['POP_SIZE']            = str(args.pop_size)

    _EST, _COMP = _load_dataset(args.data_dir)
    score_label = (f'window-{args.score_window}' if args.score_window > 0
                   else 'full-history cumulative')
    print(f"  Loaded estimation set : {len(_EST)} problems")
    print(f"  Loaded competition set: {len(_COMP)} problems")
    print(f"  n_agents={args.n_agents}  pop_size={args.pop_size}  "
          f"shape_weight={args.shape_weight}  cum_shape_weight={args.cum_shape_weight}  "
          f"score={score_label}  warm_start={args.warm_start}")
    print(f"  freeze_p_reveal={args.freeze_p_reveal}  freeze_theta={args.freeze_theta}  "
          f"fit_condition={FIT_CONDITION or 'pooled load_0'}")

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
            'score_window': args.score_window,
            'cum_shape_weight': args.cum_shape_weight,
            'freeze_p_reveal': args.freeze_p_reveal,
            'freeze_theta': args.freeze_theta,
            'fit_condition': args.fit_condition or None,
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
        print(f"  Checkpoint updated: {model_name} saved")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE -- {run_label}")
    print(f"{'-'*65}")
    print(f"  {'Model':<16}  {'loss':>10}  {'R-MSD':>8}  {'R-win':>8}  {'R-cum':>8}  k")
    print(f"{'-'*65}")
    for name, res in all_results.items():
        k = ALL_MODELS[name].n_params()
        rcorr = res.get('train_r_corr', float('nan'))
        rcum  = res.get('train_r_corr_cum', float('nan'))
        loss  = res.get('train_total_loss', res.get('train_total_msd', float('nan')))
        print(f"  {name:<16}  {loss:>10.5f}  "
              f"{res['train_r_msd']:>8.5f}  {rcorr:>+8.3f}  {rcum:>+8.3f}  {k}")
    print(f"{'='*65}")
    print(f"  Results saved → {run_dir}")
    print(f"  Next step: python evaluate.py --run-dir {run_dir}")


if __name__ == '__main__':
    main()
