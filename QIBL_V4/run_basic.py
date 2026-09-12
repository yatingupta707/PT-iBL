"""experiments/run_basic.py v4 — Examples for all v4 features."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')

from configs import ModelConfig
from pipeline import build_pipeline
from utils import compute_metrics, ParameterEstimator, GroupParameterEstimator

def run_basic():
    cfg = ModelConfig(n_trials=150, seed=42, log_experiments=True,
                      experiment_name='v4_basic', log_dir='logs',
                      ibl_decay_d=0.5, ibl_noise_sigma=0.25, ibl_p_inertia=0.0)
    results = build_pipeline(cfg).run_experiment(run_id='demo')
    m = compute_metrics(results)
    print(f"\n[Basic v4]  risk={m['risk_rate']:.3f}  alt={m['alternation_rate']:.3f}  |I|={results['interference'].abs().mean():.4f}")
    return results

def run_tpt_random():
    from components.dataloader import DictDataLoader
    situations = [
        {'safe_value':2.0,'risky_values':[0,4],'risky_probs':[0.5,0.5]},
        {'safe_value':1.5,'risky_values':[0,6],'risky_probs':[0.4,0.6]},
        {'safe_value':3.0,'risky_values':[1,5],'risky_probs':[0.3,0.7]},
    ] * 50
    loader = DictDataLoader(situations)
    cfg = ModelConfig(n_trials=150, seed=42, dataloader_sampling_mode='random',
                      log_experiments=True, experiment_name='v4_tpt', log_dir='logs')
    results = build_pipeline(cfg, dataloader=loader).run_experiment(run_id='random')
    print(f"\n[TPT-Random] risk={results['chose_risky'].mean():.3f}  situations={sorted(results['trial_safe_value'].unique())}")
    return results

def run_mle_individual():
    print("\n[MLE Individual]")
    true_cfg = ModelConfig(pt_alpha=0.75, pt_lambda=2.8, ibl_decay_d=0.6,
                           ibl_noise_sigma=0.3, ibl_p_inertia=0.1,
                           n_trials=80, seed=1, log_experiments=False)
    human = build_pipeline(true_cfg).run_experiment()
    print(f"  Synthetic: {len(human)} trials, risk={human['chose_risky'].mean():.3f}")

    base = ModelConfig(n_trials=80, seed=1, log_experiments=False)
    est  = ParameterEstimator(base, n_restarts=2, max_iter=300, verbose=True)
    learned = est.fit(human[['trial','action','raw_outcome']])
    print(f"\n  True  d={true_cfg.ibl_decay_d} σ_s={true_cfg.ibl_noise_sigma} p={true_cfg.ibl_p_inertia}")
    print(f"  Fit   d={learned.ibl_decay_d:.3f} σ_s={learned.ibl_noise_sigma:.3f} p={learned.ibl_p_inertia:.3f}")
    print(f"  NLL={learned.fit_nll:.4f}")
    return learned

def run_metric_group():
    print("\n[Group Metric Fitting]")
    participants = [
        build_pipeline(ModelConfig(n_trials=50, seed=i, log_experiments=False,
                                   pt_alpha=0.8+i*0.02, ibl_decay_d=0.5)).run_experiment()
        for i in range(5)
    ]
    from utils.behaviour_metrics import compute_group_metrics
    group_m = compute_group_metrics(participants)
    print(f"  Group targets: risk={group_m['risk_rate']:.3f}  alt={group_m['alternation_rate']:.3f}")

    gcfg = ModelConfig(n_trials=50, seed=42, log_experiments=False,
                       fitting_mode='group', training_objective='metric',
                       n_metric_sims=5, metrics_to_use=['risk_rate','alternation_rate'])
    gest = GroupParameterEstimator(gcfg, n_restarts=2, max_iter=200, verbose=True)
    gleaned = gest.fit(participants)
    print(f"  Fitted loss={gleaned.fit_loss:.5f}  α={gleaned.pt_alpha:.3f}  d={gleaned.ibl_decay_d:.3f}")
    return gleaned

if __name__ == '__main__':
    r1 = run_basic()
    r2 = run_tpt_random()
    r3 = run_mle_individual()
    r4 = run_metric_group()
    print("\nAll v4 experiments complete.")
