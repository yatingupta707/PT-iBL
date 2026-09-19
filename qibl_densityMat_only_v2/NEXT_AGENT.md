# Handoff — tDCS 3-choice R/A plots

Read this before changing loss weights, agent count, or GPU code.

## What is already done

- Original 2-choice TPT pipeline understood: risk = 1 iff risky; alteration = previous choice differs; both averaged then `cumsum/t`. Plots aligned because **MSD on those cumulative series was the training objective**.
- Adapted to `tdcs_load_60_final_comb.csv` (A=safe, B=risky, R=reveal, 50 trials, 4 conditions).
- Reveal-aware alteration: skip R; compare last committed A/B to next committed A/B. Risk still 1 iff B (R counts as 0, stays in the denominator).
- Code: `human_metrics.py`, `models.py` (`risk_series`, `alternation_series`, optional `p_reveal` default 0), `evaluate.py`, `plot_results.py`, `density_ibl_quantum_v3.py`.
- `*ORIGINAL.py` files are the old 2-choice TPT scripts; leave them.
- **Current PNGs in `runs/tdcs_3choice_eval/` are NOT a tDCS fit.** They are exp2 TPT params (IBL = defaults) evaluated on tDCS humans. That is why all four condition figures look bad. Pooled `plot_estimation_set.png` is the only fair overlay for those params, and R-rate there is already close.

Human mean P(B): tDCS_0_load_0 **0.37**, tDCS_1_load_0 **0.60**, tDCS_0_load_1 **0.30**, tDCS_1_load_1 **0.55**. One vector cannot hit these.

## What NOT to do

- Do not raise `n_agents` as the first lever. It only averages MC noise.
- Do not rewrite for GPU. Fitting is SciPy DE on **CPU** (`workers=-1`).
- Do not enable coin-flip `p_reveal`. It scored as R=0 and inverted unique-seed slopes in earlier iterations.
- Do not retune `shape_weight` / windowed loss. This round uses original MSD-on-cumulative.
- Do not judge a pooled or TPT vector on `plot_tDCS_0_*`.

## What Harshbir should run next (SSH, CPU)

Four **separate** fits, 10 agents, 50 DE gens, warm-start from `qiblruns/exp2_30epochs` (IBL has no warm file → latin hypercube).

```bash
tmux new -s qibl
cd qibl_densityMat_only_v2
# tdcs_load_60_final_comb.csv must sit in the parent folder

for cond in tDCS_0_load_0 tDCS_1_load_0 tDCS_0_load_1 tDCS_1_load_1; do
  python -u density_ibl_quantum_v3.py \
    --run-name tdcs_${cond} \
    --fit-condition ${cond} \
    --n-epochs 50 --n-agents 10 --pop-size 10 \
    --optimizer de \
    --warm-start qiblruns/exp2_30epochs \
    --models IBL PTiBL IBLQuantum PTIBLQuantum

  python evaluate.py --run-dir runs/tdcs_${cond}_50epochs --n-sims 5 --n-agents 10
  python plot_results.py --run-dir runs/tdcs_${cond}_50epochs --no-show
done
```

Judge **that run’s** `plot_${cond}.png` (R left, A right). Checkpointing skips finished models if the same command is rerun.

## After those four runs (next agent)

1. Compare each `plot_tDCS_*.png` to `runs/tdcs_3choice_eval/plot_tDCS_*.png`. Level should move toward that cell’s humans. If tDCS_0 and tDCS_1 still share the same model curves, `--fit-condition` did not take.
2. Open `*/loss_history.json`. If total MSD still falling in the last 10 gens → same 10 agents, 100 epochs, `--warm-start` that condition’s 50-epoch folder. If flat by ~gen 20 with wrong slope → **stop scaling compute**.
3. Read `theta` in PTIBLQuantum `best_params.json`. If still ~0, nested freeze-theta is the next test, not more epochs.
4. `tDCS_1_load_1` humans **rise** in late R. If the fitted model still only falls/settles, that is IBL equilibrium (capacity), not agent count. Report it; do not add a coin-flip reveal to fake the slope.
5. Only then consider a value-dependent reveal (third option or utility-gated sample). Not a constant `p_reveal`.

## Files to start from

- This note
- `runs/tdcs_3choice_eval/eval_all_summary.json`
- `density_ibl_quantum_v3.py` (`--fit-condition`, `--warm-start`)
- Human CSV: parent `tdcs_load_60_final_comb.csv`
