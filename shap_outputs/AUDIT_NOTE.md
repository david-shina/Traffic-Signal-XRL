# LIME + SHAP Output Audit

- Method: PNGs could not be viewed directly (model limitation); audited programmatically via PIL pixel/color stats plus a gradient-sensitivity check of P(switch) against `policy_checkpoints_phase2/best_policy/variables/variables.index`.
- Checkpoint health: Q-values are diverged (range [-429.5, -0.15], mean ~-170), which destabilises the softmax the explainers wrap.
- Action distribution: P(switch) spans [0,1] on uniform states (mean 0.24, 22.5% argmax-switch) but only 2.4% on real rollout states — the boundary is real yet concentrated in rare regions.
- Feature sensitivity: `mean_wait` (waiting time) and `time_since_switch` dominate; `pressure`/`demand` are weak (0.09/0.05), so the surrogate tree's pressure split is a depth-2 artefact.
- Structural: `shap_waterfall_switch.png` is absent (skipped because max P(switch)<0.5); LIME PNGs are written under `shap_outputs/` (no `lime_outputs/` exists); no `.npz` raw data is committed.
- Verdict: explainers are faithful but low-signal because Q-values diverged and switch is rare; re-train to stabilise Q-values and increase action diversity, then regenerate.
