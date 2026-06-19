# Task 2 Reproduction Results

This report separates the required reproduction task, the transfer extension, and the method-development experiments.

# Part 1: Required Reproduction

Fit on the cosine LR schedule and test on the WSD LR schedule. The `cosine -> cosine` rows are fit-quality checks; the `cosine -> WSD` rows are the requested cross-schedule prediction results.

| experiment | model | fit_scheduler | target | role | direction | mae | rmse | mape | r2 | final_abs_error | final_relative_error | selected_velocity_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cosine_fit | Tissue2024 | cosine | cosine | fit_check | cosine -> cosine | 0.0373065 | 0.0490254 | 0.0128612 | 0.964941 | 0.0213224 | 0.0080578 |  |
| cosine_fit | Tissue2024 | cosine | wsd | transfer | cosine -> wsd | 0.0567873 | 0.0695504 | 0.0196105 | 0.922965 | 0.0297001 | 0.0112683 |  |
| cosine_fit | LuoMPL | cosine | cosine | fit_check | cosine -> cosine | 0.0391676 | 0.0518825 | 0.0134705 | 0.960736 | 0.036646 | 0.0138487 |  |
| cosine_fit | LuoMPL | cosine | wsd | transfer | cosine -> wsd | 0.0560156 | 0.0687512 | 0.0193143 | 0.924725 | 0.00497959 | 0.00188926 |  |

## Reproduction Conclusion

- The reproduced baselines reach average MAPE 1.9462% on the requested `cosine -> WSD` transfer.
- Best reproduced baseline on WSD is `LuoMPL` with MAPE 1.9314%.

# Part 2: Extension

Evaluate all six directed transfers among `cosine`, `WSD`, and `8-1-1`, then choose the best direction by the selected metric.

## Extension Transfer Metrics

| experiment | model | fit_scheduler | target | role | direction | mae | rmse | mape | r2 | final_abs_error | final_relative_error | selected_velocity_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 811_fit | LuoMPL | 811 | cosine | transfer | 811 -> cosine | 0.0509064 | 0.0700912 | 0.0175596 | 0.928339 | 0.0788345 | 0.0297918 |  |
| 811_fit | Tissue2024 | 811 | cosine | transfer | 811 -> cosine | 0.0511986 | 0.0703326 | 0.0176701 | 0.927845 | 0.080201 | 0.0303082 |  |
| 811_fit | LuoMPL | 811 | wsd | transfer | 811 -> wsd | 0.0455558 | 0.0653553 | 0.0153165 | 0.931978 | 0.0336998 | 0.0127857 |  |
| 811_fit | Tissue2024 | 811 | wsd | transfer | 811 -> wsd | 0.0455976 | 0.0653723 | 0.0153319 | 0.931942 | 0.0338329 | 0.0128363 |  |
| cosine_fit | LuoMPL | cosine | 811 | transfer | cosine -> 811 | 0.0571609 | 0.0701446 | 0.0197036 | 0.923199 | 0.00603925 | 0.00228734 |  |
| cosine_fit | Tissue2024 | cosine | 811 | transfer | cosine -> 811 | 0.0574659 | 0.0707238 | 0.0198223 | 0.921926 | 0.0236554 | 0.00895935 |  |
| cosine_fit | LuoMPL | cosine | wsd | transfer | cosine -> wsd | 0.0560156 | 0.0687512 | 0.0193143 | 0.924725 | 0.00497959 | 0.00188926 |  |
| cosine_fit | Tissue2024 | cosine | wsd | transfer | cosine -> wsd | 0.0567873 | 0.0695504 | 0.0196105 | 0.922965 | 0.0297001 | 0.0112683 |  |
| wsd_fit | LuoMPL | wsd | 811 | transfer | wsd -> 811 | 0.0448097 | 0.065366 | 0.0150423 | 0.933307 | 0.0333287 | 0.0126231 |  |
| wsd_fit | Tissue2024 | wsd | 811 | transfer | wsd -> 811 | 0.0445796 | 0.0650111 | 0.0149657 | 0.934029 | 0.0162336 | 0.00614839 |  |
| wsd_fit | LuoMPL | wsd | cosine | transfer | wsd -> cosine | 0.0498115 | 0.0695373 | 0.017168 | 0.929467 | 0.0783838 | 0.0296215 |  |
| wsd_fit | Tissue2024 | wsd | cosine | transfer | wsd -> cosine | 0.0451244 | 0.0653771 | 0.0154051 | 0.937654 | 0.0524721 | 0.0198294 |  |

## Extension Transfer Ranking

| direction | fit_scheduler | target | mean_mape | mean_rmse | mean_mae | mean_final_relative_error |
| --- | --- | --- | --- | --- | --- | --- |
| wsd -> 811 | wsd | 811 | 0.015004 | 0.0651886 | 0.0446947 | 0.00938574 |
| 811 -> wsd | 811 | wsd | 0.0153242 | 0.0653638 | 0.0455767 | 0.012811 |
| wsd -> cosine | wsd | cosine | 0.0162865 | 0.0674572 | 0.047468 | 0.0247255 |
| 811 -> cosine | 811 | cosine | 0.0176149 | 0.0702119 | 0.0510525 | 0.03005 |
| cosine -> wsd | cosine | wsd | 0.0194624 | 0.0691508 | 0.0564014 | 0.00657876 |
| cosine -> 811 | cosine | 811 | 0.019763 | 0.0704342 | 0.0573134 | 0.00562334 |

## Extension Conclusion

- Best average transfer direction by `mape` is `wsd -> 811` with mean MAPE 1.5004%.
- Per-model winners are:
| model | direction | mape | rmse | mae | final_relative_error |
| --- | --- | --- | --- | --- | --- |
| Tissue2024 | wsd -> 811 | 0.0149657 | 0.0650111 | 0.0445796 | 0.00614839 |
| LuoMPL | wsd -> 811 | 0.0150423 | 0.065366 | 0.0448097 | 0.0126231 |

# Part 3: Method Development

Compare the reproduced baselines with the proposed loss-curve fitting strategies on the original `cosine -> WSD` task.

| experiment | model | fit_scheduler | target | role | direction | mae | rmse | mape | r2 | final_abs_error | final_relative_error | selected_velocity_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cosine_fit | Tissue2024 | cosine | cosine | fit_check | cosine -> cosine | 0.0373065 | 0.0490254 | 0.0128612 | 0.964941 | 0.0213224 | 0.0080578 |  |
| cosine_fit | Tissue2024 | cosine | wsd | transfer | cosine -> wsd | 0.0567873 | 0.0695504 | 0.0196105 | 0.922965 | 0.0297001 | 0.0112683 |  |
| cosine_fit | LuoMPL | cosine | cosine | fit_check | cosine -> cosine | 0.0391676 | 0.0518825 | 0.0134705 | 0.960736 | 0.036646 | 0.0138487 |  |
| cosine_fit | LuoMPL | cosine | wsd | transfer | cosine -> wsd | 0.0560156 | 0.0687512 | 0.0193143 | 0.924725 | 0.00497959 | 0.00188926 |  |
| cosine_fit | Dev1-EffectiveTime | cosine | cosine | fit_check | cosine -> cosine | 0.0386772 | 0.0509773 | 0.0133138 | 0.962094 | 0.0316043 | 0.0119434 |  |
| cosine_fit | Dev1-EffectiveTime | cosine | wsd | transfer | cosine -> wsd | 0.054051 | 0.0669493 | 0.0186124 | 0.928619 | 0.0420497 | 0.0159537 |  |
| cosine_fit | Dev2-StepResidual | cosine | cosine | fit_check | cosine -> cosine | 0.0376858 | 0.0499523 | 0.0129689 | 0.963603 | 0.0143579 | 0.00542589 |  |
| cosine_fit | Dev2-StepResidual | cosine | wsd | transfer | cosine -> wsd | 0.0598536 | 0.0728351 | 0.0208392 | 0.915517 | 0.128187 | 0.0486342 |  |
| cosine_fit | Dev3-EffectiveResidual | cosine | cosine | fit_check | cosine -> cosine | 0.0377103 | 0.05 | 0.0129762 | 0.963534 | 0.0143311 | 0.00541576 |  |
| cosine_fit | Dev3-EffectiveResidual | cosine | wsd | transfer | cosine -> wsd | 0.0599231 | 0.0729463 | 0.0208639 | 0.915259 | 0.129243 | 0.0490348 |  |
| cosine_fit | Dev4-VelocityMatched | cosine | cosine | fit_check | cosine -> cosine | 0.0386492 | 0.0510308 | 0.0133024 | 0.962014 | 0.0317226 | 0.0119881 | 0.5 |
| cosine_fit | Dev4-VelocityMatched | cosine | wsd | transfer | cosine -> wsd | 0.0539851 | 0.0669461 | 0.0185875 | 0.928626 | 0.0421679 | 0.0159985 | 0.5 |
| cosine_fit | Dev5-HybridTissueMPL | cosine | cosine | fit_check | cosine -> cosine | 0.0373064 | 0.0490243 | 0.0128612 | 0.964943 | 0.0213223 | 0.00805779 |  |
| cosine_fit | Dev5-HybridTissueMPL | cosine | wsd | transfer | cosine -> wsd | 0.0567872 | 0.0695496 | 0.0196105 | 0.922967 | 0.0297001 | 0.0112683 |  |
| cosine_fit | Dev6-CurvatureMatched | cosine | cosine | fit_check | cosine -> cosine | 0.0386631 | 0.0510036 | 0.0133081 | 0.962055 | 0.0316681 | 0.0119675 |  |
| cosine_fit | Dev6-CurvatureMatched | cosine | wsd | transfer | cosine -> wsd | 0.054016 | 0.066945 | 0.0185992 | 0.928628 | 0.0421135 | 0.0159779 |  |
| cosine_fit | Dev7-PowerExponential | cosine | cosine | fit_check | cosine -> cosine | 0.0331021 | 0.0415094 | 0.0116422 | 0.974867 | 0.0212688 | 0.00803754 |  |
| cosine_fit | Dev7-PowerExponential | cosine | wsd | transfer | cosine -> wsd | 0.0455421 | 0.0565448 | 0.0159005 | 0.949082 | 0.0317141 | 0.0120324 |  |
| cosine_fit | Dev8-TimeEnsemble | cosine | cosine | fit_check | cosine -> cosine | 0.0385724 | 0.0500321 | 0.0133187 | 0.963487 | 0.034306 | 0.0129644 |  |
| cosine_fit | Dev8-TimeEnsemble | cosine | wsd | transfer | cosine -> wsd | 0.0568748 | 0.0694043 | 0.0196493 | 0.923288 | 0.0384283 | 0.0145798 |  |
| cosine_fit | Dev9-TissueVelocity | cosine | cosine | fit_check | cosine -> cosine | 0.0391795 | 0.0520645 | 0.0134694 | 0.96046 | 0.0367883 | 0.0139024 | 0.5 |
| cosine_fit | Dev9-TissueVelocity | cosine | wsd | transfer | cosine -> wsd | 0.0559602 | 0.0688047 | 0.0192919 | 0.924608 | 0.00614077 | 0.00232982 | 0.5 |
| cosine_fit | Dev10-LuoMPLVelocity | cosine | cosine | fit_check | cosine -> cosine | 0.0391394 | 0.0520236 | 0.0134555 | 0.960522 | 0.0365567 | 0.0138149 | 0.5 |
| cosine_fit | Dev10-LuoMPLVelocity | cosine | wsd | transfer | cosine -> wsd | 0.0557793 | 0.0686361 | 0.0192257 | 0.924977 | 0.00507088 | 0.0019239 | 0.5 |
| cosine_fit | Dev11-PowerExpVelocity | cosine | cosine | fit_check | cosine -> cosine | 0.0330878 | 0.0415039 | 0.0116367 | 0.974874 | 0.0211557 | 0.00799481 | 0.5 |
| cosine_fit | Dev11-PowerExpVelocity | cosine | wsd | transfer | cosine -> wsd | 0.0455046 | 0.0565117 | 0.0158869 | 0.949141 | 0.031601 | 0.0119895 | 0.5 |

## Method Development Conclusion

- The reproduced baselines reach average MAPE 1.9462% on the requested `cosine -> WSD` transfer.
- The proposed methods reach average MAPE 1.8824% on the same transfer.
- Best reproduced baseline on WSD is `LuoMPL` with MAPE 1.9314%.
- Best proposed method on WSD is `Dev11-PowerExpVelocity` with MAPE 1.5887%.
- Best overall method on WSD is `Dev11-PowerExpVelocity` with MAPE 1.5887%.

## Notes

- `Tissue2024` uses `L = L0 + A*S1^{-alpha} - C*S2`, with `S2` implemented as decayed LR-annealing momentum.
- `LuoMPL` follows the public MultiPowerLaw repository formula `L = L0 + A*S1^{-alpha} - B*LD`, using the repository's 100M shape parameters for `C`, `beta`, and `gamma` and fitting the remaining curve-specific parameters on the chosen source schedule.
- Main reproduction outputs: `task2_main_metrics.csv`, `task2_cosine_fit_predictions.png`.
- Extension outputs: `task2_extension_metrics.csv`, `task2_transfer_ranking.csv`, `task2_best_transfer.json`, `task2_transfer_comparison.png`, `task2_transfer_heatmap.png`, `task2_wsd_fit_predictions.png`, `task2_811_fit_predictions.png`.
- Method-development outputs: `task2_development_metrics.csv`, `task2_method_development_predictions.png`, `task2_method_development_comparison.png`.
- Combined/debug outputs: `task2_metrics.csv`, `task2_params.json`.