# Results of the revision (round 1)

Result files of the analyses added in the first revision of the paper. All use the cached ViT-L/16 features and
the PCA of the main experiments. AUROC values are fractions (0-1) in the CSV files and percentages in
`selection_result.json`.

| File | Paper | Content |
|---|---|---|
| `patchcore_full.csv` | Table 1, Section 5.1 | PatchCore with its complete scoring on the same features: greedy coreset of 10% of the training patches (random projection to 128 dimensions, seed 0) and the reweighting of the image score with b = 9 support samples (`patchcore_full`); the coreset alone with the maximum (`patchcore_coreset_max`); and PatchCore-NN, exact search with the maximum or P95 (`patchcore_exact_max`, `patchcore_exact_p95`). Per category, with the bank and coreset sizes. |
| `operating_points.csv` | Table 2, Table A10 | False-positive rate at 95% recall and recall at a 5% false-positive rate per category, detector and anomaly subset, read from the test scores (thresholds at the 5th percentile of the anomaly scores and at the 95th percentile of the normal scores). |
| `gradient_predictors.csv` | Table 5, Section 5.3 | PC-Reg_Dual on MVTec LOCO AD with the closed-form ridge predictor, the same ridge objective minimized by full-batch Adam (100, 1000 and 10000 iterations, learning rate 1e-3, W initialized at zero), and a two-layer MLP per position (256-512-256, ReLU, MSE, Adam 1e-3, mini-batches of 32, 200 epochs, seed 0). Columns: AUROC (combined, logical, structural), relative distance to the closed-form W, relative excess of the ridge objective, GPU fitting time of the predictors (s). |
| `selection_candidates.csv` | Table 6, Section 5.4 | Per-category AUROC of the 480 candidate configurations (`s1`: context and regression setting, `agg`: aggregation rule, `branch`: compositional branch, `alpha`: fusion weight). |
| `selection_result.json` | Table 6, Section 5.4 | Configuration selected without each evaluated benchmark (`lobo`), its AUROC, the AUROC of the reported configuration and its rank under the selection criterion; summary of 2000 random 16/16 splits of the 32 categories (`cv`). |
| `grids.csv` | Table A6, Section 5.3 | Compositional branch with cell grids 1x1, 1x2, 2x1, 2x2, 3x3 and 4x4, fused with the positional branch (`dual`) and alone (`comp_only`), per category. |
| `timing_fit.csv` | Table 8, Section 5.6 | Fitting time per step on one CPU thread (reference implementation in `pcreg/`), five MVTec LOCO AD categories, ViT-L/16 and ViT-H+/16. |
| `timing_patchcore.json` | Table 8, Section 5.6 | PatchCore timings on the RTX 3080 Ti laptop GPU (breakfast_box): coreset selection, nearest-neighbor search per image in the exact bank and in the coreset (median of 100 runs, batch size 1). |
