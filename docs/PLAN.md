# Project plan: transcriptional noise and ML in Alzheimer's disease

## Question

Do per-cell-type transcriptional noise measures predict a donor's Alzheimer's pathology, and do they add information **beyond** mean expression?

**Hypothesis:** some populations (e.g. microglia and vulnerable neurons) increase their cell-to-cell variability with pathology, and that increase can be detected even when the mean does not change.

## So what

- Identifies which cell types lose control of their expression first.
- Proposes noise as a class of biomarker complementary to differential expression.
- It is testable: it can be validated in a second, independent dataset.

## Data

| | Use |
|---|---|
| **Lau et al. 2020 (prefrontal cortex)** | Discovery. 21 donors (12 AD, 9 control), already preprocessed by the thesis pipeline (`.h5ad` with raw counts in `layers["counts"]`). |
| **SEA-AD (MTG, snRNA-seq)** | External validation. Per-donor neuropathology (ADNC, Braak, Thal) and age/sex. Open access via CELLxGENE. |
| **Simulated data** | Pipeline positive control (already included). |

If the atlas is very large, work only with the cell types of interest and use anndata's `backed` mode.

## Methods

1. **Preprocessing:** raw counts, downsampling to a common depth, the same number of cells per donor x cell type, a minimum number of cells per group.
2. **Noise metrics:** residual of log CV² from the mean–CV² trend (per gene) and mean distance to centroid (global). Compare with the thesis metrics and keep the most robust ones.
3. **Donor-level features:** mean (baseline), per-gene noise, global noise, combined, covariates.
4. **Models:** L1/L2-regularised logistic regression (main, interpretable) and random forest (comparison).
5. **Evaluation:** repeated nested CV, AUC; pairwise comparison on the same splits; permutation test (≥200 permutations for the final result).
6. **Interpretation:** which cell types and genes carry the most weight; functional enrichment of those genes.
7. **Extensions if time allows:** continuous target (Braak or a progression score), external validation, sensitivity to depth and cell numbers.

## Risks

- **Few donors (21 in Lau et al. 2020):** limits power. Mitigation: simple models, many repeated splits, permutation test, and treating any result as a hypothesis for SEA-AD.
- **Technical confounding:** batch, post-mortem interval, nucleus quality. Mitigation: include them as covariates or check that they do not predict the label on their own.
- **Noise adds nothing:** still a result worth showing if it is well controlled.
- **Overlap with the thesis:** agree with the thesis supervisor on which code and results can be public before uploading anything with real data.
