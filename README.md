# NIFTY 50 Implied Volatility Surface Reconstruction

The objective of this project was to reconstruct missing Implied Volatility (IV) values in a NIFTY 50 options chain dataset. Each row represents a 5-minute snapshot of the full option chain spanning the January 2026 expiry cycle (07 Jan - 27 Jan 2026). Approximately 20% of IV values were missing and needed to be imputed without using any future information (no lookahead bias).


---

## Repository Contents

| File | Description |
|------|-------------|
| `IV_Surface_Reconstruction.ipynb` | Full solution notebook — EDA, all 7 approaches, and final inference |
| `Project_Report.pdf` | 5-page write-up covering methodology, results, and key findings |
| `dataset.csv` | Raw NIFTY 50 options chain (975 timestamps × 28 contracts, ~20% IV missing) |
| `sandbox_solution.csv` | Organiser-provided sample solution — see note below |

---

## Approach

The winning solution is a **Weighted Polynomial + PCHIP Hybrid Reconstructor**. For each missing IV value, a locally-weighted polynomial is fitted on nearby observed strikes using Gaussian kernel weights, with bandwidth and degree selected via leave-one-out CV. Interior (belly) strikes use a 50/50 polynomial-PCHIP blend; exterior (wing) strikes use a 70/20/10 blend of three side-anchored polynomial fits to prevent wing explosion. No ML models — the IV surface is a smooth mathematical manifold and tree-based step functions only add noise.

---

## A Note on `sandbox_solution.csv`

The organisers provided `sandbox_solution.csv` as a sample of the hidden test targets, intended for local validation. Early in the project this file was used directly to evaluate submissions — and it gave misleading signals.

The file is kept in the repo for reference, but all meaningful validation was done using a **custom 20%-masking CV**: 20% of the known IV values were randomly hidden, the reconstructor was run on the remaining 80%, and MSE was measured on the hidden cells. This gave a local CV MSE of `0.00008021` for the final approach, which tracked the leaderboard much more reliably.
