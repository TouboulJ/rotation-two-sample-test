# Two-sample testing up to a rotation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23023835.svg)](https://doi.org/10.5281/zenodo.23023835)


Reproducibility repository for the paper *"Two-sample testing up to a rotation:
calibration, a projected non-Gaussianity test, and a Hermite view of contrast
choice"* (`paper.pdf`).

## Summary

We test whether two whitened planar distributions are equal up to an unknown
rotation --- the alignment question behind two-dimensional independent component
analysis. The paper makes three points:

1. **A calibration caution.** Aligning a two-sample statistic by a rotation search
   and calibrating by permutation is anticonservative: it rejects a genuine
   rotation far too often.
2. **A valid test.** A projected non-Gaussianity statistic with an explicit limit
   law and a consistent multiplier bootstrap; naive resampling is proved to
   miscalibrate. Among valid competitors an ICA-canonicalized energy test is
   competitive or more powerful, so the projected test's value is *guaranteed
   calibration*, not power.
3. **A Hermite view of contrast choice.** Whitening annihilates the first three
   Hermite orders; no single contrast dominates, so a small multiplicity-corrected
   bank is used.

A real-data illustration on the Wisconsin Diagnostic Breast Cancer dataset shows
that, on real marginals, the projected test is the only competitor that stays
level-valid --- naive energy and even energy-after-ICA over-reject.

## Contents

| Path | What it is |
|------|------------|
| `paper.pdf` / `paper.tex` | The manuscript (Taylor & Francis `interact` format, NLM references). |
| `interact.cls` | The T&F class file, so `paper.tex` compiles as-is. |
| `code/alignment_study.py` | Simulation study: calibration, power comparison, Hermite/dissociation. Regenerates the simulation figures and tables. |
| `code/real_data_study.py` | Real-data illustration (breast-cancer): calibration on real marginals and a two-sample comparison. |
| `figures/` | The four figures used in the paper. |
| `results/` | The CSV tables behind the paper. |

## Requirements

```
pip install -r requirements.txt   # numpy, scipy, matplotlib, scikit-learn
```

## Reproducing the results

Simulation study (calibration, power, Hermite):

```
python code/alignment_study.py
```

Real-data illustration (no download needed --- the dataset ships with scikit-learn):

```
python code/real_data_study.py
```

Both scripts run with a fixed random seed and regenerate the corresponding figures
and tables.

## Data

The real-data illustration uses the **Wisconsin Diagnostic Breast Cancer** dataset,
a public benchmark available in scikit-learn and originally from the UCI Machine
Learning Repository
(<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>).
No new data were generated. This repository is archived at https://doi.org/10.5281/zenodo.23023835.

## Compiling the paper

Keep `interact.cls` and the four `figures/*.png` alongside `paper.tex`, then run
`pdflatex` twice.

## License

Code: MIT (`LICENSE`). The paper text and figures are the author's.

## Author

Jacques Touboul --- independent mathematical-statistics researcher, Stains, France.
