# Unicode-Robust Text Classification HWp

This repository contains a notebook-first comparative study of word, character, BPE-subword, and DistilBERT text classifiers under adversarial Unicode perturbations. It evaluates BANKING77 and coarse-label TREC under homoglyph, invisible-character, compatibility, canonical-composition, and mixed attacks, followed by lightweight test-time defenses.

## Reproduce

Requirements: macOS/Linux, `uv`, and `tectonic`. The current project is configured for Python 3.12.

```bash
uv sync
uv run python -m ipykernel install --user --name unicode-hwp --display-name "Python (Unicode HWp)"
uv run jupyter lab
```

Run the notebooks in numeric order, from either the repository root or the `notebooks` directory. Each notebook contains its own imports and helpers and runs in a fresh kernel using saved inputs from earlier stages. Comments identify intentionally repeated helpers; no shared Python module is required.

1. `00_environment_and_smoke_tests.ipynb`
2. `01_data_and_eda.ipynb`
3. `02_unicode_attacks.ipynb`
4. `03_classical_models.ipynb`
5. `04_transformer_models.ipynb`
6. `05_defenses.ipynb`
7. `06_results_and_statistics.ipynb`
8. `07_tokenization_analysis.ipynb`
9. `08_error_analysis_and_report_assets.ipynb`
10. `09_reproducibility_and_packaging.ipynb`

The YAML configuration holds the experiment settings. Notebook 04 reuses a Transformer run when both its final checkpoint and raw prediction partition exist; if only the checkpoint exists, predictions are rebuilt without retraining. Notebook 05 similarly reuses existing defended partitions and computes only missing ones. Delete a stale partition explicitly when it needs to be regenerated.

Notebook 00 records the environment; attack examples are shown in 02 and repair examples in 05. Outputs focus on result tables and report figures. Batching and reuse of unchanged predictions keep evaluation practical; notebook 09 packages the submission without separate inventories or manifests.

To execute a notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_environment_and_smoke_tests.ipynb --ExecutePreprocessor.timeout=-1
```

Compile the report after notebook 08:

```bash
cd report
tectonic --keep-logs --synctex main.tex --outdir ../output/pdf
```

## Outputs

- `artifacts/tables/`: compact metrics, uncertainty estimates, regressions, and qualitative examples.
- `artifacts/figures/`: report-ready PDF figures.
- `output/pdf/main.pdf`: compiled report.
- `output/unicode_robust_text_classification_hwp.zip`: submission archive produced by notebook 09.

Raw datasets, caches, and model checkpoints are excluded from Git and the submission archive. Metadata records the downloaded dataset sources and file hashes, plus the installed package versions. Compile the report after notebook 08, then run notebook 09 to build the ZIP.

## Scientific cautions

The confusable-repair defense is an English-specific heuristic. Unicode confusable skeletons are security-analysis representations, not universally safe display-text normalization. Canonical-composition attacks have zero eligible test examples in these ASCII-heavy datasets, so the full-set no-change result is reported and the eligible-subset estimate is explicitly undefined.
