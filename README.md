# Unicode-Robust Text Classification HWp

This project compares word, character and BPE TF-IDF classifiers with DistilBERT on BANKING77 and TREC. It tests Unicode attacks, lightweight input defenses, and an optional zero-shot comparison with Jev.

## Inspect the supplied results

Start with `output/pdf/main.pdf`. The report uses the supplied ACL2015 course template; its source, official `acl2015.sty`, generated tables and referenced figures are included. You can also inspect saved notebook outputs and the CSVs in `artifacts/tables/` without rerunning anything.

**The ZIP excludes raw datasets, fitted checkpoints and prediction caches, including Jev responses.** The included results are ready to read, but running notebooks 06 or 07 from an extracted ZIP needs the earlier stages' local artifacts. Jev additionally needs its response cache or explicit opt-in to paid API calls.

## Reproduce the local experiments

Requirements: Python 3.12 and `uv`. Use `tectonic` to compile the report.

```bash
uv sync
uv run jupyter lab
```

Run notebooks 00–06 in order, from the project root or the `notebooks` directory. Each runs in a fresh kernel and reads saved inputs from earlier stages.

1. `00_environment_and_smoke_tests.ipynb` — check the environment.
2. `01_data_and_eda.ipynb` — prepare and inspect the datasets.
3. `02_unicode_attacks.ipynb` — generate attacked inputs.
4. `03_classical_models.ipynb` — train the TF-IDF classifiers.
5. `04_transformer_models.ipynb` — train and evaluate DistilBERT.
6. `05_defenses.ipynb` — evaluate input repair.
7. `06_results_and_statistics.ipynb` — compare clean performance, attacks and defenses.

Notebook `07_jev_robustness.ipynb` is optional; see below.

Configuration is in `config/experiment.yaml`. On a fresh copy, notebooks 01–05 download data and model weights, train the classifiers and generate predictions. This needs network access for downloads and enough time for local training and evaluation. When those artifacts already exist, notebook 04 reuses saved checkpoints and predictions, and notebook 05 reuses completed defense files. Notebook 06 reports accuracy, macro-F1, attack success rate, score drops and defense recovery, with descriptive variation across seeds. It also plots model performance across every saved attack strength, breaks the curves down by attack family, and compares 20%, 40% and 75% in bar charts. The supplied tables and plots cover all four baselines through 75% strength; the underlying raw predictions are kept locally and excluded from the ZIP. Ground truth labels and their probabilities use `gt_label` and `gt_probability` in saved predictions.

After creating or restoring the prediction files, you can execute the results notebook from a terminal:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_results_and_statistics.ipynb --ExecutePreprocessor.timeout=-1
```

## Jev robustness check

Notebook 07 uses the pinned `jev-1.13.0` model, fixed label descriptions and no demonstrations. It evaluates all 3,080 BANKING77 and 500 TREC test examples: clean text once, plus four attack families at 5%, 10%, 20%, 40% and 75% strength, attack seed 13, without defenses. All 75,180 evaluation rows must be present before results are exported. Each attacked row is matched to its clean counterpart, and each attack family has equal weight in the summary. Composition is excluded because it changes no inputs.

The Jev-only table and plot show the full strength sweep. `jev_strength_condition_metrics.csv` contains 42 condition rows, and `jev_strength_summary.csv` contains 12 clean/strength rows. The four original 20% comparison CSVs are also retained with their existing format and values; baseline training seeds are scored separately before averaging.

`RUN_LIVE = False` is the default. Complete cached responses replay without API calls. Restore them under `artifacts/jev/responses/`, together with the processed inputs and baseline predictions from notebooks 01–04. The ZIP does not include these caches; with the default settings and no cache, notebook 07 stops rather than making requests. To request missing responses, set `RUN_LIVE = True` and provide `TYPESAFE_API_KEY` in the environment or the ignored `.env.local` file; the environment takes precedence. `JEV_OFFLINE=1` always disables live requests. Live requests use a paid hosted API. Run only one copy of the notebook against the cache at a time.

The cache includes the exact model, original Unicode text and question. Successful responses are saved immediately, and a failed run can resume. Missing or mismatched rows stop evaluation before the comparison is exported. The API's returned Choice is the prediction; rounded probabilities are retained even if their argmax differs. The gt-probability is the probability assigned to the ground truth (gt) label.

## Outputs and report

- `artifacts/tables/`: retained experiment metrics and qualitative examples.
- `artifacts/figures/`: report figures, including the Jev comparison.
- `report/`: LaTeX source, official ACL2015 style and generated tables needed to compile it.
- `output/pdf/main.pdf`: compiled report.
- `output/unicode_robust_text_classification_hwp.zip`: submission package.

Raw datasets, model checkpoints, prediction caches, API credentials and private authoring tools are excluded from Git and the submission. The report's generated assets are included, so the delivered source compiles directly from an extracted ZIP:

```bash
cd report
tectonic --keep-logs --synctex main.tex --outdir ../output/pdf
```

Report authoring and packaging notebooks are kept locally under the ignored `private/notebooks/` folder. They are not required to run the delivered experiments or compile the included report. For local authoring, run the report helper after notebooks 00–07, compile the report, then run the packaging helper.

Run the offline checks with:

```bash
uv run python -m unittest discover -s tests -v
```

## Interpretation

The confusable-repair defense is an English-specific heuristic covering the attack whitelist. Canonical-composition attacks change no test inputs in these datasets. The report shows mean scores without standard-deviation annotations for readability. The project does not establish a causal relationship between tokenization changes and classification failures. Jev uses different training and capacity from the local baselines, and a cached response does not guarantee the same answer from a future API call.

## Use of GenAI and LLM tools

I used ChatGPT and Codex to refine the project design, locate and cross-check literature and documentation, scaffold and review notebooks, assist with debugging and statistical checks, and revise the report's prose, tables and figures. The main controlled experiments use local classifiers. The Jev comparison uses separate calls to a paid hosted classification API; these model evaluations are distinct from assistance with writing and coding. I remain responsible for running the experiments, verifying sources and results, and approving the submitted report and code.
