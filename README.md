# Blightwilt Optimizer

An optimizer comparison for **Phytophthora blight and wilt classification in strawberry images**, using a **four-block convolutional neural network (CNN)** trained from scratch. The experiment compares **SGD, Adagrad, RMSprop, and Adam** with learning rates selected by validation macro-F1.

Across five final seeds per optimizer, **RMSprop achieved the highest mean test macro-F1 (0.867 ± 0.013)**. **Adam reached the validation macro-F1 target of 0.80 fastest on average (83.23 seconds)**. The saved statistical analysis did not establish a single overall winner under its decision rule.

## Four-block CNN

The model receives **128 × 128 RGB images** and applies four convolutional blocks:

```text
Input: 3 × 128 × 128
  → Conv 3×3, 16 channels  → ReLU → MaxPool 2×2
  → Conv 3×3, 32 channels  → ReLU → MaxPool 2×2
  → Conv 3×3, 64 channels  → ReLU → MaxPool 2×2
  → Conv 3×3, 128 channels → ReLU → MaxPool 2×2
  → Global average pooling
  → Dense 128 → 64 → ReLU
  → Dense 64 → 2 class logits (blight, wilt)
```

Training uses cross-entropy loss and channel normalization calculated from the training set. For each seed, the comparison code gives every optimizer the same initial weights and batch order. SGD uses no momentum; optimizer settings other than learning rate use library defaults.

## CNN comparison results

### Final comparison: five seeds per optimizer

The [final results](docs/results/final_results.csv) contain **20 runs**, using seeds **0–4**, the `smallcnn` model, **128 × 128** inputs, a **40-epoch maximum**, and **CUDA**. Each run is evaluated at its best validation macro-F1 checkpoint. Macro-F1 is the unweighted mean of the two class F1 scores.

Test macro-F1 is reported as **mean ± sample standard deviation** across five seeds (`ddof=1`). Accuracy and per-class F1 are means across those same runs.

| Optimizer | Selected learning rate | Test macro-F1 | Test accuracy | Blight F1 | Wilt F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSprop | 0.003 | 0.867 ± 0.013 | 87.42% | 0.837 | 0.897 |
| Adam | 0.003 | 0.852 ± 0.011 | 85.76% | 0.824 | 0.880 |
| Adagrad | 0.01 | 0.848 ± 0.011 | 85.50% | 0.817 | 0.880 |
| SGD | 0.1 | 0.761 ± 0.089 | 77.06% | 0.718 | 0.804 |

![Test performance across five seeds per optimizer](docs/figures/Figure_2_test_performance.png)

**What the results show:**

- **RMSprop leads in observed test performance.** It has the highest mean macro-F1 and mean F1 for both classes. Its macro-F1 advantage is 0.015 over Adam, 0.019 over Adagrad, and 0.106 over SGD.
- **Adam and Adagrad are close in test performance.** Their mean macro-F1 values are 0.852 and 0.848, with sample standard deviations of about 0.011.
- **SGD varies most across seeds.** Its macro-F1 standard deviation is 0.089, compared with 0.011–0.013 for the adaptive optimizers.
- **Wilt has higher mean F1 than blight for every optimizer.** The test split shown in the confusion matrices contains 689 blight images and 1,114 wilt images (1,803 total). Class imbalance is relevant context, but these results alone do not establish why the class scores differ.

### Convergence and training time

The final runs use a target of **validation macro-F1 ≥ 0.80**. The averages below include **only runs that reached the target**; the success counts are shown explicitly. Time includes the training and validation work measured by the trainer through the first epoch that reaches the target.

| Optimizer | Runs reaching target | Mean epochs to target | Mean time to target (seconds) |
| --- | ---: | ---: | ---: |
| Adam | 5/5 | 9.8 | 83.23 |
| RMSprop | 5/5 | 11.2 | 94.64 |
| Adagrad | 5/5 | 14.2 | 120.25 |
| SGD | 2/5 | 31.5 | 265.30 |

![Epochs and time to the 0.80 validation macro-F1 target](docs/figures/Figure_3_convergence_and_time.png)

Adam reaches the target fastest on average, followed by RMSprop and Adagrad. All five runs reach the target for each of these three optimizers. **Three of five SGD runs never reach it**, so SGD's averages describe only its two successful runs. Median epoch times are similar across optimizers, approximately **8.45–8.49 seconds** on the recorded setup; most of the observed time-to-target difference comes from the number of epochs needed. These timings are specific to this experiment's hardware and environment.

### Training behavior and generalization

The [training curves](docs/figures/Figure_1_training_curves.png) show declining training loss and increasing training accuracy, while validation loss fluctuates. The adaptive optimizers' mean training accuracies at their selected checkpoints are approximately **95.5–97.4%**, compared with **85.5–87.4%** on the test set. This gap is consistent with overfitting and warrants attention even when validation and test scores track each other.

Validation and test macro-F1 have a pooled correlation of **r = 0.968** across the 20 final runs, as shown in the [generalization figure](docs/figures/Figure_S1_generalisation.png). This is a descriptive association across optimizers and seeds, not proof of generalization to a different farm or acquisition setting.

The [confusion matrices](docs/figures/Figure_4_confusion_matrices.png) show one representative run per optimizer, selected by median test macro-F1. For RMSprop (seed 4), 602/689 blight images and 982/1,114 wilt images are correctly classified. These are single-run counts, not averages or pooled counts across seeds.

### Statistical interpretation

The [saved analysis](docs/results/decision.txt) reports a Friedman test statistic of **12.12, p = 0.0070**, indicating an overall difference among optimizers. However, the [Holm-adjusted paired tests](docs/results/significance.csv) comparing RMSprop's test macro-F1 with each other optimizer all have **p = 0.2185**, above 0.05.

The decision rule then compares convergence and time. It finds Adam reaches the target in significantly fewer epochs than SGD (**adjusted p = 0.0031**, with unsuccessful runs assigned the 40-epoch budget plus one). It does not distinguish Adam, RMSprop, and Adagrad at the adjusted 0.05 threshold in the remaining comparisons.

**Conclusion:** RMSprop has the strongest observed average classification performance, and Adam has the fastest observed average convergence to 0.80. The five-seed analysis does **not** establish a unique overall winner. The original report uses “statistically equivalent” for unresolved comparisons; a nonsignificant difference does not establish equivalence.

### Learning-rate sweep: separate from the final results

The [sweep results](docs/results/sweep_results.csv) contain **28 runs**: four optimizers × seven learning rates (`0.0003`, `0.001`, `0.003`, `0.01`, `0.03`, `0.1`, `0.3`), all at **seed 100**. The selected rates in [best_lr.json](docs/results/best_lr.json) maximize **validation** macro-F1: SGD 0.1, Adagrad 0.01, RMSprop 0.003, and Adam 0.003.

![Learning-rate sensitivity in the single-seed sweep](docs/figures/o2_f1_grid.png)

The sweep illustrates sensitivity to learning rate: the supplied plots mark seven runs as collapsing to single-class predictions, including Adam at 0.01–0.3 and RMSprop at 0.1–0.3. Adagrad has no marked collapse across the seven tested rates. This is a single-seed observation and does not establish stability across other seeds.

The sweep uses a **0.85 validation macro-F1 convergence target**, whereas final runs use **0.80**. Six of the 28 sweep runs reach 0.85; RMSprop at 0.003 reaches it fastest (17 epochs, 143.51 seconds). **Do not compare these sweep times directly with the final-run times above.** The two targets are retained as recorded in the source data. The plotted test-score grid is descriptive; it is not the criterion used to choose learning rates.

The [figure bundle](docs/figures/All_figures.pdf) contains all five final-comparison figures. The [figure audit](docs/results/data_checks.json) records five runs per optimizer, sample standard deviations, and unsmoothed plots.

## Repository layout

```text
Blightwilt_Optimizer/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── scripts/
│   ├── prepare_aihub.py      # Filter labels, split groups, and crop images
│   ├── train_optimizers.py   # Smoke check, learning-rate sweep, and final runs
│   ├── analyze_optimizers.py # Statistics and decision report
│   └── make_figures.py      # Figures as PNG/PDF
└── docs/
    ├── Notes.docx           # Original research notes
    ├── figures/             # Selected supplied figures and combined PDF
    └── results/             # Small result snapshots and their provenance
```

Raw datasets, prepared crops, new experiment runs, and checkpoints stay local. The selected results in `docs/` support this README and are tracked in Git. Default data/output paths are anchored at the repository root; run the commands below from that root so explicit relative paths resolve consistently.

## Setup

Use Python 3.10 or newer in a virtual environment. Dependencies are listed in `requirements.txt`; a pinned environment for the supplied experiment was not provided.

```bash
git clone https://github.com/isabelhendrick-sudo/Blightwilt_Optimizer.git
cd Blightwilt_Optimizer
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The trainer automatically selects CUDA, then Apple MPS, then CPU. Use `--device cpu`, `--device cuda`, or `--device mps` to choose explicitly. For a particular GPU build, install a compatible PyTorch/torchvision pair for your platform.

## Prepare the AI Hub blight/wilt dataset

Obtain the source dataset separately through AI Hub under its access terms. Keep its ZIP files intact beneath this directory; the preparation script searches subfolders recursively:

```text
data/
└── 305.시설작물개체이미지 및 시설작물질병이미지/
    └── 01-1.정식개방데이터/
        └── ... ZIP archives ...
```

Label ZIP names start with `TL_` or `VL_`, and image ZIP names with `TS_` or `VS_`. Archive names must contain `역병` (blight) or `시들음병` (wilt).

```bash
python scripts/prepare_aihub.py --count-only
python scripts/prepare_aihub.py
```

Preparation defaults keep symptomatic boxes at least 32 pixels wide and high, restrict both diseases to their shared date window, and keep images at least 24 hours apart per plant. Whole plant groups within each disease are assigned to an approximate 70/15/15 train/validation/test split, with seed 42 and a 10% crop margin. These are the current script defaults; the result CSVs do not independently record the preparation settings of the supplied experiment.

Outputs are `aihub_cls/{train,validation,test}/{blight,wilt}/`, `labels_all.csv`, `manifest.csv`, and `prep_summary.txt`. Review counts, missing images, and orientation mismatches before training. `--count-only` writes a label cache and a planned-split report without creating crops. A full preparation run **replaces existing crop split folders**. Use `--reparse` after changing label archives.

## Run the four-block CNN comparison

The following commands explicitly match the recorded image size, epoch budget, seed sets, learning-rate grid, and stage-specific convergence targets. The CSVs do not record every setting or dependency version, so these commands are a workflow for the documented configuration, not a guarantee of identical numerical results.

```bash
# 1. Two epochs per optimizer to check the pipeline
python scripts/train_optimizers.py smoke --data aihub_cls --out runs/aihub_smallcnn --target-f1 0.80

# 2. Seven learning rates per optimizer, seed 100, sweep target 0.85
python scripts/train_optimizers.py sweep --data aihub_cls --out runs/aihub_smallcnn --img 128 --epochs 40 --sweep-seed 100 --lr-grid 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 --target-f1 0.85

# 3. Selected learning rates over five final seeds, final target 0.80
python scripts/train_optimizers.py final --data aihub_cls --out runs/aihub_smallcnn --img 128 --epochs 40 --seeds 0 1 2 3 4 --target-f1 0.80 --save-weights

# 4. Summarize and plot new final runs
python scripts/analyze_optimizers.py runs/aihub_smallcnn/final
python scripts/make_figures.py --runs runs/aihub_smallcnn/final --out figures/aihub_smallcnn
```

The default model is the four-block `smallcnn`. The current trainer defaults to batch size 32 and early-stopping patience 10 on validation macro-F1. Without explicit options, its learning-rate grid has six rates and its target is 0.85; use the commands above for the settings recorded in these results. Always supply `--data aihub_cls` for this dataset, since the legacy default points to `strawberry_cls/`.

The `final` phase reads rates from `<out>/sweep/best_lr.json`; other settings must be supplied separately. Completed runs are skipped when their `result.json` exists. Run names contain optimizer, learning rate, and seed, so **use a new output directory when changing data or experiment settings**. For a new study, define the target and other comparison criteria before running.

Each phase generates aggregate `results.csv`, `history.csv`, and `predictions.csv`. Final runs optionally save `best.pt`. Analysis creates `summary.csv`, `significance.csv`, `per_class_metrics.csv`, `decision.txt`, and pooled confusion matrices under `<out>/final/analysis/`. The figure script uses representative-run confusion matrices instead, so their counts differ from pooled analysis figures.

## Interpretation and reproducibility limits

- Learning-rate selection uses validation macro-F1, but the trainer also evaluates and logs test metrics during smoke and sweep runs. Those metrics should not guide tuning; a strictly untouched-test protocol requires changing that behavior.
- Five seeds quantify variation within this dataset split. They do not capture uncertainty across new farms, cameras, seasons, or independently collected datasets.
- All three resized image splits are loaded into RAM. Large datasets can require substantial memory.
- Exact hardware model, library versions, batch size, and preparation settings are not fully documented in the supplied CSVs. Record these alongside the Git revision for future experiments.
- The supplied figures and result snapshots are included; raw images, trained weights, full per-epoch histories, and per-image predictions are not. Recreating every figure requires the full local run outputs or a new training run.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the Git workflow and checks. Keep raw datasets, checkpoints, and full generated outputs out of Git. Small curated results in `docs/results/` and figures in `docs/figures/` should include provenance when updated.

The repository currently has no license file. Code licensing and AI Hub dataset access terms are separate considerations.
