# Working on Blightwilt Optimizer

## Git workflow

Keep `main` as the shared working version. Use a short-lived branch for each focused change; a separate permanent development branch is unnecessary for this small project.

```bash
git switch main
git pull --ff-only
git switch -c docs/improve-setup
# Edit files and run the relevant checks.
git status --short
git diff --check
git diff
git add README.md
git commit -m "docs: clarify dataset setup"
git push -u origin docs/improve-setup
```

Open a pull request into `main` describing the change and how it was checked. Merge after review. Suggested branch prefixes are `docs/`, `feat/`, and `fix/`; commit examples are `docs: explain experiment outputs`, `feat: add a model option`, and `fix: preserve dataset paths`.

## Where changes belong

- `scripts/`: executable data preparation, training, analysis, and plotting code.
- `docs/`: research notes and supporting written documentation.
- Repository root: setup, dependencies, and contributor guidance.
- `data/`, `aihub_cls/`, `strawberry_cls/`, `runs/`, and `figures/`: local inputs and generated outputs, ignored by Git.

Default data/output paths should resolve from the repository root, even when a script is launched from another working directory. Explicit relative command-line paths are resolved from the current working directory. Update README commands when moving or renaming scripts.

## Checks

For documentation and file-layout changes:

```bash
python -m compileall -q scripts
git diff --check
```

With the dependencies installed, check the command-line interfaces:

```bash
python scripts/prepare_aihub.py --help
python scripts/train_optimizers.py --help
python scripts/make_figures.py --help
```

`analyze_optimizers.py` accepts a positional final-run directory and requires completed result files; it does not implement `--help`.

For training changes, run the two-epoch `smoke` phase against prepared data in a new output directory. For analysis/plotting changes, use a completed experiment with matching seeds for every optimizer. Syntax checks alone do not verify model behavior or statistical correctness.

## Reproducible experiments

Before a full experiment, record the Git commit, preparation settings, exact commands, model, device, dependency versions, and seed choices. Fix the learning-rate grid and convergence target before comparing optimizers. Keep the sweep seed separate from final seeds, and use consistent settings between phases.

```bash
git rev-parse HEAD
python -m pip freeze > runs/aihub_smallcnn/environment.txt
```

The output directory must already exist for the second command. Use a fresh directory for a new configuration because completed runs are skipped based on their existing result files. Inspect the preparation report and manifest before training, and keep them with the results when sharing an experiment.

Check `git status` before committing. Commit code and documentation; store datasets, checkpoints, and full experiment outputs separately. If selected small results are later added to the repository, place them in a deliberately documented location and include provenance.
