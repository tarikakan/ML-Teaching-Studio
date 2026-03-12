# ML-Teaching Studio

ML-Teaching Studio is a production-style Python package and PySide6 desktop application for teaching machine learning as an interactive lab. It combines theory pages, dataset exploration, model explanations, guided hyperparameter teaching, training workflows, visual diagnostics, saved-run comparison, and quiz practice inside one desktop interface.

The application is designed to answer four questions at the same time:

- Which model performed better?
- Why did it perform better?
- How did hyperparameters change the behavior?
- What should the learner conclude from the experiment?

## Author

Tarik Akan

## Core Features

- Built-in theory lessons covering ML foundations, preprocessing, validation, metrics, bias-variance, overfitting, and hyperparameter tuning.
- Built-in datasets including Iris, Wine, Breast Cancer, California Housing, synthetic classification/regression data, and noisy overfitting demos.
- CSV import with target selection, feature preview, and dataset summaries.
- Model zoo for linear models, KNN, trees, ensembles, SVMs, naive Bayes, and MLPs.
- Interactive live examples inside the model and hyperparameter pages, so learners can adjust key settings and watch plots update immediately.
- Dedicated hyperparameter teaching page with plain-language and algorithmic explanations.
- End-to-end training workflow with preprocessing options, background execution, metrics, and plot outputs.
- Visualization gallery for data understanding, model performance, model behavior, comparison, and hyperparameter sensitivity.
- Hyperparameter Lab for one- and two-parameter sweeps with train/test gap analysis.
- Compare Runs view for reviewing saved experiments side by side.
- Quiz / Practice section for concept reinforcement.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the App

```bash
ml-teaching-studio
```

Or:

```bash
python -m ml_teaching_studio.main
```

## One-Click Launchers

The project root now includes clickable launchers for each desktop platform:

- `ML-Teaching Studio.app` and `ML-Teaching Studio.command` for macOS
- `ML-Teaching Studio.bat` for Windows
- `ML-Teaching Studio.desktop` and `ML-Teaching Studio.sh` for Linux

All launchers call `scripts/launch_studio.py`, which:

- adds the local `src/` directory automatically
- prefers a local `.venv` interpreter when available
- falls back to system `python3` or `python`
- shows a user-facing error message if dependencies are missing

You can validate the launcher environment without opening the GUI:

```bash
python scripts/launch_studio.py --check
```

## Run Tests

```bash
pytest
```

## Typical Workflow

1. Open `Datasets` and load a built-in dataset or import a CSV.
2. Choose the target column and feature subset.
3. Open `Theory` or `Models` to review the concept you want to study.
4. Move to `Training`, choose preprocessing, pick a model, and adjust hyperparameters.
5. Train the model and inspect metrics, plots, and interpretation text.
6. Save the run.
7. Open `Hyperparameter Lab` to sweep one or two parameters and inspect validation curves.
8. Use `Compare Runs` to compare saved experiments fairly.

## Screenshots

Screenshot placeholders are documented in:

- [docs/examples/workflows.md](docs/examples/workflows.md)
- [docs/theory/lesson_map.md](docs/theory/lesson_map.md)

## Package Structure

```text
ml_teaching_studio/
  src/ml_teaching_studio/
    core/            Dataset loading, preprocessing, training, sweeps, explanations
    educational/     Lessons, glossary, model notes, quizzes
    gui/             Main window, pages, widgets, dialogs, workers
    models/          Model constructors and registry metadata
    plotting/        Matplotlib plots for datasets, performance, sweeps, comparisons
    utils/           Configuration, helpers, I/O, logging
  tests/             Core logic tests
  docs/              Theory and workflow notes
  assets/            Placeholder folders for logos, icons, and banners
```

## Educational Design Notes

- The theory module is intentionally connected to the interactive pages.
- Result pages include interpretation panels so learners do not stop at raw metrics.
- Hyperparameters are treated as first-class teaching objects, not hidden advanced options.
- Synthetic noisy datasets are included specifically for bias-variance and overfitting lessons.

## Developer Notes

- The project uses a `src` layout and modular separation between GUI, logic, plotting, and content.
- GUI work that may block the interface runs through worker objects on background threads.
- New models can be added by updating `models/model_registry.py`, the constructor modules, and educational help content.
- Core tests focus on non-GUI logic so the ML engine can be validated independently from PySide6 rendering.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
