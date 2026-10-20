# How to explain and rerun the project

## 1. Verify the environment
The project uses PyTorch on the RTX 4070 Laptop GPU. `screenshots/01_setup.png` records the checked runtime and the passing cell tests. The local environment is `.venv-gpu`; its CUDA libraries reuse an existing installation. To install independently, run `setup.ps1` with Python 3.12 available.

## 2. Understand the dataset
The input is review text. The output is one of three classes derived from the rating. Negative means 1-4, neutral means 5-6, and positive means 7-10. Explain these as your design choices. The model never receives the rating as an input.

Review `results/data_summary.json`, `results/dataset_provenance.json` and `screenshots/02_data.png`. The source files contain many repeated reviews, including overlap across source train and test partitions. We remove exact normalized duplicates and exclude overlapping test reviews before drawing subsets. This matters because memorizing the same text would inflate the apparent test performance.

## 3. Understand the cells
Read `models.py` alongside the appendix. A gate is a learned vector of values between zero and one. The forget gate controls retained memory, the input gate controls new content, and the output gate controls the exposed hidden state.

- Vanilla learns all three gates independently.
- CIFG sets the input gate to `1 - forget_gate`, reducing parameters.
- WMC adds transformed memory values to the gates. The output gate uses the current updated memory.

Run `python test_cells.py` in the selected environment. `screenshots/03_models.png` shows the implementation and the validation record. Trailing padding must not change a prediction.

## 4. Explain the experiments
Run `python run.py`. It trains three configurations for each of the three architectures. Compare learning rates while holding batch size 128 fixed. Compare batch sizes while holding learning rate 0.001 fixed. All runs share splits and training rules.

The epoch with the highest validation macro-F1 supplies each run's checkpoint. The highest-scoring configuration for each architecture supplies its final checkpoint. Only then does the script evaluate the test data. `screenshots/run_1.png` through `run_9.png` record individual configurations and their measured epoch histories.

## 5. Interpret results
Read `results/comparison.csv`, per-class classification JSON files, confusion matrices and learning curves. Accuracy is the overall fraction correct. Macro-F1 gives equal importance to each class, helping expose poor performance on the smaller neutral class. Compare with `results/baseline.json` before saying the model is useful.

Run `python analyze_errors.py` and inspect the resulting error examples. Discuss at least one mixed or ambiguous review in the defense. Avoid claiming that one cell is always better from this small single-seed comparison.

## 6. Prepare the defense
The slides include speaker notes with approximate timings for a ten-minute English presentation. Explain the task, exploratory analysis, model equations, hyperparameter choices, final results and improvements. The PDF report is five pages and uses the saved UIR logo and student details.

Before submitting, add the remaining team members, review every result, verify the public repository link and rehearse the talk. The deadline includes 25 October 2026. Submission is the student's final action.
