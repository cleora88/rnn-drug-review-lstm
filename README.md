# Comparing manual LSTM cells on drug reviews

Advanced Natural Language Processing, UIR, 2026/2027.
Team (TP B): Zouhair Eouichak, Aymane El Youssi, Malak Bartal, Sara Baalla. The guidelines require 5-6 members; confirm the remaining member(s) before submission.

## Task
Predict rating-derived sentiment from review text only: negative (1-4), neutral (5-6), positive (7-10).
These thresholds are our modeling choice, not a requirement of the assignment.
Dataset: UCI Drugs.com, https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
Download: https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip
Credit: Kallumadi and Grasser (2018), Drug Review Dataset (Drugs.com), UCI Machine Learning Repository.

## Reproduce (PowerShell, Python 3.12)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
New-Item -ItemType Directory -Force data
Invoke-WebRequest 'https://raw.githubusercontent.com/semnan-university-ai/Drug-Review-Dataset/master/drugsComTrain_raw.tsv' -OutFile data/drugsComTrain_raw.tsv
Invoke-WebRequest 'https://raw.githubusercontent.com/semnan-university-ai/Drug-Review-Dataset/master/drugsComTest_raw.tsv' -OutFile data/drugsComTest_raw.tsv
.venv\Scripts\python test_cells.py
.venv\Scripts\python run.py
```
For a CPU-only environment, install `requirements.txt` from the default PyPI index instead of the CUDA lock file.

## Protocol
The provided training and test partitions remain separate. Normalize text, drop exact normalized duplicates within each source, then remove test reviews also present in training. Split training 85/15 with stratification and seed 42. Use fixed stratified subsets of 12,000 training, 2,000 validation and 4,000 test examples to make manual-cell training practical. Vocabulary is learned only from the training subset (maximum 12,000 words), sequences are capped at 96 tokens. Padding does not change the recurrent state.

Vanilla, CIFG and WMC cells use nn.Linear, with no nn.LSTM. WMC output-gate memory connection uses the newly computed c_t. All models have 64-dimensional embeddings and hidden states, dropout 0.2 and a three-class head. Compare three Adam configurations: (learning rate 0.001, batch size 128), (0.003, 128), and (0.001, 64), for nine runs total. Each run uses four epochs, class-weighted cross entropy and gradient clipping at 1.0. Select each architecture's checkpoint and configuration by validation macro-F1. Evaluate only selected configurations on the test set. Report accuracy, macro-F1, per-class metrics, confusion matrices and trainable parameters. Parameter totals include the embedding and classifier.

The recorded experiments used Python 3.14 and CUDA-enabled PyTorch 2.9.1 on an RTX 4070 Laptop GPU. The exact package versions and device details are recorded in `requirements-lock.txt` and `results/environment.json`. Use the installation instructions above for a separate environment.

At setup time the current UCI download endpoint returned an error. The TSV files were obtained from the mirror https://github.com/semnan-university-ai/Drug-Review-Dataset. Row counts match the standard 161,297 / 53,766 partitions. Source URLs and SHA-256 hashes are recorded in `results/dataset_provenance.json`. These checks document provenance but do not independently prove byte equality with a fresh UCI download.

## Outputs
`results/` contains split files, vocabulary, environment details, epoch histories, checkpoints, predictions, metrics and figures. `screenshots/` contains browser views of the saved experiment records and previews of the report and slides.

The public repository excludes raw TSV data, split/prediction CSVs, model checkpoints and virtual environments. The local project archive includes split files and checkpoints. `run.py` recreates them from the documented data source. For the recorded dependency versions, install `requirements-lock.txt` with `--extra-index-url https://download.pytorch.org/whl/cu128`.

After training, run `python verify_results.py`, `python analyze_errors.py`, and `python capture_steps.py` to verify the measured outputs and capture the main stages. Capture individual runs with `python capture_steps.py run_1 run_2 run_3 run_4 run_5 run_6 run_7 run_8 run_9`. Use `python predict.py "review text" --model cifg` for an inference demonstration. See `WALKTHROUGH.md` for defense preparation.

## Code organization

| File | Responsibility |
|---|---|
| `preprocessing.py` | Text normalization, sampling, vocabulary and token encoding |
| `models.py` | Manual recurrent cells and review classifier |
| `run.py` | Experiment configuration, training, model selection and evaluation |
| `predict.py` | Command-line inference from a checkpoint |
| `style_figures.py` | Figures generated from recorded results |
| `test_cells.py` | Cell equations, gradients and padding checks |
| `verify_results.py` | Split, metric and parameter-count verification |

Run a small pipeline check without overwriting the recorded experiments:

```powershell
python run.py --train-size 90 --validation-size 30 --test-size 30 --epochs 1 --output-dir .build/smoke
```

Code checks:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python test_cells.py
python test_preprocessing.py
python verify_results.py
```

## Measured test results

Each architecture's configuration was selected by validation macro-F1.

| Model | Accuracy | Macro-F1 | Parameters |
|---|---:|---:|---:|
| Vanilla | 0.58875 | 0.47448 | 801,219 |
| CIFG | 0.63950 | 0.50008 | 792,963 |
| WMC | 0.62925 | 0.49236 | 813,507 |
| Majority class | 0.66400 | 0.26603 | - |

All three models improve macro-F1 over the majority predictor, while their accuracy remains below its 0.6640. This limited training run provides an architecture comparison but modest predictive performance. CIFG has the highest validation macro-F1 and the highest observed test macro-F1. This is a result for this protocol and seed, not a general claim about LSTM variants.

## Limits and next steps
One random seed and a small learning-rate grid provide a limited comparison. Repeat with several seeds, more epochs, longer sequences and larger subsets before making strong architecture claims. Neutral reviews may be hard to distinguish; inspect errors. Rating-derived sentiment is an imperfect proxy. Exact text deduplication does not eliminate near-duplicate or shared-drug effects. No clinical decision-making use is intended.

## Submission checklist
- [ ] Add all 5-6 team members from the same lab group.
- [ ] Review experiments, report and slides; rehearse a 10-minute English talk.
- [x] Public repository: https://github.com/cleora88/rnn-drug-review-lstm
- [ ] Submit before the end of 25 October 2026: https://forms.cloud.microsoft/r/78AEQ7pj2d
- [ ] Attend lab 6 defense.
