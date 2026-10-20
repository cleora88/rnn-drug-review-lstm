from pathlib import Path

import nbformat as nbf

p = Path(__file__).parent
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Manual LSTM comparison on drug reviews\n"
        "Zouhair Eouichak, Aymane El Youssi, Malak Bartal and Sara Baalla. TP B.\n\n"
        "See README.md for the experimental protocol. Running the training cell "
        "repeats all nine experiments and replaces the files in results/."
    ),
    nbf.v4.new_markdown_cell(
        "## Protocol\n"
        "The three architectures share fixed stratified splits, preprocessing and "
        "evaluation metrics. Checkpoints and configurations are selected by "
        "validation macro-F1 before test evaluation."
    ),
    nbf.v4.new_code_cell("%run test_cells.py"),
    nbf.v4.new_code_cell("%run run.py"),
    nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('results/comparison.csv')"),
    nbf.v4.new_code_cell(
        "from IPython.display import Image, display\ndisplay(Image('results/eda.png'))\ndisplay(Image('results/learning_curves.png'))"
    ),
]
nbf.write(nb, p / "RNN_project.ipynb")
