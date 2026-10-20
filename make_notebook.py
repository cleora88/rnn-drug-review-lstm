import nbformat as nbf
from pathlib import Path
p=Path(__file__).parent
nb=nbf.v4.new_notebook()
nb.cells=[nbf.v4.new_markdown_cell('# Manual LSTM comparison on drug reviews\nZouhair Eouichak, TP B. Run the cells in order. See README.md for the experimental protocol.'),
nbf.v4.new_code_cell('%run test_cells.py'),
nbf.v4.new_code_cell('%run run.py'),
nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('results/comparison.csv')"),
nbf.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image('results/eda.png'))\ndisplay(Image('results/learning_curves.png'))")]
nbf.write(nb,p/'RNN_project.ipynb')
