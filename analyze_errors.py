import json
from pathlib import Path
import pandas as pd
R=Path(__file__).parent
runs=json.loads((R/'results/experiments.json').read_text())
selected=max(runs,key=lambda r:r['best_validation_macro_f1'])['model']
df=pd.read_csv(R/f'results/{selected}_predictions.csv')
errors=df[df.actual!=df.predicted]
examples=errors.groupby(['actual','predicted'],group_keys=False).head(2)
examples.to_csv(R/'results/error_examples.csv',index=False)
lines=[f'# Error examples for {selected.upper()}','', 'Model selected using validation macro-F1. Examples below are fixed first errors for each label pair, not a representative sample. Rating-derived labels may differ from the sentiment expressed in text.','']
labels=['negative','neutral','positive']
for _,row in examples.iterrows():
    lines.extend([f"Actual: {labels[row.actual]}; predicted: {labels[row.predicted]}",'',str(row.text),''])
(R/'results/error_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
print(f'Saved {len(examples)} error examples from {selected}')
