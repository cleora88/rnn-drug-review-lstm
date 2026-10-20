$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install pandas scikit-learn matplotlib pillow jupyter playwright reportlab
New-Item -ItemType Directory -Force data | Out-Null
$datasetMirror = 'https://raw.githubusercontent.com/semnan-university-ai/Drug-Review-Dataset/master/'
foreach ($datasetFile in @('drugsComTrain_raw.tsv','drugsComTest_raw.tsv')) {
    Invoke-WebRequest ($datasetMirror + $datasetFile) -OutFile (Join-Path data $datasetFile)
}
& .\.venv\Scripts\python.exe test_cells.py
Write-Host 'Setup complete. Run .venv\Scripts\python run.py to train.'
