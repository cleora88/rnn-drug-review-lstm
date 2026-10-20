$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }

$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

New-Item -ItemType Directory -Force data | Out-Null
$datasetMirror = 'https://raw.githubusercontent.com/semnan-university-ai/Drug-Review-Dataset/master/'
foreach ($datasetFile in @('drugsComTrain_raw.tsv', 'drugsComTest_raw.tsv')) {
    Invoke-WebRequest ($datasetMirror + $datasetFile) -OutFile (Join-Path data $datasetFile)
}

& $projectPython test_cells.py
if ($LASTEXITCODE -ne 0) { throw 'Recurrent cell checks failed.' }
& $projectPython test_preprocessing.py
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing checks failed.' }
Write-Host 'Setup complete. Run .venv\Scripts\python run.py to train.'
