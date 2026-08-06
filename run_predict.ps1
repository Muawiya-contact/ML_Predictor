# run_predict.ps1
# Activate the repository `.venv` (if present) and run `predict_batch.py`.
param(
    [string]$Input = "sample_100_patients.xlsx",
    [string]$Output = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..."
    . $venvActivate
}
else {
    Write-Warning "Virtual environment not found at $venvActivate. Running with system Python."
}

if ($Output -ne "") {
    python $scriptDir\predict_batch.py $Input $Output
}
else {
    python $scriptDir\predict_batch.py $Input
}
