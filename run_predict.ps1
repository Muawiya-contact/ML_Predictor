# run_predict.ps1
# Activate the repository `.venv` (if present) and run `predict_batch.py`.
# NOTE: the input parameter is NOT called $Input. `$Input` is a PowerShell
# automatic variable (the pipeline enumerator); declaring it as a parameter
# binds it and then PowerShell immediately overwrites it with the empty
# pipeline enumerator, so `-Input my_patients.xlsx` was silently discarded and
# predict_batch.py fell back to its own default file. -Input is kept as an
# alias so the documented call still works.
param(
    [Alias('Input')]
    [string]$InputPath = "sample_100_patients.xlsx",
    [Alias('Output')]
    [string]$OutputPath = ""
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

$predictScript = Join-Path $scriptDir "predict_batch.py"

if ($OutputPath -ne "") {
    python $predictScript $InputPath $OutputPath
}
else {
    python $predictScript $InputPath
}
