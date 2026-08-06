# setup_env.ps1
# Create a Python virtual environment in `.venv` and install dependencies.
param(
    [switch]$Recreate
)

Write-Host "Setting up Python virtual environment (Windows PowerShell)..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found in PATH. Please install Python and enable 'Add Python to PATH'. See HOW_TO_RUN.md"
    exit 1
}

$venvPath = Join-Path -Path $PSScriptRoot -ChildPath ".venv"

if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host "Recreating virtual environment at $venvPath..."
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

Write-Host "Activating virtual environment for this session..."
. (Join-Path $venvPath 'Scripts\Activate.ps1')

Write-Host "Upgrading pip and installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

# PowerShell escapes with a backtick, not a backslash - "\n" printed literally.
Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "To activate the venv in a new PowerShell session:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To run the batch example now:" -ForegroundColor Yellow
Write-Host "  python predict_batch.py sample_100_patients.xlsx" -ForegroundColor Cyan
