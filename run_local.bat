@echo off
REM =====================================================================
REM run_local.bat - start the triage app on Windows.
REM
REM Save this, double-click it whenever you want the project. Safe to run
REM repeatedly: every step checks whether it is already done.
REM
REM Windows needs no equivalent of run_gui.sh: tkinter ships with the
REM python.org installer, so there is no private Tk runtime to point at.
REM =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Roman Urdu Emergency Triage - starting locally
echo ============================================================

REM --- 1. Python environment ------------------------------------------
REM Checked first: everything below is pointless without it, and its own
REM error message is the least self-explanatory of the possible failures.
if not exist ".venv\Scripts\python.exe" (
    echo   FAIL  .venv is missing. Create it once with:
    echo             python -m venv .venv
    echo             .venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    echo             .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    echo         Install torch from the CPU index FIRST. Left to itself pip
    echo         fetches the CUDA build - several GB of NVIDIA libraries
    echo         this project never uses. It is CPU-only by design.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('.venv\Scripts\python --version 2^>^&1') do echo   OK    python  %%v

REM --- 2. tkinter ------------------------------------------------------
REM Included with the python.org installer. If it is missing, Python came
REM from somewhere that omitted it - the Microsoft Store build, usually.
.venv\Scripts\python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo   FAIL  tkinter is missing.
    echo         Reinstall Python from python.org - the Microsoft Store
    echo         build omits it. Tick "tcl/tk and IDLE" during setup.
    pause
    exit /b 1
)
echo   OK    tkinter available

REM --- 3. Ollama service -----------------------------------------------
REM Started if absent rather than merely reported. Without it nothing can
REM be translated, and translation is the first step of every prediction.
curl -sf --max-time 3 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo   FAIL  Ollama is not installed.
        echo         Download it from https://ollama.com/download
        pause
        exit /b 1
    )
    echo   ..    ollama  not running - starting it
    start "" /min ollama serve
    REM Give it time to bind the port. A fixed wait rather than a poll
    REM loop, because batch has no clean way to retry a curl for N tries.
    timeout /t 8 /nobreak >nul
    curl -sf --max-time 3 http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo   FAIL  ollama did not come up. Start it in another window:
        echo             ollama serve
        pause
        exit /b 1
    )
    echo   OK    ollama  started
) else (
    echo   OK    ollama  already running
)

REM --- 4. A translation model ------------------------------------------
REM Any capable model will do, so this only offers a 2 GB download when
REM nothing usable is installed at all.
ollama list 2>nul | findstr /r /c:"[a-z]" >nul
if errorlevel 1 (
    echo   ..    no model installed
    set /p ANS="        Download llama3.2 now? About 2 GB, one time. [y/N] "
    if /i "!ANS!"=="y" (
        ollama pull llama3.2
    ) else (
        echo   FAIL  Without a model nothing can be translated.
        pause
        exit /b 1
    )
)
echo   OK    model   installed

REM --- 5. Trained classifiers ------------------------------------------
REM These ship with the project, so a miss here means an incomplete copy -
REM a ZIP made without them - not a setup step someone forgot.
if not exist "triage_model_embedding_english\model.pkl" (
    echo   FAIL  triage_model_embedding_english\model.pkl is missing.
    echo         The trained model ships with the project; this copy is
    echo         incomplete. Re-clone, or unzip the full archive.
    pause
    exit /b 1
)
echo   OK    model   triage_model_embedding_english\ present

echo ============================================================
echo Starting the app. First prediction takes about 11 seconds -
echo that is the local translation, not a hang.
echo.
.venv\Scripts\python triage_gui.py %*
pause
