#!/usr/bin/env bash
# =====================================================================
# run_local.sh - start everything the triage app needs, then the app.
#
# Save this, run it whenever you want the project. It is safe to run
# repeatedly: every step checks whether it is already done first.
#
#     ./run_local.sh
#
# Unlike run_gui.sh, which only launches the window and assumes the rest
# is already up, this script checks Ollama, checks the model, checks the
# Python environment, and reports what it found before starting anything.
# =====================================================================
set -u
cd "$(dirname "$0")"

GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELL=$'\033[0;33m'; OFF=$'\033[0m'
ok()   { echo "  ${GREEN}OK${OFF}    $1"; }
warn() { echo "  ${YELL}..${OFF}    $1"; }
fail() { echo "  ${RED}FAIL${OFF}  $1"; }

echo "Roman Urdu Emergency Triage - starting locally"
echo "============================================================"

# --- 1. Python environment -------------------------------------------
# Checked first because everything else is pointless without it, and the
# error it gives ("no such file") is the least self-explanatory of the
# failures this script can hit.
if [ ! -x ".venv/bin/python" ]; then
    fail ".venv is missing. Create it once with:"
    echo "        python -m venv .venv"
    echo "        .venv/bin/python -m pip install \\"
    echo "            --index-url https://download.pytorch.org/whl/cpu torch"
    echo "        .venv/bin/python -m pip install -r requirements.txt"
    echo
    echo "        Install torch from the CPU index FIRST. Left to itself pip"
    echo "        fetches the CUDA build - several GB of NVIDIA libraries"
    echo "        this project never uses. It is CPU-only by design."
    exit 1
fi
ok "python  $(.venv/bin/python --version 2>&1)"

# --- 2. Ollama service ------------------------------------------------
# Started if absent rather than merely reported. This service has stopped
# on its own repeatedly during development, and "start it yourself" is a
# poor answer when the script can just do it.
if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "ollama  already running"
else
    if ! command -v ollama >/dev/null 2>&1; then
        fail "ollama is not installed. Get it from https://ollama.com/download"
        echo "        The app cannot translate without it, and translation is"
        echo "        the first step of every prediction."
        exit 1
    fi
    warn "ollama  not running - starting it"
    nohup ollama serve >/tmp/ollama-triage.log 2>&1 &
    for _ in $(seq 1 20); do
        sleep 1
        curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    done
    if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "ollama  started"
    else
        fail "ollama did not come up. See /tmp/ollama-triage.log"
        exit 1
    fi
fi

# --- 3. A translation model -------------------------------------------
# Any of several models will do, so this reports what is there rather
# than insisting on one. Only offers to download when NOTHING usable is
# installed - a 2 GB pull should never be triggered by a script that
# could have used a model already on disk.
MODELS=$(curl -sf --max-time 5 http://localhost:11434/api/tags \
         | .venv/bin/python -c "import sys,json;print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null)
if [ -z "${MODELS:-}" ]; then
    warn "no model installed"
    read -r -p "        Download llama3.2 now? About 2 GB, one time. [y/N] " ans
    case "$ans" in
        [yY]*) ollama pull llama3.2 || { fail "pull failed"; exit 1; } ;;
        *) fail "Without a model nothing can be translated, so nothing can be scored."; exit 1 ;;
    esac
    ok "model   llama3.2 pulled"
else
    ok "model   $MODELS"
fi

# --- 4. Trained classifiers -------------------------------------------
# These ship with the repository, so a miss here means an incomplete
# copy - a ZIP made without them, usually - not a setup step forgotten.
if [ ! -f "triage_model_embedding_english/model.pkl" ]; then
    fail "triage_model_embedding_english/model.pkl is missing."
    echo "        The trained model ships with the project; this copy is"
    echo "        incomplete. Re-clone, or unzip the full archive."
    exit 1
fi
ok "model   triage_model_embedding_english/ present"

# --- 5. tkinter -------------------------------------------------------
# The single most common failure on a new machine, and on this one after
# a Python upgrade: tkinter is not a pip package and ships separately on
# most Linux distributions.
if ! .venv/bin/python -c "import tkinter" >/dev/null 2>&1; then
    # run_gui.sh keeps a private Tk runtime for exactly this case; try it
    # before giving up, since it may well be all that is needed.
    if [ -d ".venv/tk-runtime" ]; then
        warn "tkinter needs the bundled Tk runtime - using run_gui.sh"
        exec ./run_gui.sh "$@"
    fi
    fail "tkinter is missing. Install it with your system package manager:"
    echo "        Fedora/RHEL    sudo dnf install python3-tkinter"
    echo "        Ubuntu/Debian  sudo apt install python3-tk"
    echo "        macOS (brew)   brew install python-tk"
    echo "        Windows        already included with python.org Python"
    exit 1
fi
ok "tkinter available"

echo "============================================================"
echo "Starting the app. First prediction takes about 11 seconds -"
echo "that is the local translation, not a hang."
echo
exec .venv/bin/python triage_gui.py "$@"
