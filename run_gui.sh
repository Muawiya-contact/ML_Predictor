#!/usr/bin/env bash
# ============================================================
# Launch the triage GUI on this machine.
#
# Why this wrapper exists: Fedora does not ship tkinter with the
# base python3 package, and installing it needs root. The project
# venv therefore carries a private copy of the Tk runtime under
# .venv/tk-runtime/ (extracted from the tk RPM, nothing installed
# system-wide). Tk needs two environment variables to find its
# script library, so plain `python triage_gui.py` will not work
# until `sudo dnf install python3-tkinter` has been run.
#
#   ./run_gui.sh
#
# If you later run `sudo dnf install -y python3-tkinter`, this
# wrapper is no longer needed and `python triage_gui.py` works
# directly.
# ============================================================
set -e
cd "$(dirname "$0")"

TKRT="$PWD/.venv/tk-runtime"
if [ -d "$TKRT" ]; then
    export LD_LIBRARY_PATH="$TKRT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TK_LIBRARY="$TKRT/share/tk9.0"
    export TCL_LIBRARY="${TCL_LIBRARY:-/usr/share/tcl9.0}"
fi

exec "$PWD/.venv/bin/python" triage_gui.py "$@"
