---
name: run-triage-gui
description: Launch and drive the Roman Urdu triage Tkinter GUI on this machine, including screenshots. Use when asked to run, start, screenshot, or confirm a change in the actual app rather than in tests. Covers the Fedora Tk runtime, driving the app without xdotool, and the screenshot redraw race.
---

# Running the triage GUI

Verified working on this machine (Fedora, X11, `DISPLAY=:0`).

## Interactive launch

```bash
cd /home/muawiya/ML_Predictor && ./run_gui.sh
```

Always use the wrapper, never `python triage_gui.py`. Fedora does not ship
tkinter with base `python3`, so the venv carries a private Tk runtime under
`.venv/tk-runtime/` and Tk needs two env vars to find its script library.
Without them the import dies with:

```
ImportError: libtcl9tk9.0.so: cannot open shared object file
```

To launch by hand (e.g. from a driver script), set them yourself:

```bash
TKRT="/home/muawiya/ML_Predictor/.venv/tk-runtime"
export LD_LIBRARY_PATH="$TKRT/lib64"
export TK_LIBRARY="$TKRT/share/tk9.0"
export TCL_LIBRARY="${TCL_LIBRARY:-/usr/share/tcl9.0}"
```

If `sudo dnf install -y python3-tkinter` has been run, none of this is needed.

## Driving it

This machine has **no `xdotool`, `wmctrl`, `xwininfo`, or `Xvfb`** — only
`xprop` and ImageMagick `import`. So you cannot synthesize clicks. Drive the
app by calling the same handlers the buttons call, from a script that owns the
Tk instance. `_do_predict` is the "Triage this patient" handler; `_do_explore`
is Analyse; `_do_batch` is Run batch triage.

The model loads on a **background thread** and the result arrives through
`_work_queue`, drained by an `after()` callback. So you must pump the event
loop until `app.artifacts` is set — roughly 10-30s on a warm
sentence-transformer cache. Do not assume it is ready after a fixed sleep.

```python
import os, subprocess, sys, time
sys.path.insert(0, '/home/muawiya/ML_Predictor')
os.chdir('/tmp')                 # optional: proves the path fix (see below)
import triage_gui as tg

app = tg.TriageGUI()
t0 = time.time()
while app.artifacts is None and time.time() - t0 < 300:
    app.update(); time.sleep(0.05)

app.nb.select(0)                 # 0 Triage, 1 Pipeline, 2 Stop Words,
                                 # 3 Batch, 4 Results, 5 Model Score
app._set_complaint('subah se pait mein tez dard hai lekin bukhar nahi')
app._do_predict()
print(app.status.get())          # -> "Predicted Level 1 (EMERGENCY) at 74.7% confidence."
app.destroy()
```

## Screenshots

Find the window id via `xprop` (there is no `xwininfo` here), or just use
`hex(app.winfo_id())` from inside the driver. Capture the **window**, not
`-window root` — the root grab would capture the user's whole desktop.

```python
def shot(app, path):
    app.lift(); app.update_idletasks(); app.update()
    end = time.time() + 2.0                     # let X actually paint
    while time.time() < end:
        app.update_idletasks(); app.update(); time.sleep(0.02)
    subprocess.run(['import', '-window', hex(app.winfo_id()), path], check=True)
```

**The redraw race is real.** `import` will happily grab a stale frame: a first
attempt with a 0.6s settle produced screenshots showing a tab three positions
away from the one that had been selected. Pump for ~2s before grabbing, and
record what was actually selected at capture time so a mislabelled image is
obvious:

```python
tab = app.nb.tab(app.nb.select(), 'text').strip()
```

From outside the process, find the id like this:

```bash
python - <<'PY'
import subprocess, re
ids = re.findall(r'0x[0-9a-f]+', subprocess.run(
    ['xprop','-root','_NET_CLIENT_LIST'], capture_output=True, text=True).stdout)
for wid in ids:
    name = subprocess.run(['xprop','-id',wid,'WM_NAME'],
                          capture_output=True, text=True).stdout
    if 'riage' in name: print(wid, name.strip())
PY
```

**Look at the screenshot.** A blank or partial frame means the grab beat the
paint, not that the app is broken.

## Worth exercising

Project files resolve against the code, not the cwd (`resolve_project_file()`
in `triage_pipeline.py`). Driving from `os.chdir('/tmp')` is the cheapest
regression test for that: a healthy run from a foreign directory still shows

- `model_dir` = `.../triage_model_embedding`, method `C) Embeddings + preprocessing`
- 10 learned stop words, not 0
- Results tab tables populated, **not** "not generated yet"

Use a complaint containing deployed stop words (`baad bhi jaisa ki lekin nahi
saath se tak tez`) to prove stage 5 fires — e.g. the example above reports
`stop words dropped: se, tez, lekin, nahi`. A complaint without any, such as
`seena mein shadeed dard aur pasina aa raha hai`, correctly reports "no
learned stop words were present"; that is not a bug.

## Cleanup

`run_gui.sh` in the background exits **143** when killed — that is SIGTERM
from your own `kill`, not a crash. Check no instance is left behind:

```bash
ps -eo pid,comm,args | awk '$2 ~ /^python/ && /triage_gui\.py/'
```

Do **not** use `pgrep -af triage_gui.py` for this. The tool call's own wrapper
shell carries the script text in its argv, so pgrep matches itself and always
reports a phantom instance. Matching on `comm` being python avoids that.
