"""Build the real GUI, drive every tab, report exceptions.

Constructs TriageApp itself rather than mocking it, so a tab that raises
during construction or on its own button handler is caught here.
"""
import os
import sys
import traceback

sys.path.insert(0, "/home/muawiya/Desktop/ML_Predictor")
os.chdir("/home/muawiya/Desktop/ML_Predictor")

import tkinter as tk

import triage_gui

RESULTS = []


def rec(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))


app = triage_gui.TriageGUI()
app.withdraw()               # keep it off-screen; we only exercise logic

# The model loads on a background thread. Pump the event loop until the
# artifacts land, or a tab audited before then would be testing a half-built
# window rather than the real thing.
import time
deadline = time.time() + 180
while app.artifacts is None and time.time() < deadline:
    app.update()
    time.sleep(0.1)
rec("load  background model load", app.artifacts is not None,
    f"artifacts loaded, model_dir={app.model_dir}")

# ---- tab inventory -------------------------------------------------------
labels = [app.nb.tab(i, "text").strip() for i in range(app.nb.index("end"))]
expected = ["Triage a Patient", "Pipeline Explorer", "Stop Words",
            "Batch File", "Results", "Cluster Analysis"]
rec("tabs  inventory", labels == expected, f"{len(labels)}: {labels}")

# ---- every tab must select and render without raising --------------------
for i, label in enumerate(labels):
    try:
        app.nb.select(i)
        app.update()
        rec(f"tab{i+1}  {label}: select + render", True, "no exception")
    except Exception as e:
        rec(f"tab{i+1}  {label}: select + render", False,
            f"{type(e).__name__}: {e}")
        traceback.print_exc()

# ---- Tab 3: the stop-word table must hold the SERVING list ---------------
try:
    n = len(app.stopword_report["stopwords"]) if app.stopword_report else 0
    rows = len(app.stop_tree.get_children())
    rec("tab3  Stop Words shows the serving 68-token list",
        n == 68 and rows > 0,
        f"report has {n} stopwords, table rendered {rows} rows")
except Exception as e:
    rec("tab3  Stop Words shows the serving 68-token list", False,
        f"{type(e).__name__}: {e}")

# ---- Tab 1: drive a real prediction through the form --------------------
try:
    app.complaint.delete("1.0", "end")
    app.complaint.insert("1.0", "seena mein shadeed dard aur pasina aa raha hai")
    app._do_predict()
    app.update()
    banner = app.level_text.cget("text")
    ok = "Level" in banner
    rec("tab1  Triage a Patient: full submit", ok,
        f"banner={banner!r}, status={app.status.get()[:60]!r}")
except Exception as e:
    rec("tab1  Triage a Patient: full submit", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---- Tab 1: junk complaint must cap and warn ----------------------------
try:
    app.complaint.delete("1.0", "end")
    app.complaint.insert("1.0", "n/a")
    app._do_predict()
    app.update()
    status = app.status.get()
    pct = float(status.split("at ")[1].split("%")[0]) if "at " in status else 100.0
    rec("tab1  junk complaint capped at 50%", pct <= 50.0,
        f"status={status[:80]!r}")
except Exception as e:
    rec("tab1  junk complaint capped at 50%", False, f"{type(e).__name__}: {e}")

# ---- Tab 2: Pipeline Explorer must render all five stages ---------------
try:
    app.explore_var.set("paet may darad aur bukar hai")
    app._do_explore()
    app.update()
    texts = []

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, tk.Label):
                texts.append(c.cget("text"))
            walk(c)
    walk(app.explore_out)
    joined = "\n".join(texts)
    stages = [s for s in ("0  Raw input", "1  Dictionary + fuzzy",
                          "2  Local Ollama translation",
                          "3  Anatomical assertion gate",
                          "4  Sentence-transformer") if s in joined]
    rec("tab2  Pipeline Explorer renders 5 stages", len(stages) == 5,
        f"{len(stages)}/5 present: {[s.split('  ')[1][:16] for s in stages]}")
except Exception as e:
    rec("tab2  Pipeline Explorer renders 5 stages", False,
        f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---- no dangling references to the removed sections ---------------------
try:
    gone = [n for n in ("_results_section_methods", "_results_section_embedding",
                        "_score_section_model", "_score_section_pairs",
                        "_score_section_demo") if hasattr(app, n)]
    rec("clean  removed sections leave no attributes", not gone,
        "none present" if not gone else f"still defined: {gone}")
except Exception as e:
    rec("clean  removed sections leave no attributes", False, str(e))

app.destroy()

print("\n" + "=" * 100)
print(f"{'CHECK':<52} {'STATUS':<8} DETAILS")
print("=" * 100)
for name, status, detail in RESULTS:
    print(f"{name:<52} {status:<8} {str(detail)[:110]}")
print("=" * 100)
failed = [r for r in RESULTS if r[1] != "PASS"]
print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
