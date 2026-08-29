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

# ---- Tab 4: drive the batch file end to end -----------------------------
try:
    import pandas as pd
    fixture = os.path.join("tests", "fixtures", "batch_sample.csv")
    app.batch_path.set(os.path.abspath(fixture))
    app._do_batch()
    # _do_batch hands off to a background thread; pump until the worker
    # drains its queue rather than sleeping a fixed guess.
    out_csv = os.path.splitext(os.path.abspath(fixture))[0] + "_predictions.csv"
    if os.path.exists(out_csv):
        os.remove(out_csv)
    deadline = time.time() + 900
    while not os.path.exists(out_csv) and time.time() < deadline:
        app.update()
        time.sleep(0.2)
    time.sleep(1.0)
    app.update()

    src_rows = len(pd.read_csv(fixture))
    res = pd.read_csv(out_csv)
    problems = []
    if len(res) != src_rows:
        problems.append(f"{len(res)} rows out of {src_rows} in")
    for col in ("Predicted_Triage_Level", "Confidence", "Notes",
                "Translation", "Gate_Status", "Gate_Detail"):
        if col not in res.columns:
            problems.append(f"missing column {col}")
    if not problems:
        # every row must carry a gate verdict
        if res["Gate_Status"].isna().any():
            problems.append("a row has no gate verdict")
        # blocked rows must carry no triage level
        blocked = res[res["Gate_Status"] != "PASS"]
        if len(blocked) and blocked["Predicted_Triage_Level"].notna().any():
            problems.append("a blocked row still carries a triage level")
        # the unreadable vital must be reported, not silently mean-filled
        if not res["Notes"].fillna("").str.contains("Heart_Rate").any():
            problems.append("unreadable Heart_Rate not reported in Notes")
        # the junk complaint must be capped
        junk = res[res["Complaint_Text"].fillna("").str.contains("n/a|unknown",
                                                                 case=False)]
        scored = junk["Confidence"].dropna()
        if len(scored) and (scored > 0.5 + 1e-9).any():
            problems.append(f"junk complaint scored {scored.max():.3f} > cap")
    counts = res["Gate_Status"].value_counts().to_dict() if "Gate_Status" in res else {}
    rec("tab4  Batch File: end-to-end on the fixture", not problems,
        "; ".join(problems) or
        f"{len(res)} rows scored, gate {counts}, notes on substituted vitals")
except Exception as e:
    rec("tab4  Batch File: end-to-end on the fixture", False,
        f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---- Tab 3: the removed list must be the default view -------------------
try:
    n_removed = len(app.stopword_report["stopwords"])
    rows = len(app.stop_tree.get_children())
    summary = app.stop_summary.cget("text")
    problems = []
    if app.stop_filter.get() != "stop":
        problems.append(f"default filter is {app.stop_filter.get()!r}, not 'stop'")
    if rows != n_removed:
        problems.append(f"{rows} rows rendered for {n_removed} stop words")
    if f"These {n_removed} English stop-words are removed" not in summary:
        problems.append("header text does not state the serving-path claim")
    rec("tab3  default view shows exactly the removed tokens", not problems,
        "; ".join(problems) or
        f"filter='stop', {rows} rows == {n_removed} stop words, header states "
        f"the serving path")
except Exception as e:
    rec("tab3  default view shows exactly the removed tokens", False,
        f"{type(e).__name__}: {e}")

# ---- Tab 4: the button must lock during a run and unlock after ----------
try:
    states = {"before": str(app.batch_btn.cget("state"))}
    app._batch_progress = {"done": 3, "total": 10, "text": "x",
                           "finished": False}
    app.batch_btn.config(state="disabled")
    app.batch_progress_box.pack(fill="x")
    app._poll_batch_progress()
    app.update()
    states["during"] = str(app.batch_btn.cget("state"))
    bar_max = int(app.batch_bar.cget("maximum"))
    bar_val = int(app.batch_bar.cget("value"))
    label = app.batch_progress_label.cget("text")
    app._end_batch_ui()
    app.update()
    states["after"] = str(app.batch_btn.cget("state"))
    problems = []
    if states["during"] != "disabled":
        problems.append(f"button not locked during run ({states['during']})")
    if states["after"] == "disabled":
        problems.append("button left disabled after the run - tab is dead")
    if (bar_max, bar_val) != (10, 3):
        problems.append(f"bar shows {bar_val}/{bar_max}, expected 3/10")
    if "Processing patient 4 of 10" not in label:
        problems.append(f"label reads {label[:60]!r}")
    if "Ollama" not in label:
        problems.append("label does not explain the delay")
    rec("tab4  progress bar, button lock, and unlock", not problems,
        "; ".join(problems) or
        f"bar {bar_val}/{bar_max}, 'Processing patient 4 of 10', button "
        f"{states['during']} -> {states['after']}")
except Exception as e:
    rec("tab4  progress bar, button lock, and unlock", False,
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
