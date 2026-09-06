"""
triage_gui.py
=======================================================================
Desktop GUI for the Roman Urdu emergency triage system.
=======================================================================

Built for UNDERSTANDING the system, not just running it. Every tab shows
the working, not only the answer:

  1. Triage a Patient   - enter one patient, get the triage level AND see
                          what the text pipeline did to the complaint at
                          each stage, plus which words drove the result.
  2. Pipeline Explorer  - type any complaint and watch it move through
                          clean -> fuzzy -> stop-word removal, with the
                          statistics that justify every removed word.
  3. Stop Words         - the learned list (Contribution 1) as a sortable
                          table: document frequency, mutual information,
                          chi-square, and the reason each token was kept
                          or dropped.
  4. Batch File         - triage a whole Excel/CSV of patients, see the
                          results table and the level distribution.
  5. Results            - the model comparison and the embedding
                          effectiveness study (Contribution 2) as charts.

Uses only the Python standard library for the interface (tkinter), so it
adds NO new dependencies and runs fully offline, in keeping with the
project's lightweight CPU-only goal.

A pleasant side effect: tkinter renders Unicode properly, so the
diacritized canonical forms (dárd, bukhār, sēna) display correctly here
even though the Windows console cannot print them.

-----------------------------------------------------------------------
RUN
-----------------------------------------------------------------------
    python triage_gui.py

-----------------------------------------------------------------------
WHICH MODEL IS LIVE
-----------------------------------------------------------------------
The app predicts with triage_model_embedding/ (the offline
sentence-embedding pipeline) when that directory exists, and falls back
to the dictionary + Bag-of-Words model in triage_model/ when
sentence-transformers is not installed. Whichever is live is named in
the status bar, on the Triage a Patient tab, and on the Model Score tab
- it is never left to be inferred from a results table.

Two different systems produce numbers in this app. Every table states
which one produced it:
  * dictionary + Bag-of-Words counts with domain attention weights
  * offline sentence-transformer embeddings (384 dims)

The Results tab additionally reads the CSV/JSON files produced by
train_embedding_pipeline.py and embedding_evaluation.py, and simply says
so if they have not been generated yet.
=======================================================================
"""

import csv
import json
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

# See prediction.py: a desktop shortcut, a launcher script or an IDE runner
# may not put this file's folder on sys.path, and every `from
# triage_pipeline import ...` below (they are deliberately lazy, inside the
# methods that need them) would then fail at the moment the user clicks a
# button rather than at startup.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from triage_pipeline import resolve_project_file

# ---------------------------------------------------------------- theme
BG = "#f4f6f8"
CARD = "#ffffff"
INK = "#1f2933"
MUTED = "#7b8794"
LINE = "#d7dde3"
ACCENT = "#2563a8"

# Triage level colours: 0 = emergency (red) ... 3 = non-urgent (green)
LEVEL_COLOURS = ["#c0392b", "#e08e0b", "#2f7fbf", "#2e9e5b"]
#: Measured on this machine: 10.3 / 10.8 / 11.6 seconds for three
#: complaints through llama3.2. Used only to render an ETA, so being wrong
#: costs a misleading countdown, not a wrong prediction.
SECONDS_PER_ROW = 11

LEVEL_NAMES = ["EMERGENCY", "URGENT", "STANDARD", "NON-URGENT"]
LEVEL_BLURB = [
    "Immediate attention required",
    "Seen within 15 minutes",
    "Seen within 60 minutes",
    "Can wait or be redirected",
]

# Every one of these is a file that SHIPS WITH the project, so each is
# resolved against the code rather than against the working directory. The
# app is launched from a shortcut, a wrapper script or an IDE as often as
# from a terminal sitting in the project folder; with bare relative names
# every Results panel reported "not generated yet" and offered a command to
# regenerate files that were already there.
MODEL_DIR = resolve_project_file("triage_model")
EMBED_MODEL_DIR = resolve_project_file("triage_model_embedding")
STOPWORDS_FILE = resolve_project_file("learned_stopwords.json")

# ======================================================================
# SINGLE PIPELINE
#
# The mode toggle is gone. The GUI now has ONE triage path: translate the
# typed complaint with Ollama on localhost, then score it with the
# English-trained bundle. No network call is made either way.
#
# READ THIS BEFORE QUOTING A NUMBER FROM THIS APP. Removing the toggle also
# removed access to the 10,000-row submitted model - that bundle expects
# Roman Urdu and cannot be fed translated English without train/serve skew,
# so it could not simply be pointed at the new path. What the GUI scores
# with now is the 2,252-row English bundle, which the banner has always
# marked EXPERIMENTAL. That is a real reduction in training data and it is
# deliberate; triage_model_embedding/ is still on disk and still what
# predict_batch.py and the docs refer to as the submitted system.
#
# Translation failures are reported, never silently swapped for a different
# model. There is no second pipeline to fall back to, and an operator who
# cannot tell which model produced a triage level is worse off than one
# facing an honest error.
ENGLISH_MODEL_DIR = "triage_model_embedding_english"
CLUSTERS_FILE = resolve_project_file("evaluation_clusters.json")
DATASET_FILE = resolve_project_file("triage_mixed_language_dataset.csv")

# PIPELINE_RESULTS / EVAL_RESULTS / EVAL_NEIGHBOURS lived here and are gone.
# They fed the method-comparison and embedding-evaluation panels, which were
# removed once the mode toggle went: both scored bundles the operator can no
# longer reach. The CSVs are still on disk and still regenerated by
# embedding_evaluation.py; nothing in the app reads them.

# Plain-English name for every text representation, so a table row always
# states which pipeline produced its numbers.
REPRESENTATION_LABEL = {
    "dictionary_bow": "dictionary + BoW counts",
    "embeddings_raw": "sentence-transformer (raw text)",
    "embeddings_preprocessed": "sentence-transformer (preprocessed)",
    "hybrid": "dictionary + BoW  AND  sentence-transformer",
}

# When a file the app needs is missing, print the command that produces it
# rather than inventing a plausible-looking number.
MISSING_HINT = {
    os.path.join(EMBED_MODEL_DIR, "triage_metrics.json"): "python train_embedding_pipeline.py",
    os.path.join(resolve_project_file(ENGLISH_MODEL_DIR), "triage_metrics.json"):
        "python train_embedding_pipeline.py --english",
}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def enable_dpi_awareness():
    """Render crisply on high-DPI Windows displays.

    Without this, Windows scales the whole window as a bitmap: text is
    blurry and the layout overflows the screen on a 125%/150% display,
    which is the default on most modern laptops. Silently ignored on
    other platforms and on Windows versions without the API.
    """
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # per-monitor
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()        # older Windows
    except (ImportError, AttributeError, OSError):
        pass


def card(parent, **kw):
    """A white panel with a hairline border."""
    return tk.Frame(parent, bg=CARD, highlightbackground=LINE,
                    highlightthickness=1, **kw)


def heading(parent, text, size=13):
    return tk.Label(parent, text=text, bg=CARD, fg=INK,
                    font=("Segoe UI Semibold", size), anchor="w")


def body(parent, text="", size=10, fg=INK, bg=CARD, **kw):
    return tk.Label(parent, text=text, bg=bg, fg=fg,
                    font=("Segoe UI", size), anchor="w", justify="left", **kw)


def missing_notice(parent, path, extra=""):
    """Say which command produces a missing results file.

    Used everywhere a number would otherwise be shown. The app never
    substitutes a placeholder or example value for a real measurement.
    """
    cmd = MISSING_HINT.get(path, "see README.md")
    holder = tk.Frame(parent, bg="#fdf3f2", highlightbackground="#f0c8c3",
                      highlightthickness=1)
    inner = tk.Frame(holder, bg="#fdf3f2")
    inner.pack(fill="x", padx=12, pady=9)
    tk.Label(inner, text=f"'{path}' has not been generated yet.",
             bg="#fdf3f2", fg="#8c2f22", font=("Segoe UI Semibold", 9),
             anchor="w").pack(fill="x")
    tk.Label(inner, text=f"Run this first:    {cmd}",
             bg="#fdf3f2", fg="#8c2f22", font=("Consolas", 9),
             anchor="w").pack(fill="x", pady=(2, 0))
    if extra:
        tk.Label(inner, text=extra, bg="#fdf3f2", fg=MUTED,
                 font=("Segoe UI", 8), anchor="w", justify="left",
                 wraplength=900).pack(fill="x", pady=(4, 0))
    return holder


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------
# Why a token was kept
#
# The learner drops a token only when THREE things hold: it is frequent
# enough, its normalized mutual information is at or below the threshold,
# AND its Cramer's V effect size is at or below the threshold. The tab used
# to collapse all three into one sentence, "kept - carries triage signal",
# shown whenever the token was not judged uninformative.
#
# (Historical note: the decision used to hinge on the chi-square p-value
# instead of Cramer's V. P-values shrink as the corpus grows, so filler
# like "hai" and "mein" - genuinely uninformative by MI - kept being
# vetoed on larger datasets. Effect size does not scale with n, which is
# why the criterion was changed.) Name the criterion that decided each
# row instead of a vague catch-all.
# ---------------------------------------------------------------------



# ======================================================================
# GLOSSARY + HOVER TOOLTIPS
#
# The Results tab is dense with machine-learning vocabulary, and a triage
# nurse reading it has no reason to know what "macro average" means. Every
# definition below is written for that reader: what the number is, and what
# it means HERE - on a triage screen - rather than a textbook restatement.
#
# Kept as data in one place so the wording cannot drift between the column
# header, the summary line and the table that uses the same term.
# ======================================================================

GLOSSARY = {
    "precision": (
        "PRECISION\n\n"
        "Of all the patients the model CALLED this level, how many really "
        "were?\n\n"
        "Low precision means false alarms: the model puts patients into this "
        "level who do not belong there. 0.90 means 9 out of every 10 it "
        "labelled this way were correct."
    ),
    "recall": (
        "RECALL  (also called sensitivity)\n\n"
        "Of all the patients who REALLY were this level, how many did the "
        "model catch?\n\n"
        "Low recall means missed cases. On the emergency row this is the "
        "number that matters most - a missed emergency is the dangerous "
        "failure, not a false alarm."
    ),
    "f1": (
        "F1 SCORE\n\n"
        "A metric used to evaluate classification models by combining "
        "precision and recall into a single number.\n\n"
        "It is their harmonic mean, so it only goes high when BOTH are high - "
        "a model that catches every case by labelling everything this level "
        "would have perfect recall but terrible precision, and F1 exposes "
        "that. 1.0 is perfect, 0.0 is useless."
    ),
    "support": (
        "SUPPORT\n\n"
        "How many test patients truly had this level.\n\n"
        "Read every other number on the row against this one. A class with "
        "80 patients moves several percentage points on a handful of "
        "predictions, so its precision and recall are far less stable than "
        "a class with 800."
    ),
    "macro": (
        "MACRO AVERAGE\n\n"
        "The plain average across the four triage levels, counting each "
        "level equally regardless of how many patients it has.\n\n"
        "This is the honest summary when classes are unbalanced: the rare "
        "NON-URGENT level counts as much as the common URGENT one, so a "
        "model that ignores rare levels cannot hide behind them."
    ),
    "weighted": (
        "WEIGHTED AVERAGE\n\n"
        "The average across levels, weighted by how many patients each has.\n\n"
        "It tracks overall accuracy closely, which also means it is "
        "flattered by the common levels. If weighted is much higher than "
        "macro, the model is doing well on the frequent levels and poorly "
        "on the rare ones."
    ),
    "accuracy": (
        "ACCURACY\n\n"
        "The share of test patients given the correct triage level.\n\n"
        "It treats every mistake alike, which triage does not - calling an "
        "emergency patient non-urgent and calling a non-urgent patient an "
        "emergency both cost one point here, but only one of them is "
        "dangerous. Read it next to under-triage."
    ),
    "under_triage": (
        "UNDER-TRIAGE\n\n"
        "The share of patients rated LESS urgent than they really are.\n\n"
        "This is the dangerous error: a Level 1 patient sent to the Level 3 "
        "queue waits while their condition worsens. It is reported "
        "separately from accuracy precisely because accuracy hides it."
    ),
    "over_triage": (
        "OVER-TRIAGE\n\n"
        "The share of patients rated MORE urgent than they really are.\n\n"
        "Wasteful rather than dangerous - it consumes a bed, a clinician and "
        "a slot that someone sicker needed. Preferable to under-triage, but "
        "not free."
    ),
    "confusion_matrix": (
        "CONFUSION MATRIX\n\n"
        "A grid of true level against predicted level. The diagonal is what "
        "the model got right; everything off the diagonal is a mistake, and "
        "WHERE it sits tells you the kind.\n\n"
        "Cells below the diagonal are over-triage, cells above it are "
        "under-triage. Every precision, recall and F1 figure on this page is "
        "derived from this grid."
    ),
    "safety_grade": (
        "SAFETY GRADE\n\n"
        "A band on the under-triage rate alone, not on accuracy:\n"
        "  A+ under 5%   A under 10%   B under 15%   C under 20%   F above\n\n"
        "A model can be accurate and still graded poorly if the mistakes it "
        "does make are the dangerous kind."
    ),
}


class _TooltipManager:
    """One tooltip window for the whole app, shown near the cursor.

    THE BUG THIS REPLACES: the first version created a Tooltip object per
    hover target, and the Treeview variant created a NEW one on every
    column the pointer crossed. Each instance re-bound <Enter>, <Leave>
    and <ButtonPress> to the same tree with add="+", so the handlers piled
    up - by the time you had swept the header a few times the widget was
    firing a dozen stale callbacks, and the popup flickered or stuck.

    One window, one set of bindings per widget, and a single pending timer
    that is always cancelled before a new one starts.
    """

    DELAY_MS = 400
    WRAP = 400
    CURSOR_DX, CURSOR_DY = 16, 22

    def __init__(self):
        self._win = None
        self._label = None
        self._after = None
        self._owner = None

    def _ensure(self, widget):
        if self._win is not None and self._win.winfo_exists():
            return
        self._win = tk.Toplevel(widget.winfo_toplevel())
        self._win.wm_overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._label = tk.Label(
            self._win, justify="left", anchor="w", bg="#2b3a4a", fg="white",
            font=("Segoe UI", 9), relief="solid", bd=1, padx=10, pady=8,
            wraplength=self.WRAP)
        self._label.pack()
        self._win.withdraw()

    def schedule(self, widget, text, x_root, y_root):
        """Show `text` after the delay, unless cancelled first."""
        self.cancel()
        if not text:
            return
        self._owner = widget
        self._after = widget.after(
            self.DELAY_MS, lambda: self._show(widget, text, x_root, y_root))

    def _show(self, widget, text, x_root, y_root):
        self._after = None
        if not widget.winfo_exists():
            return
        self._ensure(widget)
        self._label.configure(text=text)
        self._win.update_idletasks()
        w, h = self._win.winfo_reqwidth(), self._win.winfo_reqheight()
        sw, sh = self._win.winfo_screenwidth(), self._win.winfo_screenheight()
        # Keep it on screen: flip to the other side of the cursor near an edge.
        x = min(x_root + self.CURSOR_DX, sw - w - 8)
        y = y_root + self.CURSOR_DY
        if y + h > sh - 8:
            y = y_root - h - 8
        self._win.wm_geometry(f"+{max(8, x)}+{max(8, y)}")
        self._win.deiconify()
        self._win.lift()

    def cancel(self):
        if self._after is not None:
            try:
                if self._owner is not None and self._owner.winfo_exists():
                    self._owner.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def hide(self, _event=None):
        self.cancel()
        if self._win is not None and self._win.winfo_exists():
            self._win.withdraw()

    @property
    def visible(self):
        return (self._win is not None and self._win.winfo_exists()
                and self._win.state() != "withdrawn")

    @property
    def text(self):
        return self._label.cget("text") if self._label else ""


TOOLTIP = _TooltipManager()


def attach_tooltip(widget, key_or_text):
    """Hover help on a plain widget. Binds once, never re-binds."""
    text = GLOSSARY.get(key_or_text, key_or_text)
    if getattr(widget, "_tooltip_bound", False):
        widget._tooltip_text = text
        return
    widget._tooltip_bound = True
    widget._tooltip_text = text
    widget.bind(
        "<Enter>",
        lambda e, w=widget: TOOLTIP.schedule(w, w._tooltip_text, e.x_root, e.y_root),
        add="+")
    widget.bind("<Leave>", TOOLTIP.hide, add="+")
    widget.bind("<ButtonPress>", TOOLTIP.hide, add="+")


def attach_header_tooltips(tree, column_keys):
    """Per-column hover help on a Treeview heading row.

    A Treeview is a single widget, so <Enter> cannot say which heading the
    pointer is over. Motion is tracked instead, and the tooltip is only
    rescheduled when the pointer actually moves to a DIFFERENT column -
    otherwise every pixel of movement would restart the timer and the
    popup would never appear.
    """
    if getattr(tree, "_hdr_tooltip_bound", False):
        return
    tree._hdr_tooltip_bound = True
    state = {"col": None}

    def on_motion(event):
        if tree.identify_region(event.x, event.y) != "heading":
            if state["col"] is not None:
                state["col"] = None
                TOOLTIP.hide()
            return
        col = tree.identify_column(event.x)
        if col == state["col"]:
            return                      # same column - let the timer run
        state["col"] = col
        if col == "#0":
            key = column_keys.get("#0")
        else:
            try:
                key = column_keys.get(tree["columns"][int(col[1:]) - 1])
            except (ValueError, IndexError):
                key = None
        if not key:
            TOOLTIP.hide()
            return
        TOOLTIP.schedule(tree, GLOSSARY.get(key, key), event.x_root, event.y_root)

    def on_leave(_event=None):
        state["col"] = None
        TOOLTIP.hide()

    tree.bind("<Motion>", on_motion, add="+")
    tree.bind("<Leave>", on_leave, add="+")


def per_class_metrics(confusion):
    """Precision, recall, F1 and support per class, from the confusion matrix.

    Derived rather than stored. The saved metrics file records accuracy,
    under/over-triage and the confusion matrix but no per-class table, and
    the matrix already contains one exactly - precision is the column
    share, recall the row share. Deriving beats re-training to add a field,
    and beats hardcoding numbers that would silently go stale the next time
    the model changes.

    Verified against the stored accuracy: trace/total reproduces the
    metrics file's own figure to four decimals.
    """
    cm = [[float(v) for v in row] for row in confusion]
    n = len(cm)
    total = sum(sum(r) for r in cm)
    rows, macro_p, macro_r, macro_f = [], 0.0, 0.0, 0.0
    weighted_p = weighted_r = weighted_f = 0.0
    for i in range(n):
        tp = cm[i][i]
        col = sum(cm[r][i] for r in range(n))
        support = sum(cm[i])
        prec = tp / col if col else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"index": i, "precision": prec, "recall": rec,
                     "f1": f1, "support": int(support)})
        macro_p += prec; macro_r += rec; macro_f += f1
        if total:
            weighted_p += prec * support / total
            weighted_r += rec * support / total
            weighted_f += f1 * support / total
    accuracy = (sum(cm[i][i] for i in range(n)) / total) if total else 0.0
    return {
        "per_class": rows,
        "macro": {"precision": macro_p / n, "recall": macro_r / n, "f1": macro_f / n},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f},
        "accuracy": accuracy,
        "total": int(total),
    }


def keep_reason(stat, thresholds):
    """The specific criterion that stopped this token becoming a stop word."""
    if stat["is_stopword"]:
        return "STOP WORD - removed"
    if stat["clinically_protected"] and stat["is_uninformative"]:
        return "kept - clinical safety guard (medical vocabulary)"
    if not stat["is_high_frequency"]:
        return "kept - not frequent enough to be tested"

    mi_ok = stat["normalized_mutual_information"] <= thresholds["mi_threshold"]
    v_ok = stat["cramers_v"] <= thresholds["cramers_v_threshold"]
    if not mi_ok and not v_ok:
        return "kept - both tests say it tracks the triage level"
    if not mi_ok:
        return (f"kept - mutual information "
                f"{stat['normalized_mutual_information']:.5f} is above the "
                f"{thresholds['mi_threshold']} threshold")
    # mi_ok and not v_ok: low MI but a real association by effect size.
    return (f"kept - Cramer's V {stat['cramers_v']:.3f} is above the "
            f"{thresholds['cramers_v_threshold']} cutoff, so its association "
            f"with triage is too strong to drop, even though its MI is low")


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------

class TriageGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Roman Urdu Emergency Triage  -  Offline Decision Support")
        self.configure(bg=BG)

        # Fit the window to the screen rather than assuming a large monitor:
        # a fixed 1180x790 overflows a 1366x768 laptop once the taskbar and
        # title bar are taken into account.
        want_w, want_h = 1180, 800
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(want_w, screen_w - 80)
        height = min(want_h, screen_h - 100)
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(940, 620)

        self.artifacts = None
        self.stopword_report = None
        self.model_info = None          # which method is actually deployed
        self._ollama_model_note = None  # last Ollama tag reported to the user
        self._pull_active = False
        self.model_dir = None
        self._work_queue = queue.Queue()

        self._build_style()
        self._build_header()
        self._build_tabs()
        self._build_statusbar()

        # Loading the model takes a moment; do it off the UI thread so the
        # window paints immediately instead of appearing frozen.
        self._run_async(self._load_model, "Loading model and encoders...")
        self.after(80, self._drain_queue)

    # ---------------------------------------------------------- styling
    def _build_style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(18, 9),
                    font=("Segoe UI", 10), background="#e4e9ee", foreground=INK)
        s.map("TNotebook.Tab",
              background=[("selected", CARD)],
              foreground=[("selected", ACCENT)])
        s.configure("Treeview", rowheight=24, fieldbackground=CARD,
                    background=CARD, foreground=INK, font=("Segoe UI", 9))
        s.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))
        s.configure("Accent.TButton", font=("Segoe UI Semibold", 10),
                    padding=(16, 8))

    def _build_header(self):
        # 58px fitted the title alone. The mode switch added a second row of
        # controls to the same bar, and with pack_propagate(False) the frame
        # refuses to grow - so the radio buttons and the title were clipped
        # top and bottom instead. Sized for the tallest child now.
        bar = tk.Frame(self, bg=ACCENT, height=76)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="  Roman Urdu Emergency Triage",
                 bg=ACCENT, fg="white",
                 font=("Segoe UI Semibold", 15)).pack(side="left", padx=(14, 0))
        self.header_note = tk.Label(
            bar, text="English via local Ollama  |  offline  |  CPU only  |  research prototype",
            bg=ACCENT, fg="#cfe0f2", font=("Segoe UI", 9))
        self.header_note.pack(side="right", padx=(0, 10))

    def _build_statusbar(self):
        self.status = tk.StringVar(value="Starting...")
        bar = tk.Frame(self, bg="#e9edf1", height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status, bg="#e9edf1", fg=MUTED,
                 font=("Segoe UI", 9), anchor="w").pack(side="left", padx=10)

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        self.tab_predict = tk.Frame(self.nb, bg=BG)
        self.tab_pipeline = tk.Frame(self.nb, bg=BG)
        self.tab_stops = tk.Frame(self.nb, bg=BG)
        self.tab_batch = tk.Frame(self.nb, bg=BG)
        self.tab_results = tk.Frame(self.nb, bg=BG)
        self.tab_score = tk.Frame(self.nb, bg=BG)
        for frame, label in [
            (self.tab_predict, "  Triage a Patient  "),
            (self.tab_pipeline, "  Pipeline Explorer  "),
            (self.tab_stops, "  Stop Words  "),
            (self.tab_batch, "  Batch File  "),
            (self.tab_results, "  Results  "),
            (self.tab_score, "  Cluster Analysis  "),
        ]:
            self.nb.add(frame, text=label)

        self._build_predict_tab()
        self._build_pipeline_tab()
        self._build_stopwords_tab()
        self._build_batch_tab()
        self._build_results_tab()
        self._build_score_tab()
        self.nb.select(0)

    # ------------------------------------------------ background worker
    def _run_async(self, fn, status_text):
        self.status.set(status_text)

        def worker():
            try:
                result = fn()
                self._work_queue.put(("ok", fn.__name__, result))
            except Exception:
                self._work_queue.put(("err", fn.__name__, traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_queue(self):
        try:
            while True:
                kind, name, payload = self._work_queue.get_nowait()
                if kind == "err":
                    self.status.set(f"Error in {name}")
                    if name == "_batch_worker":
                        self._end_batch_ui()
                    # Re-enable any button the failed job had disabled, so a
                    # single failure does not leave the tab permanently dead.
                    messagebox.showerror("Error", payload)
                else:
                    handler = getattr(self, f"_done{name}", None)
                    if handler:
                        handler(payload)
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

    def _load_model(self):
        from triage_pipeline import describe_model, load_artifacts, resolve_model_dir

        # Which model this app predicts with is now resolved, reported and
        # shown on screen. It used to be the hard-coded string "triage_model"
        # - the dictionary + Bag-of-Words model - while the Results tabs
        # showed embedding scores beside it with nothing saying that the
        # live predictions came from neither of the embedding rows.
        model_dir, note = resolve_model_dir()
        art = load_artifacts(model_dir)
        info = describe_model(model_dir)

        # Warm the sentence-transformer here, on the background thread. It is
        # loaded lazily on first use, and first use is inside predict_one() on
        # the UI thread - which would freeze the window for several seconds on
        # the first "Triage this patient" click.
        if info["uses_embeddings"]:
            from triage_pipeline import build_text_features
            build_text_features(art, ["warmup"])

        # The SERVING bundle's own stop-word report, not the project-root
        # one. That file belongs to whichever model trained last, and since
        # the mode toggle went it is not the model that scores anything: it
        # listed 22 Roman Urdu tokens while the bundle actually running
        # removes 68 English ones. The tab was showing a list that never
        # touches a single prediction. Same schema in both, so nothing
        # downstream changes.
        report = None
        served_stops = os.path.join(resolve_project_file(ENGLISH_MODEL_DIR),
                                    "learned_stopwords.json")
        for candidate in (served_stops, STOPWORDS_FILE):
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    report = json.load(f)
                break
        return art, report, info, model_dir, note

    def _done_load_model(self, payload):
        (self.artifacts, self.stopword_report,
         self.model_info, self.model_dir, note) = payload
        # The bundle's own manifest, so the banner can state provenance and
        # scope from what was actually deployed rather than a hardcoded string.
        self.manifest = (self.artifacts or {}).get("manifest", {})
        n_stops = len(self.stopword_report["stopwords"]) if self.stopword_report else 0
        # NAME THE BUNDLE THAT SCORES, not the one discovered at startup.
        # self.model_dir is triage_model_embedding/ - the 10,000-row Roman
        # Urdu bundle - and since the mode toggle was removed that is NOT
        # what scores a patient. The status bar still announced it, so the
        # line along the bottom of every screen named a different model from
        # the Results tab directly above it, and from the banner beside it.
        # An operator reconciling the two had no way to tell which was
        # right.
        #
        # The row count comes from the SERVING manifest for the same reason:
        # quoting 10,000 rows beside a model trained on 2,252 overstates the
        # evidence behind every number on screen.
        serving = getattr(self, "model_info_en", None) or self.model_info
        man = self.active_manifest() or {}
        rows = (man.get("dataset") or {}).get("rows")
        rows_txt = f"{rows:,} rows" if isinstance(rows, int) else "unknown rows"
        self.status.set(
            f"Ready.  Serving: {ENGLISH_MODEL_DIR}/  ({rows_txt}, "
            f"{serving['method']})  |  {n_stops} learned stop words  |  "
            f"translation and scoring both run locally."
            + (f"  |  {note}" if note else ""))
        self._fill_dropdowns()
        self._populate_stopwords()
        self._refresh_deployed_banners(note)
        self.predict_btn.config(state="normal")
        self.batch_btn.config(state="normal")


    # ---------------- mode plumbing ----------------

    def in_english_mode(self):
        """Always true now. Kept as a method rather than inlined at ~12 call
        sites, so restoring a second pipeline stays a one-line change."""
        return True

    def active_artifacts(self):
        """The English bundle - the only one the GUI predicts with."""
        if getattr(self, "artifacts_en", None) is None:
            from triage_pipeline import load_artifacts, describe_model
            d = resolve_project_file(ENGLISH_MODEL_DIR)
            self.artifacts_en = load_artifacts(d)
            self.model_info_en = describe_model(d)
            self.model_dir_en = ENGLISH_MODEL_DIR
        return self.artifacts_en

    def active_model_dir(self):
        """The directory of the bundle that actually serves predictions.

        self.model_dir is the Roman Urdu bundle discovered at startup. Since
        the mode toggle was removed, that is NOT what scores a patient, so
        any tab describing "the deployed model" from it is describing a
        model the operator can no longer reach.
        """
        return resolve_project_file(ENGLISH_MODEL_DIR)

    def active_manifest(self):
        art = self.active_artifacts()
        return (art or {}).get("manifest", {}) if art else {}

    def translate_for_mode(self, text, allow_blocked=False):
        """Returns (english_text, error_message).

        allow_blocked=True returns the translation even when the anatomical
        gate rejects it, and leaves the verdict in self._last_gate. The batch
        path needs that: aborting a 500-row file because one row drifted
        would be a worse failure than the drift, and the operator cannot act
        on rows they are never shown. The single-patient path keeps the
        default, where a blocked translation yields no prediction at all.

        Routed through the LOCAL Ollama service. Every failure path here
        returns a message rather than raising: this is called from the
        "Triage this patient" handler, and an exception escaping would
        take the button down with it. A dead translator must degrade to a
        clear message, never to a dead UI.
        """
        try:
            from src.offline_pipeline import (OLLAMA_URL, ollama_available,
                                              ollama_models,
                                              select_translation_model,
                                              translate_roman_urdu)
        except Exception as e:
            return None, f"Offline pipeline failed to import: {e}"

        try:
            if not ollama_available():
                return None, (
                    f"Ollama is not reachable at {OLLAMA_URL}.\n\n"
                    f"Start it with:\n    ollama serve\n\n"
                    f"Translation is the only triage path now, so nothing "
                    f"can be scored until it is running.")

            have = ollama_models()
            model = select_translation_model(have)
            if model is None:
                # No model at all - this is the case worth offering to fix.
                self._offer_model_pull()
                return None, (
                    "Ollama is running but has no models installed.\n\n"
                    "Use the download prompt. Without a model the app "
                    "cannot translate, and no prediction will be made.")

            if model != self._ollama_model_note:
                # Say so once when a fallback is in play, so a different
                # translator is never mistaken for the configured one.
                self._ollama_model_note = model
                self.status.set(f"translating locally with {model}")

            out = translate_roman_urdu(text, model=model)
        except Exception as e:
            return None, (f"Translation failed: {type(e).__name__}: {e}\n\n"
                          f"No prediction was made.")

        if not out:
            from src.offline_pipeline import (has_medical_signal,
                                              is_non_latin_script,
                                              fuzzy_normalize_roman_urdu)
            if is_non_latin_script(text):
                return None, (
                    "This complaint is not in the Latin alphabet.\n\n"
                    "The app reads Roman Urdu (Urdu written in English "
                    "letters) and English. Urdu script is not supported - "
                    "the dictionary, the vocabulary and the safety checks "
                    "are all Latin-alphabet.\n\n"
                    "What to do: type the complaint in Roman Urdu, for "
                    "example \"seena mein dard\" rather than \u0633\u06cc\u0646\u06d2 \u0645\u06cc\u06ba \u062f\u0631\u062f.")
            # NORMALIZED, not raw. translate_roman_urdu() repairs spelling
            # before it checks, so checking the raw string here disagreed
            # with the pipeline: with Ollama down, "bukar" (a misspelling of
            # bukhar/fever) was reported as "not a complaint" when the real
            # problem was the dead translator, sending the operator off to
            # rewrite text that was fine.
            if not has_medical_signal(fuzzy_normalize_roman_urdu(text, verbose=False)):
                return None, (
                    "This does not look like a complaint.\n\n"
                    f"{text!r} contains no symptom and no body part, so there "
                    f"is nothing to translate. Asked to translate it anyway, "
                    f"the model invents a symptom - which then gets scored as "
                    f"though the patient reported it.\n\n"
                    f"What to do: describe the symptom and where it is, for "
                    f"example \"seena mein dard\" or \"pait mein dard aur "
                    f"ulti\".")
            return None, (
                f"{model} returned nothing. The console shows the reason.\n\n"
                f"No prediction was made - the app never falls back to a "
                f"different model, so the level you see is never from a "
                f"pipeline you did not ask for.")

        # Deterministic anatomical gate. The GUI has no second pipeline to
        # fall back to any more, so a drifted translation must BLOCK rather
        # than quietly score: a stomach complaint rendered as chest pain
        # would otherwise produce a confident cardiac triage level with
        # nothing on screen to suggest the body part had changed.
        from src.offline_pipeline import (fuzzy_normalize_roman_urdu,
                                          verify_anatomical_integrity)
        ok, failures = verify_anatomical_integrity(
            fuzzy_normalize_roman_urdu(text, verbose=False), out)
        # Recorded so the batch path can report per row instead of inferring
        # the reason from an error string.
        self._last_gate = (ok, failures, out)
        if not ok and not allow_blocked:
            return None, (
                "Anatomical check failed - the English does not match the "
                "body part in the complaint.\n\n"
                + "\n".join(failures) +
                f"\n\nThe translator produced: {out!r}\n\n"
                f"No triage level was produced, on purpose. Scoring this "
                f"would attribute a body part to the patient that they did "
                f"not report.\n\n"
                f"What to do: write the complaint more fully - name the body "
                f"part and the symptom, for example \"seena mein dard\" "
                f"rather than a fragment.")
        return out, None

    def _offer_model_pull(self, model="llama3.2"):
        """Offer to download a model, with a live progress bar.

        The pull runs on a worker thread; Tk is not thread-safe, so the
        worker only stores numbers and a poller on the main thread paints
        them. Declining leaves the app on the offline Roman Urdu model,
        which is the whole point - the user is never stuck.
        """
        if getattr(self, "_pull_active", False):
            return
        if not messagebox.askyesno(
                "No Ollama model installed",
                f"English (Local LLM) mode needs a model, and Ollama has "
                f"none installed.\n\n"
                f"Download {model} now? It is about 2 GB and needs a network "
                f"connection for the download only - translation afterwards "
                f"is fully offline.\n\n"
                f"Without it the app cannot translate, and triage will "
                f"report a translation error rather than guess."):
            self.status.set(f"{model} not installed - triage will fail until "
                            f"it is pulled.")
            return

        self._pull_active = True
        win = tk.Toplevel(self)
        win.title(f"Downloading {model}")
        win.transient(self)
        win.resizable(False, False)
        frame = tk.Frame(win, bg=CARD, padx=18, pady=16)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=f"Pulling {model} via Ollama",
                 bg=CARD, fg=INK, font=("Segoe UI Semibold", 10),
                 anchor="w").pack(fill="x")
        status = tk.Label(frame, text="starting...", bg=CARD, fg=MUTED,
                          font=("Segoe UI", 9), anchor="w", width=52)
        status.pack(fill="x", pady=(4, 6))
        bar = ttk.Progressbar(frame, length=380, mode="determinate", maximum=100)
        bar.pack(fill="x")
        tk.Label(frame, text="Triage stays unavailable until this finishes.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                 anchor="w", wraplength=380).pack(fill="x", pady=(8, 0))

        shared = {"pct": 0.0, "text": "starting...", "done": False, "ok": None,
                  "msg": ""}

        def on_progress(st, completed, total):
            shared["pct"] = (100.0 * completed / total) if total else 0.0
            shared["text"] = (f"{st}  {completed/1e9:.2f} / {total/1e9:.2f} GB"
                              if total else st)

        def worker():
            from src.offline_pipeline import pull_model
            ok, msg = pull_model(model, progress=on_progress)
            shared["ok"], shared["msg"], shared["done"] = ok, msg, True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if not win.winfo_exists():
                return
            bar["value"] = shared["pct"]
            status.configure(text=shared["text"][:64])
            if shared["done"]:
                self._pull_active = False
                win.destroy()
                if shared["ok"]:
                    messagebox.showinfo(
                        "Download complete",
                        f"{model} is installed. English (Local LLM) mode is "
                        f"ready - translation now runs entirely on this "
                        f"machine.")
                    self.status.set(f"{model} installed - English mode ready.")
                else:
                    messagebox.showerror(
                        "Download failed",
                        f"{shared['msg']}\n\nTriage will report a "
                        f"translation error until a model is installed.")
                return
            win.after(300, poll)

        win.after(300, poll)

    def _deployed_line(self):
        """Four aligned lines: model, pipeline, data, status.

        This was a twelve-line paragraph of asterisk-fenced banners, built up
        by prepending one warning in front of another, plus a branch for a
        mode that no longer exists. Everything in it was true and almost none
        of it was read - a wall of shouting text is skipped exactly like no
        text at all. The facts that must survive a glance are: which bundle
        is scoring, that nothing leaves the machine, that the data is
        synthetic, and that this is not a medical device. Those are the four
        lines; nothing else earns a place here.
        """
        if not self.model_info:
            return "model      loading..."

        i = getattr(self, "model_info_en", None) or self.model_info
        man = self.active_manifest() or getattr(self, "manifest", None) or {}
        ds = man.get("dataset", {})
        prov = ds.get("provenance", {})

        rows = ds.get("rows")
        model_bits = [f"{rows:,} rows" if isinstance(rows, int) else None,
                      i.get("method"), ENGLISH_MODEL_DIR + "/"]
        if i.get("uses_embeddings"):
            model_bits.insert(2, f"{i['embedding_model'].split('/')[-1]} "
                                 f"({i['embedding_dim']}d)")
        lines = ["model      " + "  ·  ".join(b for b in model_bits if b),
                 "pipeline   Roman Urdu  ->  Ollama (local)  ->  anatomical "
                 "gate  ->  classifier   ·   no network call"]

        data_bits = [ds.get("file")]
        if prov.get("synthetic") is True:
            data_bits.append("SYNTHETIC - not real patient records")
        elif prov.get("synthetic") == "unknown":
            data_bits.append("provenance UNKNOWN")
        scope = (man.get("scope") or {}).get("clinical_scope")
        if scope:
            data_bits.append(scope)
        lines.append("data       " + "  ·  ".join(b for b in data_bits if b))

        status = "research prototype - not a medical device"
        if man.get("experiment"):
            status = "EXPERIMENTAL bundle, not the submitted model   ·   " + status
        lines.append("status     " + status)
        return "\n".join(lines)

    def _refresh_deployed_banners(self, note=""):
        """Fill in everything that depends on WHICH model finished loading."""
        # Redraw the classification report. It is built at startup so the
        # window paints immediately, but the model loads on a background
        # thread, so the numbers are only correct after this call. The two
        # method-comparison tables that used to be redrawn here described
        # models that no longer serve and have been removed with them; this
        # list is kept as a list so adding another deferred table is a
        # one-line change rather than a restructure.
        for box, render in [(getattr(self, "_results_classification_box", None),
                             self._results_section_classification)]:
            if box is None:
                continue
            for child in box.winfo_children():
                child.destroy()
            render(box)

        live = []
        for label in getattr(self, "_deployed_labels", []):
            if not label.winfo_exists():
                continue
            text = self._deployed_line()
            if note:
                text += f"\n{note}"
            label.configure(text=text)
            live.append(label)
        self._deployed_labels = live

    def _deployed_banner(self, parent, prefix=""):
        """A prominent, always-visible statement of what is actually running."""
        holder = tk.Frame(parent, bg="#eaf3ea", highlightbackground="#b7d7b7",
                          highlightthickness=1)
        inner = tk.Frame(holder, bg="#eaf3ea")
        inner.pack(fill="x", padx=12, pady=9)
        if prefix:
            tk.Label(inner, text=prefix, bg="#eaf3ea", fg=MUTED,
                     font=("Segoe UI", 8), anchor="w").pack(fill="x")
        label = tk.Label(inner, text=self._deployed_line(), bg="#eaf3ea",
                         fg="#1e5c2e", font=("Segoe UI Semibold", 9),
                         anchor="w", justify="left", wraplength=940)
        label.pack(fill="x")
        if not hasattr(self, "_deployed_labels"):
            self._deployed_labels = []
        self._deployed_labels.append(label)
        return holder

    # =================================================================
    # TAB 1 - Triage a patient
    # =================================================================
    def _build_predict_tab(self):
        root = self.tab_predict
        # grid with uniform columns forces a true 50/50 split. With pack the
        # input card's natural width wins and squeezes the result pane down to
        # a few hundred pixels, clipping the probability bars.
        root.columnconfigure(0, weight=1, uniform="half")
        root.columnconfigure(1, weight=1, uniform="half")
        root.rowconfigure(0, weight=1)
        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=6)
        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=6)

        # ---- input card ----
        inp = card(left)
        inp.pack(fill="both", expand=True)
        pad = tk.Frame(inp, bg=CARD)
        pad.pack(fill="both", expand=True, padx=16, pady=14)

        heading(pad, "Patient details").pack(fill="x")
        body(pad, "Chief complaint in Roman Urdu or English.",
             fg=MUTED, size=9).pack(fill="x", pady=(2, 8))

        self.complaint = tk.Text(pad, height=3, font=("Segoe UI", 11),
                                 wrap="word", relief="solid", bd=1,
                                 highlightthickness=0)
        self.complaint.pack(fill="x")
        self.complaint.insert("1.0", "seena mein shadeed dard aur pasina aa raha hai")

        ex = tk.Frame(pad, bg=CARD)
        ex.pack(fill="x", pady=(6, 10))
        body(ex, "Examples:", fg=MUTED, size=9).pack(side="left")
        for label, text in [
            ("chest pain", "seena mein shadeed dard aur pasina aa raha hai"),
            ("breathless", "saans phool rahi hai lekin baat kar sakti hai"),
            ("fever", "bukhaar aur khaansi teen din se"),
            ("trauma", "accident mein haddi toot gayi aur khoon beh raha hai"),
        ]:
            tk.Button(ex, text=label, font=("Segoe UI", 8), bg="#eef2f6",
                      fg=ACCENT, relief="flat", cursor="hand2", padx=7,
                      command=lambda t=text: self._set_complaint(t)
                      ).pack(side="left", padx=3)

        grid = tk.Frame(pad, bg=CARD)
        grid.pack(fill="x")
        self.fields = {}

        numeric = [
            ("Age", "years", 65), ("Heart_Rate", "bpm", 118),
            ("Systolic_BP", "mmHg", 160), ("Diastolic_BP", "mmHg", 95),
            ("Temperature", "C", 37.2), ("SpO2", "%", 94),
        ]
        for i, (name, unit, default) in enumerate(numeric):
            r, c = divmod(i, 3)
            cell = tk.Frame(grid, bg=CARD)
            cell.grid(row=r, column=c, sticky="ew", padx=(0, 12), pady=5)
            grid.columnconfigure(c, weight=1)
            body(cell, f"{name.replace('_', ' ')}  ({unit})",
                 fg=MUTED, size=9).pack(fill="x")
            var = tk.StringVar(value=str(default))
            tk.Entry(cell, textvariable=var, font=("Segoe UI", 10),
                     relief="solid", bd=1).pack(fill="x")
            self.fields[name] = var

        self.combos = {}
        for i, (name, label) in enumerate([
            ("Gender", "Gender"), ("Mode_of_Arrival", "Mode of arrival"),
            ("AVPU", "AVPU (consciousness)"), ("ECG_Status", "ECG status"),
        ]):
            r, c = divmod(i, 2)
            cell = tk.Frame(grid, bg=CARD)
            cell.grid(row=2 + r, column=c, columnspan=1, sticky="ew",
                      padx=(0, 12), pady=5)
            body(cell, label, fg=MUTED, size=9).pack(fill="x")
            cb = ttk.Combobox(cell, state="readonly", font=("Segoe UI", 10))
            cb.pack(fill="x")
            self.combos[name] = cb

        self.predict_btn = ttk.Button(pad, text="Triage this patient",
                                      style="Accent.TButton",
                                      state="disabled", command=self._do_predict)
        self.predict_btn.pack(fill="x", pady=(14, 0))

        # Read the complaint back aloud, slowly. In a noisy resus room a
        # nurse cannot always read a screen, and 0.8x is deliberate rather
        # than decorative - it is the rate at which a clinical phrase stays
        # intelligible over background noise.
        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        self.speak_btn = ttk.Button(row, text="Speak complaint  (0.8x)",
                                    command=self._do_speak)
        self.speak_btn.pack(side="left")
        self._last_spoken = None
        tk.Label(row, text="offline, via eSpeak NG", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(10, 0))

        # ---- result card ----
        res = card(right)
        res.pack(fill="both", expand=True)
        rpad = tk.Frame(res, bg=CARD)
        rpad.pack(fill="both", expand=True, padx=16, pady=14)

        heading(rpad, "Result").pack(fill="x")
        self._deployed_banner(
            rpad, "This tab's predictions come from:").pack(fill="x", pady=(6, 0))

        # No fixed height: at high DPI the two stacked labels are taller than
        # any hard-coded value, and the subtitle gets clipped.
        self.level_banner = tk.Frame(rpad, bg="#e9edf1")
        self.level_banner.pack(fill="x", pady=(10, 4))
        self.level_text = tk.Label(self.level_banner, text="—",
                                   bg="#e9edf1", fg=MUTED,
                                   font=("Segoe UI Semibold", 19))
        self.level_text.pack(pady=(14, 0))
        self.level_sub = tk.Label(self.level_banner, text="Enter a patient and press Triage",
                                  bg="#e9edf1", fg=MUTED, font=("Segoe UI", 9),
                                  wraplength=520)
        self.level_sub.pack(pady=(2, 14))

        body(rpad, "Confidence across triage levels", fg=MUTED,
             size=9).pack(fill="x", pady=(10, 4))
        self.proba_canvas = tk.Canvas(rpad, height=118, bg=CARD,
                                      highlightthickness=0)
        self.proba_canvas.pack(fill="x")
        # Redraw on resize so the bars track the window instead of keeping
        # the width they happened to have when the prediction was made.
        self._last_proba = None
        self._cluster_vectors = None
        self._last_batch_results = None
        self._last_cluster_result = None
        self._cluster_labels = []
        self.proba_canvas.bind(
            "<Configure>",
            lambda e: self._draw_proba(*self._last_proba) if self._last_proba else None)

        body(rpad, "What the text pipeline did", fg=MUTED,
             size=9).pack(fill="x", pady=(12, 4))
        # Scrollbar, because this panel outgrew its box. It now prints six
        # pipeline stages at two lines each, the dropped stop words, which
        # stage the model was actually fed, and any input-quality warning -
        # comfortably past 15 lines. Without a scrollbar the tail was simply
        # invisible, which is how "the sentence-transformer received step 5"
        # ended up cut off at the bottom edge.
        stages_wrap = tk.Frame(rpad, bg=CARD)
        stages_wrap.pack(fill="both", expand=True)
        stages_sb = ttk.Scrollbar(stages_wrap, orient="vertical")
        self.stages = tk.Text(stages_wrap, height=7, font=("Consolas", 9),
                              wrap="word", relief="solid", bd=1,
                              bg="#fbfcfd", state="disabled",
                              yscrollcommand=stages_sb.set)
        stages_sb.config(command=self.stages.yview)
        stages_sb.pack(side="right", fill="y")
        self.stages.pack(side="left", fill="both", expand=True)

    def _set_complaint(self, text):
        self.complaint.delete("1.0", "end")
        self.complaint.insert("1.0", text)

    def _fill_dropdowns(self):
        mapping = {
            # From the SERVING bundle. The two bundles' categorical
            # vocabularies happen to be identical today, so this is latent
            # rather than active - but a dropdown offering a category the
            # scoring model never saw would silently fall back to a default
            # and the operator would never know which value was used.
            "Gender": self.active_artifacts()["le_gender"],
            "Mode_of_Arrival": self.active_artifacts()["le_mode"],
            "AVPU": self.active_artifacts()["le_avpu"],
            "ECG_Status": self.active_artifacts()["le_ecg"],
        }
        # ECG_Status defaults to "Normal", NOT to an infarct pattern.
        #
        # THE BUG THIS FIXES: the form opened with "ST elevation" preselected,
        # so anyone who typed a complaint and pressed Predict without opening
        # the dropdown got EMERGENCY whatever they had written. Measured:
        # "halka sa seena mein dabao hai, rest se theek ho jata hai" (mild
        # pressure, resolves with rest) returned Level 1 at 63.7% on the old
        # defaults and Level 4 at 81.0% once the ECG was set to Normal. The
        # model was reading the complaint correctly the whole time; the form
        # was answering for it.
        #
        # A pre-filled emergency ECG is the wrong direction to be wrong in on
        # a screen whose output is a triage level, so the safe reading is the
        # default and any abnormality is a deliberate act by the operator.
        defaults = {"Gender": "Male", "Mode_of_Arrival": "Ambulance",
                    "AVPU": "A", "ECG_Status": "Normal"}
        for name, enc in mapping.items():
            values = list(enc.classes_)
            self.combos[name]["values"] = values
            want = defaults.get(name)
            self.combos[name].set(want if want in values else values[0])

    def _do_speak(self):
        """Read the standardised English aloud, or the raw complaint.

        Prefers the translation when one exists, since that is the text the
        model actually scored. Never raises - a missing audio device must
        not take the window with it.
        """
        try:
            from src.offline_pipeline import speak, tts_available
        except Exception as e:
            messagebox.showerror("Speech unavailable",
                                 f"Could not load the speech module:\n{e}")
            return
        text = self._last_spoken or self.complaint.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to speak",
                                "Enter a complaint, or run a triage first.")
            return
        if not tts_available():
            messagebox.showerror(
                "eSpeak NG not installed",
                "Offline speech needs eSpeak NG.\n\nInstall it with:\n"
                "    sudo dnf install espeak-ng")
            return
        ok, msg = speak(text)
        self.status.set(f"Speaking: {text[:48]}..." if ok
                        else f"Speech failed: {msg}")
        if not ok:
            messagebox.showerror("Speech failed", msg)

    def _do_predict(self):
        from triage_pipeline import predict_one

        text = self.complaint.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No complaint", "Please enter a complaint.")
            return

        numbers = {}
        for name, var in self.fields.items():
            value = fnum(var.get())
            if value is None:
                messagebox.showwarning(
                    "Invalid number",
                    f"'{var.get()}' is not a valid value for "
                    f"{name.replace('_', ' ')}.")
                return
            numbers[name] = value

        original_text = text
        english_text = None
        self._last_similarity = None
        self._last_spoken = text
        if self.in_english_mode():
            self.status.set("translating locally via Ollama...")
            self.update_idletasks()
            english_text, err = self.translate_for_mode(text)
            if err:
                # A gate block and a dead translator arrive through the same
                # return value and are completely different events. The old
                # title called both "English (Local LLM) mode failed", which
                # names a mode that no longer exists and reads as the app
                # breaking - when a gate block is the app doing its job.
                # THREE different events arrive through this one return
                # value and must not share a message. A refusal by design
                # ("this is not a complaint", "the body part changed") is the
                # app working; a dead translator is the app unable to work.
                # Calling all three "failed" taught the operator to retry
                # until something got through, which is the opposite of what
                # a safety refusal should invite.
                if err.startswith("Anatomical check failed"):
                    title = "Refused: the translation changed the body part"
                    short = ("Refused - the English named a body part the "
                             "complaint did not. No prediction made.")
                    messagebox.showwarning(title, err)
                elif err.startswith("This complaint is not in the Latin"):
                    title = "Script not supported"
                    short = ("Refused - Urdu script is not supported. Type "
                             "the complaint in Roman Urdu.")
                    messagebox.showwarning(title, err)
                elif err.startswith("This does not look like a complaint"):
                    title = "Not a complaint"
                    short = ("Refused - no symptom or body part in the text. "
                             "No prediction made.")
                    messagebox.showwarning(title, err)
                else:
                    title = "Cannot translate this complaint"
                    short = "Translation unavailable - no prediction made."
                    messagebox.showerror(title, err)
                self.status.set(short)
                return
            text = english_text
            self._last_spoken = english_text

            # The cosine fallback that used to live here is gone. It never
            # ran: it called get_text_encoder() without importing it, and
            # its own except-clause swallowed the NameError, so for its
            # entire life this block computed nothing and fell back never.
            # It is not restored, because the number it gated on could not
            # tell a correct translation (0.8054) from "My leg is broken
            # after a fall" (0.7922). The anatomical gate in
            # translate_for_mode() has already blocked drift by this point.
            #
            # Similarity is still computed, for display only, and this time
            # with the encoder imported.
            try:
                import numpy as _np
                from triage_pipeline import get_text_encoder
                enc = get_text_encoder(self.active_artifacts())
                v = enc.encode([original_text, english_text],
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
                self._last_similarity = float(_np.dot(v[0], v[1]))
            except Exception:
                # Display-only: never let it take the prediction down.
                self._last_similarity = None

        input_warnings = []
        try:
            level, confidence, proba = predict_one(
                self.active_artifacts(), text,
                numbers["Age"], numbers["Heart_Rate"], numbers["Systolic_BP"],
                numbers["Diastolic_BP"], numbers["Temperature"], numbers["SpO2"],
                self.combos["Gender"].get(), self.combos["Mode_of_Arrival"].get(),
                self.combos["AVPU"].get(), self.combos["ECG_Status"].get(),
                warnings=input_warnings)
        except Exception:
            messagebox.showerror("Prediction failed", traceback.format_exc())
            return

        colour = LEVEL_COLOURS[level]
        self.level_banner.configure(bg=colour)
        self.level_text.configure(
            bg=colour, fg="white",
            text=f"Level {level + 1}  -  {LEVEL_NAMES[level]}")
        self.level_sub.configure(
            bg=colour, fg="#f2f6fa",
            text=f"{LEVEL_BLURB[level]}    ·    confidence {confidence * 100:.1f}%")

        self._draw_proba(proba, level, shown_confidence=confidence)

        # The stage list must name every stage that ran. It used to skip the
        # dictionary pass entirely and assert that "the Roman Urdu dictionary
        # and stop-word stages do not apply" - half right. The stop-word list
        # does not apply; the dictionary does, and runs before the translator.
        from src.offline_pipeline import fuzzy_normalize_roman_urdu
        normalized = fuzzy_normalize_roman_urdu(original_text, verbose=False)

        self.stages.config(state="normal")
        self.stages.delete("1.0", "end")
        self.stages.insert("end", "0. raw input (as typed)\n", "h")
        self.stages.insert("end", f"   {original_text}\n")
        self.stages.insert("end", "1. dictionary + fuzzy normalization"
                                  f"{'' if normalized != original_text else '   (no change)'}\n", "h")
        self.stages.insert("end", f"   {normalized}\n")
        self.stages.insert("end", "2. local Ollama translation\n", "h")
        self.stages.insert("end", f"   {english_text}\n")
        self.stages.insert("end", "3. anatomical gate\n", "h")
        self.stages.insert("end", "   passed - every body part named in the\n"
                                  "   complaint survives into the English\n")
        self.stages.insert("end", "4. sentence-transformer\n", "h")
        self.stages.insert("end", "   English encoded directly - the learned\n"
                                  "   stop-word list does not apply to this bundle\n")
        # Input-quality warnings. These lived only in the branch below,
        # which is now unreachable; a capped confidence with no stated
        # reason reads as a weak case rather than a bad input.
        if input_warnings:
            self.stages.insert("1.0", "!! " + "\n!! ".join(input_warnings) + "\n\n")
        self.stages.tag_config("h", foreground=ACCENT,
                               font=("Consolas", 9, "bold"))
        self.stages.config(state="disabled")
        self.status.set(
            f"Level {level + 1} ({LEVEL_NAMES[level]}) at "
            f"{confidence * 100:.1f}% confidence."
            + ("   |  " + input_warnings[0].split(':')[0] if input_warnings else ""))
        return


    def _draw_proba(self, proba, chosen, shown_confidence=None):
        """Bars for every class, plus a note when the headline was capped.

        The bars are the model's RAW output and stay that way - redrawing
        them to match a capped headline would misreport what the classifier
        said. But leaving the contradiction unexplained was worse: a
        complaint with no usable text showed "confidence 50.0%" above a bar
        labelled 94.0%, and the cap exists precisely so a blank complaint
        cannot look confident.
        """
        self._last_proba = (proba, chosen, shown_confidence)
        c = self.proba_canvas
        c.delete("all")
        width = c.winfo_width()
        if width < 60:                      # not laid out yet; Configure will call back
            return
        # Percentages are anchored to the RIGHT edge, so they can never run off
        # the canvas however narrow the pane gets. The bar takes what is left.
        right_pad = 58
        label_w = min(96, max(0, width - right_pad - 60))
        bar_w = max(20, width - label_w - right_pad)
        for i, p in enumerate(proba):
            y = 8 + i * 28
            if label_w > 40:
                c.create_text(0, y + 8, anchor="w", text=f"L{i + 1} {LEVEL_NAMES[i]}",
                              font=("Segoe UI", 8), fill=MUTED)
            c.create_rectangle(label_w, y, label_w + bar_w, y + 16,
                               fill="#eef1f4", outline="")
            filled = max(2, int(bar_w * float(p)))
            capped_here = (i == chosen and shown_confidence is not None
                           and float(p) - shown_confidence > 1e-6)
            c.create_rectangle(label_w, y, label_w + filled, y + 16,
                               fill=LEVEL_COLOURS[i], outline="")
            c.create_text(width - 2, y + 8, anchor="e",
                          text=f"{float(p) * 100:.1f}%",
                          font=("Segoe UI Semibold" if i == chosen else "Segoe UI", 9),
                          fill=INK if i == chosen else MUTED)
            if capped_here:
                # Say WHY the headline disagrees with this bar, on the bar.
                c.create_text(label_w + 6, y + 8, anchor="w",
                              text=f"raw {float(p) * 100:.1f}%  →  reported "
                                   f"{shown_confidence * 100:.1f}% (capped: "
                                   f"no usable complaint text)",
                              font=("Segoe UI", 8), fill="#7a2f28")

    # =================================================================
    # TAB 2 - Pipeline explorer
    # =================================================================
    def _build_pipeline_tab(self):
        root = self.tab_pipeline
        wrap = card(root)
        wrap.pack(fill="both", expand=True, padx=4, pady=6)
        pad = tk.Frame(wrap, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Text pipeline explorer").pack(fill="x")
        body(pad,
             "Type any complaint and see it move through every stage. This is the same "
             "code path training and prediction use, so what you see here is exactly what "
             "the model receives.",
             fg=MUTED, size=9, wraplength=980).pack(fill="x", pady=(2, 12))

        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x")
        self.explore_var = tk.StringVar(
            value="subah se pait mein tez dard hai lekin bukhar nahi")
        entry = tk.Entry(row, textvariable=self.explore_var,
                         font=("Segoe UI", 12), relief="solid", bd=1)
        entry.pack(side="left", fill="x", expand=True, ipady=4)
        entry.bind("<Return>", lambda e: self._do_explore())
        ttk.Button(row, text="Analyse", style="Accent.TButton",
                   command=self._do_explore).pack(side="left", padx=(8, 0))

        self.explore_out = tk.Frame(pad, bg=CARD)
        self.explore_out.pack(fill="both", expand=True, pady=(16, 0))

    def _do_explore(self):
        """Show the five stages a complaint actually passes through.

        This tab drifted badly from the code it claims to document. It had
        two branches, one per mode; with the toggle gone the Roman Urdu
        branch became unreachable and its ~60 lines described a pipeline
        nobody could run. Worse, the surviving branch told the operator the
        "Roman Urdu dictionary, fuzzy matching and learned stop-word stages
        are skipped" - which stopped being true the moment
        fuzzy_normalize_roman_urdu() was wired in ahead of the translator.
        A pipeline explorer that misreports the pipeline is worse than no
        explorer, because it is believed.

        Every stage below is produced by calling the SAME function the
        serving path calls, never a re-implementation.
        """
        from src.offline_pipeline import (fuzzy_normalize_roman_urdu,
                                          verify_anatomical_integrity)

        for w in self.explore_out.winfo_children():
            w.destroy()

        raw = self.explore_var.get().strip()
        if not raw:
            body(self.explore_out,
                 "Type a complaint above and press Analyse. "
                 "There is nothing to run the pipeline on yet.",
                 fg=MUTED, size=9).pack(fill="x")
            return

        def panel(title, text, note, tone=None):
            blk = tk.Frame(self.explore_out, bg=CARD)
            blk.pack(fill="x", pady=(0, 10))
            head = tk.Frame(blk, bg=CARD)
            head.pack(fill="x")
            tk.Label(head, text=title, bg=CARD, fg=ACCENT,
                     font=("Segoe UI Semibold", 10), anchor="w").pack(side="left")
            if tone:
                label, ok = tone
                tk.Label(head, text=label,
                         bg="#eaf3ea" if ok else "#f7e9e7",
                         fg="#1e5c2e" if ok else "#a33228",
                         font=("Segoe UI", 8), padx=6).pack(side="right")
            tk.Label(blk, text=text or "(empty)", bg="#fbfcfd", fg=INK,
                     font=("Consolas", 11), anchor="w", justify="left",
                     wraplength=1000, padx=10, pady=7, relief="solid",
                     bd=1).pack(fill="x", pady=(2, 1))
            tk.Label(blk, text=note, bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                     anchor="w", justify="left",
                     wraplength=1000).pack(fill="x")

        panel("0  Raw input (as typed)", raw,
              "exactly what the triage nurse typed - nothing has run yet")

        # 1. dictionary + fuzzy, the stage the tab used to deny existed
        normalized = fuzzy_normalize_roman_urdu(raw, verbose=False)
        changed = normalized != raw
        panel("1  Dictionary + fuzzy normalization", normalized,
              "Local and deterministic, before Ollama sees anything. Spelling "
              "variants collapse onto one canonical Roman Urdu token "
              "(\"payt\" -> \"pait\"). Fuzzy matching is deliberately timid: "
              "cutoff 0.88, a 4-character minimum and a blocklist, because at "
              "0.80 it rewrote 1,619 words in this project's own corpus - "
              "\"peene\" (to drink) became \"seene\" (chest).",
              ("changed this text" if changed else "ran - nothing to change",
               True))

        # 2. translation. allow_blocked=True on purpose: a gate rejection is
        # NOT a translation failure, and reporting it as one made this tab -
        # whose entire job is saying which stage failed - blame the wrong
        # stage and skip rendering the stage that actually rejected the row.
        en, err = self.translate_for_mode(raw, allow_blocked=True)
        if err:
            panel("2  Ollama translation  -  FAILED", err,
                  "The pipeline stops here. No prediction is made, and no "
                  "other model answers in its place.", ("failed", False))
            return
        panel("2  Local Ollama translation", en,
              "translated on this machine by Ollama on localhost - no network "
              "call. temperature 0, so the same complaint gives the same "
              "English every run.", ("translated", True))

        # 3. the gate, reported whichever way it went. This panel is the
        #    only place a blocked translation is shown beside the English
        #    that caused it.
        ok, failures = verify_anatomical_integrity(normalized, en)
        panel("3  Anatomical assertion gate",
              "PASSED - every body part named in the complaint survives "
              "into the English translation."
              if ok else "BLOCKED\n" + "\n".join(failures),
              "Deterministic, and this is the decision. It replaced cosine "
              "similarity, which could not tell a correct translation "
              "(0.8054) from \"My leg is broken after a fall\" (0.7922). "
              "The gate checks anatomy only - it cannot see a wrong symptom "
              "attached to the right body part.",
              ("passed" if ok else "blocked", ok))

        # 4. encoding
        man = self.active_manifest() or {}
        enc = man.get("embedding_model") or "sentence-transformer"
        # The previous wording claimed the stop-word list was NOT applied
        # here. It is. skip_normalization skips the Roman Urdu DICTIONARY
        # stages, not stop-word removal - the bundle carries its own list and
        # build_text_features() applies it on every prediction.
        n_stops = len(self.active_artifacts().get("stopwords") or [])
        panel("4  Sentence-transformer", en,
              f"The English text is preprocessed using the bundle's "
              f"{n_stops}-token English stop-word list and encoded directly "
              f"via {enc}. The Roman Urdu dictionary pipeline is skipped to "
              f"prevent train/serve skew - serving this bundle any other way "
              f"cost 38 points of accuracy once already.", ("encoded", True))

    def _explain_kept_fillers(self, final_text):
        if not self.stopword_report:
            return
        stats = {t["token"]: t for t in self.stopword_report["token_statistics"]}
        kept = [stats[w] for w in dict.fromkeys(final_text.split())
                if w in stats and not stats[w]["is_stopword"]
                and not stats[w]["clinically_protected"]]
        if not kept:
            return

        box = tk.Frame(self.explore_out, bg=CARD)
        box.pack(fill="x", pady=(10, 0))
        tk.Label(box, text="Common words that were KEPT, and why",
                 bg=CARD, fg=ACCENT, font=("Segoe UI Semibold", 10),
                 anchor="w").pack(fill="x")
        tk.Label(box,
                 text=('This step removes LEARNED stop words, not every filler '
                       'word. A frequent word is only dropped when the data says '
                       'it is unrelated to the triage level.'),
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w",
                 wraplength=980, justify="left").pack(fill="x")
        tree = ttk.Treeview(box, columns=("df", "mi", "v", "why"),
                            show="tree headings", height=min(6, len(kept)))
        tree.heading("#0", text="token")
        tree.column("#0", width=150)
        for col, label, w in [("df", "doc frequency", 110),
                              ("mi", "normalized MI", 110),
                              ("v", "Cramer's V", 90),
                              ("why", "why it stayed", 380)]:
            tree.heading(col, text=label)
            tree.column(col, width=w,
                        anchor="w" if col == "why" else "center")
        for s in kept:
            tree.insert("", "end", text=s["token"], values=(
                f"{s['document_frequency']:.3f}",
                f"{s['normalized_mutual_information']:.5f}",
                f"{s['cramers_v']:.3f}",
                keep_reason(s, self.stopword_report["thresholds"])))
        tree.pack(fill="x", pady=(4, 0))

    # =================================================================
    # TAB 3 - Stop words
    # =================================================================
    def _build_stopwords_tab(self):
        root = self.tab_stops
        wrap = card(root)
        wrap.pack(fill="both", expand=True, padx=4, pady=6)
        pad = tk.Frame(wrap, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Contribution 1  -  automatically learned stop words").pack(fill="x")
        self.stop_summary = body(
            pad, "Loading...", fg=MUTED, size=9, wraplength=1000)
        self.stop_summary.pack(fill="x", pady=(2, 6))

        filt = tk.Frame(pad, bg=CARD)
        filt.pack(fill="x", pady=(4, 6))
        body(filt, "Show:", fg=MUTED, size=9).pack(side="left", padx=(0, 6))
        self.stop_filter = tk.StringVar(value="stop")
        for value, label in [("all", "all tested tokens"),
                             ("stop", "removed (learned stop words)"),
                             ("kept", "kept (everything else)"),
                             ("protected", "rescued by the clinical guard")]:
            tk.Radiobutton(filt, text=label, value=value,
                           variable=self.stop_filter, bg=CARD, fg=INK,
                           font=("Segoe UI", 9), activebackground=CARD,
                           selectcolor=CARD,
                           command=self._populate_stopwords).pack(side="left", padx=6)

        cols = ("df", "count", "mi", "v", "chi", "p", "verdict")
        self.stop_tree = ttk.Treeview(pad, columns=cols, show="tree headings")
        self.stop_tree.heading("#0", text="token")
        self.stop_tree.column("#0", width=160)
        for col, label, w in [("df", "doc frequency", 105), ("count", "documents", 80),
                              ("mi", "normalized MI", 150),
                              ("v", "Cramer's V", 130), ("chi", "chi-square", 90),
                              ("p", "p-value (info only)", 110),
                              ("verdict", "decision", 380)]:
            self.stop_tree.heading(col, text=label)
            self.stop_tree.column(col, width=w, anchor="center")
        self.stop_tree.column("verdict", anchor="w")

        sb = ttk.Scrollbar(pad, orient="vertical", command=self.stop_tree.yview)
        self.stop_tree.configure(yscrollcommand=sb.set)
        self.stop_tree.pack(side="left", fill="both", expand=True, pady=(4, 0))
        sb.pack(side="right", fill="y", pady=(4, 0))

        self.stop_tree.tag_configure("stop", foreground="#c0392b")
        self.stop_tree.tag_configure("prot", foreground="#2e9e5b")

    def _populate_stopwords(self):
        if not self.stopword_report:
            return
        r = self.stopword_report
        t = r["thresholds"]
        c = r["corpus"]
        review = r.get("review_recommended") or []
        # This early-returned before filling the table, on the belief that
        # stop-word removal did not apply once complaints were translated.
        # It does. skip_normalization skips the Roman Urdu DICTIONARY stages,
        # not stop-word removal - the serving bundle carries its own list and
        # build_text_features() applies it on every prediction. The return
        # left the table permanently empty; the tab rendered its headers and
        # nothing else, which reads as "no stop words" rather than "not
        # drawn".
        n_removed = len(r["stopwords"])
        self.stop_summary.configure(text=(
            f"These {n_removed} English stop-words are removed from translated "
            f"complaints on the active serving path before sentence-transformer "
            f"encoding.\n"
            f"A token is removed only when ALL THREE hold:  document frequency "
            f">= {t['effective_df_cutoff']:.4f}"
            f"   AND   normalized mutual information <= {t['mi_threshold']}"
            f"   AND   Cramer's V <= {t['cramers_v_threshold']}\n"
            f"Read from {ENGLISH_MODEL_DIR}/learned_stopwords.json - the "
            f"serving bundle's own list, not the project-root file, which "
            f"belongs to whichever model trained last.\n"
            f"{c['n_documents']} complaints, {c['n_unique_tokens']} unique tokens, "
            f"{c['n_tokens_tested']} high-frequency tokens tested, "
            f"{r['n_stopwords']} learned as stop words: "
            f"{', '.join(r['stopwords'])}.\n"
            "The table shows the removed tokens by default. Switch to 'all "
            "tested tokens' to see the ones that were tested and KEPT - the "
            "'decision' column names the criterion that saved each one. The "
            "chi-square statistic and p-value are shown for transparency only: "
            "the decision uses Cramer's V (effect size), which does not shrink "
            "as the dataset grows the way p-values do."
            + (f"\nNeeds clinician sign-off (negation / intensity): "
               f"{', '.join(review)}" if review else "")))

        for row in self.stop_tree.get_children():
            self.stop_tree.delete(row)

        mode = self.stop_filter.get()
        for s in r["token_statistics"]:
            if mode == "stop" and not s["is_stopword"]:
                continue
            if mode == "kept" and s["is_stopword"]:
                continue
            if mode == "protected" and not (s["clinically_protected"]
                                            and s["is_uninformative"]):
                continue
            verdict = keep_reason(s, t)
            if s["is_stopword"]:
                tag = "stop"
            elif s["clinically_protected"] and s["is_uninformative"]:
                tag = "prot"
            else:
                tag = ""
            # Per-criterion PASS/FAIL, so a row explains itself without the
            # reader having to hold three thresholds in their head. Chi-square
            # and its p-value carry no flag: they are context, not criteria.
            mi_flag = "pass" if s["normalized_mutual_information"] <= t["mi_threshold"] else "FAIL"
            v_flag = "pass" if s["cramers_v"] <= t["cramers_v_threshold"] else "FAIL"
            self.stop_tree.insert("", "end", text=s["token"], tags=(tag,), values=(
                f"{s['document_frequency']:.4f}", s["document_count"],
                f"{s['normalized_mutual_information']:.5f}  ({mi_flag})",
                f"{s['cramers_v']:.4f}  ({v_flag})",
                f"{s['chi_square']:.2f}",
                f"{s['chi_square_p_value']:.4f}",
                verdict))

    # =================================================================
    # TAB 4 - Batch file
    # =================================================================
    def _build_batch_tab(self):
        root = self.tab_batch
        top = card(root)
        top.pack(fill="x", padx=4, pady=(6, 0))
        pad = tk.Frame(top, bg=CARD)
        pad.pack(fill="x", padx=18, pady=14)

        heading(pad, "Batch triage from a file").pack(fill="x")
        body(pad,
             "Pick an Excel (.xlsx) or CSV file of patients. Only Complaint_Text is "
             "required; missing numbers fall back to the training average and unknown "
             "categories fall back safely, with a note recorded per row. CSV files are "
             "read with encoding detection (utf-8, utf-8-sig, cp1252, latin-1), so a "
             "sheet exported from Excel on Windows loads instead of failing.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 10))

        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x")
        self.batch_path = tk.StringVar(
            value=resolve_project_file("sample_100_patients.xlsx"))
        tk.Entry(row, textvariable=self.batch_path, font=("Segoe UI", 10),
                 relief="solid", bd=1).pack(side="left", fill="x",
                                            expand=True, ipady=3)
        ttk.Button(row, text="Browse...",
                   command=self._browse_batch).pack(side="left", padx=6)
        # Disabled until the model is loaded, like the Triage tab's button.
        # It used to be live from the moment the window painted, so clicking it
        # during the few seconds the sentence-transformer takes to load called
        # predict_dataframe(None, df) and threw a raw TypeError traceback in a
        # dialog box.
        self.batch_btn = ttk.Button(row, text="Run batch triage",
                                    style="Accent.TButton", state="disabled",
                                    command=self._do_batch)
        self.batch_btn.pack(side="left")
        # Disabled until a run produces something. An export button that is
        # always live invites a click that can only ever say "nothing to
        # export".
        self.batch_export_btn = ttk.Button(row, text="Export CSV...",
                                           state="disabled",
                                           command=self._export_batch_csv)
        self.batch_export_btn.pack(side="left", padx=(8, 0))

        self._deployed_banner(
            pad, "Every row is triaged by:").pack(fill="x", pady=(10, 0))

        # Progress furniture. Hidden until a run starts: an idle progress bar
        # sitting at zero reads as a job that has stalled.
        self.batch_progress_box = tk.Frame(pad, bg=CARD)
        self.batch_progress_label = body(
            self.batch_progress_box,
            "Translating complaints locally via Ollama... Please wait, "
            "processing patient records.", fg=MUTED, size=9)
        self.batch_progress_label.pack(fill="x")
        self.batch_bar = ttk.Progressbar(self.batch_progress_box,
                                         mode="determinate", maximum=100)
        self.batch_bar.pack(fill="x", pady=(4, 0))

        self.batch_summary = body(pad, "", fg=INK, size=9)
        self.batch_summary.pack(fill="x", pady=(10, 0))

        bottom = card(root)
        bottom.pack(fill="both", expand=True, padx=4, pady=6)
        bpad = tk.Frame(bottom, bg=CARD)
        bpad.pack(fill="both", expand=True, padx=18, pady=14)

        # "complaint" now shows the ORIGINAL text, with the translation in
        # its own column beside it. Showing only the translation - which is
        # what Complaint_Text holds after the worker overwrites it - meant
        # the operator could not find their own row by reading it.
        cols = ("english", "level", "label", "conf", "gate", "notes")
        self.batch_tree = ttk.Treeview(bpad, columns=cols, show="tree headings")
        self.batch_tree.heading("#0", text="complaint (as typed)")
        self.batch_tree.column("#0", width=260)
        for col, label, w in [("english", "translation", 240),
                              ("level", "level", 55), ("label", "label", 110),
                              ("conf", "confidence", 85),
                              ("gate", "anatomical gate", 115),
                              ("notes", "notes", 200)]:
            self.batch_tree.heading(col, text=label)
            self.batch_tree.column(col, width=w,
                                   anchor="w" if col == "notes" else "center")
        sb = ttk.Scrollbar(bpad, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=sb.set)
        self.batch_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for i, colour in enumerate(LEVEL_COLOURS):
            self.batch_tree.tag_configure(f"L{i}", foreground=colour)

    def _browse_batch(self):
        path = filedialog.askopenfilename(
            title="Choose a patient file",
            filetypes=[("Excel or CSV", "*.xlsx *.xls *.csv"),
                       ("All files", "*.*")])
        if path:
            self.batch_path.set(path)

    # ---------------------------------------------------------------- export
    def _ask_save_csv(self, suggested):
        """Save dialog for a CSV. Returns a path, or None if cancelled."""
        return filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            initialfile=suggested,
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")])

    def _export_batch_csv(self):
        """Write the batch results exactly as shown, plus what is behind them.

        The batch worker already drops a _predictions.csv beside the input
        file, which is fine when the operator owns that folder and knows to
        look. It is no use when the sheet came from a shared drive, a
        read-only mount, or Downloads - and it never asks where to put it.
        This does.

        Everything on screen is exported, INCLUDING the rows that were not
        scored. A file that quietly contained only the successes would read
        as a complete triage of the input, which is the same failure the
        summary line was changed to avoid.
        """
        results = getattr(self, "_last_batch_results", None)
        if results is None or not len(results):
            messagebox.showinfo(
                "Nothing to export",
                "Run a batch first. There are no results to write yet.")
            return
        src = os.path.splitext(os.path.basename(
            getattr(self, "_batch_target", "batch")))[0]
        path = self._ask_save_csv(f"{src}_triage.csv")
        if not path:
            return
        try:
            # utf-8-sig, not utf-8: Excel on Windows reads a plain UTF-8 CSV
            # as cp1252 and turns every non-ASCII character into mojibake.
            # The BOM is what tells it otherwise, and these files are full of
            # Roman Urdu.
            results.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:
            messagebox.showerror("Export failed",
                                 f"Could not write {path}:\n\n{e}")
            return
        scored = int(results["Predicted_Triage_Level"].notna().sum()) \
            if "Predicted_Triage_Level" in results else len(results)
        messagebox.showinfo(
            "Exported",
            f"{len(results)} rows written to\n{path}\n\n"
            f"{scored} scored, {len(results) - scored} not scored (blocked by "
            f"the anatomical gate, or not translated). Unscored rows are "
            f"included with their reason in Gate_Status and Gate_Detail - "
            f"leaving them out would make the file read as a complete triage "
            f"of the input.")
        self.status.set(f"Exported {len(results)} rows to "
                        f"{os.path.basename(path)}")

    def _export_cluster_csv(self):
        """Write the similarity matrix and the complaints behind it.

        The heatmap is for reading; this is for keeping. Both halves go in:
        the S-labels alone would be unusable a week later, so each row
        carries its Roman Urdu original and its English translation beside
        its similarities.
        """
        res = getattr(self, "_last_cluster_result", None)
        if not res or res.get("matrix") is None:
            messagebox.showinfo(
                "Nothing to export",
                "Run the cluster analysis first. There is no matrix yet.")
            return
        path = self._ask_save_csv("cluster_similarity.csv")
        if not path:
            return
        S = res["matrix"]
        sents = res["sentences"]
        n = len(sents)
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                # A header block, because a bare matrix in a spreadsheet a
                # month from now is unreadable without knowing what produced
                # it or what the numbers mean.
                w.writerow(["# Cluster similarity matrix"])
                w.writerow(["# encoder", res.get("encoder", "")])
                w.writerow(["# vectors", f"{n} x 384, L2-normalised"])
                w.writerow(["# cosine", "dot product of unit vectors; "
                                        "1.00 = identical, 0.00 = unrelated"])
                w.writerow(["# diagonal_ok", res.get("diagonal_ok", "")])
                w.writerow(["# mean_similarity", res.get("mean_similarity", "")])
                w.writerow([])
                w.writerow(["id", "roman_urdu", "english", "translated_ok"]
                           + [f"S{j + 1}" for j in range(n)])
                for i, sent in enumerate(sents):
                    w.writerow(
                        [f"S{i + 1}", sent["raw"], sent["translated"] or "",
                         sent["translated_ok"]]
                        + [f"{float(S[i][j]):.4f}" for j in range(n)])
        except Exception as e:
            messagebox.showerror("Export failed",
                                 f"Could not write {path}:\n\n{e}")
            return
        messagebox.showinfo(
            "Exported",
            f"{n} complaints and their {n}x{n} similarity matrix written to\n"
            f"{path}\n\nEach row carries its Roman Urdu original and its "
            f"English translation, so the numbers can still be read months "
            f"from now without this window open.")
        self.cluster_status.set(f"exported to {os.path.basename(path)}")

    def _do_batch(self):
        path = self.batch_path.get().strip()
        if not os.path.exists(path):
            messagebox.showwarning("File not found", f"No such file:\n{path}")
            return
        self._batch_target = path
        # Translation is ~11s per row and strictly serial, so a 100-row sheet
        # is ~18 minutes. The old single status line never changed in that
        # time, which is indistinguishable from a hang - the most common
        # reason this tab gets reported as broken. The worker publishes a
        # counter and a poller on the UI thread reads it; Tk is not
        # thread-safe, so the worker must never touch a widget itself.
        self._batch_progress = {"done": 0, "total": 0, "text": "reading file",
                                "finished": False}
        # Disable the trigger for the duration. Two overlapping batch runs
        # write to the same _predictions.csv and interleave rows in the same
        # tree, and the second one silently wins.
        self.batch_btn.config(state="disabled")
        self.batch_progress_box.pack(fill="x", pady=(8, 0))
        self.batch_bar.configure(value=0, maximum=100)
        self.batch_progress_label.configure(
            text="Translating complaints locally via Ollama... Please wait, "
                 "processing patient records.")
        self._run_async(self._batch_worker, f"Triaging {os.path.basename(path)}...")
        self._poll_batch_progress()

    def _poll_batch_progress(self):
        """Render the worker's counter. Runs on the UI thread by design.

        Tk is not thread-safe, so the worker publishes plain ints into a dict
        and every widget touch happens here.
        """
        pr = getattr(self, "_batch_progress", None)
        if not pr:
            return
        total, done = pr.get("total", 0), pr.get("done", 0)
        if total:
            secs = int((total - done) * SECONDS_PER_ROW)
            eta = f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"
            self.batch_bar.configure(maximum=total, value=done)
            self.batch_progress_label.configure(
                text=f"Processing patient {min(done + 1, total)} of {total}...  "
                     f"about {eta} left.\n"
                     f"Translating complaints locally via Ollama - each row is "
                     f"one call to the local model, so this is slow by nature. "
                     f"Please wait.")
            self.status.set(f"Translating row {done}/{total}  -  about {eta} "
                            f"left  |  {pr.get('text', '')[:48]}")
        if pr.get("finished"):
            return
        self.after(300, self._poll_batch_progress)

    def _stage_columns(self, originals, translations):
        """Every intermediate stage, per row, for the results table and CSV.

        The batch used to export the translation under the column name
        "Complaint_Text" and nothing else about how it got there. Two
        problems with that. The operator's ORIGINAL text was overwritten and
        gone - what the nurse actually typed did not survive into the file
        at all - and a reviewer asking "why did this row score that way" had
        no stage to look at between the raw input and the level.

        Every column below is a real stage, computed with the SAME function
        the serving path uses. Nothing here re-implements the pipeline; a
        column that drifted from what actually ran would be worse than no
        column.
        """
        from src.offline_pipeline import fuzzy_normalize_roman_urdu
        from stopwords import remove_stopwords

        art = self.active_artifacts()
        man = art.get("manifest", {}) or {}
        own_stops = art.get("stopwords") or set()
        skip_norm = bool(man.get("skip_normalization"))

        normalized, encoded, dropped = [], [], []
        for raw, en in zip(originals, translations):
            # Stage 1: dictionary + fuzzy repair, on the Roman Urdu.
            norm = fuzzy_normalize_roman_urdu(str(raw or ""), verbose=False)
            normalized.append(norm)

            # Stage 4: what the ENCODER actually receives. This bundle is a
            # skip_normalization bundle, so the Roman Urdu dictionary stages
            # do not run on the English - only stop-word removal does. That
            # asymmetry is exactly the thing people get wrong about this
            # pipeline, so the column shows the result rather than asking
            # anyone to reason about it.
            source = en if en else raw
            if skip_norm:
                clean = remove_stopwords(str(source or ""), own_stops)
            else:
                from triage_pipeline import preprocess_corpus_for_embedding
                clean = preprocess_corpus_for_embedding([str(source or "")],
                                                        own_stops)[0]
            encoded.append(clean)

            # Which words the stop-word list removed. Reported explicitly
            # because "the encoder saw less than I typed" is a reasonable
            # thing to want to check, and diffing two columns by eye is not.
            before = str(source or "").lower().split()
            after = set(clean.lower().split())
            dropped.append(" ".join(w for w in before
                                    if w.strip(".,;:!?") not in after))

        enc = man.get("embedding_model", "")
        return {
            "Input_Raw": list(originals),
            "Input_Normalized": normalized,
            "Translation_English": list(translations),
            "Text_Encoded": encoded,
            "Stopwords_Removed": dropped,
            # Constant for the run, but carried per row on purpose: a CSV
            # that does not say which model and which encoder produced it
            # cannot be checked against anything six months from now.
            "Encoder": [enc] * len(originals),
            "Embedding_Dim": [man.get("embedding_dim", "")] * len(originals),
            "Model_Bundle": [ENGLISH_MODEL_DIR] * len(originals),
        }

    def _batch_worker(self):
        # read_table() is shared with predict_batch.py and tries a chain of
        # encodings. The old inline pd.read_csv(path) assumed UTF-8 and threw
        # UnicodeDecodeError ("can't decode byte 0xfb") on any CSV saved by
        # Excel on a Windows machine, which is the normal way these files
        # arrive.
        from triage_pipeline import predict_dataframe, read_table

        path = self._batch_target
        df = read_table(path)
        gate_status, gate_detail, texts = [], [], []
        if self.in_english_mode():
            # Translate the whole column first, then score the English.
            # Done up front rather than row-by-row inside predict_dataframe
            # so a mid-file API failure stops the run with a clear count
            # instead of leaving half the sheet scored by a pipeline the
            # operator did not pick.
            texts, failures = [], 0
            gate_status, gate_detail, failed_rows = [], [], []
            pr = getattr(self, "_batch_progress", {})
            pr["total"] = len(df)
            for _i, t in enumerate(df["Complaint_Text"].fillna("").astype(str)):
                pr["done"] = _i
                pr["text"] = t
                self._last_gate = (True, [], None)
                en, err = self.translate_for_mode(t, allow_blocked=True)
                ok, why, _ = self._last_gate
                if err:
                    failures += 1
                    # The ORIGINAL goes into the scored column so the row
                    # still shows what the operator typed, but the
                    # Translation column gets None - it promises English, and
                    # Roman Urdu printed there reads as its own translation.
                    failed_rows.append(len(texts))
                    texts.append(t)
                    gate_status.append("NOT TRANSLATED")
                    gate_detail.append(err.splitlines()[0][:120])
                else:
                    texts.append(en)
                    gate_status.append("PASS" if ok else "BLOCKED")
                    gate_detail.append("" if ok else "; ".join(why)[:120])
            # `len(df) and` matters: a header-only sheet has 0 rows and 0
            # failures, and 0 == 0 fired this guard, so exporting the template
            # and uploading it unedited raised "None of the 0 rows could be
            # translated ... the usual causes are Ollama not running" against
            # a perfectly healthy service. predict_dataframe has a 0-row path
            # built for exactly that upload; this made it unreachable.
            if len(df) and failures == len(df):
                # Nothing survived. This IS fatal - there is no table to show.
                raise RuntimeError(
                    f"None of the {len(df)} rows could be translated. Check "
                    f"the console: the usual causes are Ollama not running "
                    f"(start it with 'ollama serve') or the model refusing, "
                    f"which the guardrail logs.")

            # A partial failure is NOT fatal any more. It used to abort the
            # whole run, on the reasoning that mixing translated and
            # untranslated rows would put two pipelines in one table. The
            # reasoning was right; the remedy was too blunt. One "n/a"
            # complaint - which a real spreadsheet always has - destroyed the
            # other 499 rows, and the operator was shown nothing at all.
            #
            # Now every row carries its own verdict and untranslated rows are
            # scored by nobody, so the two pipelines still never mix: a row
            # either has an English translation and a triage level, or it has
            # neither and says why.
            self._batch_failures = failures

            df = df.copy()
            # Captured BEFORE the column is overwritten. Replacing
            # Complaint_Text with the translation is correct - the English
            # is what gets scored - but it destroyed the only record of what
            # the operator typed, and that is the column a reviewer always
            # wants back first.
            originals = list(df["Complaint_Text"].fillna("").astype(str))
            df["Complaint_Text"] = texts
        pr = getattr(self, "_batch_progress", {})
        pr["done"] = pr.get("total", 0)
        pr["text"] = "scoring"
        results, _ = predict_dataframe(self.active_artifacts(), df)
        pr["finished"] = True

        if gate_status:
            # Every stage the complaint passed through, in pipeline order, so
            # the CSV reads left to right the way the text actually travelled:
            # raw -> dictionary -> translation -> what the encoder saw.
            for name, col in self._stage_columns(originals, texts).items():
                results[name] = col
            # Translation_English must agree with Translation: a row that was
            # never translated has no English, and printing the untranslated
            # Roman Urdu under an "English" heading is how the old
            # Translation column came to be wrong.
            blanked = [None if i in set(failed_rows) else v
                       for i, v in enumerate(texts)]
            results["Translation"] = blanked
            results["Translation_English"] = blanked
            results["Gate_Status"] = gate_status
            results["Gate_Detail"] = gate_detail
            # A blocked row must not carry a triage level. Scoring it anyway
            # and hoping the operator reads a status column is how a stomach
            # complaint gets actioned as cardiac; the single-patient path
            # already refuses outright, and a spreadsheet should not be the
            # laxer of the two.
            blocked = [i for i, g in enumerate(gate_status) if g != "PASS"]
            for col in ("Predicted_Level_0to3", "Predicted_Triage_Level",
                        "Predicted_Label", "Confidence",
                        "P_L0", "P_L1", "P_L2", "P_L3"):
                if col in results.columns:
                    results.loc[results.index[blocked], col] = None
            import pandas as _pd
            for i in blocked:
                prev = results.at[results.index[i], "Notes"]
                prev = "" if prev is None or _pd.isna(prev) else str(prev)
                results.at[results.index[i], "Notes"] = (
                    f"{prev}NOT SCORED - {gate_status[i]}: {gate_detail[i]}; ")

        out_base = os.path.splitext(path)[0] + "_predictions"
        results.to_csv(out_base + ".csv", index=False)
        try:
            results.to_excel(out_base + ".xlsx", index=False)
        except Exception:
            pass
        return results, out_base

    def _end_batch_ui(self):
        """Restore the tab. Called on success AND on failure - a crashed run
        that leaves the button disabled makes the tab permanently dead."""
        self._batch_progress = {"finished": True}
        try:
            self.batch_progress_box.pack_forget()
            self.batch_btn.config(state="normal")
        except Exception:
            pass

    def _done_batch_worker(self, payload):
        results, out_base = payload
        # Held for the export button. Without this the only copy is the file
        # the worker dropped beside the input, which the operator may not be
        # able to write to or find.
        self._last_batch_results = results
        self._end_batch_ui()
        self.batch_export_btn.config(state="normal")
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)

        import pandas as _pd
        counts = {i: 0 for i in range(1, 5)}
        skipped = 0
        for _, r in results.iterrows():
            raw_level = r.get("Predicted_Triage_Level")
            # A row the gate blocked, or one that never translated, carries
            # no level by design. int(nan) raises, and a row shown with a
            # fabricated level would defeat the point of withholding it.
            if raw_level is None or _pd.isna(raw_level):
                skipped += 1
                self.batch_tree.insert(
                    "", "end",
                    text=str(r.get("Input_Raw") or r.get("Complaint_Text", ""))[:90],
                    values=(str(r.get("Translation_English") or "-")[:80],
                            "-", "not scored", "-",
                            r.get("Gate_Status", "NOT SCORED"),
                            str(r.get("Gate_Detail") or r.get("Notes", ""))[:90]))
                continue
            level = int(raw_level)
            counts[level] = counts.get(level, 0) + 1
            self.batch_tree.insert(
                "", "end",
                text=str(r.get("Input_Raw") or r.get("Complaint_Text", ""))[:90],
                tags=(f"L{level - 1}",),
                values=(str(r.get("Translation_English") or "")[:80],
                        level, r["Predicted_Label"], r["Confidence"],
                        r.get("Gate_Status", "PASS"),
                        str(r.get("Notes", ""))[:90]))

        total = len(results)
        parts = [f"Level {lvl} {LEVEL_NAMES[lvl - 1]}: {counts.get(lvl, 0)}"
                 for lvl in range(1, 5)]
        line = f"{total - skipped} of {total} patients triaged.   " + "    ".join(parts)
        if skipped:
            # Stated on the summary line, not buried in a column. A run that
            # silently drops rows reads as a complete triage of the file.
            line += (f"\n{skipped} row(s) NOT scored - blocked by the "
                     f"anatomical gate or not translated. See Gate_Status.")
        self.batch_summary.configure(
            text=line + f"\nSaved to {os.path.basename(out_base)}.csv and .xlsx")
        self.status.set(f"Batch complete: {total} patients.")

    # =================================================================
    # TAB 5 - Results
    # =================================================================
    def _build_results_tab(self):
        root = self.tab_results
        canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        holder = tk.Frame(canvas, bg=BG)
        holder.bind("<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=holder, anchor="nw")
        # Without this the inner frame keeps its natural width and the charts
        # are laid out narrower than the window, clipping their labels.
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=4, pady=6)
        sb.pack(side="right", fill="y", pady=6)

        def on_wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        # Bind while the pointer is over this tab only, so the wheel does not
        # hijack scrolling in the other tabs' tables.
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # The method-comparison table has to say WHICH row is deployed, and
        # that is only known once the model has finished loading on the
        # background thread. Build it into a container that gets re-rendered
        # then, rather than drawing a table that is stale from birth.
        # The A/B/C/D method comparison and the embedding evaluation used to
        # sit here. Both measured OTHER systems - the 10,000-row Roman Urdu
        # bundle and the professor baseline - and neither is reachable since
        # the mode toggle was removed. A reader saw accuracy figures on the
        # same page as a live triage level and reasonably assumed they
        # described it. Removed rather than relabelled, by decision: this
        # page now describes exactly one model, the one that serves.
        self._results_classification_box = tk.Frame(holder, bg=BG)
        self._results_classification_box.pack(fill="x")
        self._results_section_classification(self._results_classification_box)

    def _results_section_classification(self, parent):
        """Precision / recall / F1 / support for the DEPLOYED model.

        Every number is computed from the saved confusion matrix at render
        time, so it cannot drift from the model actually loaded - there are
        no placeholder or example values anywhere on this page.
        """
        # Read the DEPLOYED bundle's own metrics file, the same way the
        # other sections on this page do, so the table can never describe a
        # different model than the one serving predictions.
        # The bundle that SERVES, not the one found at startup. These are
        # different directories now, and a precision/recall table describing
        # the unreachable one is worse than no table.
        model_dir = self.active_model_dir()
        path = os.path.join(resolve_project_file(model_dir),
                            "triage_metrics.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            return
        cm = metrics.get("confusion_matrix")
        if not cm:
            return

        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Classification report  -  the deployed model").pack(fill="x")
        body(pad,
             "Precision, recall and F1 per triage level, derived from the saved "
             "confusion matrix of the held-out test patients. Precision = of the "
             "cases called this level, how many really were. Recall = of the cases "
             "that really were this level, how many were caught. F1 = their harmonic "
             "mean. Support = how many test patients truly had this level.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 10))

        m = per_class_metrics(cm)
        cols = ("precision", "recall", "f1", "support")
        tree = ttk.Treeview(pad, columns=cols, show="tree headings",
                            height=len(m["per_class"]) + 3)
        tree.heading("#0", text="triage level")
        tree.column("#0", width=210)
        for c, label, w in [("precision", "precision", 120),
                            ("recall", "recall", 120),
                            ("f1", "F1 score", 120),
                            ("support", "support", 110)]:
            tree.heading(c, text=label)
            tree.column(c, width=w, anchor="center")

        for row in m["per_class"]:
            i = row["index"]
            name = (f"L{i + 1} {LEVEL_NAMES[i]}" if i < len(LEVEL_NAMES)
                    else f"L{i + 1}")
            # Flag the weak class rather than leaving the reader to spot it.
            tag = "weak" if row["f1"] < 0.70 else ""
            tree.insert("", "end", text=name, tags=(tag,), values=(
                f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                f"{row['f1']:.3f}", row["support"]))
        tree.insert("", "end", text="", values=("", "", "", ""))
        for label, key in [("macro avg", "macro"), ("weighted avg", "weighted")]:
            tree.insert("", "end", text=label, tags=("avg",), values=(
                f"{m[key]['precision']:.3f}", f"{m[key]['recall']:.3f}",
                f"{m[key]['f1']:.3f}", m["total"]))
        tree.tag_configure("weak", foreground="#c0392b")
        tree.tag_configure("avg", font=("Segoe UI Semibold", 9))
        tree.pack(fill="x")

        # Hover help on each column heading. The terms below are the ones a
        # reader without an ML background has no way to guess at.
        attach_header_tooltips(tree, {
            "#0": "confusion_matrix",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "support": "support",
        })
        hint = body(pad, "Hover any column heading for a plain-English "
                         "definition of the term.", fg=ACCENT, size=8)
        hint.pack(fill="x", pady=(4, 0))

        acc = m["accuracy"] * 100
        stated = metrics.get("accuracy")
        note = (f"Accuracy recomputed from this matrix: {acc:.2f}%"
                + (f"  (metrics file states {stated:.2f}% - they agree)"
                   if stated is not None and abs(stated - acc) < 0.05
                   else f"  (metrics file states {stated}% - MISMATCH)"
                   if stated is not None else ""))
        weak = [r for r in m["per_class"] if r["f1"] < 0.70]
        if weak:
            names = ", ".join(f"L{r['index'] + 1}" for r in weak)
            note += (f"\nRed rows ({names}) score below 0.70 F1. Read those "
                     f"alongside their support column - a class with few test "
                     f"patients moves several points on one prediction.")
        body(pad, note, fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(8, 0))

    def _build_score_tab(self):
        root = self.tab_score
        canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        holder = tk.Frame(canvas, bg=BG)
        holder.bind("<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=holder, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=4, pady=6)
        sb.pack(side="right", fill="y", pady=6)

        def on_wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Was four sections. Three of them scored the 10,000-row Roman Urdu
        # bundle and the 185-row professor baseline - neither reachable now -
        # and the embedding demo preprocessed with a pipeline the serving
        # bundle does not use. The cluster inspector is kept: it runs the
        # CURRENT path, translating each sentence through Ollama before
        # embedding it, so it measures what actually ships.
        self._score_section_compare(holder)
        self._score_section_cluster(holder)

    # ------------------------------------------- (b) compare your own text
    def _score_section_compare(self, parent):
        """Type one complaint, see it become numbers, see what it matches.

        The matrix below this shows 100 numbers at once, which is correct
        and unreadable: it answers "how does this set hang together", not
        "what happens to MY sentence". This section answers the second
        question, one sentence at a time, and shows every intermediate value
        rather than only the verdict.
        """
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Try it  -  turn one complaint into numbers").pack(fill="x")
        body(pad,
             "Type any complaint in Roman Urdu or English. It goes through "
             "the SAME steps as a real prediction - translate, clean, encode - "
             "and is then compared against the ten reference complaints "
             "below. Every value is shown, so you can follow the arithmetic "
             "rather than trust it.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 8))

        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x", pady=(0, 8))
        self.compare_var = tk.StringVar(value="seena mein dard aur pasina")
        tk.Entry(row, textvariable=self.compare_var, font=("Segoe UI", 10),
                 relief="solid", bd=1).pack(side="left", fill="x", expand=True,
                                            ipady=4, padx=(0, 8))
        self.compare_btn = ttk.Button(row, text="Compare",
                                      style="Accent.TButton",
                                      command=self._do_compare)
        self.compare_btn.pack(side="left")
        self.compare_status = tk.StringVar(value="")
        tk.Label(row, textvariable=self.compare_status, bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 0))

        self.compare_out = tk.Frame(pad, bg=CARD)
        self.compare_out.pack(fill="x")
        body(self.compare_out,
             "Press Analyse on the cluster below first - the ten reference "
             "complaints have to be embedded before anything can be compared "
             "against them.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x")

    def _do_compare(self):
        text = self.compare_var.get().strip()
        if not text:
            return
        if getattr(self, "_cluster_vectors", None) is None:
            messagebox.showinfo(
                "Run the cluster first",
                "The ten reference complaints have not been embedded yet.\n\n"
                "Press \"Load sample cluster and analyse\" below, wait for it "
                "to finish, then Compare.")
            return
        if getattr(self, "_compare_running", False):
            return

        self._compare_running = True
        self.compare_btn.configure(state="disabled")
        self.compare_status.set("translating and encoding...")
        shared = {"done": False, "step": None, "error": None}

        def worker():
            try:
                from src.embedding_pipeline import preprocess_and_embed
                shared["step"] = preprocess_and_embed(text, translate=True)
            except Exception as e:
                shared["error"] = f"{type(e).__name__}: {e}"
            shared["done"] = True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if not shared["done"]:
                self.after(300, poll)
                return
            self._compare_running = False
            self.compare_btn.configure(state="normal")
            self.compare_status.set("")
            if shared["error"]:
                messagebox.showerror("Compare failed", shared["error"])
                return
            self._render_compare(text, shared["step"])

        self.after(300, poll)

    def _render_compare(self, raw, step):
        import numpy as np

        for w in self.compare_out.winfo_children():
            w.destroy()

        if step is None or step.get("embedding") is None:
            body(self.compare_out,
                 f"Could not encode this complaint.\n{step.get('error') if step else ''}",
                 fg="#c0392b", size=9, wraplength=1000).pack(fill="x")
            return

        vec = step["embedding"]
        norm = float(np.linalg.norm(vec))

        def stage(n, title, value, note):
            blk = tk.Frame(self.compare_out, bg=CARD)
            blk.pack(fill="x", pady=(0, 7))
            head = tk.Frame(blk, bg=CARD)
            head.pack(fill="x")
            tk.Label(head, text=f"{n}", bg=CARD, fg=ACCENT,
                     font=("Consolas", 10, "bold")).pack(side="left", padx=(0, 8))
            tk.Label(head, text=title, bg=CARD, fg=INK,
                     font=("Segoe UI Semibold", 10)).pack(side="left")
            tk.Label(blk, text=value, bg="#fbfcfd", fg=INK,
                     font=("Consolas", 10), anchor="w", justify="left",
                     wraplength=980, padx=10, pady=6, relief="solid",
                     bd=1).pack(fill="x", padx=(24, 0), pady=(2, 1))
            tk.Label(blk, text=note, bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                     anchor="w", justify="left", wraplength=980).pack(
                fill="x", padx=(24, 0))

        stage("1", "What you typed", raw,
              "the raw complaint, before anything touches it")
        stage("2", "English translation",
              step["translated"] or "(not translated - embedded as typed)",
              "by Ollama on this machine. If translation fails the original "
              "text is embedded instead, and this line says so.")
        stage("3", "Cleaned for encoding", step["normalized"],
              "lowercased, punctuation dropped, common words removed - the "
              "exact text handed to the encoder")

        preview = ", ".join(f"{float(x):+.3f}" for x in vec[:8])
        stage("4", f"The vector  -  {len(vec)} numbers",
              f"[{preview}, ...  {len(vec) - 8} more ]\n"
              f"length of this vector = {norm:.4f}",
              "Every complaint becomes exactly 384 numbers. The length is "
              "always 1.0000 - the encoder normalises them - and that is what "
              "makes the comparison below a simple multiply-and-add.")

        # ---- comparison ------------------------------------------------
        sims = self._cluster_vectors @ vec
        order = list(np.argsort(-sims))

        blk = tk.Frame(self.compare_out, bg=CARD)
        blk.pack(fill="x", pady=(6, 0))
        head = tk.Frame(blk, bg=CARD)
        head.pack(fill="x")
        tk.Label(head, text="5", bg=CARD, fg=ACCENT,
                 font=("Consolas", 10, "bold")).pack(side="left", padx=(0, 8))
        tk.Label(head, text="Compared against the ten reference complaints",
                 bg=CARD, fg=INK, font=("Segoe UI Semibold", 10)).pack(side="left")
        tk.Label(blk,
                 text="score = v1·w1 + v2·w2 + ... + v384·w384      "
                      "(1.00 = identical meaning, 0.00 = unrelated)",
                 bg=CARD, fg=MUTED, font=("Consolas", 9), anchor="w").pack(
            fill="x", padx=(24, 0), pady=(3, 4))

        table = tk.Frame(blk, bg=CARD)
        table.pack(fill="x", padx=(24, 0))
        for rank, idx in enumerate(order):
            sim = float(sims[idx])
            label = (self._cluster_labels[idx] or "")[:52]
            r = tk.Frame(table, bg=CARD)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=f"{sim:.4f}", bg=CARD,
                     fg="#1e5c2e" if rank == 0 else INK,
                     font=("Consolas", 10, "bold" if rank == 0 else "normal"),
                     width=8, anchor="w").pack(side="left")
            # a bar, because ten four-decimal numbers do not show a gap
            bar = tk.Frame(r, bg="#eef1f4", height=13, width=210)
            bar.pack(side="left", padx=(0, 10))
            bar.pack_propagate(False)
            fill = max(2, int(210 * max(0.0, min(1.0, sim))))
            tk.Frame(bar, bg="#2e9e5b" if rank == 0 else "#9db3c0",
                     height=13, width=fill).place(x=0, y=0)
            tk.Label(r, text=("closest  " if rank == 0 else "") + label,
                     bg=CARD, fg=INK if rank == 0 else MUTED,
                     font=("Segoe UI", 9), anchor="w").pack(side="left")

        best, worst = float(sims[order[0]]), float(sims[order[-1]])
        tk.Label(blk,
                 text=(f"Closest {best:.4f}, furthest {worst:.4f}, "
                       f"gap {best - worst:.4f}.  A large gap (e.g. > 0.50) "
                       f"means the encoder clearly separates this complaint "
                       f"from unrelated ones. A small gap indicates the "
                       f"encoder struggles to differentiate distinct medical "
                       f"conditions.  This is also why the safety check in "
                       f"this project is a word test rather than a similarity "
                       f"score: on cross-language pairs the gap collapses to "
                       f"0.013 and the number stops being usable."),
                 bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w",
                 justify="left", wraplength=980).pack(fill="x", padx=(24, 0),
                                                      pady=(6, 0))

    # --------------------------------------- (c) cluster inspector
    def _score_section_cluster(self, parent):
        """Pairwise similarity over a 10-complaint cluster.

        The work is slow - one local LLM call per sentence - so it runs on
        a worker thread and the table is filled from the main thread when
        it lands. Blocking here would freeze every other tab, including
        the triage button.
        """
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Cluster Embedding Inspector").pack(fill="x")
        body(pad,
             "Runs a cluster of complaints through the full offline pipeline "
             "(Ollama translation -> lowercase + stop-word removal -> "
             "MiniLM-L12-v2) and shows the pairwise cosine matrix. Because "
             "every vector is L2-normalised, the dot product IS the cosine, "
             "so the diagonal must read 1.00 - it doubles as a check that the "
             "encoder is behaving.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 8))

        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x", pady=(0, 8))
        self.cluster_btn = ttk.Button(
            row, text="Load sample cluster (10 complaints)  and analyse",
            style="Accent.TButton", command=self._do_cluster_analysis)
        self.cluster_btn.pack(side="left")
        self.cluster_export_btn = ttk.Button(row, text="Export CSV...",
                                             state="disabled",
                                             command=self._export_cluster_csv)
        self.cluster_export_btn.pack(side="left", padx=(8, 0))
        self.cluster_status = tk.StringVar(value="not run yet")
        tk.Label(row, textvariable=self.cluster_status, bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(12, 0))

        # A canvas, not a Treeview. A Treeview tag colours a whole ROW, and a
        # similarity matrix needs a colour PER CELL - so the widget could
        # never have shown the structure the old comment promised, and the
        # "hi"/"lo" tags configured for it were never even applied. It also
        # clipped: a fixed 24px row height and a fixed 300px complaint column
        # cut every sentence off mid-word on a high-DPI screen.
        wrap = tk.Frame(pad, bg=CARD)
        wrap.pack(fill="x")
        self.cluster_canvas = tk.Canvas(wrap, bg=CARD, highlightthickness=0,
                                        height=620)
        hbar = ttk.Scrollbar(wrap, orient="horizontal",
                             command=self.cluster_canvas.xview)
        self.cluster_canvas.configure(xscrollcommand=hbar.set)
        self.cluster_canvas.pack(side="top", fill="x")
        hbar.pack(side="top", fill="x")

        # Legend, and under it the complaints in full - the text the old grid
        # truncated now gets a whole line each.
        self.cluster_legend = tk.Frame(pad, bg=CARD)
        self.cluster_legend.pack(fill="x", pady=(8, 4))

        self.cluster_summary = body(pad, "", fg=MUTED, size=9, wraplength=1000)
        self.cluster_summary.pack(fill="x", pady=(8, 0))

    def _do_cluster_analysis(self):
        """Kick the analysis onto a worker thread; never block the UI."""
        if getattr(self, "_cluster_running", False):
            return
        try:
            from src.cluster_analyzer import SAMPLE_CLUSTER
        except Exception as e:
            messagebox.showerror("Cluster analysis unavailable",
                                 f"Could not import the analyser:\n{e}")
            return

        self._cluster_running = True
        self.cluster_btn.configure(state="disabled")
        shared = {"done": False, "result": None, "error": None, "msg": ""}

        def worker():
            try:
                from src.cluster_analyzer import analyze_sentence_cluster
                def prog(i, n, text):
                    shared["msg"] = f"translating {i + 1}/{n}: {text[:34]}..."
                shared["result"] = analyze_sentence_cluster(
                    SAMPLE_CLUSTER, translate=True, progress=prog)
            except Exception as e:
                shared["error"] = f"{type(e).__name__}: {e}"
            shared["done"] = True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if shared["done"]:
                self._cluster_running = False
                self.cluster_btn.configure(state="normal")
                if shared["error"]:
                    self.cluster_status.set("failed")
                    messagebox.showerror("Cluster analysis failed",
                                         shared["error"])
                else:
                    self._render_cluster(shared["result"])
                return
            self.cluster_status.set(shared["msg"] or "working...")
            self.after(400, poll)

        self.cluster_status.set("starting...")
        self.after(400, poll)

    def _fit_text(self, text, max_px, font_spec):
        """Trim text with an ellipsis so it fits max_px in the given font.

        Uses the real font metrics rather than a character count. Character
        counts are wrong for proportional fonts - "Chest pain and sweating
        variant number 1" and "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII" are the
        same length and nowhere near the same width - and wrong again at a
        different DPI, which is how the label column came to be overrun by
        every row at once.
        """
        import tkinter.font as tkfont
        f = tkfont.Font(font=font_spec)
        if f.measure(text) <= max_px:
            return text
        ell = f.measure("...")
        out = text
        while out and f.measure(out) + ell > max_px:
            out = out[:-1]
        return out.rstrip() + "..."

    def _cluster_cell_colour(self, v):
        """Similarity -> a background colour, low red through high green.

        A 10x10 matrix is 100 numbers. Read as text it is a wall; read as
        colour the structure - which complaints group, which one sits apart -
        is visible before any number is.

        Bands rather than a continuous gradient, because the eye cannot
        reliably rank a hundred shades but can rank five. The boundaries are
        the ones this project already uses: 0.85 is the pair threshold in
        cluster_analyzer, and 0.50 is the yardstick named in the panel above.
        """
        if v >= 0.95:
            return "#1e7a45", "white"      # near-identical
        if v >= 0.85:
            return "#57ab6f", "white"      # a matching pair
        if v >= 0.65:
            return "#bcdcc4", "#12171B"    # related
        if v >= 0.50:
            return "#f0e6c8", "#12171B"    # weak
        return "#f2d3ce", "#7a2f28"        # unrelated

    def _render_cluster(self, res):
        """Draw the similarity matrix as a heatmap, not a table.

        This was a ttk.Treeview, and it had two problems that were not
        cosmetic.

        First, it clipped. The global Treeview rowheight is a fixed 24px and
        the complaint column a fixed 300px, so on a high-DPI display the rows
        overlapped and every complaint was cut off mid-sentence.

        Second, and worse, the "hi" and "lo" colour tags configured for it
        could never have worked: a Treeview tag colours an entire ROW, and a
        similarity matrix needs a colour PER CELL. They were also never
        applied - only the "diag" tag was ever set - so the comment promising
        that "structure is visible without reading 100 numbers" described
        something the widget cannot do.

        A canvas can. Each cell is drawn and coloured individually, the
        diagonal is marked, and the full complaint text is listed underneath
        where it has room to be read.
        """
        c = self.cluster_canvas
        c.delete("all")
        for w in self.cluster_legend.winfo_children():
            w.destroy()

        if res is None or res.get("matrix") is None:
            self.cluster_status.set("no matrix produced")
            self.cluster_summary.configure(
                text="\n".join(res.get("errors", [])) if res else "")
            return

        S = res["matrix"]
        n = len(S)
        sents = res["sentences"]

        # Keep the vectors so the "Try it" panel can compare against them.
        self._last_cluster_result = res
        self.cluster_export_btn.config(state="normal")
        self._cluster_vectors = res.get("vectors")
        self._cluster_labels = [
            f"S{k + 1}  {(x['translated'] or x['raw'])}"
            for k, x in enumerate(sents)]

        # LAYOUT: complaints down the left, their row of the matrix beside
        # them. The row label IS the complaint, so reading across one line
        # answers "how does THIS complaint relate to the others" without
        # cross-referencing an S-number against a list further down. The
        # separate list underneath is gone; it existed only because the old
        # grid had nowhere to put the text.
        LABEL_W, CELL, HDR = 430, 52, 30
        ROW = 30                      # shorter than CELL: text, not a square
        w = LABEL_W + CELL * n + 4
        h = HDR + ROW * n + 4
        c.configure(width=w, height=h, scrollregion=(0, 0, w, h))

        # column headers, over the grid only
        for i in range(n):
            c.create_text(LABEL_W + i * CELL + CELL / 2, HDR / 2,
                          text=f"S{i + 1}", font=("Segoe UI Semibold", 9),
                          fill=MUTED)

        for r in range(n):
            y0 = HDR + r * ROW
            src = sents[r]
            english = (src["translated"] or src["raw"])
            # Truncated by MEASURED WIDTH, not character count. A 46-char
            # cut was tried first and still overran the label column by a
            # wide margin - proportional text has no fixed characters-per-
            # pixel, and the overrun lands underneath the first heatmap
            # column where it is unreadable against the cell colours.
            shown = self._fit_text(english, LABEL_W - 46, ("Segoe UI", 9))
            c.create_text(6, y0 + ROW / 2, anchor="w",
                          text=f"S{r + 1}", font=("Consolas", 9, "bold"),
                          fill=ACCENT)
            c.create_text(40, y0 + ROW / 2, anchor="w", text=shown,
                          font=("Segoe UI", 9), fill=INK)

            for col in range(n):
                v = float(S[r][col])
                bg, fg = self._cluster_cell_colour(v)
                x0 = LABEL_W + col * CELL
                # The diagonal is every complaint against itself. It must
                # read 1.00 - if it does not, the vectors are not unit
                # length and every other number here is suspect. Outlined
                # rather than recoloured, so it stays legible without
                # competing with the data.
                c.create_rectangle(x0, y0, x0 + CELL, y0 + ROW,
                                   fill=bg,
                                   outline="#12171B" if r == col else "#ffffff",
                                   width=2 if r == col else 1)
                c.create_text(x0 + CELL / 2, y0 + ROW / 2, text=f"{v:.2f}",
                              font=("Consolas", 8,
                                    "bold" if r == col else "normal"),
                              fill=fg)

        # legend, in the same bands as the cells
        tk.Label(self.cluster_legend, text="similarity:", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))
        for label, probe in [("< 0.50 unrelated", 0.2), ("0.50-0.65 weak", 0.55),
                             ("0.65-0.85 related", 0.7),
                             ("0.85+ matching pair", 0.9),
                             ("0.95+ near-identical", 0.97)]:
            bg, fg = self._cluster_cell_colour(probe)
            tk.Label(self.cluster_legend, text=f" {label} ", bg=bg, fg=fg,
                     font=("Segoe UI", 8)).pack(side="left", padx=2)

        # The Roman Urdu each row came from. Kept, but compactly and below -
        # an operator checking a suspicious score needs to see the original,
        # and it is the one thing the row label has no room for.
        origins = tk.Frame(self.cluster_legend.master, bg=CARD)
        origins.pack(fill="x", pady=(6, 0))
        tk.Label(origins, text="translated from:", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 8), anchor="w").pack(fill="x")
        for k, x in enumerate(sents):
            if not x["translated"] or x["raw"] == x["translated"]:
                continue
            row = tk.Frame(origins, bg=CARD)
            row.pack(fill="x")
            tk.Label(row, text=f"S{k + 1}", bg=CARD, fg=ACCENT,
                     font=("Consolas", 8, "bold"), width=4,
                     anchor="w").pack(side="left")
            tk.Label(row, text=x["raw"], bg=CARD, fg=MUTED,
                     font=("Segoe UI", 8), anchor="w").pack(side="left")

        n_ok = sum(1 for s in sents if s["translated_ok"])
        self.cluster_status.set(f"{n} embedded, {n_ok} translated")


def main():
    enable_dpi_awareness()
    # Either bundle is enough to start: the embedding pipeline is preferred,
    # the dictionary model is the fallback. Requiring triage_model/ here would
    # refuse to launch on a clone that only has the deployed embedding model.
    if not any(os.path.exists(os.path.join(d, "model.pkl"))
               for d in (EMBED_MODEL_DIR, MODEL_DIR)):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model not found",
            f"Found neither {EMBED_MODEL_DIR}/model.pkl nor "
            f"{MODEL_DIR}/model.pkl.\n\n"
            "Train a model first:\n"
            "    python train_embedding_pipeline.py     (deployed pipeline)\n"
            "    python train_embedding_pipeline.py")
        return
    TriageGUI().mainloop()


if __name__ == "__main__":
    main()
