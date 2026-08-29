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
PIPELINE_RESULTS = resolve_project_file("embedding_pipeline_results.csv")
EVAL_RESULTS = resolve_project_file("embedding_evaluation_results.csv")
EVAL_NEIGHBOURS = resolve_project_file("embedding_evaluation_neighbours.csv")
CLUSTERS_FILE = resolve_project_file("evaluation_clusters.json")
DATASET_FILE = resolve_project_file("triage_mixed_language_dataset.csv")

# Every figure the app shows is read from one of the files above. When a file
# is missing the app prints the command that produces it rather than inventing
# a plausible-looking number.
# Plain-English name for every text representation, so a table row always
# states which pipeline produced its numbers. Two very different pipelines
# are reported side by side in this app - a dictionary + Bag-of-Words count
# model, and an offline sentence-transformer - and without this the reader
# has no way to tell an "A" row from a "C" row apart from the letter.
REPRESENTATION_LABEL = {
    "dictionary_bow": "dictionary + BoW counts",
    "embeddings_raw": "sentence-transformer (raw text)",
    "embeddings_preprocessed": "sentence-transformer (preprocessed)",
    "hybrid": "dictionary + BoW  AND  sentence-transformer",
}

MISSING_HINT = {
    PIPELINE_RESULTS: "python train_embedding_pipeline.py",
    EVAL_RESULTS: "python embedding_evaluation.py",
    EVAL_NEIGHBOURS: "python embedding_evaluation.py",
    os.path.join(MODEL_DIR, "triage_metrics.json"): "python triage_bow_fuzzy_diac.py",
    os.path.join(EMBED_MODEL_DIR, "triage_metrics.json"): "python train_embedding_pipeline.py",
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
                    # Re-enable any button the failed job had disabled, so a
                    # single failure does not leave the tab permanently dead.
                    if hasattr(self, "demo_btn"):
                        self.demo_btn.config(state="normal")
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
        self.status.set(
            f"Ready.  Deployed: {self.model_info['method']}  "
            f"({self.model_info['basis']}) from {self.model_dir}/  |  "
            f"{n_stops} learned stop words  |  everything runs offline."
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

    def translate_for_mode(self, text):
        """Mode 2 only. Returns (english_text, error_message).

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
                self.status.set(f"English mode: translating with {model}")

            out = translate_roman_urdu(text, model=model)
        except Exception as e:
            return None, (f"Translation failed: {type(e).__name__}: {e}\n\n"
                          f"No prediction was made.")

        if not out:
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
        if not ok:
            return None, (
                "Anatomical check failed - the translation moved the "
                "complaint to a different part of the body.\n\n"
                + "\n".join(failures) +
                f"\n\nTranslation was: {out!r}\n\n"
                f"No prediction was made. Rephrase the complaint, or retry - "
                f"the translator is not deterministic across model versions.")
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
        """One-line 'this is the model making the predictions' summary."""
        if not self.model_info:
            return "Currently deployed: loading..."
        i = self.model_info
        line = (f"Currently deployed: {i['method']}   ·   "
                f"text features from {i['basis']}   ·   {self.model_dir}/")
        if i["uses_embeddings"]:
            line += f"\nembedding model: {i['embedding_model']}  ({i['embedding_dim']} dims)"

        # Provenance and scope, read straight from the bundle manifest.
        # This is deliberately on the banner rather than buried in a tab:
        # a triage screen that does not say "synthetic, cardiac only" invites
        # exactly the misreading this project cannot afford.
        english = self.in_english_mode()
        if english:
            i = getattr(self, "model_info_en", None) or i
            line = (f"Currently deployed: {i['method']}   ·   "
                    f"text features from {i['basis']}   ·   {ENGLISH_MODEL_DIR}/")
            if i["uses_embeddings"]:
                line += (f"\nembedding model: {i['embedding_model']}"
                         f"  ({i['embedding_dim']} dims)")
            line = ("*** MODE: ENGLISH (LOCAL LLM) - translated on this machine ***\n"
                    "every prediction is translated by Ollama on localhost "
                    "before it is scored - no network call\n") + line
        else:
            line = "*** MODE: ROMAN URDU (OFFLINE) - no network calls ***\n" + line

        man = self.active_manifest() or getattr(self, "manifest", None) or {}
        ds = man.get("dataset", {})
        prov = ds.get("provenance", {})

        # Experiment marker, first line and impossible to miss. This branch
        # can load an English-translation model that is NOT the submitted
        # one, and a triage screen that looks identical while running a
        # different pipeline is exactly how the wrong result gets reported.
        if man.get("experiment"):
            line = ("*** EXPERIMENTAL: " + str(man["experiment"]) +
                    " - NOT the submitted model ***\n") + line

        bits = []
        if ds.get("file"):
            bits.append(f"trained on {ds['file']} ({ds.get('rows', '?')} rows"
                        + (f", {man['trained_on_date']}" if man.get("trained_on_date") else "")
                        + ")")
        scope = (man.get("scope") or {}).get("clinical_scope")
        if scope:
            bits.append(f"scope: {scope}")
        if bits:
            line += "\n" + "   ·   ".join(bits)
        if prov.get("synthetic") is True:
            line += ("\n⚠ SYNTHETIC DATA - generated by "
                     f"{prov.get('generator', 'a script')}. NOT real patient "
                     "records. Research prototype only.")
        elif prov.get("synthetic") == "unknown":
            line += ("\n⚠ Training data provenance UNKNOWN - not verified as "
                     "real or synthetic.")
        return line

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
            self.status.set("English mode: translating locally via Ollama...")
            self.update_idletasks()
            english_text, err = self.translate_for_mode(text)
            if err:
                messagebox.showerror("English (Local LLM) mode failed", err)
                self.status.set("English mode: translation failed - no prediction made.")
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

        self._draw_proba(proba, level)

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


    def _draw_proba(self, proba, chosen):
        self._last_proba = (proba, chosen)
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
            c.create_rectangle(label_w, y, label_w + filled, y + 16,
                               fill=LEVEL_COLOURS[i], outline="")
            c.create_text(width - 2, y + 8, anchor="e",
                          text=f"{float(p) * 100:.1f}%",
                          font=("Segoe UI Semibold" if i == chosen else "Segoe UI", 9),
                          fill=INK if i == chosen else MUTED)

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

        # 2. translation
        en, err = self.translate_for_mode(raw)
        if err:
            panel("2  Ollama translation  -  FAILED", err,
                  "The pipeline stops here. No prediction is made, and no "
                  "other model answers in its place.", ("failed", False))
            return
        panel("2  Local Ollama translation", en,
              "translated on this machine by Ollama on localhost - no network "
              "call. temperature 0, so the same complaint gives the same "
              "English every run.", ("translated", True))

        # 3. the gate. translate_for_mode() already enforces it - reaching
        #    this line means it passed, so re-running it here is for display,
        #    and it is the same function, not a second opinion.
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
        panel("4  Sentence-transformer", en,
              f"The English text is encoded by {enc} directly. The learned "
              f"Roman Urdu stop-word list is NOT applied here - this bundle "
              f"was trained with skip_normalization, and serving it any other "
              f"way is the train/serve skew that cost 38 points of accuracy "
              f"once already.", ("encoded", True))

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
        self.stop_filter = tk.StringVar(value="all")
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
        if self.in_english_mode():
            self.stop_summary.configure(text=(
                "The learned Roman Urdu stop-word list below is NOT applied "
                "to the triage path: complaints are translated first, and the "
                "English text goes to the sentence-transformer directly. The "
                "table is shown for reference - it documents how the list was "
                "derived, not what runs on your input."))
            return
        self.stop_summary.configure(text=(
            f"A token is removed only when ALL THREE hold:  document frequency "
            f">= {t['effective_df_cutoff']:.4f}"
            f"   AND   normalized mutual information <= {t['mi_threshold']}"
            f"   AND   Cramer's V <= {t['cramers_v_threshold']}\n"
            f"{c['n_documents']} complaints, {c['n_unique_tokens']} unique tokens, "
            f"{c['n_tokens_tested']} high-frequency tokens tested, "
            f"{r['n_stopwords']} learned as stop words: "
            f"{', '.join(r['stopwords'])}.\n"
            "Most rows in this table are tokens that were TESTED and KEPT - the "
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

        self._deployed_banner(
            pad, "Every row is triaged by:").pack(fill="x", pady=(10, 0))

        self.batch_summary = body(pad, "", fg=INK, size=9)
        self.batch_summary.pack(fill="x", pady=(10, 0))

        bottom = card(root)
        bottom.pack(fill="both", expand=True, padx=4, pady=6)
        bpad = tk.Frame(bottom, bg=CARD)
        bpad.pack(fill="both", expand=True, padx=18, pady=14)

        cols = ("level", "label", "conf", "notes")
        self.batch_tree = ttk.Treeview(bpad, columns=cols, show="tree headings")
        self.batch_tree.heading("#0", text="complaint")
        self.batch_tree.column("#0", width=430)
        for col, label, w in [("level", "level", 70), ("label", "label", 130),
                              ("conf", "confidence", 95), ("notes", "notes", 260)]:
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

    def _do_batch(self):
        path = self.batch_path.get().strip()
        if not os.path.exists(path):
            messagebox.showwarning("File not found", f"No such file:\n{path}")
            return
        self._batch_target = path
        self._run_async(self._batch_worker, f"Triaging {os.path.basename(path)}...")

    def _batch_worker(self):
        # read_table() is shared with predict_batch.py and tries a chain of
        # encodings. The old inline pd.read_csv(path) assumed UTF-8 and threw
        # UnicodeDecodeError ("can't decode byte 0xfb") on any CSV saved by
        # Excel on a Windows machine, which is the normal way these files
        # arrive.
        from triage_pipeline import predict_dataframe, read_table

        path = self._batch_target
        df = read_table(path)
        if self.in_english_mode():
            # Translate the whole column first, then score the English.
            # Done up front rather than row-by-row inside predict_dataframe
            # so a mid-file API failure stops the run with a clear count
            # instead of leaving half the sheet scored by a pipeline the
            # operator did not pick.
            texts, failures = [], 0
            for t in df["Complaint_Text"].fillna("").astype(str):
                en, err = self.translate_for_mode(t)
                if err:
                    failures += 1
                    texts.append(t)
                else:
                    texts.append(en)
            if failures:
                raise RuntimeError(
                    f"English (local LLM) mode: {failures} of {len(df)} rows "
                    f"failed to translate. Check the console: the usual causes "
                    f"are Ollama not running (start it with 'ollama serve') or "
                    f"the model refusing, which the guardrail logs. "
                    f"No results are shown, because mixing translated "
                    f"and untranslated rows would put two different pipelines in "
                    f"one table.")
            df = df.copy()
            df["Complaint_Text"] = texts
        results, _ = predict_dataframe(self.active_artifacts(), df)

        out_base = os.path.splitext(path)[0] + "_predictions"
        results.to_csv(out_base + ".csv", index=False)
        try:
            results.to_excel(out_base + ".xlsx", index=False)
        except Exception:
            pass
        return results, out_base

    def _done_batch_worker(self, payload):
        results, out_base = payload
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)

        counts = {i: 0 for i in range(1, 5)}
        for _, r in results.iterrows():
            level = int(r["Predicted_Triage_Level"])
            counts[level] = counts.get(level, 0) + 1
            self.batch_tree.insert(
                "", "end", text=str(r.get("Complaint_Text", ""))[:110],
                tags=(f"L{level - 1}",),
                values=(level, r["Predicted_Label"], r["Confidence"],
                        str(r.get("Notes", ""))[:90]))

        total = len(results)
        parts = [f"Level {lvl} {LEVEL_NAMES[lvl - 1]}: {counts.get(lvl, 0)}"
                 for lvl in range(1, 5)]
        self.batch_summary.configure(
            text=f"{total} patients triaged.   " + "    ".join(parts)
                 + f"\nSaved to {os.path.basename(out_base)}.csv and .xlsx")
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
        self._score_section_cluster(holder)

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
        self.cluster_status = tk.StringVar(value="not run yet")
        tk.Label(row, textvariable=self.cluster_status, bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(12, 0))

        cols = tuple(f"s{i}" for i in range(10))
        self.cluster_tree = ttk.Treeview(pad, columns=cols,
                                         show="tree headings", height=12)
        self.cluster_tree.heading("#0", text="complaint")
        self.cluster_tree.column("#0", width=300)
        for i, c in enumerate(cols):
            self.cluster_tree.heading(c, text=f"S{i + 1}")
            self.cluster_tree.column(c, width=58, anchor="center")
        self.cluster_tree.pack(fill="x")
        # Colour by band so the structure is visible without reading 100
        # numbers: high similarity green, low red.
        self.cluster_tree.tag_configure("hi", foreground="#2e9e5b")
        self.cluster_tree.tag_configure("lo", foreground="#c0392b")
        self.cluster_tree.tag_configure("diag", font=("Segoe UI Semibold", 9))

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

    def _render_cluster(self, res):
        for row in self.cluster_tree.get_children():
            self.cluster_tree.delete(row)
        if res is None or res.get("matrix") is None:
            self.cluster_status.set("no matrix produced")
            self.cluster_summary.configure(
                text="\n".join(res.get("errors", [])) if res else "")
            return

        S = res["matrix"]
        for k, sent in enumerate(res["sentences"]):
            label = f"S{k + 1}  {(sent['translated'] or sent['raw'])[:34]}"
            vals, tag = [], ""
            for j in range(len(S)):
                v = S[k][j]
                vals.append(f"{v:.2f}")
                if k == j:
                    tag = "diag"
            self.cluster_tree.insert("", "end", text=label, values=tuple(vals),
                                     tags=(tag,))

        n_ok = sum(1 for s in res["sentences"] if s["translated_ok"])
        self.cluster_status.set(
            f"{len(res['sentences'])} embedded, {n_ok} translated")

        parts = [
            f"mean intra-cluster similarity {res['mean_similarity']:.4f}   "
            f"(min {res['min_similarity']:.4f}, max {res['max_similarity']:.4f})",
            f"vectors: {len(res['sentences'])} x 384, all L2-normalised "
            f"(diagonal reads 1.00: {res['diagonal_ok']})",
            f"pairs at or above {res['threshold']:.2f}: {len(res['top_pairs'])}",
        ]
        for pr in res["top_pairs"][:3]:
            parts.append(f"    {pr['similarity']:.3f}   S{pr['i'] + 1} <-> "
                         f"S{pr['j'] + 1}")
        if res.get("outlier"):
            o = res["outlier"]
            parts.append(f"outlier: S{o['index'] + 1} at mean "
                         f"{o['mean_similarity']:.3f} - {o['text'][:56]}")
            parts.append(f"    {o['note']}")
        parts.append(
            "Mean similarity alone does not grade an encoder: a model that "
            "maps every medical sentence to nearly the same vector scores "
            "well here and is useless. Compare against a second cluster and "
            "read the GAP.")
        if res.get("errors"):
            parts.append("errors: " + "; ".join(res["errors"][:2]))
        self.cluster_summary.configure(text="\n".join(parts))

    def _demo_reference_corpus(self):
        """Complaints to match against, with their meaning group if known.

        Prefers the hand-labelled evaluation clusters, because then the match
        can be shown with the meaning it belongs to. Falls back to the raw
        dataset when that file is absent.
        """
        if os.path.exists(CLUSTERS_FILE):
            try:
                with open(CLUSTERS_FILE, "r", encoding="utf-8") as f:
                    clusters = json.load(f)["clusters"]
                pairs = [(t, name) for name, items in clusters.items()
                         for t in items]
                if pairs:
                    return pairs, CLUSTERS_FILE
            except (OSError, ValueError, KeyError):
                pass
        if os.path.exists(DATASET_FILE):
            import pandas as pd
            texts = (pd.read_csv(DATASET_FILE)["Complaint_Text"]
                     .dropna().astype(str).drop_duplicates().tolist())
            return [(t, "") for t in texts], DATASET_FILE
        return [], ""

    def _run_demo(self):
        text = self.demo_var.get().strip()
        if not text:
            return
        self.demo_btn.config(state="disabled")
        self._demo_text = text
        self._run_async(self._demo_worker, "Embedding the complaint...")

    def _demo_worker(self):
        import time

        import numpy as np
        from triage_pipeline import (get_text_encoder,load_sentence_transformer,
                                     preprocess_corpus_for_embedding,
                                     preprocess_for_embedding, read_manifest)

        if self._demo_model is None:
            try:
                import sentence_transformers          # noqa: F401
            except ImportError:
                return {"error":
                        "The 'sentence-transformers' library is not installed, so "
                        "the embedding demo cannot run.\n\n"
                        "Install it once (needs internet), then try again:\n"
                        "    pip install -r requirements-embedding.txt\n\n"
                        "Everything else in this app works without it."}
            # Prefer the model the DEPLOYED bundle actually uses, so the demo
            # shows the same encoder that produced the live prediction. Fall
            # back to the one the evaluation study used.
            name = None
            if self.model_info and self.model_info.get("embedding_model"):
                name = self.model_info["embedding_model"]
            if not name:
                name = read_manifest(EMBED_MODEL_DIR).get("embedding_model")
            if not name and os.path.exists(EVAL_RESULTS):
                for r in read_csv_rows(EVAL_RESULTS):
                    if r.get("embedding_model"):
                        name = r["embedding_model"]
                        break
            name = name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Cache-first, same as the deployed predictor - see
            # triage_pipeline.load_sentence_transformer for why.
            self._demo_model = load_sentence_transformer(name)
            self._demo_model_name = name

        if self._demo_corpus is None:
            pairs, source = self._demo_reference_corpus()
            if not pairs:
                return {"error": "No reference complaints found. Expected "
                                 f"{CLUSTERS_FILE} or {DATASET_FILE}."}
            # preprocess_corpus_for_embedding loads learned_stopwords.json ONCE.
            # The per-item preprocess_for_embedding() used before re-read and
            # re-parsed that JSON for every complaint in the pool.
            texts = preprocess_corpus_for_embedding([t for t, _ in pairs])
            vecs = self._demo_model.encode(texts, batch_size=32,
                                           convert_to_numpy=True,
                                           show_progress_bar=False,
                                           normalize_embeddings=True)
            self._demo_corpus = (pairs, texts, vecs, source)

        pairs, pool_texts, vecs, source = self._demo_corpus
        raw = self._demo_text
        cleaned = preprocess_for_embedding(raw)

        # Timed so the panel can show the encode really happened on this
        # click. Nothing here is cached per input text.
        t0 = time.perf_counter()
        vec = self._demo_model.encode([cleaned], convert_to_numpy=True,
                                      normalize_embeddings=True)[0]
        encode_ms = (time.perf_counter() - t0) * 1000

        # vecs is L2-normalized and so is vec, so the dot product IS the
        # cosine similarity. Computed here, now, against the live pool.
        sims = vecs @ vec
        order = np.argsort(-sims)[:5]
        return {
            "raw": raw,
            "cleaned": cleaned,
            "vector": vec,
            "vector_norm": float(np.linalg.norm(vec)),
            "vector_sum": float(vec.sum()),
            "vector_min": float(vec.min()),
            "vector_max": float(vec.max()),
            "encode_ms": encode_ms,
            "model": self._demo_model_name,
            "source": source,
            "pool_size": len(pairs),
            "pool_groups": len({g for _, g in pairs if g}),
            "sim_mean": float(sims.mean()),
            "sim_min": float(sims.min()),
            "matches": [(pairs[i][0], pool_texts[i], pairs[i][1], float(sims[i]))
                        for i in order],
        }

    def _done_demo_worker(self, payload):
        self.demo_btn.config(state="normal")
        for w in self.demo_out.winfo_children():
            w.destroy()

        if payload is None:
            return
        if payload.get("error"):
            body(self.demo_out, payload["error"], fg="#c0392b", size=9).pack(fill="x")
            self.status.set("Embedding demo could not run.")
            return

        vec = payload["vector"]
        out = self.demo_out

        body(out, f"1.  Preprocessed exactly like training "
                  f"(clean -> fuzzy -> learned stop-word removal):",
             size=9).pack(fill="x")
        tk.Label(out, text=payload["cleaned"] or "(empty)", bg="#fbfcfd", fg=INK,
                 font=("Consolas", 10), anchor="w", justify="left",
                 wraplength=980, padx=10, pady=6, relief="solid",
                 bd=1).pack(fill="x", pady=(2, 0))

        body(out, f"2.  The model turns that into {len(vec)} numbers "
                  f"(only the first 24 are shown):",
             size=9).pack(fill="x", pady=(8, 2))

        nums = "  ".join(f"{v:+.4f}" for v in vec[:24])
        tk.Label(out, text=nums, bg="#fbfcfd", fg=INK, font=("Consolas", 9),
                 anchor="w", justify="left", wraplength=980, padx=10, pady=8,
                 relief="solid", bd=1).pack(fill="x")
        # Fingerprints of the WHOLE vector, not just the 24 shown. These move
        # whenever the text changes, which is how you can tell at a glance
        # that the numbers were computed and not fetched from a file.
        body(out,
             f"whole-vector fingerprint:   sum {payload['vector_sum']:+.6f}"
             f"    min {payload['vector_min']:+.4f}"
             f"    max {payload['vector_max']:+.4f}"
             f"    L2 norm {payload['vector_norm']:.6f}"
             f"    computed in {payload['encode_ms']:.0f} ms on this click",
             fg=MUTED, size=8).pack(fill="x", pady=(3, 0))
        body(out,
             f"That list of {len(vec)} numbers IS the complaint, as far as the "
             "model is concerned. Two complaints that mean the same thing get "
             "two similar lists. Change a word above and every one of these "
             "figures changes.",
             fg=MUTED, size=8, wraplength=1000).pack(fill="x", pady=(1, 0))

        body(out, f"3.  Compared live against all {payload['pool_size']} "
                  f"complaints in {payload['source']}"
                  + (f" ({payload['pool_groups']} meaning groups)"
                     if payload["pool_groups"] else "")
                  + ".  Closest matches (1.00 = identical meaning):",
             size=9).pack(fill="x", pady=(10, 2))

        tree = ttk.Treeview(out, columns=("prep", "group", "sim"),
                            show="tree headings", height=5)
        tree.heading("#0", text="complaint (as written in the reference file)")
        tree.column("#0", width=360)
        tree.heading("prep", text="what was actually compared (preprocessed)")
        tree.column("prep", width=330, anchor="w")
        tree.heading("group", text="meaning group")
        tree.column("group", width=150, anchor="center")
        tree.heading("sim", text="cosine similarity")
        tree.column("sim", width=110, anchor="center")
        for i, (text, prepped, group, sim) in enumerate(payload["matches"]):
            tag = "best" if i == 0 else ""
            tree.insert("", "end", text=text, tags=(tag,),
                        values=(prepped, group or "-", f"{sim:.3f}"))
        tree.tag_configure("best", foreground="#2e9e5b",
                           font=("Segoe UI Semibold", 9))
        tree.pack(fill="x")

        best_sim = payload["matches"][0][3]
        verdict = ("a confident match" if best_sim >= 0.7 else
                   "a usable match" if best_sim >= 0.5 else
                   "a WEAK match - below the 0.5 cut-off")
        body(out,
             f"Top match scores {best_sim:.3f}, which is {verdict}. "
             f"For contrast, the mean similarity across the whole pool is "
             f"{payload['sim_mean']:.3f} and the worst is {payload['sim_min']:.3f} - "
             "if the top match were not grounded in the pool these three numbers "
             "would not spread apart.",
             fg=MUTED, size=8, wraplength=1000).pack(fill="x", pady=(6, 0))
        body(out,
             f"model: {payload['model']}    reference set: {payload['source']} "
             f"({payload['pool_size']} complaints, encoded once per session, then "
             "cosine-compared on every click)",
             fg=MUTED, size=8, wraplength=1000).pack(fill="x")
        self.status.set(f"Embedded {len(vec)} dimensions in "
                        f"{payload['encode_ms']:.0f} ms; "
                        f"best match {best_sim:.3f} of {payload['pool_size']}.")


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
            "    python triage_bow_fuzzy_diac.py        (dictionary fallback)")
        return
    TriageGUI().mainloop()


if __name__ == "__main__":
    main()
