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

Needs the trained model in triage_model/ (already shipped). The Results
tab additionally reads the CSV/JSON files produced by
train_embedding_pipeline.py and embedding_evaluation.py, and simply says
so if they have not been generated yet.
=======================================================================
"""

import csv
import json
import os
import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

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

MODEL_DIR = "triage_model"
EMBED_MODEL_DIR = "triage_model_embedding"
STOPWORDS_FILE = "learned_stopwords.json"
PIPELINE_RESULTS = "embedding_pipeline_results.csv"
EVAL_RESULTS = "embedding_evaluation_results.csv"
EVAL_NEIGHBOURS = "embedding_evaluation_neighbours.csv"
CLUSTERS_FILE = "evaluation_clusters.json"
DATASET_FILE = "triage_mixed_language_dataset.csv"

# Every figure the app shows is read from one of the files above. When a file
# is missing the app prints the command that produces it rather than inventing
# a plausible-looking number.
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
        bar = tk.Frame(self, bg=ACCENT, height=58)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="  Roman Urdu Emergency Triage",
                 bg=ACCENT, fg="white",
                 font=("Segoe UI Semibold", 15)).pack(side="left", padx=(14, 0))
        tk.Label(bar,
                 text="offline  |  CPU only  |  research prototype, not a medical device  ",
                 bg=ACCENT, fg="#cfe0f2",
                 font=("Segoe UI", 9)).pack(side="right")

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
            (self.tab_score, "  Model Score & Embedding Analysis  "),
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
        from triage_pipeline import load_artifacts
        art = load_artifacts(MODEL_DIR)
        report = None
        if os.path.exists(STOPWORDS_FILE):
            with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
                report = json.load(f)
        return art, report

    def _done_load_model(self, payload):
        self.artifacts, self.stopword_report = payload
        n_stops = len(self.stopword_report["stopwords"]) if self.stopword_report else 0
        self.status.set(
            f"Ready.  Model loaded from {MODEL_DIR}/  |  "
            f"{n_stops} learned stop words  |  everything runs offline.")
        self._fill_dropdowns()
        self._populate_stopwords()
        self.predict_btn.config(state="normal")

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

        # ---- result card ----
        res = card(right)
        res.pack(fill="both", expand=True)
        rpad = tk.Frame(res, bg=CARD)
        rpad.pack(fill="both", expand=True, padx=16, pady=14)

        heading(rpad, "Result").pack(fill="x")

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
        self.stages = tk.Text(rpad, height=7, font=("Consolas", 9),
                              wrap="word", relief="solid", bd=1,
                              bg="#fbfcfd", state="disabled")
        self.stages.pack(fill="both", expand=True)

    def _set_complaint(self, text):
        self.complaint.delete("1.0", "end")
        self.complaint.insert("1.0", text)

    def _fill_dropdowns(self):
        mapping = {
            "Gender": self.artifacts["le_gender"],
            "Mode_of_Arrival": self.artifacts["le_mode"],
            "AVPU": self.artifacts["le_avpu"],
            "ECG_Status": self.artifacts["le_ecg"],
        }
        defaults = {"Gender": "Male", "Mode_of_Arrival": "Ambulance",
                    "AVPU": "A", "ECG_Status": "ST elevation"}
        for name, enc in mapping.items():
            values = list(enc.classes_)
            self.combos[name]["values"] = values
            want = defaults.get(name)
            self.combos[name].set(want if want in values else values[0])

    def _do_predict(self):
        from triage_pipeline import (normalize_roman_urdu, predict_one,
                                     preprocess_for_embedding)

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

        try:
            level, confidence, proba = predict_one(
                self.artifacts, text,
                numbers["Age"], numbers["Heart_Rate"], numbers["Systolic_BP"],
                numbers["Diastolic_BP"], numbers["Temperature"], numbers["SpO2"],
                self.combos["Gender"].get(), self.combos["Mode_of_Arrival"].get(),
                self.combos["AVPU"].get(), self.combos["ECG_Status"].get())
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

        normalized = normalize_roman_urdu(text)
        cleaned = preprocess_for_embedding(text)
        removed = [w for w in normalized.split() if w not in cleaned.split()]

        self.stages.config(state="normal")
        self.stages.delete("1.0", "end")
        self.stages.insert("end", "1. raw input\n", "h")
        self.stages.insert("end", f"   {text}\n\n")
        self.stages.insert("end", "2. cleaned + fuzzy matched + diacritized\n", "h")
        self.stages.insert("end", f"   {normalized}\n\n")
        self.stages.insert("end", "3. learned stop words removed  (fed to the embedding model)\n", "h")
        self.stages.insert("end", f"   {cleaned}\n")
        if removed:
            self.stages.insert("end", f"\n   dropped: {', '.join(removed)}\n")
        self.stages.tag_config("h", foreground=ACCENT,
                               font=("Consolas", 9, "bold"))
        self.stages.config(state="disabled")

        self.status.set(
            f"Predicted Level {level + 1} ({LEVEL_NAMES[level]}) "
            f"at {confidence * 100:.1f}% confidence.")

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
        from triage_pipeline import (apply_diacritization, fuzzy_correct_text,
                                     normalize_roman_urdu,
                                     preprocess_for_embedding)
        import re

        for w in self.explore_out.winfo_children():
            w.destroy()

        raw = self.explore_var.get().strip()
        if not raw:
            return

        lowered = re.sub(r"[^a-z\s]", " ", raw.lower())
        fuzzed = fuzzy_correct_text(lowered)
        diacritized = apply_diacritization(fuzzed)
        normalized = normalize_roman_urdu(raw)
        final = preprocess_for_embedding(raw)

        stages = [
            ("1  Raw input", raw,
             "exactly what the triage nurse typed"),
            ("2  Lowercase + punctuation stripped", " ".join(lowered.split()),
             "everything except a-z becomes a space"),
            ("3  Fuzzy spelling correction", fuzzed,
             "RapidFuzz maps misspellings onto the canonical vocabulary at >=80% similarity"),
            ("4  Diacritization", diacritized,
             "every known spelling variant collapses to one canonical form"),
            ("5  Learned stop words removed", final,
             "Contribution 1 - the list was learned from the data, not hand-written"),
        ]

        for title, text, note in stages:
            block = tk.Frame(self.explore_out, bg=CARD)
            block.pack(fill="x", pady=(0, 10))
            tk.Label(block, text=title, bg=CARD, fg=ACCENT,
                     font=("Segoe UI Semibold", 10), anchor="w").pack(fill="x")
            tk.Label(block, text=text or "(empty)", bg="#fbfcfd", fg=INK,
                     font=("Consolas", 11), anchor="w", justify="left",
                     wraplength=1000, padx=10, pady=7,
                     relief="solid", bd=1).pack(fill="x", pady=(2, 1))
            tk.Label(block, text=note, bg=CARD, fg=MUTED,
                     font=("Segoe UI", 8), anchor="w").pack(fill="x")

        # ---- word-by-word accounting ----
        dropped = [w for w in normalized.split() if w not in final.split()]
        if dropped and self.stopword_report:
            stats = {t["token"]: t for t in self.stopword_report["token_statistics"]}
            box = tk.Frame(self.explore_out, bg=CARD)
            box.pack(fill="x", pady=(4, 0))
            tk.Label(box, text="Why those words were dropped", bg=CARD, fg=ACCENT,
                     font=("Segoe UI Semibold", 10), anchor="w").pack(fill="x")
            tree = ttk.Treeview(box, columns=("df", "mi", "chi", "p"),
                                show="tree headings", height=min(6, len(dropped)))
            tree.heading("#0", text="token")
            tree.column("#0", width=150)
            for col, label, w in [("df", "doc frequency", 110),
                                  ("mi", "normalized MI", 110),
                                  ("chi", "chi-square", 100),
                                  ("p", "p-value", 90)]:
                tree.heading(col, text=label)
                tree.column(col, width=w, anchor="center")
            for token in dropped:
                s = stats.get(token)
                if s:
                    tree.insert("", "end", text=token, values=(
                        f"{s['document_frequency']:.3f}",
                        f"{s['normalized_mutual_information']:.5f}",
                        f"{s['chi_square']:.2f}",
                        f"{s['chi_square_p_value']:.4f}"))
                else:
                    tree.insert("", "end", text=token,
                                values=("-", "-", "-", "-"))
            tree.pack(fill="x", pady=(4, 2))
            tk.Label(box,
                     text=("Dropped because the token is common AND its mutual information "
                           "with the triage level is near zero AND the chi-square test is "
                           "not significant (p > 0.05)."),
                     bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                     anchor="w", wraplength=980, justify="left").pack(fill="x")
        elif not dropped:
            tk.Label(self.explore_out,
                     text="No stop words were present in this complaint.",
                     bg=CARD, fg=MUTED, font=("Segoe UI", 9),
                     anchor="w").pack(fill="x")

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
                             ("stop", "learned stop words"),
                             ("protected", "rescued by the clinical guard")]:
            tk.Radiobutton(filt, text=label, value=value,
                           variable=self.stop_filter, bg=CARD, fg=INK,
                           font=("Segoe UI", 9), activebackground=CARD,
                           selectcolor=CARD,
                           command=self._populate_stopwords).pack(side="left", padx=6)

        cols = ("df", "count", "mi", "chi", "p", "verdict")
        self.stop_tree = ttk.Treeview(pad, columns=cols, show="tree headings")
        self.stop_tree.heading("#0", text="token")
        self.stop_tree.column("#0", width=160)
        for col, label, w in [("df", "doc frequency", 110), ("count", "documents", 90),
                              ("mi", "normalized MI", 115), ("chi", "chi-square", 100),
                              ("p", "p-value", 95), ("verdict", "decision", 230)]:
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
        self.stop_summary.configure(text=(
            f"Rule:  document frequency >= {t['effective_df_cutoff']:.4f}"
            f"   AND   normalized mutual information <= {t['mi_threshold']}"
            f"   AND   chi-square p > {t['alpha']}\n"
            f"{c['n_documents']} complaints, {c['n_unique_tokens']} unique tokens, "
            f"{c['n_tokens_tested']} high-frequency tokens tested, "
            f"{r['n_stopwords']} learned as stop words."
            + (f"\nNeeds clinician sign-off (negation / intensity): "
               f"{', '.join(review)}" if review else "")))

        for row in self.stop_tree.get_children():
            self.stop_tree.delete(row)

        mode = self.stop_filter.get()
        for s in r["token_statistics"]:
            if mode == "stop" and not s["is_stopword"]:
                continue
            if mode == "protected" and not (s["clinically_protected"]
                                            and s["is_uninformative"]):
                continue
            if s["is_stopword"]:
                verdict, tag = "STOP WORD - removed", "stop"
            elif s["clinically_protected"] and s["is_uninformative"]:
                verdict, tag = "kept - clinical safety guard", "prot"
            elif not s["is_uninformative"]:
                verdict, tag = "kept - carries triage signal", ""
            else:
                verdict, tag = "kept", ""
            self.stop_tree.insert("", "end", text=s["token"], tags=(tag,), values=(
                f"{s['document_frequency']:.4f}", s["document_count"],
                f"{s['normalized_mutual_information']:.5f}",
                f"{s['chi_square']:.2f}", f"{s['chi_square_p_value']:.4f}",
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
             "categories fall back safely, with a note recorded per row.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 10))

        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x")
        self.batch_path = tk.StringVar(value="sample_100_patients.xlsx")
        tk.Entry(row, textvariable=self.batch_path, font=("Segoe UI", 10),
                 relief="solid", bd=1).pack(side="left", fill="x",
                                            expand=True, ipady=3)
        ttk.Button(row, text="Browse...",
                   command=self._browse_batch).pack(side="left", padx=6)
        ttk.Button(row, text="Run batch triage", style="Accent.TButton",
                   command=self._do_batch).pack(side="left")

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
        import pandas as pd
        from triage_pipeline import predict_dataframe

        path = self._batch_target
        ext = os.path.splitext(path)[1].lower()
        df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
        results, _ = predict_dataframe(self.artifacts, df)

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

        self._results_section_methods(holder)
        self._results_section_embedding(holder)

    def _results_section_methods(self, parent):
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Which text representation wins?").pack(fill="x")
        body(pad,
             "Same train/test split, same vital signs, same Logistic Regression - only "
             "the text representation changes. Produced by train_embedding_pipeline.py.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 10))

        if not os.path.exists(PIPELINE_RESULTS):
            body(pad, f"Not generated yet. Run:  python train_embedding_pipeline.py",
                 fg="#c0392b", size=9).pack(fill="x")
            return

        rows = read_csv_rows(PIPELINE_RESULTS)
        cols = ("acc", "under", "over", "grade", "feat")
        tree = ttk.Treeview(pad, columns=cols, show="tree headings",
                            height=len(rows) + 1)
        tree.heading("#0", text="method")
        tree.column("#0", width=330)
        for col, label, w in [("acc", "accuracy %", 100),
                              ("under", "under-triage %", 120),
                              ("over", "over-triage %", 115),
                              ("grade", "safety", 80), ("feat", "features", 90)]:
            tree.heading(col, text=label)
            tree.column(col, width=w, anchor="center")

        best_acc = max(fnum(r["accuracy"], 0) for r in rows)
        min_under = min(fnum(r["under_triage_pct"], 99) for r in rows)
        for r in rows:
            acc, under = fnum(r["accuracy"], 0), fnum(r["under_triage_pct"], 0)
            tags = []
            if under <= min_under + 0.5 and acc == max(
                    fnum(x["accuracy"], 0) for x in rows
                    if fnum(x["under_triage_pct"], 99) <= min_under + 0.5):
                tags = ["sel"]
            elif acc == best_acc:
                tags = ["acc"]
            tree.insert("", "end", text=r["method"], tags=tags, values=(
                f"{acc:.2f}", f"{under:.2f}",
                f"{fnum(r['over_triage_pct'], 0):.2f}",
                r.get("safety_grade", ""), r.get("n_features", "")))
        tree.tag_configure("sel", foreground="#2e9e5b",
                           font=("Segoe UI Semibold", 9))
        tree.tag_configure("acc", foreground=ACCENT)
        tree.pack(fill="x")

        body(pad,
             "Green = selected for deployment.  Blue = most accurate.\n"
             "They differ on purpose: under-triage means a critically ill patient is sent "
             "to the back of the queue, so the selection rule minimises it first and uses "
             "accuracy only as the tie-breaker.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(8, 0))

    def _results_section_embedding(self, parent):
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Contribution 2  -  how good is the embedding generator?").pack(fill="x")
        body(pad,
             "Within-cluster similarity per manual meaning-cluster. A good embedding "
             "generator puts same-meaning complaints close together (most pairs above "
             "0.5). Produced by embedding_evaluation.py.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 10))

        if not os.path.exists(EVAL_RESULTS):
            body(pad, "Not generated yet. Run:  python embedding_evaluation.py",
                 fg="#c0392b", size=9).pack(fill="x")
            return

        rows = read_csv_rows(EVAL_RESULTS)
        clusters = [r for r in rows if r["cluster"] != "OVERALL"]
        overall = next((r for r in rows if r["cluster"] == "OVERALL"), None)

        pct_key = next((k for k in rows[0] if k.startswith("pct_above_")), None)
        chart = tk.Canvas(pad, bg=CARD, highlightthickness=0,
                          height=26 * len(clusters) + 16)
        chart.pack(fill="x")

        def draw_clusters(_event=None):
            chart.delete("all")
            width = chart.winfo_width()
            if width < 80:                  # Configure will call back once sized
                return
            # As with the probability bars, the right-hand figures are anchored
            # to the east edge so they cannot be clipped at any window width.
            right_pad = 200
            label_w = min(190, max(0, width - right_pad - 40))
            bar_w = max(20, width - label_w - right_pad)
            for i, r in enumerate(clusters):
                y = 6 + i * 26
                pct = fnum(r.get(pct_key), 0.0)
                sim = fnum(r["mean_cosine_similarity"], 0.0)
                colour = ("#2e9e5b" if pct >= 80 else
                          "#e08e0b" if pct >= 50 else "#c0392b")
                if label_w > 60:
                    chart.create_text(0, y + 9, anchor="w", text=r["cluster"],
                                      font=("Segoe UI", 9), fill=INK)
                chart.create_rectangle(label_w, y, label_w + bar_w, y + 18,
                                       fill="#eef1f4", outline="")
                chart.create_rectangle(label_w, y,
                                       label_w + max(2, bar_w * pct / 100),
                                       y + 18, fill=colour, outline="")
                chart.create_text(width - 2, y + 9, anchor="e",
                                  text=f"{pct:.1f}% of {r['n_pairs']} pairs"
                                       f"    sim {sim:.3f}",
                                  font=("Segoe UI", 8), fill=MUTED)

        # Draw once the real width is known, and again on every resize.
        chart.bind("<Configure>", draw_clusters)

        if overall:
            gap = fnum(overall.get("separation_gap"), 0.0)
            closed = fnum(overall.get("nn_efficiency_closed_pool_pct"), 0.0)
            open_pool = fnum(overall.get("nn_efficiency_open_pool_pct"))
            summary = tk.Frame(pad, bg="#f7f9fb", highlightbackground=LINE,
                               highlightthickness=1)
            summary.pack(fill="x", pady=(12, 0))
            inner = tk.Frame(summary, bg="#f7f9fb")
            inner.pack(fill="x", padx=14, pady=10)
            tk.Label(inner, text="Headline numbers for the paper", bg="#f7f9fb",
                     fg=ACCENT, font=("Segoe UI Semibold", 10),
                     anchor="w").pack(fill="x")
            lines = [
                f"within-cluster similarity   {fnum(overall['mean_cosine_similarity'], 0):.3f}"
                f"      across-cluster   {fnum(overall.get('across_cluster_similarity'), 0):.3f}"
                f"      separation gap   {gap:+.3f}",
                f"pairs above threshold       {overall.get(pct_key, '?')}%"
                f"  of {overall['n_pairs']} pairs",
                f"embedding generator efficiency   {closed:.1f}%  (closed pool)"
                + (f"      {open_pool:.1f}%  (open pool, with distractors)"
                   if open_pool is not None else ""),
                f"model  {overall.get('embedding_model', '?')}"
                f"   ({overall.get('embedding_dim', '?')} dims)",
            ]
            for line in lines:
                tk.Label(inner, text=line, bg="#f7f9fb", fg=INK,
                         font=("Consolas", 9), anchor="w",
                         justify="left").pack(fill="x")
            tk.Label(inner,
                     text=("Round-trip fidelity is measured as nearest-neighbour agreement: "
                           "sentence embeddings are lossy and one-way, so the vector cannot "
                           "be turned back into its sentence. Instead each complaint's "
                           "closest neighbour is checked for the same meaning."),
                     bg="#f7f9fb", fg=MUTED, font=("Segoe UI", 8), anchor="w",
                     justify="left", wraplength=960).pack(fill="x", pady=(6, 0))


    # =================================================================
    # TAB 6 - Model Score & Embedding Analysis
    # =================================================================
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

        self._score_section_model(holder)
        self._score_section_pairs(holder)
        self._score_section_demo(holder)

    # ----------------------------------------------------- (a) score
    def _score_section_model(self, parent):
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Model score").pack(fill="x")
        body(pad,
             "Measured on the held-out test patients the model never saw during "
             "training. Every figure below is read from the saved result files - "
             "nothing here is typed in by hand.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 12))

        shown = 0
        for path, title, note in [
            (os.path.join(EMBED_MODEL_DIR, "triage_metrics.json"),
             "Selected pipeline", "chosen by the safety-first rule"),
            (os.path.join(MODEL_DIR, "triage_metrics.json"),
             "Dictionary model (shipped in triage_model/)",
             "the model the prediction tabs actually use"),
        ]:
            if not os.path.exists(path):
                missing_notice(pad, path).pack(fill="x", pady=(0, 8))
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    m = json.load(f)
            except (OSError, ValueError):
                missing_notice(pad, path,
                               "The file exists but could not be read.").pack(
                    fill="x", pady=(0, 8))
                continue

            # The two files use different key names and different scales
            # (0-1 vs 0-100), so normalise before displaying.
            if "overall_under_triage_rate" in m:
                acc = fnum(m.get("accuracy"), 0.0) * 100
                under = fnum(m.get("overall_under_triage_rate"), 0.0)
                over = fnum(m.get("overall_over_triage_rate"), 0.0)
                subtitle = m.get("winning_method", "")
                n_test = m.get("total_patients")
            else:
                acc = fnum(m.get("accuracy"), 0.0)
                under = fnum(m.get("under_triage_pct"), 0.0)
                over = fnum(m.get("over_triage_pct"), 0.0)
                subtitle = m.get("winning_method", "")
                n_test = m.get("n_test")

            self._score_card(pad, title, note, subtitle, acc, under, over,
                             n_test, m, path)
            shown += 1

        if shown:
            body(pad,
                 "Under-triage is the dangerous error: the patient is rated LESS "
                 "urgent than they really are, so a critically ill person waits. "
                 "Over-triage only wastes staff time. That is why the selection "
                 "rule minimises under-triage first and treats accuracy as the "
                 "tie-breaker.",
                 fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(10, 0))

    def _score_card(self, parent, title, note, subtitle, acc, under, over,
                    n_test, raw, path):
        wrap = tk.Frame(parent, bg="#f7f9fb", highlightbackground=LINE,
                        highlightthickness=1)
        wrap.pack(fill="x", pady=(0, 10))
        inner = tk.Frame(wrap, bg="#f7f9fb")
        inner.pack(fill="x", padx=14, pady=12)

        tk.Label(inner, text=title, bg="#f7f9fb", fg=ACCENT,
                 font=("Segoe UI Semibold", 11), anchor="w").pack(fill="x")
        line = note if not subtitle else f"{note}  ·  {subtitle}"
        tk.Label(inner, text=line, bg="#f7f9fb", fg=MUTED,
                 font=("Segoe UI", 8), anchor="w",
                 wraplength=960, justify="left").pack(fill="x", pady=(0, 8))

        stats = tk.Frame(inner, bg="#f7f9fb")
        stats.pack(fill="x")
        cells = [
            ("Accuracy", f"{acc:.2f}%", ACCENT, "correct triage decisions"),
            ("Under-triage", f"{under:.2f}%", "#c0392b", "rated too LOW - dangerous"),
            ("Over-triage", f"{over:.2f}%", "#e08e0b", "rated too HIGH - wasteful"),
        ]
        grade = raw.get("safety_grade")
        if grade:
            cells.append(("Safety grade", str(grade), "#2e9e5b", "under-triage band"))
        for i, (label, value, colour, hint) in enumerate(cells):
            stats.columnconfigure(i, weight=1, uniform="stat")
            cell = tk.Frame(stats, bg="#f7f9fb")
            cell.grid(row=0, column=i, sticky="ew", padx=(0, 10))
            tk.Label(cell, text=label, bg="#f7f9fb", fg=MUTED,
                     font=("Segoe UI", 8), anchor="w").pack(fill="x")
            tk.Label(cell, text=value, bg="#f7f9fb", fg=colour,
                     font=("Segoe UI Semibold", 17), anchor="w").pack(fill="x")
            tk.Label(cell, text=hint, bg="#f7f9fb", fg=MUTED,
                     font=("Segoe UI", 8), anchor="w").pack(fill="x")

        bits = [f"source: {path}"]
        if n_test:
            bits.append(f"{n_test} held-out test patients")
        if raw.get("embedding_model"):
            bits.append(f"{raw['embedding_model']} ({raw.get('embedding_dim', '?')} dims)")
        tk.Label(inner, text="   |   ".join(bits), bg="#f7f9fb", fg=MUTED,
                 font=("Segoe UI", 8), anchor="w", wraplength=960,
                 justify="left").pack(fill="x", pady=(8, 0))

        cm = raw.get("confusion_matrix")
        if cm:
            tk.Label(inner, text="Confusion matrix  (rows = true level, "
                                 "columns = predicted level)",
                     bg="#f7f9fb", fg=MUTED, font=("Segoe UI", 8),
                     anchor="w").pack(fill="x", pady=(8, 2))
            grid = tk.Frame(inner, bg="#f7f9fb")
            grid.pack(fill="x")
            for j in range(4):
                tk.Label(grid, text=f"pred L{j + 1}", bg="#f7f9fb", fg=MUTED,
                         font=("Segoe UI", 8), width=9).grid(row=0, column=j + 1)
            for i, row in enumerate(cm):
                tk.Label(grid, text=f"true L{i + 1}", bg="#f7f9fb", fg=MUTED,
                         font=("Segoe UI", 8), width=9,
                         anchor="e").grid(row=i + 1, column=0, sticky="e")
                for j, v in enumerate(row):
                    on_diag = i == j
                    tk.Label(grid, text=str(v), bg="#f7f9fb",
                             fg=INK if on_diag else MUTED, width=9,
                             font=("Consolas", 9, "bold") if on_diag
                             else ("Consolas", 9)).grid(row=i + 1, column=j + 1)

    # ------------------------------------------------ (b) the 45 maths
    def _score_section_pairs(self, parent):
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Where the number 45 comes from").pack(fill="x")
        body(pad,
             "To check whether the AI groups similar complaints together, every "
             "complaint in a group is compared against every other complaint in "
             "that group. Comparing 10 sentences does not give 10 comparisons - "
             "it gives 45.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 12))

        maths = tk.Frame(pad, bg="#f7f9fb", highlightbackground=LINE,
                         highlightthickness=1)
        maths.pack(fill="x")
        mi = tk.Frame(maths, bg="#f7f9fb")
        mi.pack(fill="x", padx=14, pady=12)

        for text, note in [
            ("Each of the 10 sentences is compared with the other 9",
             "10  x  9  =  90"),
            ("But A-vs-B is the same comparison as B-vs-A, so halve it",
             "90  /  2  =  45"),
        ]:
            r = tk.Frame(mi, bg="#f7f9fb")
            r.pack(fill="x", pady=2)
            tk.Label(r, text=text, bg="#f7f9fb", fg=INK, font=("Segoe UI", 9),
                     anchor="w").pack(side="left")
            tk.Label(r, text=note, bg="#f7f9fb", fg=ACCENT,
                     font=("Consolas", 11, "bold"), anchor="e").pack(side="right")

        tk.Frame(mi, bg=LINE, height=1).pack(fill="x", pady=8)
        tk.Label(mi, text="The same thing written the formal way "
                          "(the 'n choose 2' formula):",
                 bg="#f7f9fb", fg=MUTED, font=("Segoe UI", 8),
                 anchor="w").pack(fill="x")
        tk.Label(mi,
                 text="C(10,2)  =  10!  /  ( 2!  x  8! )  =  "
                      "3628800 / (2 x 40320)  =  45",
                 bg="#f7f9fb", fg=INK, font=("Consolas", 11),
                 anchor="w").pack(fill="x", pady=(2, 0))
        tk.Label(mi,
                 text="A group of 9 sentences gives C(9,2) = 9 x 8 / 2 = 36 pairs, "
                      "not 45. The table below shows the real count for each group.",
                 bg="#f7f9fb", fg=MUTED, font=("Segoe UI", 8), anchor="w",
                 wraplength=960, justify="left").pack(fill="x", pady=(6, 0))

        # ---- real per-cluster numbers ----
        body(pad, "Real results, per meaning-group", size=10).pack(
            fill="x", pady=(14, 2))

        if not os.path.exists(EVAL_RESULTS):
            missing_notice(pad, EVAL_RESULTS,
                           "This table shows the measured pair counts and pass "
                           "rates, so it stays empty until the study has been "
                           "run.").pack(fill="x")
            return

        rows = read_csv_rows(EVAL_RESULTS)
        clusters = [r for r in rows if r["cluster"] != "OVERALL"]
        overall = next((r for r in rows if r["cluster"] == "OVERALL"), None)
        pct_key = next((k for k in rows[0] if k.startswith("pct_above_")), None)
        above_key = next((k for k in rows[0] if k.startswith("pairs_above_")), None)
        threshold = (pct_key or "pct_above_0.5").rsplit("_", 1)[-1]

        cols = ("n", "calc", "pairs", "above", "pct")
        tree = ttk.Treeview(pad, columns=cols, show="tree headings",
                            height=len(clusters) + 1)
        tree.heading("#0", text="meaning group")
        tree.column("#0", width=210)
        for col, label, w in [("n", "sentences", 90),
                              ("calc", "calculation", 150),
                              ("pairs", "pairs", 80),
                              ("above", f"above {threshold}", 110),
                              ("pct", "pass rate", 100)]:
            tree.heading(col, text=label)
            tree.column(col, width=w, anchor="center")

        for r in clusters:
            n = int(fnum(r["n_complaints"], 0))
            pairs = int(fnum(r["n_pairs"], 0))
            pct = fnum(r.get(pct_key), 0.0)
            tag = ("good" if pct >= 80 else "mid" if pct >= 50 else "low")
            tree.insert("", "end", text=r["cluster"], tags=(tag,), values=(
                n, f"{n} x {n - 1} / 2", pairs,
                r.get(above_key, "-"), f"{pct:.1f}%"))
        if overall:
            tree.insert("", "end", text="ALL GROUPS", tags=("total",), values=(
                int(fnum(overall["n_complaints"], 0)), "sum of the above",
                int(fnum(overall["n_pairs"], 0)),
                overall.get(above_key, "-"),
                f"{fnum(overall.get(pct_key), 0.0):.1f}%"))
        tree.tag_configure("good", foreground="#2e9e5b")
        tree.tag_configure("mid", foreground="#e08e0b")
        tree.tag_configure("low", foreground="#c0392b")
        tree.tag_configure("total", font=("Segoe UI Semibold", 9))
        tree.pack(fill="x", pady=(2, 0))

        body(pad,
             f"'Pass rate' is the share of pairs scoring above {threshold} "
             "similarity - the cut-off for treating two complaints as meaning "
             "the same thing. Green is 80% or better, amber 50-79%, red below "
             "50%.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(8, 0))

    # --------------------------------------------- (c) embedding demo
    def _score_section_demo(self, parent):
        box = card(parent)
        box.pack(fill="x", padx=4, pady=(0, 10))
        pad = tk.Frame(box, bg=CARD)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        heading(pad, "Embedding demo  -  see a sentence become numbers").pack(fill="x")
        body(pad,
             "Type any complaint. The AI model turns it into a list of numbers, "
             "then the closest matching complaint from the evaluation set is "
             "found by comparing those numbers. This is exactly how the model "
             "decides two complaints mean the same thing. Runs offline.",
             fg=MUTED, size=9, wraplength=1000).pack(fill="x", pady=(2, 10))

        row = tk.Frame(pad, bg=CARD)
        row.pack(fill="x")
        self.demo_var = tk.StringVar(value="seena mein dard aur pasina")
        entry = tk.Entry(row, textvariable=self.demo_var, font=("Segoe UI", 12),
                         relief="solid", bd=1)
        entry.pack(side="left", fill="x", expand=True, ipady=4)
        entry.bind("<Return>", lambda e: self._run_demo())
        self.demo_btn = ttk.Button(row, text="Embed and match",
                                   style="Accent.TButton", command=self._run_demo)
        self.demo_btn.pack(side="left", padx=(8, 0))

        self.demo_out = tk.Frame(pad, bg=CARD)
        self.demo_out.pack(fill="both", expand=True, pady=(12, 0))
        body(self.demo_out,
             "The embedding model loads the first time you press the button "
             "(a few seconds).",
             fg=MUTED, size=9).pack(fill="x")

        self._demo_model = None
        self._demo_model_name = ""
        self._demo_corpus = None
        self._demo_text = self.demo_var.get()

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
        import numpy as np
        from triage_pipeline import preprocess_for_embedding

        if self._demo_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return {"error":
                        "The 'sentence-transformers' library is not installed, so "
                        "the embedding demo cannot run.\n\n"
                        "Install it once (needs internet), then try again:\n"
                        "    pip install -r requirements-embedding.txt\n\n"
                        "Everything else in this app works without it."}
            name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Read the model actually used by the study, so the demo matches
            # the reported numbers instead of quietly using a different model.
            if os.path.exists(EVAL_RESULTS):
                for r in read_csv_rows(EVAL_RESULTS):
                    if r.get("embedding_model"):
                        name = r["embedding_model"]
                        break
            self._demo_model = SentenceTransformer(name, device="cpu")
            self._demo_model_name = name

        if self._demo_corpus is None:
            pairs, source = self._demo_reference_corpus()
            if not pairs:
                return {"error": "No reference complaints found. Expected "
                                 f"{CLUSTERS_FILE} or {DATASET_FILE}."}
            texts = [preprocess_for_embedding(t) for t, _ in pairs]
            vecs = self._demo_model.encode(texts, batch_size=32,
                                           convert_to_numpy=True,
                                           show_progress_bar=False,
                                           normalize_embeddings=True)
            self._demo_corpus = (pairs, vecs, source)

        pairs, vecs, source = self._demo_corpus
        raw = self._demo_text
        cleaned = preprocess_for_embedding(raw)
        vec = self._demo_model.encode([cleaned], convert_to_numpy=True,
                                      normalize_embeddings=True)[0]
        sims = vecs @ vec
        order = np.argsort(-sims)[:5]
        return {
            "raw": raw,
            "cleaned": cleaned,
            "vector": vec,
            "model": self._demo_model_name,
            "source": source,
            "matches": [(pairs[i][0], pairs[i][1], float(sims[i])) for i in order],
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

        body(out, f"1.  After cleaning:   {payload['cleaned']}",
             size=9).pack(fill="x")
        body(out, f"2.  The model turns that into {len(vec)} numbers "
                  f"(only the first 24 are shown):",
             size=9).pack(fill="x", pady=(8, 2))

        nums = "  ".join(f"{v:+.4f}" for v in vec[:24])
        tk.Label(out, text=nums, bg="#fbfcfd", fg=INK, font=("Consolas", 9),
                 anchor="w", justify="left", wraplength=980, padx=10, pady=8,
                 relief="solid", bd=1).pack(fill="x")
        body(out,
             f"That list of {len(vec)} numbers IS the complaint, as far as the "
             "model is concerned. Two complaints that mean the same thing get "
             "two similar lists.",
             fg=MUTED, size=8).pack(fill="x", pady=(3, 0))

        body(out, "3.  Closest matching complaints (1.00 = identical meaning):",
             size=9).pack(fill="x", pady=(10, 2))

        tree = ttk.Treeview(out, columns=("group", "sim"),
                            show="tree headings", height=5)
        tree.heading("#0", text="complaint")
        tree.column("#0", width=520)
        tree.heading("group", text="meaning group")
        tree.column("group", width=180, anchor="center")
        tree.heading("sim", text="similarity")
        tree.column("sim", width=100, anchor="center")
        for i, (text, group, sim) in enumerate(payload["matches"]):
            tag = "best" if i == 0 else ""
            tree.insert("", "end", text=text, tags=(tag,),
                        values=(group or "-", f"{sim:.3f}"))
        tree.tag_configure("best", foreground="#2e9e5b",
                           font=("Segoe UI Semibold", 9))
        tree.pack(fill="x")

        best_sim = payload["matches"][0][2]
        verdict = ("a confident match" if best_sim >= 0.7 else
                   "a usable match" if best_sim >= 0.5 else
                   "a WEAK match - below the 0.5 cut-off")
        body(out,
             f"Top match scores {best_sim:.3f}, which is {verdict}.    "
             f"model: {payload['model']}    reference set: {payload['source']}",
             fg=MUTED, size=8, wraplength=1000).pack(fill="x", pady=(6, 0))
        self.status.set(f"Embedded {len(vec)} dimensions; "
                        f"best match {best_sim:.3f}.")


def main():
    enable_dpi_awareness()
    if not os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model not found",
            f"Could not find {MODEL_DIR}/model.pkl.\n\n"
            "Train the model first:\n    python triage_bow_fuzzy_diac.py")
        return
    TriageGUI().mainloop()


if __name__ == "__main__":
    main()
