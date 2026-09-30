"""Build the eight-page digest of the recorded study, without historical appendices."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
import reportlab

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports/triage_10000"
REF = Path(__file__).parent / "reference"
WORK = ROOT / "tmp/pdfs/short_report"
WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "output/pdf/Triage_Classifier_Concise_Report.pdf"
F = json.loads((DATA / "final_results.json").read_text())
F = sorted(F, key=lambda r: r["id"] != "selected_system")
A = json.loads((DATA / "data_audit.json").read_text())
S = F[0]
fonts = Path(reportlab.__file__).parent / "fonts"
for name, file in [("Vera", "Vera.ttf"), ("VeraBold", "VeraBd.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
styles = getSampleStyleSheet()
for s in styles.byName.values():
    s.fontName = "Vera"
    s.textColor = colors.black
styles["Title"].fontName = "VeraBold"
styles["Title"].fontSize = 18
styles["Title"].leading = 23
styles["Heading2"].fontName = "VeraBold"
styles["Heading2"].fontSize = 11
styles["Heading2"].leading = 14
styles["BodyText"].fontSize = 9
styles["BodyText"].leading = 12.5
styles.add(styles["BodyText"].clone("Small"))
styles["Small"].fontSize = 7.7
styles["Small"].leading = 10
story = []


def p(text, style="BodyText"):
    story.extend([Paragraph(text, styles[style]), Spacer(1, 6)])


def page(title):
    if story:
        story.append(PageBreak())
    p(title, "Title")


def table(rows, widths, size=8, pad=4):
    cellstyle = styles["Small"].clone("cell")
    cellstyle.fontSize = size
    cellstyle.leading = size + 2
    rows = [[Paragraph(str(v), cellstyle) for v in row] for row in rows]
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), pad),
                ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
            ]
        )
    )
    story.extend([t, Spacer(1, 9)])


def img(path, w, h):
    story.extend([Image(str(path), width=w, height=h), Spacer(1, 6)])


def pct(x):
    return f"{100*x:.2f}%"


def num(x):
    return f"{x:.4f}"


clf = {"hgb": "HGB", "logreg": "LR", "rf": "RF"}


def desc(c):
    f = dict(c["features"])
    if f["view"] == "structured":
        f.pop("encoder", None)
        f.pop("pca", None)
    view = {"structured": "Patient features", "fused": "Combined", "text": "Text only"}[
        f["view"]
    ]
    enc = {
        "sapbert_concept": "SapBERT",
        "mpnet_concept": "MPNet",
        "minilm_complaint": "MiniLM",
    }.get(f.get("encoder"), "")
    dims = (" / " + str(f["pca"]) + "D") if f.get("pca") else (" / full" if enc else "")
    return (
        f"{clf[c['classifier']]}: {view}"
        + (" / " + enc if enc else "")
        + dims
        + (" + derived" if f.get("derived") else "")
    )


names = {r["id"]: f"M{i+1}" for i, r in enumerate(F)}
page("Roman Urdu Medical Triage\nClassifier Comparison")
p("Eight-page research summary | Supplied 10,000-record dataset", "Heading2")
p(
    "The study compares three classifiers for predicting triage levels from complaint representations and patient features. It uses the latest uploaded classifier settings as baselines, then evaluates additional settings under the same grouped test split. The language encoders remain frozen; training and tuning apply to the classifiers."
)
table(
    [
        ["Records", "Development", "Final test", "Target levels"],
        ["10,000", "8,001", "1,999", "1, 2 and 3"],
    ],
    [125] * 4,
    10,
    8,
)
p("Main results", "Heading2")
table(
    [
        ["Model", "Accuracy", "Macro recall", "Macro F1"],
        [
            "Development-selected HGB",
            pct(S["metrics"]["accuracy"]),
            num(S["metrics"]["recall_macro"]),
            num(S["metrics"]["macro_f1"]),
        ],
    ]
    + [
        [
            label,
            pct(next(r for r in F if r["id"] == id)["metrics"]["accuracy"]),
            num(next(r for r in F if r["id"] == id)["metrics"]["recall_macro"]),
            num(next(r for r in F if r["id"] == id)["metrics"]["macro_f1"]),
        ]
        for label, id in [
            ("Original fused HGB settings", "6be327aea023a6"),
            ("Random Forest comparison", "37444b1406c5fa"),
        ]
    ],
    [248, 84, 84, 84],
    9,
    7,
)
p(
    "The selected system was chosen using development predictions, before the final test was evaluated. It did not exceed the strongest baseline on this test. Random Forest obtained the highest comparison score, but was not selected retrospectively from test results."
)
p("Data preparation and split", "Heading2")
p(
    "Twelve input fields and the target were retained. The constant Category field and five processing metadata columns were excluded. One invalid arrival entry was treated as missing; no records were removed in this run. Labels were not changed."
)
p(
    "Records sharing the same normalized complaint or clinical concept were linked into 6,573 groups. Approximately 20% were reserved for testing. Three-fold screening and five-fold refinement used development groups only. Imputation, scaling, categorical encoding and PCA were fitted separately in each training fold."
)
table(
    [["Level", "All records", "Development", "Test"]]
    + [
        [
            k,
            A["class_counts"][k],
            A["development_class_counts"][k],
            A["test_class_counts"][k],
        ]
        for k in ["1", "2", "3"]
    ],
    [125] * 4,
)
p(
    "Methods note: the supplied target labels were assigned by an AI model, as confirmed by the dataset provider. Scores measure agreement with those labels. The dataset has no level-4 examples; independent clinical validation was not part of this experiment.",
    "Small",
)
page("Methods and final-model tables")
p(
    "SapBERT encodes clinical concepts with CLS pooling (768-D, 64-token limit). MPNet encodes concepts (768-D), and multilingual MiniLM encodes Roman Urdu complaints (384-D), both with a 128-token limit. Vectors are normalized. PCA 32, 64 and 128, and full dimensions, were tested. LR = Logistic Regression; HGB = Hist Gradient Boosting; RF = Random Forest. P/R/F1 are macro-averaged."
)
rows = [["ID / configuration", "Accuracy", "Precision", "Recall", "F1"]]
for r in F:
    rows.append(
        [
            names[r["id"]]
            + " "
            + (
                "Selected adjusted HGB"
                if r["id"] == "selected_system"
                else desc(r["config"])
            ),
            pct(r["metrics"]["accuracy"]),
            num(r["metrics"]["precision_macro"]),
            num(r["metrics"]["recall_macro"]),
            num(r["metrics"]["macro_f1"]),
        ]
    )
table(rows, [244, 64, 64, 64, 64], 7.7, 5)
p("Directional errors and agreement", "Heading2")
rows = [["ID", "QWK", "MAE", "Under-triage", "Over-triage"]]
for r in F:
    m = r["metrics"]
    rows.append(
        [
            names[r["id"]],
            num(m["qwk"]),
            num(m["mae"]),
            f"{round(m['under_triage_rate']*1999)} ({pct(m['under_triage_rate'])})",
            f"{round(m['over_triage_rate']*1999)} ({pct(m['over_triage_rate'])})",
        ]
    )
table(rows, [60, 90, 90, 130, 130], 8, 4)
p(
    "QWK = quadratic weighted kappa; MAE = mean absolute level error. Under-triage means predicting a higher numbered (less urgent) level; over-triage means predicting a lower numbered level. All ten rows use the same 1,999 test records.",
    "Small",
)
page("Accuracy and F1 comparison")
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
for ax, key, title in zip(
    axes, ["accuracy", "macro_f1"], ["Test accuracy (%)", "Test macro F1 (%)"]
):
    vals = [r["metrics"][key] * 100 for r in F]
    bars = ax.barh(
        [names[r["id"]] for r in F],
        vals,
        color=["#343434" if r["id"] == "selected_system" else "#8b8b8b" for r in F],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_title(title)
    ax.set_xlabel("Percent")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(
            v + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            fontsize=9,
        )
fig.tight_layout()
fig.savefig(WORK / "scores.png", dpi=180)
plt.close(fig)
img(WORK / "scores.png", 490, 490)
p(
    "Read model IDs using the full configuration table on page 2. Text-only baselines are substantially lower than models that include patient features. Accuracy and macro F1 describe different aspects of performance, so both are retained."
)
p("Selected-system uncertainty", "Heading2")
table(
    [["Metric", "Estimate", "95% group-bootstrap interval"]]
    + [
        [
            label,
            pct(S["metrics"][key]),
            " to ".join(pct(v) for v in S["group_bootstrap_95_ci"][key]),
        ]
        for label, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]
    ],
    [140, 140, 220],
    9,
    7,
)
p(
    "Intervals use 1,000 resamples of held-out groups. They describe sampling uncertainty within this dataset, rather than performance on a new hospital or independently labelled cohort.",
    "Small",
)
page("Confusion matrices: all final models")
p(
    "Rows are reference levels; columns are predicted levels. Model IDs match page 2. Each matrix contains all 1,999 test records.",
    "Small",
)
fig, axes = plt.subplots(5, 2, figsize=(8, 10))
for ax, r in zip(axes.flat, F):
    cm = np.array(r["confusion"])
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=1000)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > 550 else "black",
            )
    ax.set_xticks([0, 1, 2], ["1", "2", "3"])
    ax.set_yticks([0, 1, 2], ["1", "2", "3"])
    ax.set_title(names[r["id"]], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Reference", fontsize=8)
fig.tight_layout(pad=1.1)
fig.savefig(WORK / "matrices.png", dpi=200)
plt.close(fig)
img(WORK / "matrices.png", 496, 620)
page("Per-class performance")
p(
    "Precision, recall and F1 are shown separately for every final model and triage level. Each model has support 677 / 943 / 379 for levels 1 / 2 / 3.",
    "Small",
)
rows = [["Model", "Level", "Precision", "Recall", "F1", "Support"]]
for r in F:
    for level in ["1", "2", "3"]:
        q = r["report"][level]
        rows.append(
            [
                names[r["id"]],
                level,
                num(q["precision"]),
                num(q["recall"]),
                num(q["f1-score"]),
                int(q["support"]),
            ]
        )
table(rows, [65, 65, 100, 100, 100, 70], 8, 3)
p(
    "The selected system made 19 errors: nine under-triage and ten over-triage errors. Its class-specific results should be read alongside the aggregate scores, particularly for level 1.",
    "Small",
)
page("Development search and model selection")
p(
    "The search completed 136 three-fold configurations and 22 five-fold refinements. It varied classifier parameters, class balancing, embedding representation, PCA size, derived features and text weight. Probability combinations and small class multipliers were selected using out-of-fold development predictions."
)
cv = pd.read_csv(DATA / "cv5_leaderboard.csv")
rows = [["Five-fold configuration", "Accuracy", "Macro F1"]]
for _, r in cv.iterrows():
    q = json.loads((DATA / "cv5" / f"{r['id']}.json").read_text())
    rows.append(
        [
            desc(q["config"]) + f" [{str(r['id'])[:6]}]",
            pct(q["mean"]["accuracy"]),
            num(q["mean"]["macro_f1"]),
        ]
    )
table(rows, [340, 80, 80], 7.6, 3)
p("Selected settings", "Heading2")
p(
    "SapBERT concepts reduced to 128 PCA components, combined with patient features and three derived quantities: pulse pressure, heart-rate/systolic-pressure ratio, and a mean-pressure proxy. HGB uses 250 iterations, learning rate 0.1, 15 maximum leaves, L2 regularization 1 and balanced sample weights. Class probability multipliers are 0.8 / 1.0 / 0.8. This produces 153 input features."
)
p(
    "The six-character suffix identifies each configuration in the saved CV tables. The complete parameter records and all 136 screening results remain in the accompanying repository, avoiding dozens of near-repeated rows here. Development selection scores may be optimistic; final results use the separate test set.",
    "Small",
)
page("Feature checks and learning behaviour")
sens = json.loads((DATA / "sensitivity_results.json").read_text())
table(
    [["Development sensitivity test", "Accuracy", "Macro F1"]]
    + [
        [
            "Omit " + ", ".join(r["config"]["features"]["exclude_cat"]),
            pct(r["mean"]["accuracy"]),
            num(r["mean"]["macro_f1"]),
        ]
        for r in sens
    ],
    [300, 100, 100],
    9,
    6,
)
p(
    "These are development-only HGB feature-removal checks. The dataset contains strong relationships between patient features and labels; removing ECG, alone or together with arrival, reduces performance more than removing gender. These tests do not establish causal relationships."
)
diag = pd.read_csv(DATA / "development_diagnostics.csv")
rows = [["Diagnostic", "Training fraction", "Accuracy", "Macro F1"]]
for (name, fraction), g in diag.groupby(["diagnostic", "fraction"], dropna=False):
    rows.append(
        [
            name.replace("_", " "),
            str(fraction),
            pct(g.accuracy.mean()),
            num(g.macro_f1.mean()),
        ]
    )
table(rows, [220, 100, 90, 90], 8, 5)
img(DATA / "plots/learning_curve.png", 490, 223)
p(
    "Training-label shuffling is a negative control; learning curves compare group-based subsets of development training data. All diagnostic decisions remain separate from final-test selection.",
    "Small",
)
p("Reproducibility", "Heading2")
p(
    "Split seeds: 42 for held-out groups and 123 for development folds. The original input checksum, encoder revisions, configuration JSON, full-precision metrics and verification record accompany the code. The saved experiment artifacts do not replace the deployed translation or inference pipeline.",
    "Small",
)
page("Historical and literature comparisons")
p(
    "Historical experiments use different records, four target levels and different split protocols. Their scores provide context and are not controlled estimates of improvement on the new dataset. All historical final-model rows are retained below.",
    "Small",
)
old = pd.read_csv(REF / "all_comparison_metrics.csv")
prior = pd.read_csv(REF / "sapbert_metrics.csv")
rows = [["Earlier model / inputs", "Accuracy", "Recall", "F1"]]
cn = {"LogisticRegression": "LR", "HistGradientBoosting": "HGB", "RandomForest": "RF"}
for _, r in old.iterrows():
    rows.append(
        [
            "SBERT "
            + cn[r.classifier]
            + " / "
            + r.feature_set.replace("_", " ")
            + " / "
            + r.representation.replace("_", " "),
            pct(r.accuracy),
            num(r.recall_macro),
            num(r.f1_macro),
        ]
    )
for _, r in prior.iterrows():
    rows.append(
        [
            "SapBERT " + r.model.replace("_", " "),
            pct(r.accuracy),
            num(r.recall_macro),
            num(r.macro_f1),
        ]
    )
table(rows, [302, 66, 66, 66], 7.1, 2)
p("Published comparison: Model / Accuracy / Recall / F1", "Heading2")
table(
    [
        ["Model [1]", "Accuracy", "Recall", "F1"],
        ["TF-IDF + Logistic Regression", "0.750", "0.988", "0.544"],
        ["TF-IDF + Random Forest", "0.751", "0.964", "0.582"],
        ["BiLSTM", "0.746", "0.846", "0.670"],
        ["CNN", "0.723", "0.787", "0.667"],
        [
            "Current selected HGB (this study)",
            num(S["metrics"]["accuracy"]),
            num(S["metrics"]["recall_macro"]),
            num(S["metrics"]["macro_f1"]),
        ],
    ],
    [302, 66, 66, 66],
    7.4,
    2,
)
p(
    "[1] Seo et al. (2025). Artificial intelligence for severity triage based on conversations in an emergency department in Korea. Scientific Reports 15, 16870. DOI: 10.1038/s41598-025-99874-0, Table 2. Published results use binary KTAS and ten-fold CV; averaging and task differ from the present three-class macro metrics. These are separate models, not a CNN-BiLSTM hybrid.",
    "Small",
)


def footer(canvas, doc):
    canvas.setFont("Vera", 7)
    canvas.drawString(45, 23, "ML_Predictor | Concise research comparison")
    canvas.drawRightString(A4[0] - 45, 23, f"{doc.page} / 8")


OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=45,
    rightMargin=45,
    topMargin=32,
    bottomMargin=37,
    title="Roman Urdu Medical Triage - Concise Classifier Comparison",
).build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
