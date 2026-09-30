"""Create the complete research report from saved CV and frozen-test predictions."""

from pathlib import Path
import json, math
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import reportlab
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
    KeepTogether,
)
from study_paths import OUTPUT as HERE, REFERENCES

OUT = HERE / "Triage_10000_Optimized_Comparison.pdf"
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)
A = json.loads((HERE / "data_audit.json").read_text())
final = json.loads((HERE / "final_results.json").read_text())
chosen = next(r for r in final if r["id"] == "selected_system")
cv3 = pd.read_csv(HERE / "cv3_leaderboard.csv")
cv5 = pd.read_csv(HERE / "cv5_leaderboard.csv")
fonts = Path(reportlab.__file__).parent / "fonts"
for name, file in [("Vera", "Vera.ttf"), ("VeraBold", "VeraBd.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
styles = getSampleStyleSheet()
for s in styles.byName.values():
    s.fontName = "VeraBold" if s.name in ["Title", "Heading2", "Heading3"] else "Vera"
    s.textColor = colors.black
styles["Title"].fontSize = 18
styles["Title"].leading = 23
styles["BodyText"].fontSize = 9
styles["BodyText"].leading = 12
styles["Heading2"].fontSize = 12
styles["Heading2"].leading = 16
styles["Heading3"].fontSize = 10
styles["Heading3"].leading = 13
story = []


def p(t, s="BodyText"):
    story.extend([Paragraph(str(t), styles[s]), Spacer(1, 6)])


def page(t):
    if story:
        story.append(PageBreak())
    p(t, "Title")


def table(data, widths, size=8, padding=5):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Vera"),
                ("FONTNAME", (0, 0), (-1, 0), "VeraBold"),
                ("FONTSIZE", (0, 0), (-1, -1), size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        )
    )
    story.extend([t, Spacer(1, 10)])


def percent(x):
    return f"{x:.2%}"


def picture(path, w, h):
    story.extend([Image(str(path), width=w, height=h), Spacer(1, 8)])


CLF = {
    "logreg": "Logistic Regression",
    "hgb": "Hist Gradient Boosting",
    "rf": "Random Forest",
}
ENC = {
    "sapbert_concept": "SapBERT concepts",
    "mpnet_concept": "MPNet concepts",
    "minilm_complaint": "MiniLM complaints",
}


def description(c):
    f = c["features"]
    text = (
        "Patient features"
        if f["view"] == "structured"
        else ENC[f["encoder"]] + (" + patient features" if f["view"] == "fused" else "")
    )
    if f["view"] != "structured":
        text += " / " + ("full dimensions" if not f.get("pca") else f"PCA-{f['pca']}")
    if f.get("derived"):
        text += " / derived features"
    if f.get("exclude_cat"):
        text += " / excludes " + ", ".join(f["exclude_cat"])
    return CLF[c["classifier"]] + "; " + text


references = {
    r["id"]: json.loads((HERE / "cv5" / f"{r['id']}.json").read_text())["config"]
    for _, r in cv5.iterrows()
}
page("Triage Classifier: 10,000-Record Study")
p(
    "Complete comparison using the latest uploaded classifier and the newly supplied dataset",
    "Heading2",
)
p(
    "The study retains the uploaded classifier's Logistic Regression, Hist Gradient Boosting and SapBERT baseline settings. It adds Random Forest, alternative text encoders, fold-local preprocessing, dimensionality comparisons, class balancing, feature tests and development-only ensemble selection. Existing application models were not replaced."
)
p(
    f"Data: {A['rows']:,} records; {A['development_rows']:,} development and {A['test_rows']:,} final test records. The target has three labels: 1, 2 and 3. There are no level-4 examples in this file."
)
p("Selected system: final held-out results", "Heading2")
table(
    [["Metric", "Value", "95% group-bootstrap interval"]]
    + [
        [
            k,
            percent(chosen["metrics"][key]),
            " - ".join(percent(x) for x in chosen["group_bootstrap_95_ci"][key]),
        ]
        for k, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]
    ]
    + [
        ["Macro precision", percent(chosen["metrics"]["precision_macro"]), "-"],
        ["Macro recall", percent(chosen["metrics"]["recall_macro"]), "-"],
    ],
    [150, 115, 235],
    size=9,
    padding=8,
)
p(
    "Selection was frozen using development predictions before evaluating the final test. Its components are:"
)
for i, w in zip(chosen["selection"]["members"], chosen["selection"]["weights"]):
    p(f"Weight {w:.3f}: {description(references[i])}.")
p(
    "Class-probability multipliers for labels 1, 2 and 3: "
    + str(chosen["selection"]["class_probability_multipliers"])
    + ". These decision adjustments were tuned on development out-of-fold predictions only; adjusted scores are not calibrated probabilities."
)
p("How to interpret the scores", "Heading2")
p(
    "This dataset is different from the earlier 821-record, four-class study. A higher score here cannot be attributed to model tuning alone. The user confirmed that the labels were assigned by an AI model. Scores measure agreement with those AI-generated labels, not clinician agreement. No clinician validation or real-patient provenance was provided."
)
p(
    "Strong ECG and vital-sign relationships with the labels are documented later in this report. They can make this particular dataset easier to classify. Development CV is used for selection and is not an unbiased final performance estimate."
)
p("Outcome of tuning", "Heading2")
p(
    "The frozen selected system did not outperform the source-setting Hist Gradient Boosting baseline on this final test (baseline accuracy 99.15%, macro F1 0.9901 with fusion). Random Forest recorded the highest observed comparison accuracy, 99.35%, and macro F1 0.9926. Selection is not changed after examining test results. These findings do not support a claim that every tuning step improved the strongest baseline."
)
page("Previous and current experiments: overview")
p(
    "This table keeps the study settings separate. Different datasets, target classes, encoders and split methods mean that the rows are not a controlled before/after comparison."
)
old = pd.read_csv(REFERENCES / "all_comparison_metrics.csv")
prior = pd.read_csv(REFERENCES / "sapbert_metrics.csv")
table(
    [
        ["Study", "Records", "Classes", "Test rows", "Accuracy*", "Macro F1*"],
        [
            "Earlier SBERT",
            "2,252",
            "4",
            "451",
            percent(old.accuracy.max()),
            percent(old.f1_macro.max()),
        ],
        [
            "Earlier SapBERT",
            "821",
            "4",
            "212",
            percent(prior.accuracy.max()),
            percent(prior.macro_f1.max()),
        ],
        [
            "Current selected system",
            "10,000",
            "3",
            str(A["test_rows"]),
            percent(chosen["metrics"]["accuracy"]),
            percent(chosen["metrics"]["macro_f1"]),
        ],
    ],
    [160, 65, 55, 65, 80, 80],
    size=8,
    padding=9,
)
p(
    "*For historical studies, accuracy and macro F1 are the highest values separately reported and may come from different classifiers. The current row reports both metrics for the single system frozen before final-test evaluation."
)
p("What each study tested", "Heading2")
p(
    "Earlier SBERT: all-mpnet-base-v2 embeddings of English translations; 768-D versus PCA-64; text alone and fusion; three classifiers. A stratified random row split was used, so repeated complaint phrases could occur across the split."
)
p(
    "Earlier SapBERT: 821 repository records with matched clinical concepts; 609/212 concept-grouped holdout; source Logistic Regression and Hist Gradient Boosting, with added 768-D and Random Forest comparisons. Its reported CV used preprocessing fitted before the CV folds and was exploratory."
)
p(
    "Current study: 10,000 supplied records with user-confirmed AI-assigned labels; three frozen encoders; grouped three-fold screening and five-fold refinement; preprocessing fitted within folds; separate frozen final test; sensitivity checks, label-shuffle control, learning curves and uncertainty intervals."
)
p("Complete historical reports", "Heading2")
p(
    "The original SBERT comparison and the expanded SapBERT comparison are included unchanged as appendices after the new study. Their earlier page numbers and values are preserved. The expanded SapBERT report contains the original six configurations plus the additional comparisons and class-level results."
)
p(
    "The new experiment tables report measured values only. Any further table supplied by the user can be added without rerunning or retuning the final test."
)
page("Dataset preparation and audit")
table(
    [
        ["Item", "Result"],
        ["Uploaded size", f"{A['raw_shape'][0]:,} rows x {A['raw_shape'][1]} columns"],
        ["Retained size", f"{A['rows']:,} rows x {len(A['retained_columns'])} columns"],
        ["Exact duplicate rows removed", str(len(A["removed_duplicate_source_rows"]))],
        ["Normalized text groups", str(A["groups"])],
        ["Largest group", str(A["largest_group"])],
        ["Shared development/test groups", str(A["shared_groups"])],
    ],
    [275, 225],
    size=9,
    padding=7,
)
p("Features retained", "Heading2")
p(
    "Age; gender; arrival mode; original Roman Urdu complaint; generated clinical concept; heart rate; systolic and diastolic blood pressure; ECG status; temperature; oxygen saturation; AVPU. Labels is the target and never a predictor."
)
p("Columns removed", "Heading2")
p(
    "Category (constant Cardiac), fuzzy_glossary, min_confidence, unresolved_tokens, seconds and examples_used. These are a constant category or processing metadata, not inputs used to select triage severity."
)
p(
    "One arrival-mode value, roman_urdu_reference at source CSV row 21, was set to missing. The row and its label were retained. Missing-value imputation is learned within each training fold. No difficult examples or label classes were removed to improve scores."
)
table(
    [["Label", "Entire dataset", "Development", "Final test"]]
    + [
        [
            str(k),
            str(A["class_counts"][str(k)]),
            str(A["development_class_counts"][str(k)]),
            str(A["test_class_counts"][str(k)]),
        ]
        for k in [1, 2, 3]
    ],
    [100, 135, 135, 130],
    size=9,
    padding=8,
)
p("Source checksum (SHA-256): " + A["input_sha256"])
page("Evaluation and optimization methods")
items = [
    (
        "1. Frozen grouped holdout",
        "Normalize case, punctuation and whitespace in both text columns. Connected components link rows sharing either normalized complaint or concept. A seeded stratified grouped split reserves approximately 20% as the final test set. This prevents exact normalized text overlap; it does not guarantee independence of semantically similar generated templates.",
    ),
    (
        "2. Frozen text encoders",
        "SapBERT uses CLS pooling on clinical concepts (768 dimensions, 64-token limit). MPNet uses its Sentence Transformer pooling on concepts (768 dimensions, 128-token limit). Multilingual MiniLM embeds the original Roman Urdu complaints (384 dimensions, 128-token limit). All embeddings are L2-normalized and computed locally with frozen cached checkpoints.",
    ),
    (
        "3. Training-only preprocessing",
        "Numeric median imputation and standardization, categorical imputation and one-hot encoding, and PCA are fitted inside each CV training fold. PCA sizes 32, 64 and 128, plus full embeddings, are compared. Source baselines retain the original automatic PCA solver; exploratory candidates explicitly use a randomized solver.",
    ),
    (
        "4. Model and feature search",
        "Logistic Regression regularization, tree settings, balanced versus unweighted training, PCA size and text contribution weight are compared. Optional deterministic features are systolic minus diastolic pressure, heart-rate/systolic-pressure ratio and (systolic + 2 x diastolic)/3. They are transformations of existing inputs, not new measured clinical variables.",
    ),
    (
        "5. Grouped cross-validation",
        "Three-fold grouped screening ranks candidates. Five-fold grouped refinement evaluates the leading configurations and leading classifier/input families. Configurations and per-fold scores are saved. The additional ECG/arrival/gender sensitivity tests and learning curves use development rows only.",
    ),
    (
        "6. Ensemble and decision adjustment",
        "Compare leading single models and probability averages of classifier-family winners. Small class-probability adjustments are selected on five-fold out-of-fold development predictions, prioritizing macro F1 and then accuracy. This tuning makes those development selection scores optimistic; the separate final test provides the reported external-to-selection check.",
    ),
    (
        "7. Final evaluation",
        "Freeze model members, weights and decision adjustments. Refit on all development records, evaluate the final test once, and report macro/per-class metrics and confusion matrices. Group bootstrap with 1,000 replicates gives uncertainty intervals for the selected system. Baselines and CV-family winners are comparisons, not candidates selected by final test scores.",
    ),
]
for title, text in items:
    p(title, "Heading3")
    p(text)
page("Final held-out comparison")
p(
    "All rows below use the same "
    + str(A["test_rows"])
    + " final test records. B = source-setting baseline; C = classifier-family winner selected by CV; S = final selected system. Precision, recall and F1 are macro-averaged."
)
ids = {
    r[
        "id"
    ]: f"{('S' if r['id']=='selected_system' else 'B' if r['role']=='source_baseline' else 'C')}{j+1}"
    for j, r in enumerate(final)
}
table(
    [["Ref.", "Accuracy", "Precision", "Recall", "F1", "QWK", "MAE"]]
    + [
        [
            ids[r["id"]],
            *[
                percent(r["metrics"][k])
                for k in ["accuracy", "precision_macro", "recall_macro", "macro_f1"]
            ],
            f"{r['metrics']['qwk']:.3f}",
            f"{r['metrics']['mae']:.3f}",
        ]
        for r in final
    ],
    [40, 80, 80, 80, 80, 70, 70],
    size=8,
    padding=7,
)
p("Model key", "Heading2")
for r in final:
    p(
        ids[r["id"]]
        + ": "
        + (
            "Selected system, specified on page 1."
            if r["id"] == "selected_system"
            else description(r["config"]) + f"; {r['feature_count']} predictors."
        )
    )
p(
    "A baseline score here is measured on this new dataset and grouped holdout. The earlier study's 73-76% values are not valid denominators for attributing improvement to these changes."
)
page("Performance comparison of ED triage models")
p(
    "Article-style table matching the supplied screenshot: Model, Accuracy, Recall and F1. Values below are measured on the current frozen final test set and shown on a 0-1 scale. Recall and F1 are macro-averaged."
)
article_rows = [["Model", "Accuracy", "Recall", "F1"]]
for r in final:
    name = (
        "Proposed system (selected on development data)"
        if r["id"] == "selected_system"
        else ids[r["id"]] + ": " + description(r["config"])
    )
    article_rows.append(
        [
            Paragraph(name, styles["BodyText"]),
            *[
                f"{r['metrics'][k]:.3f}"
                for k in ["accuracy", "recall_macro", "macro_f1"]
            ],
        ]
    )
table(article_rows, [290, 70, 70, 70], size=9, padding=8)
p(
    "The proposed-system row uses the actual frozen selection, which may be a single model or an ensemble. Its name is not forced to SapBERT if another configuration was selected. No published comparator is measured on this same test set."
)
page("Published context and current result")
p(
    "Performance comparison in the requested Model / Accuracy / Recall / F1 format. Literature rows are reproduced from the cited primary study; the final row is measured in this experiment."
)
lit = json.loads((HERE / "literature_sources.json").read_text())
comparison = [["Model", "Accuracy", "Recall", "F1"]] + [
    [
        r["model"] + " [L1]",
        f"{r['accuracy']:.3f}",
        f"{r['recall']:.3f}",
        f"{r['f1']:.3f}",
    ]
    for r in lit["rows"]
]
comparison.append(
    [
        "Proposed system (current study)",
        *[
            f"{chosen['metrics'][k]:.3f}"
            for k in ["accuracy", "recall_macro", "macro_f1"]
        ],
    ]
)
table(comparison, [290, 70, 70, 70], size=9, padding=10)
p(
    "[L1] reports tenfold cross-validation of binary Korean severity classification: KTAS 3 versus KTAS 4-5. Recall and F1 are retained as published; they are not converted to our macro averages. BiLSTM and CNN are separate models, not a hybrid."
)
p(
    "Our result uses three AI-assigned labels and a separate grouped holdout. These rows provide context, not evidence of superiority on a shared benchmark. The screenshot scores and its unidentified references [40]-[42] were not carried over because their sources could not be verified."
)
p("Reference", "Heading2")
p(
    '<link href="https://doi.org/10.1038/s41598-025-99874-0">[L1] Seo et al. (2025). Artificial intelligence for severity triage based on conversations in an emergency department in Korea. Scientific Reports 15, 16870. Table 2. DOI: 10.1038/s41598-025-99874-0.</link>'
)
page("Final accuracy and macro F1")
for metric, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]:
    fig, ax = plt.subplots(figsize=(9, 4.7))
    vals = [r["metrics"][metric] for r in final]
    ax.barh(
        range(len(final)),
        vals,
        color=["#1c5a85" if r["id"] == "selected_system" else "#7899af" for r in final],
    )
    ax.set_yticks(range(len(final)), [ids[r["id"]] for r in final])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xlabel(label)
    ax.set_title("Same frozen final test set")
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, percent(v), va="center", fontsize=9)
    fig.tight_layout()
    path = PLOTS / f"final_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    picture(path, 490, 256)
page("Final precision and ordinal errors")
table(
    [["Ref.", "Macro precision", "Under-triage", "Over-triage", "Errors / records"]]
    + [
        [
            ids[r["id"]],
            percent(r["metrics"]["precision_macro"]),
            percent(r["metrics"]["under_triage_rate"]),
            percent(r["metrics"]["over_triage_rate"]),
            f"{int(np.array(r['confusion']).sum()-np.trace(np.array(r['confusion'])))}/{A['test_rows']}",
        ]
        for r in final
    ],
    [45, 115, 115, 115, 110],
    size=8,
    padding=7,
)
p(
    "Under-triage predicts a numerically higher level (less urgency) than the supplied label. Over-triage predicts a lower level. These terms describe disagreement with the CSV labels, not independently reviewed patient harm."
)
p(
    "QWK is quadratic weighted kappa; MAE is the average absolute difference between predicted and supplied label numbers. Accuracy alone can hide class-specific errors, so precision, recall, F1 and all confusion matrices are included."
)
for start in range(0, len(final), 2):
    page("Confusion matrices and class-level results")
    p(
        "True labels are rows; predicted labels are columns. All matrices use the same final test set. Support is the number of records carrying each supplied label."
    )
    for r in final[start : start + 2]:
        p(
            ids[r["id"]]
            + ": "
            + (
                "Selected system"
                if r["id"] == "selected_system"
                else description(r["config"])
            ),
            "Heading3",
        )
        m = np.array(r["confusion"])
        assert m.sum() == A["test_rows"]
        assert np.isclose(np.trace(m) / m.sum(), r["metrics"]["accuracy"])
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(3), [1, 2, 3])
        ax.set_yticks(range(3), [1, 2, 3])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for (yy, xx), v in np.ndenumerate(m):
            ax.text(
                xx,
                yy,
                str(v),
                ha="center",
                va="center",
                color="white" if v > m.max() / 2 else "black",
            )
        fig.tight_layout()
        path = PLOTS / f"cm_{r['id']}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        rep = r["report"]
        cells = [["Class", "Precision", "Recall", "F1", "N"]] + [
            [
                k,
                *[percent(rep[k][v]) for v in ["precision", "recall", "f1-score"]],
                str(int(rep[k]["support"])),
            ]
            for k in ["1", "2", "3", "macro avg", "weighted avg"]
        ]
        small = Table(cells, colWidths=[65, 55, 55, 55, 35])
        small.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Vera"),
                    ("FONTNAME", (0, 0), (-1, 0), "VeraBold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        layout = Table(
            [[Image(str(path), width=225, height=188), small]], colWidths=[235, 270]
        )
        layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.extend([layout, Spacer(1, 20)])
page("Sensitivity checks and dataset patterns")
sens = json.loads((HERE / "sensitivity_results.json").read_text())
table(
    [["Removed features", "CV accuracy", "CV macro F1"]]
    + [
        [
            ", ".join(r["config"]["features"]["exclude_cat"]),
            percent(r["mean"]["accuracy"]),
            percent(r["mean"]["macro_f1"]),
        ]
        for r in sens
    ],
    [280, 110, 110],
    size=8,
    padding=8,
)
p(
    "These are three-fold grouped development comparisons using the same boosting settings. The presence of a predictive input is not automatically leakage: ECG and vital signs are legitimate prediction-time features. However, simple label-specific patterns suggest the dataset may be easier than new clinical cases."
)
ec = pd.read_csv(HERE / "development_ecg_by_label.csv", index_col=0)
table(
    [["ECG category", "Label 1", "Label 2", "Label 3"]]
    + [[str(i), *[str(int(x)) for x in row]] for i, row in ec.iterrows()],
    [260, 80, 80, 80],
    size=8,
    padding=6,
)
p(
    "This table uses development records only. Several ECG categories occur under just one label. Numeric feature ranges also differ strongly by label; they are retained in development_feature_ranges.csv. The AI-assigned label source is user-confirmed. Independent clinician-labelled data is needed before interpreting strong internal scores as general clinical performance."
)
page("Learning curve and label-shuffle control")
diag = pd.read_csv(HERE / "development_diagnostics.csv")
curve = (
    diag[diag.diagnostic == "learning_curve"]
    .groupby("fraction")
    .agg(
        train_rows=("train_rows", "mean"),
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
    )
    .reset_index()
)
table(
    [["Training-group fraction", "Mean rows", "Accuracy", "Macro F1", "F1 std."]]
    + [
        [
            percent(r.fraction),
            f"{r.train_rows:.0f}",
            percent(r.accuracy),
            percent(r.macro_f1),
            f"{r.f1_std:.4f}",
        ]
        for _, r in curve.iterrows()
    ],
    [155, 80, 90, 90, 85],
    size=8,
    padding=8,
)
fig, ax = plt.subplots(figsize=(8.8, 4))
ax.errorbar(curve.train_rows, curve.macro_f1, yerr=curve.f1_std, marker="o", capsize=4)
ax.set_xlabel("Mean training rows per development fold")
ax.set_ylabel("Macro F1")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.2)
fig.tight_layout()
path = PLOTS / "learning_curve.png"
fig.savefig(path, dpi=180)
plt.close(fig)
picture(path, 490, 223)
control = diag[diag.diagnostic == "shuffled_training_labels"]
p("Training-label shuffle control", "Heading2")
p(
    f"With only training labels randomly permuted, mean development accuracy was {percent(control.accuracy.mean())} and macro F1 was {percent(control.macro_f1.mean())}. Validation labels were not changed. This is a diagnostic control, not a candidate model or a complete proof against every possible form of leakage."
)
p(
    "Protocol assertions also verified group separation in every split, no exact normalized text overlap in the final split, training-only PCA means, handling of unseen categories, and invariance of model features when target labels are changed."
)
for folds, frame in [(5, cv5), (3, cv3)]:
    for start in range(0, len(frame), 23):
        page(
            f"{folds}-fold grouped CV: "
            + ("refined models" if folds == 5 else "screening candidates")
        )
        p(
            f"Candidates {start+1}-{min(start+23,len(frame))} of {len(frame)}. Scores are means across development folds; SD is the population standard deviation of fold macro F1. These scores guided selection and should not be treated as independent final-test estimates."
        )
        data = [["Trial", "Model / inputs", "PCA", "Accuracy", "Macro F1", "F1 SD"]]
        for _, r in frame.iloc[start : start + 23].iterrows():
            enc = str(r.get("encoder", ""))
            enc = {
                "sapbert_concept": "Sap",
                "mpnet_concept": "MPNet",
                "minilm_complaint": "MiniLM",
            }.get(enc, "")
            desc = f"{r.classifier} / {r['view']} {enc}"
            pc = r.get("pca", float("nan"))
            pc = "-" if pd.isna(pc) else ("full" if pc == 0 else str(int(pc)))
            data.append(
                [
                    r.id[:8],
                    desc,
                    pc,
                    percent(r.accuracy),
                    percent(r.macro_f1),
                    f"{r.macro_f1_std:.4f}",
                ]
            )
        table(data, [70, 160, 45, 80, 80, 65], size=7.5, padding=6)
        p(
            "Trial IDs map to saved configuration JSON files, including class weights, regularization, tree settings, derived features and text weights. Full-precision per-fold accuracy, precision, recall, F1, QWK, MAE and directional-error values are available in cv3/ and cv5/."
        )
page("Selected settings and reproducibility")
for i, w in zip(chosen["selection"]["members"], chosen["selection"]["weights"]):
    c = references[i]
    p(f"Trial {i}; ensemble weight {w:.3f}", "Heading2")
    p(description(c))
    p("Classifier parameters: " + json.dumps(c["params"]))
    p("Feature parameters: " + json.dumps(c["features"]))
p("Reproduction files", "Heading2")
p(
    "prepare_data.py; encode_text.py; research_engine.py; validate_protocol.py; sensitivity.py; diagnostics.py; run_study.py; build_report.py. The uploaded Python file is preserved as triage_classifier_uploaded.py. Data checksums, split assignments, encoder revisions, search plans, per-fold predictions, frozen selection, final predictions, fitted preprocessors and classifiers are retained alongside this report."
)
p(
    "The saved models are research artifacts. Their feature objects refer to row-aligned embedding arrays in the experiment directory; they are not a replacement for the application's live translation/inference pipeline. Supporting files must be kept with the models to reproduce predictions on these records."
)
p("Boundaries of the study", "Heading2")
p(
    "The study covers only labels 1-3 and one uploaded dataset. It does not demonstrate level-4 performance, live Roman Urdu translation quality, calibration, temporal generalization, clinical usefulness, or literature superiority. No published-comparator scores are invented. Strong internal scores require independent data and verified label provenance before broader claims."
)
p(
    "Reported gains must be separated into dataset differences and within-dataset model differences. All final comparisons in this report use identical test rows; earlier reports used different records, class counts and split protocols."
)


def footer(c, d):
    c.setFont("Vera", 8)
    c.drawString(42, 24, "ML_Predictor | 10,000-record research study")
    c.drawRightString(A4[0] - 42, 24, str(d.page))


OUT.parent.mkdir(parents=True, exist_ok=True)
core = HERE / "report_core.pdf"
SimpleDocTemplate(
    str(core),
    pagesize=A4,
    leftMargin=42,
    rightMargin=42,
    topMargin=36,
    bottomMargin=42,
    title="Triage Classifier: 10,000-Record Study",
).build(story, onFirstPage=footer, onLaterPages=footer)
from pypdf import PdfWriter

writer = PdfWriter()
writer.append(str(core), outline_item="Current 10,000-record study")
previous = [
    (
        "Appendix A: original SBERT comparison",
        REFERENCES / "SBERT_Classifier_Comparison.pdf",
    ),
    (
        "Appendix B: expanded SapBERT comparison",
        REFERENCES / "SapBERT_Complete_Classifier_Comparison.pdf",
    ),
]
for j, (title, path) in enumerate(previous):
    if not path.exists():
        raise FileNotFoundError(path)
    divider = HERE / f"appendix_{j+1}_cover.pdf"
    contents = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 25),
        Paragraph(
            "Historical report reproduced unchanged. Its dataset, target classes, split and scores belong to the earlier experiment. Do not interpret its values as a controlled comparison with the new 10,000-record study.",
            styles["BodyText"],
        ),
    ]
    SimpleDocTemplate(
        str(divider),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=70,
        bottomMargin=42,
    ).build(contents)
    writer.append(str(divider), outline_item=title)
    writer.append(str(path), import_outline=False)
writer.add_metadata(
    {"/Title": "Triage Classifier: Complete Current and Historical Comparisons"}
)
with OUT.open("wb") as out:
    writer.write(out)
print(OUT)
