"""Audit the uploaded CSV, retain prediction-time inputs, and freeze grouped splits."""

from pathlib import Path
import hashlib, json, re, unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from study_paths import OUTPUT as HERE
import argparse

COLS = [
    "Age",
    "Gender",
    "Mode_of_Arrival",
    "chief_complaint",
    "Clinical_Concept",
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "ECG_Status",
    "Temperature",
    "SpO2",
    "AVPU",
    "Labels",
]
NUM = ["Age", "Heart_Rate", "Systolic_BP", "Diastolic_BP", "Temperature", "SpO2"]


def normal(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def main(source):
    SOURCE = Path(source).resolve()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if (HERE / "data_audit.json").exists():
        audit = json.loads((HERE / "data_audit.json").read_text())
        if audit["input_sha256"] != digest:
            raise ValueError(
                "Output belongs to another dataset; choose a fresh TRIAGE_STUDY_OUTPUT directory."
            )
        if (HERE / "dataset_with_splits.csv").exists():
            print("Matching prepared dataset already exists; preserving frozen splits.")
            return
        raise ValueError(
            "Incomplete existing study directory; choose a fresh output directory."
        )
    if any(HERE.iterdir()):
        raise ValueError("Preparation requires an empty output directory.")
    raw = pd.read_csv(SOURCE)
    df = raw[COLS].copy()
    df["source_row"] = np.arange(len(df)) + 2
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [
        "chief_complaint",
        "Clinical_Concept",
        "Gender",
        "Mode_of_Arrival",
        "ECG_Status",
        "AVPU",
    ]:
        df[c] = df[c].astype("string").str.strip()
    invalid = ~df.Mode_of_Arrival.isin(["Walk-in", "Ambulance", "Wheelchair"])
    issues = df.loc[invalid, ["source_row", "Mode_of_Arrival"]].to_dict("records")
    df.loc[invalid, "Mode_of_Arrival"] = pd.NA
    target = pd.to_numeric(df.Labels, errors="coerce")
    if target.isna().any() or set(target.unique()) != {1, 2, 3}:
        raise ValueError("Unrecognized triage labels; manual review required")
    df.Labels = target.astype(int)
    dup = df.duplicated(subset=COLS, keep="first")
    removed = df.loc[dup, "source_row"].tolist()
    df = df.loc[~dup].reset_index(drop=True)
    # Connected components prevent either repeated concepts OR repeated complaints crossing folds.
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(a)] = root(b)

    for col in ["chief_complaint", "Clinical_Concept"]:
        seen = {}
        for i, value in enumerate(df[col]):
            key = normal(value)
            if not key:
                continue
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    df["group"] = [root(i) for i in range(len(df))]
    y = df.Labels.to_numpy()
    g = df.group.to_numpy()
    development, test = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(df, y, g)
    )
    partition = np.full(len(df), "development", dtype=object)
    partition[test] = "test"
    df["partition"] = partition
    df["row_id"] = np.arange(len(df))
    for folds in [3, 5]:
        assignments = np.full(len(df), -1)
        for fold, (_, val) in enumerate(
            StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=123).split(
                df.iloc[development], y[development], g[development]
            )
        ):
            assignments[development[val]] = fold
        df[f"cv{folds}"] = assignments
    df[COLS].to_csv(HERE / "trimmed_dataset.csv", index=False)
    df.to_csv(HERE / "dataset_with_splits.csv", index=False)
    df[["row_id", "source_row", "group", "partition", "cv3", "cv5"]].to_csv(
        HERE / "split_manifest.csv", index=False
    )
    manifest = {
        "input_file": SOURCE.name,
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "raw_shape": list(raw.shape),
        "retained_columns": COLS,
        "excluded_columns": [c for c in raw if c not in COLS],
        "removed_duplicate_source_rows": removed,
        "invalid_arrival_values_set_missing": issues,
        "rows": len(df),
        "groups": int(df.group.nunique()),
        "largest_group": int(df.group.value_counts().max()),
        "class_counts": df.Labels.value_counts().sort_index().to_dict(),
        "development_rows": len(development),
        "test_rows": len(test),
        "development_class_counts": df.iloc[development]
        .Labels.value_counts()
        .sort_index()
        .to_dict(),
        "test_class_counts": df.iloc[test].Labels.value_counts().sort_index().to_dict(),
        "shared_groups": len(set(g[development]) & set(g[test])),
        "target": "Labels",
        "provenance": "User confirmed that Labels were assigned by an AI model; no clinician validation was provided.",
    }
    assert manifest["shared_groups"] == 0
    (HERE / "data_audit.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    main(p.parse_args().csv)
