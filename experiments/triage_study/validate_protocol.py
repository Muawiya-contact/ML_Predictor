"""Verify split isolation and training-only preprocessing with adversarial validation rows."""

import numpy as np, pandas as pd
from research_engine import HERE, Features
from prepare_data import normal


def main():
    d = pd.read_csv(HERE / "dataset_with_splits.csv")
    tr = d[d.partition == "development"]
    te = d[d.partition == "test"]
    assert len(tr) + len(te) == len(d)
    assert set(tr.group).isdisjoint(te.group)
    for c in ["chief_complaint", "Clinical_Concept"]:
        assert set(tr[c].map(normal)).isdisjoint(te[c].map(normal))
    for n in (3, 5):
        assert (te[f"cv{n}"] == -1).all()
        for fold in range(n):
            a = tr[tr[f"cv{n}"] != fold]
            b = tr[tr[f"cv{n}"] == fold]
            assert set(a.group).isdisjoint(b.group) and len(a) + len(b) == len(tr)
            assert set(a.Labels) == set(b.Labels) == {1, 2, 3}
    a = tr.iloc[:200].copy()
    b = tr.iloc[200:220].copy()
    f = Features(view="fused", encoder="sapbert_concept", pca=32).fit(a)
    z = f.transform(a)
    mean = f.pca_.mean_.copy()
    assert np.allclose(mean, f.embedding(a).mean(axis=0))
    b["Age"] = 9999
    b["Gender"] = "unseen_category"
    b["Labels"] = 999
    f.transform(b)
    assert np.allclose(z, f.transform(a)) and np.allclose(mean, f.pca_.mean_)
    changed = a.copy()
    changed["Labels"] = np.arange(len(a))
    assert np.allclose(f.transform(a), f.transform(changed))
    assert not any(
        c in ["Labels", "Category", "min_confidence", "seconds"]
        for _, transform, cols in f.structured_.transformers_
        if transform != "drop"
        for c in (cols if isinstance(cols, list) else [])
    )
    print(
        "PASS: all grouped splits; no held-out text duplicates; fold-local PCA; unknown categories; target exclusion; validation cannot alter fitted transforms."
    )


if __name__ == "__main__":
    main()
