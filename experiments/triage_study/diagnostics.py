"""Development-only label-shuffle control and grouped learning-curve diagnostics."""

import json, numpy as np, pandas as pd
from research_engine import HERE, matrices, fit_model, metrics, config

D = pd.read_csv(HERE / "dataset_with_splits.csv")
Y = D.Labels.to_numpy()
dev = np.flatnonzero(D.partition.eq("development"))
rng = np.random.default_rng(42)
c = config(
    "hgb",
    {"view": "structured"},
    max_iter=300,
    max_leaf_nodes=15,
    l2_regularization=1,
    learning_rate=0.1,
)
rows = []
for fold in range(3):
    va = np.flatnonzero(D.cv3.eq(fold))
    tr = np.setdiff1d(dev, va)
    a, b = matrices(D, tr, va, c["features"])
    m = fit_model(c, a, rng.permutation(Y[tr]))
    p = m.predict(b)
    rows.append(
        {
            "diagnostic": "shuffled_training_labels",
            "fold": fold,
            "fraction": 1,
            "train_rows": len(tr),
            **metrics(Y[va], p),
        }
    )
    groups = D.iloc[tr].group.unique()
    rng.shuffle(groups)
    for fraction in [0.25, 0.5, 1.0]:
        chosen = groups[: max(1, int(len(groups) * fraction))]
        sub = tr[D.iloc[tr].group.isin(chosen).to_numpy()]
        a, b = matrices(D, sub, va, c["features"])
        m = fit_model(c, a, Y[sub])
        p = m.predict(b)
        rows.append(
            {
                "diagnostic": "learning_curve",
                "fold": fold,
                "fraction": fraction,
                "train_rows": len(sub),
                **metrics(Y[va], p),
            }
        )
        print(
            f'Learning curve fold {fold+1} fraction {fraction}: {rows[-1]["macro_f1"]:.4f}',
            flush=True,
        )
pd.DataFrame(rows).to_csv(HERE / "development_diagnostics.csv", index=False)
