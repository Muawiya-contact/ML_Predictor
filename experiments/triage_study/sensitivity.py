"""Development-only sensitivity tests; these do not alter the final test split."""

import json
from research_engine import HERE, evaluate_cv, config

results = []
for excluded in [("ECG_Status",), ("ECG_Status", "Mode_of_Arrival"), ("Gender",)]:
    c = config(
        "hgb",
        {"view": "structured", "exclude_cat": excluded},
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1,
        learning_rate=0.1,
    )
    r = evaluate_cv(c, 3)
    results.append(r)
(HERE / "sensitivity_results.json").write_text(json.dumps(results, indent=2))
