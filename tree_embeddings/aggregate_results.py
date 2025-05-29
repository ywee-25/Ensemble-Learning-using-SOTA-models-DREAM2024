import json
from pathlib import Path
from warnings import warn

import joblib
import pandas as pd

DIR_PRED = Path(__file__).parents[2] / "pred"
PATH_OUT = DIR_PRED / "all_scores.csv"

exclude = [
    "tree_embeddings_molecules_and_mixtures_similarity",
    "mix_graph_closeness",
]

records = []
for score_path in DIR_PRED.glob("*/scores.json"):
    if any(s in score_path.parent.stem for s in exclude):
        print(f"Skipping {score_path}.")
        continue
    try:
        with score_path.open() as f:
            record = json.load(f)
    except json.JSONDecodeError:
        warn(f"Failed to load {score_path}.")
        continue
    if "cv_test_neg_RMSE" not in record:
        # warn(f"Skipping {score_path}: not latest version of scoring.")
        continue
    # if len(record["cv_test_neg_RMSE"]) != 20:
    if len(record["cv_test_neg_RMSE"]) != 10:
        # warn(f"Skipping {score_path}: not latest version of scoring.")
        continue

    record["run"] = score_path.parent.name

    model_path = score_path.with_name("model.joblib")
    if model_path.exists():
        record["model"] = str(joblib.load(model_path))

    print(f"Loaded {score_path}.")
    records.append(record)

df = pd.DataFrame.from_records(records)
df[["name", "version"]] = df["run"].str.rsplit("_v", n=1, expand=True)
df.to_csv(PATH_OUT, index=False)
print(f"Saved scores to {PATH_OUT}.")
