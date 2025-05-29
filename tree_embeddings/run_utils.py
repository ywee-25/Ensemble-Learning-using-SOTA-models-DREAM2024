import sys
from warnings import warn
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from typing import Union, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
import utils

__all__ = [
    "generate_mixture_binary_composition",
    "get_mixture_binary_composition",
    "generate_mixture_molecular_features",
    "get_embedded_train_set",
    "get_embedded_leaderboard_set",
    "get_embedded_test_set",
    "make_submission_files",
]

DIR_ROOT = Path(__file__).parents[2].resolve()
PATH_BINARY_COMPOSITION = DIR_ROOT / "data/mixture_binary_composition.csv"


def _vstack(arrays, use_sparse=True):
    if use_sparse:
        return sparse.vstack(arrays)
    return np.vstack(arrays)


def _hstack(arrays, use_sparse=True):
    if use_sparse:
        return sparse.hstack(arrays)
    return np.hstack(arrays)


def generate_mixture_binary_composition(out: Path = PATH_BINARY_COMPOSITION):
    f"""Generates a binary matrix indicating the presence of each molecule in each mixture.

    Saves the resulting matrix to a CSV file.

    Parameters
    ----------
    out : Path, optional
        Path to save the binary matrix, by default {PATH_BINARY_COMPOSITION}

    Returns
    -------
    pd.DataFrame
        A binary matrix with mixtures as rows and molecules as columns.
    """
    mixtures = utils.load_mixtures()

    stacked_mixtures = mixtures.stack()
    stacked_mixtures = stacked_mixtures[stacked_mixtures != 0]

    mixture_binary_composition = pd.crosstab(
        stacked_mixtures.index.get_level_values(0),
        stacked_mixtures.astype(str),  # Use CID as string to match other dataframes
    )
    mixture_binary_composition.to_csv(out)

    return mixture_binary_composition


def get_mixture_binary_composition(out: Path = PATH_BINARY_COMPOSITION):
    if out.exists():
        return pd.read_csv(out, index_col=0)
    return generate_mixture_binary_composition(out)


def generate_mixture_molecular_features(mixture_binary_composition, molecule_features):
    missing_molecules = set(mixture_binary_composition.columns) - set(
        molecule_features.index
    )
    if missing_molecules:
        warn(f"Missing descriptors for CIDs: {missing_molecules}")

    molecule_features = molecule_features.loc[mixture_binary_composition.columns]

    mixture_features = (
        mixture_binary_composition
        @ molecule_features
        / mixture_binary_composition.values.sum(axis=1, keepdims=True)
    )
    return mixture_features


def get_embedded_train_set(
    mixture_features: pd.DataFrame, combine_mixture_pair: callable, use_sparse=True
):
    df_train = utils.load_train()

    X_train = _vstack(
        [  # TODO: use df.apply
            combine_mixture_pair(
                mixture_features.loc[row["mixture1"]],
                mixture_features.loc[row["mixture2"]],
            )
            for _, row in tqdm(df_train.iterrows(), total=df_train.shape[0])
        ],
        use_sparse=use_sparse,
    )
    X_train = _hstack(
        [
            df_train.Dataset.str.startswith("Bushdid").values.astype(int)[:, None],
            X_train,
        ],
        use_sparse=use_sparse,
    )
    y_train = df_train["distance"]

    # Remove Bushdid mixtures from training set  # TODO
    # bushdid_mask = df_train.Dataset.str.startswith("Bushdid")
    # X_train = X_train[~bushdid_mask]
    # y_train = y_train[~bushdid_mask]

    return X_train, y_train


def get_embedded_leaderboard_set(
    mixture_features: pd.DataFrame, combine_mixture_pair: callable, use_sparse=True
):
    df_leaderboard = pd.read_csv(DIR_ROOT / "data/Leaderboard_set_Submission_form.csv")

    X_leaderboard = _vstack(
        [
            combine_mixture_pair(
                mixture_features.loc[f"{row.Dataset}/{row.Mixture_1:03d}"],  # HACK
                mixture_features.loc[f"{row.Dataset}/{row.Mixture_2:03d}"],  # HACK
            )
            for _, row in df_leaderboard.iterrows()
        ],
        use_sparse=use_sparse,
    )
    X_leaderboard = _hstack(
        [
            df_leaderboard.Dataset.str.startswith("Bushdid").values.astype(int)[:, None],
            X_leaderboard,
        ],
        use_sparse=use_sparse,
    )

    y_targets = pd.read_csv(DIR_ROOT / "data/leaderboard_targets.csv")

    # NOTE: Keeping just in case the file changes, but this should not be necessary
    # anymore, we saved leaderboard_targets.csv in the correct index order.
    y_targets = (
        y_targets.set_index(["Dataset", "Mixture_1", "Mixture_2"])
        .reindex(df_leaderboard[["Dataset", "Mixture_1", "Mixture_2"]])
        .reset_index()
    )

    y_targets = y_targets.Experimental_value

    return X_leaderboard, y_targets


def get_embedded_test_set(
    mixture_features: pd.DataFrame, combine_mixture_pair: callable, use_sparse=True
):
    df_test = pd.read_csv(DIR_ROOT / "data/Test_set_Submission_form.csv")

    X_test = _vstack(
        [
            combine_mixture_pair(
                mixture_features.loc[f"test/{row.Mixture_1:02d}"],  # HACK
                mixture_features.loc[f"test/{row.Mixture_2:02d}"],  # HACK
            )
            for _, row in df_test.iterrows()
        ],
        use_sparse=use_sparse,
    )
    X_test = _hstack(  # Will be all 0
        [
            df_test.Dataset.str.startswith("Bushdid").values.astype(int)[:, None],
            X_test,
        ],
        use_sparse=use_sparse,
    )

    return X_test


def make_submission_files(
    pred_leaderboard: Union[np.ndarray, pd.Series],
    #pred_test: Union[np.ndarray, pd.Series],
    model=None,
    dir_submission: Path = Path(DIR_ROOT / "pred/tree_embeddings_morgan/"),
    versioning: bool = True,
    scores: Optional[dict] = None,
):
    if versioning:
        existing_submissions = dir_submission.parent.glob(dir_submission.name + "_v*")
        current_version = max(
            (
                int(submission.name.rsplit("_v", maxsplit=1)[-1])
                for submission in existing_submissions
            ),
            default=-1,
        )
        dir_submission = dir_submission.with_name(
            dir_submission.name + f"_v{current_version + 1}"
        )

    # df_test = pd.read_csv(DIR_ROOT / "data/Test_set_Submission_form.csv")
    df_leaderboard = pd.read_csv(DIR_ROOT / "data/Leaderboard_set_Submission_form.csv")

    leaderboard_submission = df_leaderboard.assign(
        Predicted_Experimental_Values=pred_leaderboard,
    )
    # test_submission = df_test.assign(
    #     Predicted_Experimental_Values=pred_test,
    # )

    dir_submission.mkdir(exist_ok=True, parents=True)
    leaderboard_submission.to_csv(
        dir_submission / "Leaderboard_set_Submission_form.csv", index=False
    )
    # test_submission.to_csv(dir_submission / "Test_set_Submission_form.csv", index=False)

    if model is not None:
        print("Saving model...", end=" ")
        joblib.dump(model, dir_submission / "model.joblib")
        print("Done.")

    if scores is not None:
        with open(dir_submission / "scores.json", "w") as f:
            json.dump(scores, f)

    print("Submission files saved to", dir_submission)
