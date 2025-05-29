import sys
from pathlib import Path
import pickle
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
from scipy.special import softmax
from sklearn.base import clone
from sklearn.metrics import make_scorer, check_scoring
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, LeaveOneOut
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import utils, run_utils
from run_utils import (
    get_mixture_binary_composition,
    generate_mixture_molecular_features,
    get_embedded_train_set,
    get_embedded_leaderboard_set,
    get_embedded_test_set,
    make_submission_files,
)

N_JOBS = cpu_count() - 1
CV_N_ESTIMATORS = 100
FINAL_N_ESTIMATORS = 1000
RSTATE = 42
KFOLDS = 10
DIR_ROOT = Path(__file__).parents[2].resolve()
DEF_SCORING = {
    "R2": "r2",
    "neg_RMSE": "neg_root_mean_squared_error",
    "Pearson": make_scorer(lambda y, y_pred: stats.pearsonr(y, y_pred)[0]),
}


def normalize(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# NOTE: this old file leads to leakage of test labels!
def load_mixture_graph_closeness():
    mixture_similarity = np.load(DIR_ROOT / "data/graph_closeness_new.npy")
    with open(DIR_ROOT / "data/graph_mix_id_map_new.pickle", "rb") as f:
        mixture_similarity_index = pickle.load(f)
    mixture_similarity_index = pd.Series(mixture_similarity_index).sort_values()

    # HACK: standardize the index
    mixture_similarity_index = pd.Series(
        mixture_similarity_index.index,
        index=mixture_similarity_index.index,
    )
    test_instances = mixture_similarity_index.str.startswith("test")
    number = (
        mixture_similarity_index[test_instances]
        .str.extract(r"test_(\d\d?)", expand=False)
        .str.zfill(2)
    )
    mixture_similarity_index[test_instances] = "test/" + number
    mixture_similarity_index.index = mixture_similarity_index

    mixture_similarity = pd.DataFrame(
        mixture_similarity,
        index=mixture_similarity_index.index,
        columns=mixture_similarity_index.index,
    )
    return mixture_similarity


def combine_molecule_embeddings(molecule_matrix):
    return pd.concat(
        {
            "mean": molecule_matrix.mean(axis=0),
            "std": molecule_matrix.std(axis=0),
            "min": molecule_matrix.min(),
            "max": molecule_matrix.max(),
            "q10": molecule_matrix.quantile(0.1),
            "q25": molecule_matrix.quantile(0.25),
            "q50": molecule_matrix.quantile(0.5),
            "q75": molecule_matrix.quantile(0.75),
            "q90": molecule_matrix.quantile(0.9),
            "iqr": molecule_matrix.apply(stats.iqr),
            "hmean": molecule_matrix.apply(stats.hmean),
            "gmean": molecule_matrix.apply(stats.gmean),
            "sqmean": molecule_matrix.apply(stats.pmean, p=2),
            "expmean": molecule_matrix.apply(
                lambda x: np.log(np.exp(x).sum()) - np.log(len(x))
            ),
            "skew": molecule_matrix.apply(stats.skew),
            "kurtosis": molecule_matrix.apply(stats.kurtosis),
            "entropy": molecule_matrix.apply(stats.entropy),
        }
    )


def gen_complex_mix_features(binary_composition, molecule_embeddings):
    # Ensure the same order of mixtures
    molecule_embeddings = molecule_embeddings.loc[binary_composition.columns]
    # Min-max normalization  # XXX
    molecule_embeddings = molecule_embeddings - molecule_embeddings.min()
    molecule_embeddings /= molecule_embeddings.max() - molecule_embeddings.min()

    rows = [
        combine_molecule_embeddings(molecule_embeddings.loc[row.astype(bool)])
        for _, row in tqdm(binary_composition.iterrows(), total=len(binary_composition))
    ]

    return pd.DataFrame.from_records(
        rows,
        index=binary_composition.index,
    )


def apply_model(
    model,
    mixture_tree_embeddings,
    combine_mixture_pair,
    dir_submission,
    scoring=None,
    final_n_estimators=FINAL_N_ESTIMATORS,
    cv_n_estimators=CV_N_ESTIMATORS,
    n_jobs=N_JOBS,
    sparse=True,
):
    scorer = check_scoring(model, scoring or DEF_SCORING)

    X_train, y_train = get_embedded_train_set(
        mixture_tree_embeddings,
        combine_mixture_pair,
        use_sparse=sparse,
    )
    X_leaderboard, y_leaderboard = get_embedded_leaderboard_set(
        mixture_tree_embeddings,
        combine_mixture_pair,
        use_sparse=sparse,
    )
    X_test = get_embedded_test_set(
        mixture_tree_embeddings,
        combine_mixture_pair,
        use_sparse=sparse,
    )

    print("Leaderboard fit...")
    model.set_params(n_estimators=FINAL_N_ESTIMATORS)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)

    model.fit(X_train, y_train)

    y_hat_leaderboard = model.predict(X_leaderboard)


    val_scores = scorer(model, X_leaderboard, y_leaderboard)
    val_scores = {"val_" + k: v for k, v in val_scores.items()}
    print("Leaderboard scores:", val_scores)

    # Append leaderboard set to training set
    # X_train = run_utils._vstack([X_train, X_leaderboard], use_sparse=sparse)
    # y_train = np.hstack([y_train, y_leaderboard])

    # Generate the final test predictions for submission
    # print("Test fit...")
    # model.fit(X_train, y_train)
    # y_hat_test = model.predict(X_test)

    # Set a lower number of estimators for CV and LOO and let paralelism to be
    # based on the folds.
    # cv_model = clone(model).set_params(
    #     n_estimators=cv_n_estimators, verbose=0
    # )
    # if "n_jobs" in cv_model.get_params():
    #     cv_model.set_params(n_jobs=n_jobs)

    # We perform LOO only for metrics that support it.
    # print("Validating model with leave-one-out...")
    # try:
    # #     cv_raw_scores = cross_validate(
    # #         cv_model,
    # #         X_train,
    # #         y_train,
    # #         scoring=scorer,
    # #         # scoring={
    # #         #     "neg_RMSE": "neg_root_mean_squared_error",
    # #         # },
    # #         # cv=LeaveOneOut(),
    # #         cv=KFold(
    # #             n_splits=KFOLDS,
    # #             shuffle=True,
    # #             random_state=RSTATE,
    # #         ),
    # #         n_jobs=n_jobs,
    # #         verbose=50,
    # #     )
    # #     cv_raw_scores = {"cv_" + k: list(v) for k, v in cv_raw_scores.items()}
    # #
    # #     # Calculate mean and std of cross-validation scores
    # #     cv_mean_scores = {}
    # #     for k, v in cv_raw_scores.items():
    # #         cv_mean_scores[k + "_mean"] = np.mean(v)
    # #         cv_mean_scores[k + "_std"] = np.std(v)
    #
    # except KeyboardInterrupt:
    #     print("Interrupted.")
    #     cv_mean_scores = {}
    #     cv_raw_scores = {}

    # # We then use 5x20 CV for the Pearson correlation, to ensure larger test
    # # sets and thus more stable estimates.
    # print("Cross-validating model...")
    # cv_raw_scores |= cross_validate(
    #     cv_model,
    #     X_train,
    #     y_train,
    #     scoring={
    #         "R2": "r2",
    #         "Pearson": make_scorer(lambda y, y_pred: pearsonr(y, y_pred)[0]),
    #     },
    #     cv=RepeatedKFold(
    #         n_splits=5,
    #         n_repeats=40,
    #         random_state=RSTATE,
    #     ),
    #     n_jobs=n_jobs,
    #     verbose=50,
    # )

    # print("Leaderboard scores:", val_scores)
    # print("CV scores:", cv_mean_scores)

    # Save the model and scores
    make_submission_files(
        pred_leaderboard=y_hat_leaderboard,
        #pred_test=y_hat_test,
        model=model,
        dir_submission=dir_submission,
        versioning=True
        #scores = {**val_scores, **cv_mean_scores, **cv_raw_scores},
    )


def plot_validation_curve(
    model, mixture_tree_embeddings, combine_mixture_pair, out, n_jobs=N_JOBS
):
    from sklearn.model_selection import ValidationCurveDisplay
    import matplotlib.pyplot as plt

    X_train, y_train = get_embedded_train_set(
        mixture_tree_embeddings, combine_mixture_pair
    )
    X_leaderboard, y_leaderboard = get_embedded_leaderboard_set(
        mixture_tree_embeddings, combine_mixture_pair
    )

    # Append leaderboard set to training set
    X_train = sparse.vstack([X_train, X_leaderboard])
    y_train = np.hstack([y_train, y_leaderboard])

    model = clone(model).set_params(n_jobs=1, warm_start=True)

    ValidationCurveDisplay.from_estimator(
        model,
        X_train,
        y_train,
        param_name="n_estimators",
        param_range=[10, 25, 50, 100, 500, 1000, 5000],
        cv=KFold(
            n_splits=10,
            shuffle=True,
            random_state=RSTATE,
        ),
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    plt.savefig(out)


def experiment_1():
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    # molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]  # Performs worse

    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )
        # return np.abs(a - b)
        # return sparse.hstack([a, b], format="csr")

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )
    # model = GradientBoostingRegressor(  # Worse.
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     # max_depth="sqrt",
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_embeddings_morgan"),
    )


def experiment_2():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.hstack(
        [mixture_tree_embeddings, mixture_molecular_features],
        format="csr",
    )
    mixture_tree_embeddings = PCA(random_state=RSTATE).fit_transform(
        mixture_tree_embeddings
    )
    mixture_tree_embeddings = pd.Series(
        [sparse.csr_array(row[None, :]) for row in mixture_tree_embeddings],
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )
        # return np.abs(a - b)

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(
            DIR_ROOT / "pred/tree_embeddings_morgan_with_molecular_features"
        ),
    )


def experiment_3():
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.hstack(
        [
            mixture_tree_embeddings,
            mixture_binary_composition.index.str.startswith("Bushdid").astype(int)[
                :, None
            ],
        ],
        format="csr",
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )
        # return np.abs(a - b)

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_embeddings_morgan_bushdid_indicator"),
    )


def experiment_4():
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return a + b  # XXX

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_embeddings_morgan_ternary_pair"),
    )


def experiment_5():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]
    molecule_features.index = molecule_features.index.astype(str)

    # Added a third step, also encoding morgan fingerprints with tree-embeddings
    molecule_embedder = ForestEmbedder(
        RandomForestRegressor(
            n_estimators=100,
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        max_pvalue=0.05,
        node_weights="log_node_size",  # Will be considered in the impurity of mixture embedder
    )
    print("Fitting molecule embedder...")
    molecule_tree_embeddings = molecule_embedder.fit_transform(
        molecule_features,
        molecule_features,
    )
    print("Fitting PCA to compress molecule embedders...")
    molecule_tree_embeddings = PCA(n_components=100, random_state=RSTATE).fit_transform(
        molecule_tree_embeddings
    )
    molecule_tree_embeddings = pd.DataFrame(
        molecule_tree_embeddings,
        index=molecule_features.index,
    )

    print("Combining molecule embeddings into mixtures...")
    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_tree_embeddings,
    )

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
    )
    print("Fitting mixture embedder...")
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        # return np.abs(a - b)
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_embeddings_molecules_and_mixtures"),
    )


def experiment_6():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]
    molecule_features.index = molecule_features.index.astype(str)

    # Added a third step, also encoding morgan fingerprints with tree-embeddings
    molecule_embedder = ForestEmbedder(
        RandomForestRegressor(
            n_estimators=100,
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        max_pvalue=0.05,
        node_weights="log_node_size",  # Will be considered in the impurity of mixture embedder
    )
    print("Fitting molecule embedder...")
    molecule_tree_embeddings = molecule_embedder.fit_transform(
        molecule_features,
        molecule_features,
    )
    print("Fitting PCA to compress molecule embedders...")
    molecule_tree_embeddings = PCA(n_components=100, random_state=RSTATE).fit_transform(
        molecule_tree_embeddings
    )
    molecule_tree_embeddings = pd.DataFrame(
        molecule_tree_embeddings,
        index=molecule_features.index,
    )

    print("Combining molecule embeddings into mixtures...")
    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_tree_embeddings,
    )

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
    )
    print("Fitting mixture embedder...")
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_molecular_features,  # XXX
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):  # one-hot encode
        return sparse.hstack(
            [
                np.abs(a - b).reshape(1, -1),
                (a * b).reshape(1, -1),
                # ((1 - a) * (1 - b)).reshape(1, -1)
            ],
            format="csr",
        )
        # return np.abs(a - b)

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_embeddings_everywhere"),
    )


def experiment_7():
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        node_weights="log_node_size",  # Indifferent
        max_pvalue=0.01,  # Indifferent
    )
    print("Fitting mixture embedder...")
    # NOTE: Other comibnations of input features and target features did not improve the model (see exp. 6).
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_binary_composition,
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                (a + b == 0).reshape(1, -1),
                (a + b == 1).reshape(1, -1),
                (a + b == 2).reshape(1, -1),
            ],
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/only_binary_composition"),
    )


def experiment_8():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        node_weights="log_node_size",  # Indifferent
        max_pvalue=0.01,  # Indifferent
    )
    print("Fitting mixture embedder...")
    # NOTE: Other comibnations of input features and target features did not improve the model (see exp. 6).
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_binary_composition,
        # sample_weight=~mixture_binary_composition.index.str.startswith("test"),  # Worse.  # TODO: Weird!
    )

    # Second step
    for _ in range(1):  # Once was best
        mixture_tree_embeddings = mixture_embedder.fit_transform(
            # mixture_binary_composition,  # Worse.
            mixture_tree_embeddings.toarray(),
            PCA(n_components=10, random_state=RSTATE).fit_transform(
                mixture_tree_embeddings
            ),
        )

    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        n_estimators=1000,  # TODO: 1000 trees is worse. Why?
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/only_binary_composition_2step"),
    )


def experiment_9():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    # from xgboost import XGBRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        node_weights="log_node_size",  # Important
        # node_weights=lambda node_size: node_size,  # Worse.
        # node_weights=lambda node_size: 1 / node_size,  # Worse.
        max_pvalue=1e-4,  # Important
    )
    print("Fitting mixture embedder...")
    # NOTE: Other comibnations of input features and target features did not improve the model (see exp. 6).
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_binary_composition,
        # sample_weight=~mixture_binary_composition.index.str.startswith("test"),  # Worse.  # TODO: Weird!
    )

    # Second step
    for _ in range(1):  # Once was best
        mixture_tree_embeddings = mixture_embedder.fit_transform(
            # mixture_binary_composition,  # Worse.
            mixture_tree_embeddings.toarray(),
            PCA(n_components=10, random_state=RSTATE).fit_transform(
                mixture_tree_embeddings
            ),
        )

    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = GradientBoostingRegressor(
        n_estimators=5_000,  # 50_000 made no difference.
        learning_rate=0.1,  # 0.01 is worse.
        min_impurity_decrease=1e-6,
        loss="squared_error",
        max_features="sqrt",  # log2 is worse.
        max_depth=4,  # 4 better than 5, 6, 7, 8. But 2, 3 are worse.
        random_state=RSTATE,
        verbose=50,
    )
    # model = XGBRegressor(  # Worse: 0.62
    #     n_estimators=1_000,
    #     num_parallel_tree=10,
    #     max_depth=4,
    #     n_jobs=10,
    #     random_state=RSTATE,
    #     verbosity=3,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/only_binary_composition_2step_gbm"),
    )


def experiment_10():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    # from xgboost import XGBRegressor
    from tree_embedder import ForestEmbedder

    print("Loading molecule features...")
    molecule_features_human = pd.read_csv(
        DIR_ROOT / "data/percept_single.csv", index_col=0
    )
    # molecule_features_pred = pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0)
    molecule_features_pred = pd.read_csv(
        DIR_ROOT / "data/perception_prediction_new.csv", index_col=0
    )
    molecule_features_human = molecule_features_human.loc[
        :, molecule_features_pred.columns
    ]
    molecule_features = molecule_features_human.combine_first(molecule_features_pred)
    molecule_features.index = molecule_features.index.astype(str)

    mixture_binary_composition = get_mixture_binary_composition()
    # mixture_binary_composition = mixture_binary_composition.drop(columns="8858")  # FIXME: get perceptions for this molecule

    print("Averaging molecule features to represent mixtures...")
    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # node_weights="log_node_size",  # Important  # FIXME: was on!
        # node_weights=lambda node_size: node_size,  # Worse.
        # node_weights=lambda node_size: 1 / node_size,  # Worse.
        max_pvalue=1e-4,  # Important
    )
    print("Fitting mixture tree-embedder...")
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
        # sample_weight=~mixture_binary_composition.index.str.startswith("test"),  # Do not use test mixtures to generate embeddings
    )

    # Second step
    # for _ in range(1):  # Once was best
    #     mixture_tree_embeddings = mixture_embedder.fit_transform(
    #         # mixture_binary_composition,  # Worse.
    #         mixture_tree_embeddings.toarray(),
    #         PCA(n_components=10, random_state=RSTATE).fit_transform(mixture_tree_embeddings),
    #         # sample_weight=~mixture_binary_composition.index.str.startswith("test"),  # Do not use test mixtures to generate embeddings
    #     )

    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = GradientBoostingRegressor(
        n_estimators=5_000,
        learning_rate=0.1,
        min_impurity_decrease=1e-6,
        loss="squared_error",
        max_features="sqrt",
        max_depth=4,
        random_state=RSTATE,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/smell_embeddings_2step_gbm"),
    )


def experiment_11():  # Based on experiment 7
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from tree_embedder import ForestEmbedder

    print("Loading molecule features...")
    molecule_features_human = pd.read_csv(
        DIR_ROOT / "data/percept_single.csv", index_col=0
    )
    # molecule_features_pred = pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0)
    molecule_features_pred = pd.read_csv(
        DIR_ROOT / "data/perception_prediction_new.csv", index_col=0
    )
    molecule_features_human = molecule_features_human.loc[
        :, molecule_features_pred.columns
    ]
    molecule_features = molecule_features_human.combine_first(molecule_features_pred)

    # Worse.
    # molecule_features = pd.read_csv(DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0)
    # molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]

    molecule_features.index = molecule_features.index.astype(str)

    mixture_binary_composition = get_mixture_binary_composition()
    molecule_similarity = cosine_similarity(
        molecule_features.loc[mixture_binary_composition.columns]
    )

    mixture_similarity_composition = (
        mixture_binary_composition
        @ molecule_similarity
        / mixture_binary_composition.values.sum(axis=1, keepdims=True)
    )

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # node_weights="log_node_size",  # Indifferent  # FIXME
        max_pvalue=0.01,  # Indifferent
    )
    pretext_y = PCA(n_components=10, random_state=RSTATE).fit_transform(
        mixture_similarity_composition
    )
    # pretext_y /= pretext_y.std(axis=0)  # Normalize (worse)

    print("Fitting mixture embedder...")
    # NOTE: Other comibnations of input features and target features did not improve the model (see exp. 6).
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        # mixture_similarity_composition,
        # mixture_similarity_composition,
        mixture_binary_composition,
        pretext_y,
    )
    # Reduce dimensionality (worse)
    # mixture_tree_embeddings = PCA(n_components=50, random_state=RSTATE).fit_transform(mixture_tree_embeddings)
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        # [sparse.csr_array(row[None, :]) for row in mixture_tree_embeddings],
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                ((a + b) == 0).reshape(1, -1),
                ((a + b) == 1).reshape(1, -1),
                ((a + b) == 2).reshape(1, -1),
            ],
            format="csr",
            dtype=np.int8,
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        n_estimators=50,
        max_features="sqrt",
        random_state=RSTATE,
        # n_jobs=19,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.5,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/composition_from_mol_similarity"),
    )


def experiment_12():  # Based on experiment 1
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    print("Loading molecule features...")
    molecule_features_human = pd.read_csv(
        DIR_ROOT / "data/percept_single.csv", index_col=0
    )
    # molecule_features_pred = pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0)
    molecule_features_pred = pd.read_csv(
        DIR_ROOT / "data/perception_prediction_new.csv", index_col=0
    )
    molecule_features_human = molecule_features_human.loc[
        :, molecule_features_pred.columns
    ]
    molecule_features = molecule_features_human.combine_first(molecule_features_pred)

    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )
    mixture_molecular_features /= mixture_molecular_features.std(
        axis=0
    )  # Normalize (important)

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                ((a + b) == 0).reshape(1, -1),
                ((a + b) == 1).reshape(1, -1),
                ((a + b) == 2).reshape(1, -1),
            ],
            format="csr",
            dtype=np.int8,
        )
        # return sparse.hstack([np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr")
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        # n_estimators=1000,
        n_estimators=50,
        max_features="sqrt",
        random_state=RSTATE,
        # n_jobs=19,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.9,  # Worse.
    #     # max_depth=6,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_embeddings_perception"),
    )
    # plot_validation_curve(
    #     model,
    #     mixture_tree_embeddings,
    #     combine_mixture_pair,
    #     out="./validation_curve.png",
    # )


def experiment_13():  # Based on experiment 1
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    print("Loading molecule features...")
    molecule_features_human = pd.read_csv(
        DIR_ROOT / "data/percept_single.csv", index_col=0
    )
    # molecule_features_pred = pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0)
    molecule_features_pred = pd.read_csv(
        DIR_ROOT / "data/perception_prediction_new.csv", index_col=0
    )
    molecule_features_human = molecule_features_human.loc[
        :, molecule_features_pred.columns
    ]
    molecule_features = molecule_features_human.combine_first(molecule_features_pred)

    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )
    mixture_molecular_features /= mixture_molecular_features.std(
        axis=0
    )  # Normalize (important)

    def combine_mixture_pair(a, b):
        return sparse.csr_array(pd.concat([np.abs(a - b), (a * b)]).values[None, :])
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        random_state=RSTATE,
        n_jobs=10,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.9,  # Worse.
    #     # max_depth=6,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_molecular_features,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/tree_only_perception"),
    )


def experiment_14():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    print("Loading molecule fingerprints...")
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]
    molecule_features.index = molecule_features.index.astype(str)

    print("Loading molecule smell data...")
    molecule_smell = pd.read_csv(DIR_ROOT / "data/percept_single.csv", index_col=0)
    molecule_smell = normalize(molecule_smell)
    molecule_smell.index = molecule_smell.index.astype(str)

    # Added a third step, also encoding morgan fingerprints with tree-embeddings
    molecule_embedder = ForestEmbedder(
        RandomForestRegressor(
            n_estimators=100,
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
        node_weights="log_node_size",  # Will be considered in the impurity of mixture embedder
    )

    print("Fitting molecule embedder...")
    molecule_embedder.fit(
        molecule_features.loc[
            molecule_smell.index
        ],  # Only use molecules with smell data
        molecule_smell,
    )
    molecule_tree_embeddings = molecule_embedder.transform(molecule_features)

    print("Fitting PCA to compress molecule embedders...")
    # molecule_tree_embeddings = PCA(n_components=100, random_state=RSTATE).fit_transform(molecule_tree_embeddings)
    molecule_tree_embeddings = PCA(n_components=20, random_state=RSTATE).fit_transform(
        molecule_tree_embeddings
    )
    # TODO: no normalization?
    molecule_tree_embeddings = normalize(molecule_tree_embeddings)

    molecule_tree_embeddings = pd.DataFrame(
        molecule_tree_embeddings,
        index=molecule_features.index,
    )

    print("Combining molecule embeddings into mixtures...")
    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_tree_embeddings,
    )
    mixture_molecular_features = PCA(
        n_components=20, random_state=RSTATE
    ).fit_transform(mixture_molecular_features)
    # TODO: no normalization?
    mixture_molecular_features = normalize(mixture_molecular_features)

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
    )
    print("Fitting mixture embedder...")
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    # XXX
    # mixture_tree_embeddings = PCA(n_components=10, random_state=RSTATE).fit_transform(mixture_tree_embeddings)
    # mixture_tree_embeddings = normalize(mixture_tree_embeddings)

    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    # TODO: mixture similarities

    def combine_mixture_pair(a, b):
        # return sparse.hstack(
        #     [
        #         np.abs(a - b).reshape(1, -1),
        #         (a * b).reshape(1, -1),
        #     ],
        #     format="csr",
        #     dtype=np.int8,
        # )
        # return np.abs(a - b)
        return sparse.hstack(
            [
                ((a + b) == 0).reshape(1, -1),
                ((a + b) == 1).reshape(1, -1),
                ((a + b) == 2).reshape(1, -1),
            ],
            format="csr",
            dtype=np.int8,
        )

    model = RandomForestRegressor(
        # n_estimators=1000,
        n_estimators=50,
        max_features="sqrt",
        random_state=RSTATE,
        # n_jobs=19,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=DIR_ROOT / "pred/tree_embeddings_molecules_and_mixtures_smell",
    )
    # plot_validation_curve(
    #     model,
    #     mixture_tree_embeddings,
    #     combine_mixture_pair,
    #     out=DIR_ROOT / "pred/tree_embeddings_molecules_and_mixtures_smell/validation_curve.png",
    # )


def experiment_15():
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from tree_embedder import ForestEmbedder

    # print("Loading molecule features...")
    # molecule_features_human = pd.read_csv(DIR_ROOT / "data/percept_single.csv", index_col=0)
    # molecule_features_pred = pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0)
    # molecule_features_human = molecule_features_human.loc[:, molecule_features_pred.columns]
    # molecule_features = molecule_features_human.combine_first(molecule_features_pred)

    # Worse.
    # molecule_features = pd.read_csv(DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0)
    # molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]

    # molecule_features.index = molecule_features.index.astype(str)

    mixture_binary_composition = get_mixture_binary_composition()

    # molecule_similarity = cosine_similarity(molecule_features.loc[mixture_binary_composition.columns])

    mixture_similarity = np.load(DIR_ROOT / "data/graph_closeness.npy")
    with open(DIR_ROOT / "data/graph_mix_id_map.pickle", "rb") as f:
        mixture_similarity_index = pickle.load(f)
    mixture_similarity_index = pd.Series(mixture_similarity_index).sort_values()
    mixture_similarity = pd.DataFrame(
        mixture_similarity,
        index=mixture_similarity_index.index,
    )
    mixture_similarity_index = mixture_similarity_index.index

    # Using new closeness instead
    # mixture_similarity = load_mixture_graph_closeness()  # new closeness
    # mixture_similarity_index = mixture_similarity.index

    # mixture_similarity = PCA(n_components=50, random_state=RSTATE).fit_transform(mixture_similarity)  # second best
    # mixture_similarity = PCA(random_state=RSTATE).fit_transform(mixture_similarity)
    # mixture_similarity = PCA(n_components=30, random_state=RSTATE).fit_transform(mixture_similarity)  # best
    mixture_similarity = PCA(n_components=20, random_state=RSTATE).fit_transform(
        mixture_similarity
    )
    mixture_similarity = normalize(mixture_similarity)  # Important.

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.01,  # Indifferent
    )

    print("Fitting mixture embedder...")
    # TODO: mixture closeness for every mixture
    mixture_embedder.fit(
        mixture_binary_composition.loc[mixture_similarity_index],
        mixture_similarity,
    )
    mixture_tree_embeddings = mixture_embedder.transform(mixture_binary_composition)
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        # [sparse.csr_array(row[None, :]) for row in mixture_tree_embeddings],
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                (a + b == 0).reshape(1, -1),
                (a + b == 1).reshape(1, -1),
                (a + b == 2).reshape(1, -1),
            ],
            format="csr",
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.5,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/mix_graph_closeness"),
    )


def experiment_16():
    import pickle

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    # print("Loading molecule features...")
    molecule_features_human = pd.read_csv(
        DIR_ROOT / "data/percept_single.csv", index_col=0
    )
    # molecule_features_pred = pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0)
    molecule_features_pred = pd.read_csv(
        DIR_ROOT / "data/perception_prediction_new.csv", index_col=0
    )
    molecule_features_human = molecule_features_human.loc[
        :, molecule_features_pred.columns
    ]
    molecule_features = molecule_features_human.combine_first(molecule_features_pred)
    molecule_features = normalize(molecule_features)

    # Worse.
    # molecule_features = pd.read_csv(DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0)
    # molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]

    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )
    mixture_similarity = cosine_similarity(mixture_molecular_features)
    # XXX
    mixture_similarity = PCA(n_components=30, random_state=RSTATE).fit_transform(
        mixture_similarity
    )
    mixture_similarity = normalize(mixture_similarity)  # Important.

    # # Graph closeness
    # mixture_similarity = np.load(DIR_ROOT / "data/graph_closeness.npy")
    # with open(DIR_ROOT / "data/graph_mix_id_map.pickle", "rb") as f:
    #     mixture_similarity_index = pickle.load(f)
    # mixture_similarity_index = pd.Series(mixture_similarity_index).sort_values()
    # mixture_similarity = pd.DataFrame(
    #     mixture_similarity,
    #     index=mixture_similarity_index.index,
    # )
    # # mixture_similarity = PCA(n_components=50, random_state=RSTATE).fit_transform(mixture_similarity))

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )

    print("Fitting mixture embedder...")
    # TODO: mixture closeness for every mixture
    mixture_embedder.fit(
        mixture_binary_composition,
        mixture_similarity,
    )
    mixture_tree_embeddings = mixture_embedder.transform(mixture_binary_composition)
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        # [sparse.csr_array(row[None, :]) for row in mixture_tree_embeddings],
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                (a + b == 0).reshape(1, -1),
                (a + b == 1).reshape(1, -1),
                (a + b == 2).reshape(1, -1),
            ],
            format="csr",
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.5,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/mix_smell_similarity"),
    )


def experiment_17():
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    mixture_binary_composition = get_mixture_binary_composition()
    mixture_binary_composition = pd.Series(
        list(sparse.csr_array(mixture_binary_composition.values)),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                (a + b == 0).reshape(1, -1),
                (a + b == 1).reshape(1, -1),
                (a + b == 2).reshape(1, -1),
            ],
            format="csr",
        )

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.5,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_binary_composition,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/simply_binary"),
    )


def experiment_18():  # almost the same as 16
    import pickle

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    # print("Loading molecule features...")
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/chemprop_embeddings.csv", index_col=0
    )  # v0
    # molecule_features = pd.read_csv(DIR_ROOT / "data/embeddings_molecule.csv", index_col=0) # v1
    molecule_features = normalize(molecule_features)
    molecule_features.index = molecule_features.index.astype(str)

    # FIXME
    common_index = molecule_features.index.intersection(
        mixture_binary_composition.columns
    )
    mixture_binary_composition = mixture_binary_composition.loc[:, common_index]

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )
    mixture_similarity = cosine_similarity(mixture_molecular_features)
    # XXX
    mixture_similarity = PCA(n_components=30, random_state=RSTATE).fit_transform(
        mixture_similarity
    )
    mixture_similarity = normalize(mixture_similarity)  # Important.

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )

    print("Fitting mixture embedder...")
    # TODO: mixture closeness for every mixture
    mixture_embedder.fit(
        mixture_binary_composition,
        mixture_similarity,
    )
    mixture_tree_embeddings = mixture_embedder.transform(mixture_binary_composition)
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        # [sparse.csr_array(row[None, :]) for row in mixture_tree_embeddings],
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                (a + b == 0).reshape(1, -1),
                (a + b == 1).reshape(1, -1),
                (a + b == 2).reshape(1, -1),
            ],
            format="csr",
        )
        # return np.abs(a - b)
        # return a + b  # Worse.

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=5000,
    #     max_features="sqrt",
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.5,
    #     max_depth=4,
    #     random_state=RSTATE,
    #     verbose=50,
    # )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=Path(DIR_ROOT / "pred/mix_chemprop_embeddings_similarity"),
    )


def experiment_19():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor

    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    print("Loading molecule fingerprints...")
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]
    molecule_features.index = molecule_features.index.astype(str)

    print("Loading molecule smell data...")
    # molecule_smell = pd.read_csv(DIR_ROOT / "data/percept_single.csv", index_col=0)
    molecule_smell = pd.read_csv(
        DIR_ROOT / "data/perception_prediction_new.csv", index_col=0
    )
    molecule_smell = normalize(molecule_smell)
    molecule_smell.index = molecule_smell.index.astype(str)

    mixture_similarity = utils.load_mixture_graph_closeness()
    mixture_similarity_index = mixture_similarity.index  # Will be lost after PCA

    # NOTE: previous best, but with data leakage
    # mixture_similarity = np.load(DIR_ROOT / "data/graph_closeness.npy")
    # with open(DIR_ROOT / "data/graph_mix_id_map.pickle", "rb") as f:
    #     mixture_similarity_index = pickle.load(f)
    # mixture_similarity_index = pd.Series(mixture_similarity_index).sort_values()
    # mixture_similarity = pd.DataFrame(
    #     mixture_similarity,
    #     index=mixture_similarity_index.index,
    # )

    # mixture_similarity = PCA(n_components=50, random_state=RSTATE).fit_transform(mixture_similarity)  # second best
    # mixture_similarity = PCA(random_state=RSTATE).fit_transform(mixture_similarity)
    mixture_similarity = PCA(n_components=30, random_state=RSTATE).fit_transform(
        mixture_similarity
    )  # best
    mixture_similarity = normalize(mixture_similarity)  # Important.

    # v3
    # Added a third step, also encoding morgan fingerprints with tree-embeddings
    molecule_embedder = ForestEmbedder(
        RandomForestRegressor(
            n_estimators=100,
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
        # node_weights="log_node_size",  # Will be considered in the impurity of mixture embedder
    )
    # FIXME
    molecules_common_index = molecule_features.index.intersection(molecule_smell.index)

    print("Fitting molecule embedder...")
    molecule_embedder.fit(
        molecule_features.loc[molecules_common_index],  # v3
        molecule_smell.loc[molecules_common_index],  # v3
    )
    # ensure aligned indices
    molecule_tree_embeddings = molecule_embedder.transform(
        molecule_features.loc[mixture_binary_composition.columns]
    )

    # print("Fitting PCA to compress molecule embedders...")
    # # molecule_tree_embeddings = PCA(n_components=100, random_state=RSTATE).fit_transform(molecule_tree_embeddings)
    # # molecule_tree_embeddings = PCA(n_components=20, random_state=RSTATE).fit_transform(molecule_tree_embeddings)
    # # TODO: no normalization?
    # # molecule_tree_embeddings = normalize(molecule_tree_embeddings)

    print("Combining molecule embeddings into mixtures...")
    # ensure aligned indices
    mixture_molecular_features = (
        mixture_binary_composition.loc[mixture_similarity_index].values
        @ molecule_tree_embeddings
    )

    # cannot do inplace because of typing
    mixture_molecular_features = (
        mixture_molecular_features
        / mixture_binary_composition.loc[mixture_similarity_index].values.sum(
            axis=1, keepdims=True
        )
    )  # v4
    # mixture_molecular_features = (mixture_molecular_features > 0).astype(int)  #v3
    # mixture_molecular_features = PCA(n_components=20, random_state=RSTATE).fit_transform(mixture_molecular_features)
    # # TODO: no normalization?
    # mixture_molecular_features = normalize(mixture_molecular_features)

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
    )
    print("Fitting mixture embedder...")
    mixture_tree_embeddings = mixture_embedder.fit(
        mixture_molecular_features,
        mixture_similarity,
    )

    all_mixture_molecular_features = (
        mixture_binary_composition.values @ molecule_tree_embeddings
    )
    # cannot do inplace because of typing
    all_mixture_molecular_features = (
        all_mixture_molecular_features
        / mixture_binary_composition.values.sum(axis=1, keepdims=True)
    )  # v4
    # all_mixture_molecular_features = (all_mixture_molecular_features > 0).astype(int)  # v3

    mixture_tree_embeddings = mixture_embedder.transform(all_mixture_molecular_features)
    # XXX
    # mixture_tree_embeddings = PCA(n_components=10, random_state=RSTATE).fit_transform(mixture_tree_embeddings)
    # mixture_tree_embeddings = normalize(mixture_tree_embeddings)

    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [
                ((a + b) == 0).reshape(1, -1),
                ((a + b) == 1).reshape(1, -1),
                ((a + b) == 2).reshape(1, -1),
            ],
            format="csr",
            dtype=np.int8,
        )

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        combine_mixture_pair,
        dir_submission=DIR_ROOT
        / "pred/tree_embeddings_molecules_and_mixtures_similarity",
    )
    # plot_validation_curve(
    #     model,
    #     mixture_tree_embeddings,
    #     combine_mixture_pair,
    #     out=DIR_ROOT / "pred/tree_embeddings_molecules_and_mixtures_smell/validation_curve.png",
    # )


def experiment_20():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    from tree_embedder import ForestEmbedder

    mixture_binary_composition = get_mixture_binary_composition()

    print("Loading molecule fingerprints...")
    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]
    molecule_features.index = molecule_features.index.astype(str)

    molecule_emb = pd.read_csv(DIR_ROOT / "data/chemprop_embeddings.csv", index_col=0)
    molecule_emb.index = molecule_emb.index.astype(str)

    print("Loading molecule smell data...")
    molecule_smell = pd.read_csv(DIR_ROOT / "data/percept_single.csv", index_col=0)
    # molecule_smell = molecule_smell.combine_first(pd.read_csv(DIR_ROOT / "data/perception_prediction_new.csv", index_col=0))
    # molecule_smell = pd.read_csv(DIR_ROOT / "data/perception_prediction.csv", index_col=0)
    molecule_smell = normalize(molecule_smell)
    molecule_smell.index = molecule_smell.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_emb,
    )

    mixture_similarity = pd.DataFrame(
        euclidean_distances(mixture_molecular_features),
        index=mixture_binary_composition.index,
        columns=mixture_binary_composition.index,
    )

    # mixture_similarity = PCA(n_components=50, random_state=RSTATE).fit_transform(mixture_similarity)  # second best
    # mixture_similarity = PCA(random_state=RSTATE).fit_transform(mixture_similarity)
    mixture_similarity = PCA(n_components=30, random_state=RSTATE).fit_transform(
        mixture_similarity
    )  # best
    mixture_similarity = normalize(mixture_similarity)  # Important.
    mixture_similarity_index = mixture_binary_composition.index

    # v3
    # Added a third step, also encoding morgan fingerprints with tree-embeddings
    molecule_embedder = ForestEmbedder(
        RandomForestRegressor(
            n_estimators=100,
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
        # node_weights="log_node_size",  # Will be considered in the impurity of mixture embedder
    )
    # FIXME
    molecules_common_index = molecule_features.index.intersection(molecule_smell.index)

    print("Fitting molecule embedder...")
    molecule_embedder.fit(
        molecule_features.loc[molecules_common_index],  # v3
        # normalize(PCA(n_components=50, random_state=RSTATE).fit_transform(cosine_similarity(molecule_smell.loc[molecules_common_index]))),  # v7
        molecule_smell.loc[molecules_common_index],  # v3
    )
    # ensure aligned indices
    molecule_tree_embeddings = molecule_embedder.transform(
        molecule_features.loc[mixture_binary_composition.columns]
    )

    # print("Fitting PCA to compress molecule embedders...")
    # # molecule_tree_embeddings = PCA(n_components=100, random_state=RSTATE).fit_transform(molecule_tree_embeddings)
    # # molecule_tree_embeddings = PCA(n_components=20, random_state=RSTATE).fit_transform(molecule_tree_embeddings)
    # # TODO: no normalization?
    # # molecule_tree_embeddings = normalize(molecule_tree_embeddings)

    print("Combining molecule embeddings into mixtures...")
    # ensure aligned indices
    mixture_molecular_features = (
        mixture_binary_composition.loc[mixture_similarity_index].values
        @ molecule_tree_embeddings
    )

    # cannot do inplace because of typing
    mixture_molecular_features = (
        mixture_molecular_features
        / mixture_binary_composition.loc[mixture_similarity_index].values.sum(
            axis=1, keepdims=True
        )
    )  # v4
    # mixture_molecular_features = (mixture_molecular_features > 0).astype(int)  #v3
    # mixture_molecular_features = PCA(n_components=20, random_state=RSTATE).fit_transform(mixture_molecular_features)
    # # TODO: no normalization?
    # mixture_molecular_features = normalize(mixture_molecular_features)

    # Self-supervised learning to generate tree embeddings
    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        # max_pvalue=0.05,
    )
    print("Fitting mixture embedder...")
    mixture_tree_embeddings = mixture_embedder.fit(
        mixture_molecular_features,
        mixture_similarity,
    )

    all_mixture_molecular_features = (
        mixture_binary_composition.values @ molecule_tree_embeddings
    )
    # cannot do inplace because of typing
    all_mixture_molecular_features = (
        all_mixture_molecular_features
        / mixture_binary_composition.values.sum(axis=1, keepdims=True)
    )  # v4
    # all_mixture_molecular_features = (all_mixture_molecular_features > 0).astype(int)  # v3

    mixture_tree_embeddings = mixture_embedder.transform(all_mixture_molecular_features)
    # XXX
    # mixture_tree_embeddings = PCA(n_components=10, random_state=RSTATE).fit_transform(mixture_tree_embeddings)
    # mixture_tree_embeddings = normalize(mixture_tree_embeddings)

    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    # TODO: mixture similarities

    def combine_mixture_pair(a, b):
        # return sparse.hstack(
        #     [
        #         np.abs(a - b).reshape(1, -1),
        #         (a * b).reshape(1, -1),
        #     ],
        #     format="csr",
        #     dtype=np.int8,
        # )
        # return np.abs(a - b).reshape(1, -1)  # v6
        return sparse.hstack(
            [
                ((a + b) == 0).reshape(1, -1),
                ((a + b) == 1).reshape(1, -1),
                ((a + b) == 2).reshape(1, -1),
            ],
            format="csr",
            dtype=np.int8,
        )

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )

    apply_model(
        model,
        mixture_tree_embeddings,
        # a bit worse v2
        # pd.Series(list(sparse.csr_array(mixture_molecular_features)), index=mixture_molecular_features.index),
        # pd.Series(list(sparse.csr_array(all_mixture_molecular_features)), index=mixture_binary_composition.index),  # v5,6
        combine_mixture_pair,
        dir_submission=DIR_ROOT / "pred/te_molecules_and_mixsim",
    )


def experiment_21():
    from sklearn.ensemble import RandomForestRegressor

    # mol_features_path = DIR_ROOT / "data/mixture_stats_Morgan_Fingerprint_radius2_fpSize2048.csv"
    mol_features_path = DIR_ROOT / "data/mixture_stats_chemprop_embeddings.csv"
    # mol_features_path = DIR_ROOT / "data/mixture_stats_perception_prediction_new.csv"

    mixture_features = pd.read_csv(
        mol_features_path,
        index_col=0,
        header=[0, 1],
    )

    def combine_mixture_pair(a, b):
        both = np.vstack((a.values, b.values))
        return np.hstack(
            [
                np.abs(a - b).values,
                (a * b).values,
                both.mean(axis=0),
                both.max(axis=0),
                both.min(axis=0),
            ],
        )

    model = RandomForestRegressor(
        # max_features="sqrt",
        max_features=.3,
        random_state=RSTATE,
        verbose=50,
    )

    apply_model(
        model,
        mixture_features,
        combine_mixture_pair,
        dir_submission=DIR_ROOT / "pred/mix_stats",
        sparse=False,
    )


def experiment_22():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.decomposition import PCA

    mol_features_path = DIR_ROOT / "data/chemprop_embeddings.csv"

    molecule_emb = pd.read_csv(mol_features_path, index_col=0)
    molecule_emb = molecule_emb.loc[:, molecule_emb.nunique() > 1]  # Just for Morgan
    molecule_emb.index = molecule_emb.index.astype(str)

    n_components = 20

    pca = pd.DataFrame(
        PCA(n_components=n_components).fit_transform(molecule_emb),
        index=molecule_emb.index,
        columns=[f"pc{i}" for i in range(n_components)],
    )

    molecule_emb = pd.concat([molecule_emb, pca], axis=1)

    mix_bin_comp = get_mixture_binary_composition()
    molecule_emb = molecule_emb.loc[mix_bin_comp.columns]
    molecule_emb = (
        (molecule_emb - molecule_emb.min())
        / (molecule_emb.max() - molecule_emb.min())
    )

    mix_matrices = [
        molecule_emb.loc[row.astype(bool)]
        for _, row in tqdm(mix_bin_comp.iterrows(), total=len(mix_bin_comp))
    ]
    mix_matrices = pd.Series(
        mix_matrices,
        index=mix_bin_comp.index,
    )

    def combine_mixture_pair(a, b):
        # return stats.mannwhitneyu(a, b).statistic
        return stats.mannwhitneyu(a, b).pvalue

    model = RandomForestRegressor(
        max_features="sqrt",
        random_state=RSTATE,
        verbose=50,
    )

    apply_model(
        model,
        mix_matrices,
        combine_mixture_pair,
        dir_submission=DIR_ROOT / "pred/mix_mannwhitneyu",
        use_sparse=False,
    )


def experiment_23():  # combines 22 and 21
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    )
    from sklearn.decomposition import PCA
    from xgboost import XGBRegressor

    mol_features_path = DIR_ROOT / "data/chemprop_embeddings.csv"

    molecule_emb = pd.read_csv(mol_features_path, index_col=0)
    molecule_emb = molecule_emb.loc[:, molecule_emb.nunique() > 1]  # Just for Morgan
    molecule_emb.index = molecule_emb.index.astype(str)

    n_components = 10

    pca = pd.DataFrame(
        PCA(n_components=n_components).fit_transform(molecule_emb),
        index=molecule_emb.index,
        columns=[f"pc{i}" for i in range(n_components)],
    )

    molecule_emb = pd.concat([molecule_emb, pca], axis=1)

    mix_bin_comp = get_mixture_binary_composition()
    molecule_emb = molecule_emb.loc[mix_bin_comp.columns]
    molecule_emb = (
        (molecule_emb - molecule_emb.min())
        / (molecule_emb.max() - molecule_emb.min())
    )

    mix_matrices = [
        molecule_emb.loc[row.astype(bool)]
        for _, row in tqdm(mix_bin_comp.iterrows(), total=len(mix_bin_comp))
    ]
    mix_matrices = pd.Series(
        mix_matrices,
        index=mix_bin_comp.index,
    ).rename(("mat", "mat2"))

    # mol_features_path = DIR_ROOT / "data/mixture_stats_Morgan_Fingerprint_radius2_fpSize2048.csv"
    mix_features_path = DIR_ROOT / "data/mixture_stats_chemprop_embeddings.csv"
    # mol_features_path = DIR_ROOT / "data/mixture_stats_perception_prediction_new.csv"

    mixture_features = pd.read_csv(
        mix_features_path,
        index_col=0,
        header=[0, 1],
    )

    mixture_features = pd.concat({"stats": mixture_features, "matrix": mix_matrices}, axis=1)
    mixture_features = mixture_features.fillna(0)

    def combine_mixture_pair(a, b):
        mat_a, mat_b = a["matrix"].iloc[0].astype(np.float32), b["matrix"].iloc[0].astype(np.float32)
        stats_a, stats_b = a["stats"].values.astype(np.float32), b["stats"].values.astype(np.float32)
        u = stats.mannwhitneyu(mat_a, mat_b).statistic
        both_stats = np.vstack((stats_a, stats_b))

        result = np.hstack(
            [
                np.abs(stats_a - stats_b),
                (stats_a * stats_b),
                both_stats.mean(axis=0),
                both_stats.max(axis=0),
                both_stats.min(axis=0),
                u,
            ],
        )
        result = result.astype(np.float32)
        return result

    # model = RandomForestRegressor(
    #     # max_features="sqrt",
    #     max_features=.3,
    #     random_state=RSTATE,
    #     verbose=50,
    # )
    # model = GradientBoostingRegressor(
    #     n_estimators=1000,
    #     # max_features="sqrt",
    #     max_features=.3,
    #     min_impurity_decrease=1e-5,
    #     # subsample=0.5,
    #     max_depth=3,
    #     random_state=RSTATE,
    #     verbose=50
    # )
    # model = ExtraTreesRegressor(
    #     n_estimators=1000,
    #     random_state=RSTATE,
    #     verbose=50
    # )
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.1,
        random_state=RSTATE,
        n_jobs=10,
        verbosity=3,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
    )

    apply_model(
        model,
        mixture_features,
        combine_mixture_pair,
        dir_submission=DIR_ROOT / "pred/mix_stats_mannwhitneyu",
        sparse=False,
    )
def experiment_21_oof():
    from sklearn.ensemble import RandomForestRegressor

    mol_features_path = DIR_ROOT / "data/mixture_stats_chemprop_embeddings.csv"
    mixture_features = pd.read_csv(
        mol_features_path,
        index_col=0,
        header=[0, 1],
    )

    def combine_mixture_pair(a, b):
        both = np.vstack((a.values, b.values))
        return np.hstack(
            [
                np.abs(a - b).values,
                (a * b).values,
                both.mean(axis=0),
                both.max(axis=0),
                both.min(axis=0),
            ],
        )

    X_train, y_train = get_embedded_train_set(
        mixture_features,
        combine_mixture_pair,
        use_sparse=False,
    )

    n_samples = X_train.shape[0]
    oof_pred = np.zeros(n_samples)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RSTATE)
    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr = y_train[train_idx]

        model = RandomForestRegressor(
            max_features=.3,
            random_state=RSTATE,
            verbose=50,
            n_estimators=FINAL_N_ESTIMATORS,
        )
        model.fit(X_tr, y_tr)
        oof_pred[valid_idx] = model.predict(X_val)

    out_df = pd.DataFrame({
        "oof_pred": oof_pred,
        "true": y_train
    })
    out_df.to_csv(DIR_ROOT / "pred/belfaction_oof_pred_exp21.csv", index=False)
    print("Saved oof prediction for exp21!")


def experiment_23_oof_feat():
    from xgboost import XGBRegressor
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from scipy import stats

    mol_features_path = DIR_ROOT / "data/chemprop_embeddings.csv"
    molecule_emb = pd.read_csv(mol_features_path, index_col=0)
    molecule_emb = molecule_emb.loc[:, molecule_emb.nunique() > 1]
    molecule_emb.index = molecule_emb.index.astype(str)
    n_components = 10
    pca = pd.DataFrame(
        PCA(n_components=n_components).fit_transform(molecule_emb),
        index=molecule_emb.index,
        columns=[f"pc{i}" for i in range(n_components)],
    )
    molecule_emb = pd.concat([molecule_emb, pca], axis=1)

    mix_bin_comp = get_mixture_binary_composition()
    molecule_emb = molecule_emb.loc[mix_bin_comp.columns]
    molecule_emb = (molecule_emb - molecule_emb.min()) / (molecule_emb.max() - molecule_emb.min())

    mix_matrices = [
        molecule_emb.loc[row.index[row.astype(bool)]]
        for _, row in tqdm(mix_bin_comp.iterrows(), total=len(mix_bin_comp))
    ]
    mix_matrices = pd.Series(mix_matrices, index=mix_bin_comp.index).rename(("mat", "mat2"))

    mix_features_path = DIR_ROOT / "data/mixture_stats_chemprop_embeddings.csv"
    mixture_features = pd.read_csv(mix_features_path, index_col=0, header=[0, 1])
    mixture_features = pd.concat({"stats": mixture_features, "matrix": mix_matrices}, axis=1).fillna(0)

    def combine_mixture_pair(a, b):
        mat_a = a["matrix"].iloc[0]
        mat_b = b["matrix"].iloc[0]
        stats_a = a["stats"].values.astype(np.float32)
        stats_b = b["stats"].values.astype(np.float32)

        u = stats.mannwhitneyu(mat_a, mat_b).statistic
        both_stats = np.vstack((stats_a, stats_b))

        result = np.hstack([
            np.abs(stats_a - stats_b),
            stats_a * stats_b,
            both_stats.mean(axis=0),
            both_stats.max(axis=0),
            both_stats.min(axis=0),
            u,
        ])
        return result.astype(np.float32)

    X_train, y_train = get_embedded_train_set(
        mixture_features,
        combine_mixture_pair,
        use_sparse=False,
    )
    n_samples = X_train.shape[0]
    oof_pred = np.zeros(n_samples)
    n_folds = 10
    all_folds_preds = np.zeros((n_samples, n_folds))

    kf = KFold(n_splits=10, shuffle=True, random_state=RSTATE)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr = y_train[train_idx]

        model = XGBRegressor(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.1,
            random_state=RSTATE,
            n_jobs=10,
            verbosity=0,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        oof_pred[valid_idx] = y_pred
        all_folds_preds[:, fold] = model.predict(X_train)

    df_X_train = pd.DataFrame(
        X_train,
        columns=[f"exp23_feat_{i}" for i in range(X_train.shape[1])]
    )
    df_X_train["exp23_oof"] = oof_pred
    df_X_train["target"]     = y_train

    df_X_train.to_csv(DIR_ROOT / "pred/belfaction_feats_and_oof_exp23.csv", index=False)
    print("belfaction_feats_and_oof_exp23.csv is saved.")

    df_oof_all_folds = pd.DataFrame(all_folds_preds)
    df_oof_all_folds.to_csv(DIR_ROOT / "pred/belfaction_train_oof_all_folds_exp23.csv", index=False)

def experiment_23_leaderboard_feat():
    from sklearn.decomposition import PCA
    from scipy import stats
    import os

    mol_features_path   = DIR_ROOT / "data/chemprop_embeddings.csv"
    mix_features_path   = DIR_ROOT / "data/mixture_stats_chemprop_embeddings.csv"
    train_cols_file     = DIR_ROOT / "pred/train_molecule_columns.txt"
    pred_dir            = DIR_ROOT / "pred"
    os.makedirs(pred_dir, exist_ok=True)

    molecule_emb = pd.read_csv(mol_features_path, index_col=0)
    molecule_emb = molecule_emb.loc[:, molecule_emb.nunique() > 1]
    molecule_emb.index = molecule_emb.index.astype(str)
    n_components = 10
    pca = pd.DataFrame(
        PCA(n_components=n_components).fit_transform(molecule_emb),
        index=molecule_emb.index,
        columns=[f"pc{i}" for i in range(n_components)],
    )
    molecule_emb = pd.concat([molecule_emb, pca], axis=1)

    mix_bin_comp = get_mixture_binary_composition()

    mix_matrices = [
        molecule_emb.loc[row.index[row.astype(bool)]]
        for _, row in tqdm(mix_bin_comp.iterrows(), total=len(mix_bin_comp))
    ]
    mix_matrices = pd.Series(mix_matrices, index=mix_bin_comp.index)

    mix_matrices = mix_matrices.rename(("matrix", "matrix"))

    mixture_stats_df = pd.read_csv(
        mix_features_path,
        index_col=0,
        header=[0, 1],
    )

    mixture_features = pd.concat(
        {"stats": mixture_stats_df, "matrix": mix_matrices},
        axis=1,
    ).fillna(0)

    def combine_mixture_pair(a, b):
        mat_a = a["matrix"].iloc[0]
        mat_b = b["matrix"].iloc[0]
        stats_a = a["stats"].values.astype(np.float32)
        stats_b = b["stats"].values.astype(np.float32)

        u = stats.mannwhitneyu(mat_a, mat_b).statistic
        both = np.vstack((stats_a, stats_b))

        return np.hstack([
            np.abs(stats_a - stats_b),
            stats_a * stats_b,
            both.mean(axis=0),
            both.max(axis=0),
            both.min(axis=0),
            u,
        ])

    X_lb, y_lb = get_embedded_leaderboard_set(
        mixture_features,
        combine_mixture_pair,
        use_sparse=False,
    )
    feat_len = X_lb.shape[1]
    print(feat_len)

    df_X_lb = pd.DataFrame(
        X_lb,
        columns=[f"exp23_feat_{i}" for i in range(feat_len)]
    )

    df_X_lb.to_csv(pred_dir / "belfaction_feats_exp23_leaderboard.csv", index=False)
    print("belfaction_feats_exp23_leaderboard.csv is saved.")


def experiment_5_oof():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder


    mixture_binary_composition = get_mixture_binary_composition()

    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features = molecule_features.loc[:, molecule_features.nunique() > 1]
    molecule_features.index = molecule_features.index.astype(str)

    molecule_embedder = ForestEmbedder(
        RandomForestRegressor(
            n_estimators=100,
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
        max_pvalue=0.05,
        node_weights="log_node_size",
    )
    molecule_tree_embeddings = molecule_embedder.fit_transform(
        molecule_features,
        molecule_features,
    )
    molecule_tree_embeddings = PCA(n_components=100, random_state=RSTATE).fit_transform(
        molecule_tree_embeddings
    )
    molecule_tree_embeddings = pd.DataFrame(
        molecule_tree_embeddings,
        index=molecule_features.index,
    )

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_tree_embeddings,
    )

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )

    try:
        mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    except AttributeError:
        mixture_tree_embeddings = sparse.csr_matrix(mixture_tree_embeddings)

    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )

    # oof
    X_train, y_train = get_embedded_train_set(
        mixture_tree_embeddings,
        combine_mixture_pair,
        use_sparse=True,
    )


    if hasattr(X_train, "tocsc") or hasattr(X_train, "tocsr"):
        X_train = X_train.tocsr()

    n_samples = X_train.shape[0]
    oof_pred = np.zeros(n_samples)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RSTATE)
    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr = y_train[train_idx]

        model = RandomForestRegressor(
            n_estimators=1000,
            max_features="sqrt",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        )
        model.fit(X_tr, y_tr)
        oof_pred[valid_idx] = model.predict(X_val)

    out_df = pd.DataFrame({
        "oof_pred": oof_pred,
        "true": y_train
    })
    out_df.to_csv(DIR_ROOT / "pred/belfaction_oof_pred_exp5.csv", index=False)
    print("Saved oof prediction for exp5!")


def experiment_1_oof():
    from sklearn.ensemble import RandomForestRegressor
    from tree_embedder import ForestEmbedder

    N_SPLITS = 10

    mixture_binary_composition = get_mixture_binary_composition()

    molecule_features = pd.read_csv(
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv", index_col=0
    )
    molecule_features.index = molecule_features.index.astype(str)

    mixture_molecular_features = generate_mixture_molecular_features(
        mixture_binary_composition,
        molecule_features,
    )

    mixture_embedder = ForestEmbedder(
        RandomForestRegressor(
            max_features="log2",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        ),
    )
    mixture_tree_embeddings = mixture_embedder.fit_transform(
        mixture_binary_composition,
        mixture_molecular_features,
    )
    mixture_tree_embeddings = sparse.csr_array(mixture_tree_embeddings)
    mixture_tree_embeddings = pd.Series(
        list(mixture_tree_embeddings),
        index=mixture_binary_composition.index,
    )

    def combine_mixture_pair(a, b):
        return sparse.hstack(
            [np.abs(a - b).reshape(1, -1), (a * b).reshape(1, -1)], format="csr"
        )

    # oof
    X_train, y_train = get_embedded_train_set(
        mixture_tree_embeddings,
        combine_mixture_pair,
        use_sparse=True,
    )
    if hasattr(X_train, "tocsr"):
        X_train = X_train.tocsr()

    n_samples = X_train.shape[0]
    oof_pred = np.zeros(n_samples)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RSTATE)

    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr = y_train[train_idx]

        model = RandomForestRegressor(
            n_estimators=1000,
            max_features="sqrt",
            random_state=RSTATE,
            n_jobs=10,
            verbose=50,
        )
        model.fit(X_tr, y_tr)
        oof_pred[valid_idx] = model.predict(X_val)

    out_df = pd.DataFrame({
        "oof_pred": oof_pred,
        "true": y_train
    })
    out_path = DIR_ROOT / "pred/belfaction_oof_pred_exp1.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved OOF predictions for exp1 to {out_path}")


def main():
    # experiment_1()
    # experiment_2()
    # experiment_3()
    # experiment_4()
    # experiment_5()
    # experiment_6()
    # experiment_7()
    # experiment_8()
    # experiment_9()
    # experiment_10()
    # experiment_11()
    # experiment_12()
    # experiment_13()
    # experiment_14()
    # experiment_15()
    # experiment_16()
    # experiment_17()
    # experiment_18()
    # experiment_19()
    # experiment_20()
    # experiment_21()
    # experiment_22()
    # experiment_23()
    # experiment_21_oof()
    experiment_23_oof_feat()
    # experiment_23_leaderboard_feat()
    # experiment_5_oof()
    # experiment_1_oof()


if __name__ == "__main__":
    main()
