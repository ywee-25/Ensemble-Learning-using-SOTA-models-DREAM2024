import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
from tqdm import tqdm
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from run_utils import get_mixture_binary_composition

DIR_ROOT = Path(__file__).parents[2].resolve()


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
            "expmean": molecule_matrix.apply(lambda x: np.exp(x).mean()),
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


def main():
    molecular_feature_files = [
        DIR_ROOT / "data/chemprop_embeddings.csv",
        DIR_ROOT / "data/embeddings_molecule_filled.csv",
        DIR_ROOT / "data/perception_prediction_new.csv",
        DIR_ROOT / "data/Morgan_Fingerprint_radius2_fpSize2048.csv",
    ]
    mixture_binary_composition = get_mixture_binary_composition()

    for mol_features_path in molecular_feature_files:
        outpath = DIR_ROOT / f"data/mixture_stats_{mol_features_path.stem}.csv"

        if outpath.exists():
            print(f"Skipping {mol_features_path} as {outpath} already exists...")
            continue

        print(f"Processing {mol_features_path}...")
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

        try:
            mix_features = gen_complex_mix_features(
                mixture_binary_composition,
                molecule_emb,
            )
        except Exception as e:
            print(f"Failed to process {mol_features_path}: {e}")
            continue

        mix_features.to_csv(outpath)


if __name__ == "__main__":
    main()
