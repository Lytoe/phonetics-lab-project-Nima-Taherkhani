# File: src/phonetics_lab/acoustic_distances.py

from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import pinv


SELECTED_PAIRS = [
    ("e", "ɛ"),
    ("y", "u"),
    ("ø", "e"),
    ("ɑ", "a"),
]


def get_main_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["main_oral_vowel"].astype(bool)].copy()


def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("phoneme")
        .agg(
            n=("token_id", "count"),
            F1=("F1_lobanov", "mean"),
            F2=("F2_lobanov", "mean"),
        )
        .reset_index()
        .sort_values("phoneme")
    )


def compute_pooled_covariance(df: pd.DataFrame) -> np.ndarray:
    values = df[["F1_lobanov", "F2_lobanov"]].dropna().values
    return np.cov(values.T)


def compute_distance_table(df: pd.DataFrame) -> pd.DataFrame:
    centroids = compute_centroids(df)
    cov = compute_pooled_covariance(df)
    inv_cov = pinv(cov)

    rows = []

    for _, row1 in centroids.iterrows():
        for _, row2 in centroids.iterrows():
            p = row1["phoneme"]
            q = row2["phoneme"]

            v1 = np.array([row1["F1"], row1["F2"]])
            v2 = np.array([row2["F1"], row2["F2"]])

            euclidean = float(np.linalg.norm(v1 - v2))
            maha = float(mahalanobis(v1, v2, inv_cov))

            rows.append(
                {
                    "phoneme_1": p,
                    "phoneme_2": q,
                    "euclidean_distance": euclidean,
                    "mahalanobis_distance": maha,
                }
            )

    return pd.DataFrame(rows)


def pair_distance(df: pd.DataFrame, p: str, q: str) -> dict:
    subset = df[df["phoneme"].isin([p, q])].copy()

    if subset["phoneme"].nunique() < 2:
        return {
            "phoneme_1": p,
            "phoneme_2": q,
            "euclidean_distance": np.nan,
            "mahalanobis_distance": np.nan,
        }

    centroids = compute_centroids(subset)
    cov = compute_pooled_covariance(subset)
    inv_cov = pinv(cov)

    c1 = centroids[centroids["phoneme"] == p][["F1", "F2"]].iloc[0].values
    c2 = centroids[centroids["phoneme"] == q][["F1", "F2"]].iloc[0].values

    return {
        "phoneme_1": p,
        "phoneme_2": q,
        "euclidean_distance": float(np.linalg.norm(c1 - c2)),
        "mahalanobis_distance": float(mahalanobis(c1, c2, inv_cov)),
    }


def bootstrap_pair_ci(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    speakers = sorted(df["speaker_id"].unique())

    rows = []

    for p, q in pairs:
        observed = pair_distance(df, p, q)

        boot_euclidean = []
        boot_mahalanobis = []

        pair_df = df[df["phoneme"].isin([p, q])].copy()

        if pair_df.empty:
            continue

        for _ in range(n_bootstrap):
            sampled_speakers = rng.choice(speakers, size=len(speakers), replace=True)
            boot = pd.concat(
                [pair_df[pair_df["speaker_id"] == speaker] for speaker in sampled_speakers],
                ignore_index=True,
            )

            if boot["phoneme"].nunique() < 2:
                continue

            dist = pair_distance(boot, p, q)

            if not np.isnan(dist["euclidean_distance"]):
                boot_euclidean.append(dist["euclidean_distance"])

            if not np.isnan(dist["mahalanobis_distance"]):
                boot_mahalanobis.append(dist["mahalanobis_distance"])

        rows.append(
            {
                "phoneme_1": p,
                "phoneme_2": q,
                "observed_euclidean": observed["euclidean_distance"],
                "euclidean_ci_low": np.percentile(boot_euclidean, 2.5) if boot_euclidean else np.nan,
                "euclidean_ci_high": np.percentile(boot_euclidean, 97.5) if boot_euclidean else np.nan,
                "observed_mahalanobis": observed["mahalanobis_distance"],
                "mahalanobis_ci_low": np.percentile(boot_mahalanobis, 2.5) if boot_mahalanobis else np.nan,
                "mahalanobis_ci_high": np.percentile(boot_mahalanobis, 97.5) if boot_mahalanobis else np.nan,
                "n_bootstrap_successful": len(boot_euclidean),
            }
        )

    return pd.DataFrame(rows)


def write_square_matrix(distance_table: pd.DataFrame, value_col: str, out_path: Path) -> None:
    matrix = distance_table.pivot(
        index="phoneme_1",
        columns="phoneme_2",
        values=value_col,
    )

    matrix.to_csv(out_path, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    main_df = get_main_data(df)

    args.tables_dir.mkdir(parents=True, exist_ok=True)

    centroids = compute_centroids(main_df)
    centroids.to_csv(args.tables_dir / "acoustic_vowel_centroids.csv", index=False)

    distances = compute_distance_table(main_df)
    distances.to_csv(args.tables_dir / "acoustic_distance_table.csv", index=False)

    write_square_matrix(
        distances,
        "euclidean_distance",
        args.tables_dir / "Dac_euclidean.csv",
    )

    write_square_matrix(
        distances,
        "mahalanobis_distance",
        args.tables_dir / "Dac_mahalanobis.csv",
    )

    bootstrap = bootstrap_pair_ci(
        main_df,
        pairs=SELECTED_PAIRS,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )

    bootstrap.to_csv(
        args.tables_dir / "acoustic_distance_bootstrap_ci.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n=== ACOUSTIC DISTANCE MATRICES ===")
    print(f"Main vowel rows: {len(main_df)}")
    print(f"Centroids saved: {args.tables_dir / 'acoustic_vowel_centroids.csv'}")
    print(f"Distance table saved: {args.tables_dir / 'acoustic_distance_table.csv'}")
    print(f"Euclidean matrix saved: {args.tables_dir / 'Dac_euclidean.csv'}")
    print(f"Mahalanobis matrix saved: {args.tables_dir / 'Dac_mahalanobis.csv'}")
    print(f"Bootstrap CI saved: {args.tables_dir / 'acoustic_distance_bootstrap_ci.csv'}")

    print("\nSelected bootstrap CIs:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()