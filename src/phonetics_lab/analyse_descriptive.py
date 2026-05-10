from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--fig", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    group_cols = ["phoneme", "l1_status", "gender"]
    table = (
        df.groupby(group_cols, dropna=False)
        .agg(
            F1_mean=("F1_lobanov", "mean"),
            F1_median=("F1_lobanov", "median"),
            F1_sd=("F1_lobanov", "std"),
            F2_mean=("F2_lobanov", "mean"),
            F2_median=("F2_lobanov", "median"),
            F2_sd=("F2_lobanov", "std"),
            n=("phoneme", "size"),
        )
        .reset_index()
    )
    table["F1_cv"] = table["F1_sd"] / table["F1_mean"].abs()
    table["F2_cv"] = table["F2_sd"] / table["F2_mean"].abs()

    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False)

    centroids = df.groupby("phoneme").agg(F1=("F1_lobanov", "mean"), F2=("F2_lobanov", "mean")).dropna()
    plt.figure(figsize=(8, 6))
    plt.scatter(centroids["F2"], centroids["F1"])
    for phoneme, row in centroids.iterrows():
        plt.text(row["F2"], row["F1"], str(phoneme))
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel("F2 Lobanov")
    plt.ylabel("F1 Lobanov")
    plt.title("Vowel chart: phoneme centroids")
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.fig, dpi=200)


if __name__ == "__main__":
    main()
