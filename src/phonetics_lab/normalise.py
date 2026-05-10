# File: src/phonetics_lab/normalise.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FORMANT_COLUMNS = ["F1_mid", "F2_mid", "F3_mid"]


def lobanov_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lobanov normalization:
    F* = (F - speaker_mean) / speaker_sd

    Means and SDs are computed speaker-by-speaker using valid oral vowel tokens.
    """
    df = df.copy()

    valid = df["formant_valid"].astype(bool) & df["is_oral_vowel"].astype(bool)

    for formant_col in FORMANT_COLUMNS:
        norm_col = formant_col.replace("_mid", "_lobanov")

        df[norm_col] = np.nan

        stats = (
            df.loc[valid]
            .groupby("speaker_id")[formant_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{formant_col}_mean", "std": f"{formant_col}_std"})
        )

        df = df.merge(stats, left_on="speaker_id", right_index=True, how="left")

        mean_col = f"{formant_col}_mean"
        std_col = f"{formant_col}_std"

        df.loc[valid, norm_col] = (
            df.loc[valid, formant_col] - df.loc[valid, mean_col]
        ) / df.loc[valid, std_col]

        df.drop(columns=[mean_col, std_col], inplace=True)

    return df


def add_analysis_flags(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    df = df.copy()

    counts = df.groupby("phoneme")["token_id"].count().rename("phoneme_count")
    df = df.merge(counts, on="phoneme", how="left")

    df["enough_tokens_for_stats"] = df["phoneme_count"] >= min_count

    df["main_oral_vowel"] = (
        df["is_oral_vowel"].astype(bool)
        & df["formant_valid"].astype(bool)
        & df["enough_tokens_for_stats"].astype(bool)
    )

    return df


def write_summary_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    missing_report = (
        df.groupby(["phoneme", "l1_status", "gender"])
        .agg(
            n=("token_id", "count"),
            invalid_formants=("formant_valid", lambda x: int((~x.astype(bool)).sum())),
            invalid_rate=("formant_valid", lambda x: float((~x.astype(bool)).mean())),
            f0_missing_rate=("f0_mean", lambda x: float(x.isna().mean())),
        )
        .reset_index()
        .sort_values(["phoneme", "l1_status", "gender"])
    )

    missing_report.to_csv(tables_dir / "acoustic_missing_report_after_norm.csv", index=False)

    descriptive = (
        df[df["main_oral_vowel"]]
        .groupby(["phoneme", "l1_status", "gender"])
        .agg(
            n=("token_id", "count"),
            F1_mean=("F1_lobanov", "mean"),
            F1_median=("F1_lobanov", "median"),
            F1_sd=("F1_lobanov", "std"),
            F1_iqr=("F1_lobanov", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            F2_mean=("F2_lobanov", "mean"),
            F2_median=("F2_lobanov", "median"),
            F2_sd=("F2_lobanov", "std"),
            F2_iqr=("F2_lobanov", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .reset_index()
    )

    descriptive["F1_cv"] = descriptive["F1_sd"] / descriptive["F1_mean"].abs()
    descriptive["F2_cv"] = descriptive["F2_sd"] / descriptive["F2_mean"].abs()

    descriptive.to_csv(tables_dir / "descriptive_acoustic_lobanov.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    norm = lobanov_normalise(df)
    norm = add_analysis_flags(norm, min_count=args.min_count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(args.out, index=False, encoding="utf-8")

    write_summary_tables(norm, args.tables_dir)

    print("\n=== LOBANOV NORMALISATION SUMMARY ===")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(norm)}")
    print(f"Output: {args.out}")

    print("\nMain analysis vowels:")
    print(
        norm[norm["main_oral_vowel"]]
        .groupby("phoneme")
        .size()
        .sort_values(ascending=False)
    )

    print("\nExcluded because too few tokens:")
    print(
        norm[norm["is_oral_vowel"].astype(bool)]
        .groupby("phoneme")
        .size()
        .loc[lambda x: x < args.min_count]
        .sort_values()
    )


if __name__ == "__main__":
    main()