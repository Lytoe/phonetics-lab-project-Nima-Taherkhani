# File: src/phonetics_lab/gender_tests_acoustic.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def run_gender_test(df: pd.DataFrame, vowel: str, feature: str) -> dict:
    subset = df[
        (df["main_oral_vowel"].astype(bool))
        & (df["phoneme"] == vowel)
    ].copy()

    # Speaker-level means: this avoids pretending every token is independent.
    speaker_means = (
        subset.groupby(["speaker_id", "gender"])[feature]
        .mean()
        .reset_index()
        .dropna()
    )

    female = speaker_means.loc[speaker_means["gender"] == "f", feature]
    male = speaker_means.loc[speaker_means["gender"] == "m", feature]

    result = {
        "phoneme": vowel,
        "feature": feature,
        "n_female_speakers": len(female),
        "n_male_speakers": len(male),
        "mean_female": female.mean(),
        "mean_male": male.mean(),
        "difference_male_minus_female": male.mean() - female.mean(),
        "test_used": "mann_whitney_u_speaker_level",
        "statistic": np.nan,
        "p_value": np.nan,
        "rank_biserial_male_minus_female": np.nan,
    }

    if len(female) < 2 or len(male) < 2:
        result["test_used"] = "insufficient_data"
        return result

    test = mannwhitneyu(female, male, alternative="two-sided")
    result["statistic"] = float(test.statistic)
    result["p_value"] = float(test.pvalue)

    # U is for female. Positive after negation means male tends to be higher.
    r_female_minus_male = (2 * test.statistic) / (len(female) * len(male)) - 1
    result["rank_biserial_male_minus_female"] = float(-r_female_minus_male)

    return result


def add_fdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    valid = df["p_value"].notna()

    df["p_fdr"] = np.nan
    df["significant_fdr_0_05"] = False

    if valid.any():
        rejected, corrected, _, _ = multipletests(
            df.loc[valid, "p_value"],
            alpha=0.05,
            method="fdr_bh",
        )
        df.loc[valid, "p_fdr"] = corrected
        df.loc[valid, "significant_fdr_0_05"] = rejected

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/tables/acoustic_gender_tests.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    vowels = sorted(df.loc[df["main_oral_vowel"].astype(bool), "phoneme"].unique())
    features = ["F1_lobanov", "F2_lobanov"]

    rows = []

    for vowel in vowels:
        for feature in features:
            rows.append(run_gender_test(df, vowel, feature))

    results = pd.DataFrame(rows)
    results = add_fdr(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, encoding="utf-8")

    print("\n=== RESIDUAL GENDER TESTS AFTER LOBANOV ===")
    print(f"Saved: {args.out}")

    sig = results[results["significant_fdr_0_05"]]

    print("\nSignificant after FDR:")
    if sig.empty:
        print("None")
    else:
        print(
            sig[
                [
                    "phoneme",
                    "feature",
                    "difference_male_minus_female",
                    "rank_biserial_male_minus_female",
                    "p_value",
                    "p_fdr",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()