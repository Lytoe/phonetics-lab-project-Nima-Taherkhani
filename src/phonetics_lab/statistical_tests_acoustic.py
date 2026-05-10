# File: src/phonetics_lab/statistical_tests_acoustic.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def safe_shapiro(values: pd.Series) -> float:
    values = values.dropna()

    if len(values) < 3:
        return np.nan

    # Shapiro becomes overly sensitive with huge samples.
    if len(values) > 500:
        values = values.sample(500, random_state=42)

    try:
        return float(shapiro(values).pvalue)
    except Exception:
        return np.nan


def run_l1_l2_test(df: pd.DataFrame, vowel: str, feature: str) -> dict:
    subset = df[
        (df["main_oral_vowel"].astype(bool))
        & (df["phoneme"] == vowel)
    ].copy()

    l1 = subset.loc[subset["l1_status"] == "L1", feature].dropna()
    l2 = subset.loc[subset["l1_status"] == "L2", feature].dropna()

    result = {
        "phoneme": vowel,
        "feature": feature,
        "n_l1": len(l1),
        "n_l2": len(l2),
        "mean_l1": l1.mean(),
        "mean_l2": l2.mean(),
        "difference_l2_minus_l1": l2.mean() - l1.mean(),
        "shapiro_p_l1": safe_shapiro(l1),
        "shapiro_p_l2": safe_shapiro(l2),
        "levene_p": np.nan,
        "test_used": None,
        "statistic": np.nan,
        "p_value": np.nan,
    }

    if len(l1) < 3 or len(l2) < 3:
        result["test_used"] = "insufficient_data"
        return result

    try:
        result["levene_p"] = float(levene(l1, l2).pvalue)
    except Exception:
        result["levene_p"] = np.nan

    normal_enough = (
        pd.notna(result["shapiro_p_l1"])
        and pd.notna(result["shapiro_p_l2"])
        and result["shapiro_p_l1"] > 0.05
        and result["shapiro_p_l2"] > 0.05
    )

    equal_var = pd.notna(result["levene_p"]) and result["levene_p"] > 0.05

    if normal_enough:
        test = ttest_ind(l1, l2, equal_var=equal_var)
        result["test_used"] = "t_test_equal_var" if equal_var else "welch_t_test"
        result["statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)
    else:
        test = mannwhitneyu(l1, l2, alternative="two-sided")
        result["test_used"] = "mann_whitney_u"
        result["statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)

    return result


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()

    valid = results["p_value"].notna()

    results["p_fdr"] = np.nan
    results["significant_fdr_0_05"] = False

    if valid.any():
        rejected, pvals_corrected, _, _ = multipletests(
            results.loc[valid, "p_value"],
            alpha=0.05,
            method="fdr_bh",
        )

        results.loc[valid, "p_fdr"] = pvals_corrected
        results.loc[valid, "significant_fdr_0_05"] = rejected

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/tables/acoustic_l1_l2_tests.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    vowels = sorted(df.loc[df["main_oral_vowel"].astype(bool), "phoneme"].unique())
    features = ["F1_lobanov", "F2_lobanov"]

    rows = []

    for vowel in vowels:
        for feature in features:
            rows.append(run_l1_l2_test(df, vowel, feature))

    results = pd.DataFrame(rows)
    results = add_fdr(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, encoding="utf-8")

    print("\n=== ACOUSTIC L1/L2 TESTS ===")
    print(f"Saved: {args.out}")

    print("\nSignificant after FDR:")
    sig = results[results["significant_fdr_0_05"]]
    if sig.empty:
        print("None")
    else:
        print(
            sig[
                [
                    "phoneme",
                    "feature",
                    "test_used",
                    "difference_l2_minus_l1",
                    "p_value",
                    "p_fdr",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()