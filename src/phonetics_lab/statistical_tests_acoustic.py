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

    # Shapiro becomes too sensitive with large samples.
    if len(values) > 500:
        values = values.sample(500, random_state=42)

    try:
        return float(shapiro(values).pvalue)
    except Exception:
        return np.nan


def cohen_d_l2_minus_l1(l1: pd.Series, l2: pd.Series) -> float:
    """
    Cohen's d using pooled SD.
    Positive value = L2 mean is higher than L1 mean.
    """
    l1 = l1.dropna()
    l2 = l2.dropna()

    n1 = len(l1)
    n2 = len(l2)

    if n1 < 2 or n2 < 2:
        return np.nan

    sd1 = l1.std(ddof=1)
    sd2 = l2.std(ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

    if pooled_sd == 0 or pd.isna(pooled_sd):
        return np.nan

    return float((l2.mean() - l1.mean()) / pooled_sd)


def rank_biserial_l2_minus_l1(u_statistic_for_l1: float, n_l1: int, n_l2: int) -> float:
    """
    Rank-biserial correlation for Mann-Whitney U.

    scipy.stats.mannwhitneyu(l1, l2) returns U for l1.
    2U/(n1*n2)-1 is positive when L1 tends to be larger.
    We negate it so positive = L2 tends to be larger.
    """
    if n_l1 == 0 or n_l2 == 0:
        return np.nan

    r_l1_minus_l2 = (2 * u_statistic_for_l1) / (n_l1 * n_l2) - 1
    return float(-r_l1_minus_l2)


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
        "cohen_d_l2_minus_l1": cohen_d_l2_minus_l1(l1, l2),
        "rank_biserial_l2_minus_l1": np.nan,
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
        result["rank_biserial_l2_minus_l1"] = rank_biserial_l2_minus_l1(
            u_statistic_for_l1=float(test.statistic),
            n_l1=len(l1),
            n_l2=len(l2),
        )

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


def add_effect_size_interpretation(results: pd.DataFrame) -> pd.DataFrame:
    """
    Rough labels only. Do not overclaim them.
    For Cohen's d:
      0.2 small, 0.5 medium, 0.8 large
    For rank-biserial:
      0.1 small, 0.3 medium, 0.5 large
    """
    results = results.copy()

    labels = []

    for _, row in results.iterrows():
        if row["test_used"] in {"t_test_equal_var", "welch_t_test"}:
            value = abs(row["cohen_d_l2_minus_l1"])
            metric = "cohen_d"
        elif row["test_used"] == "mann_whitney_u":
            value = abs(row["rank_biserial_l2_minus_l1"])
            metric = "rank_biserial"
        else:
            labels.append("not_available")
            continue

        if pd.isna(value):
            labels.append("not_available")
        elif metric == "cohen_d":
            if value < 0.2:
                labels.append("negligible")
            elif value < 0.5:
                labels.append("small")
            elif value < 0.8:
                labels.append("medium")
            else:
                labels.append("large")
        else:
            if value < 0.1:
                labels.append("negligible")
            elif value < 0.3:
                labels.append("small")
            elif value < 0.5:
                labels.append("medium")
            else:
                labels.append("large")

    results["effect_size_label"] = labels
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
    results = add_effect_size_interpretation(results)

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
                    "cohen_d_l2_minus_l1",
                    "rank_biserial_l2_minus_l1",
                    "effect_size_label",
                    "p_value",
                    "p_fdr",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()