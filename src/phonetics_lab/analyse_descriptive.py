# File: src/phonetics_lab/analyse_descriptive.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def ensure_dirs(figures_dir: Path, tables_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)


def confidence_ellipse(x, y, ax, confidence=0.95, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return

    cov = np.cov(x, y)

    if np.linalg.det(cov) <= 0:
        return

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    scale = np.sqrt(chi2.ppf(confidence, df=2))

    width, height = 2 * scale * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=1.5,
        alpha=0.8,
        **kwargs,
    )

    ax.add_patch(ellipse)


def make_descriptive_table(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    main = df[df["main_oral_vowel"].astype(bool)].copy()

    table = (
        main.groupby(["phoneme", "l1_status", "gender"])
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

    table["F1_cv"] = table["F1_sd"] / table["F1_mean"].abs()
    table["F2_cv"] = table["F2_sd"] / table["F2_mean"].abs()

    table.to_csv(tables_dir / "descriptive_acoustic_by_vowel_group.csv", index=False)
    return table


def plot_vowel_chart(df: pd.DataFrame, figures_dir: Path) -> None:
    main = df[df["main_oral_vowel"].astype(bool)].copy()
    main["group"] = main["l1_status"] + "/" + main["gender"].str.upper()

    fig, ax = plt.subplots(figsize=(10, 8))

    groups = sorted(main["group"].dropna().unique())

    for group in groups:
        group_df = main[main["group"] == group]

        centroids = (
            group_df.groupby("phoneme")
            .agg(F1=("F1_lobanov", "mean"), F2=("F2_lobanov", "mean"))
            .reset_index()
        )

        ax.scatter(
            centroids["F2"],
            centroids["F1"],
            label=group,
            s=45,
            alpha=0.8,
        )

        for _, row in centroids.iterrows():
            ax.text(row["F2"], row["F1"], row["phoneme"], fontsize=10)

        for phoneme in sorted(group_df["phoneme"].unique()):
            phoneme_df = group_df[group_df["phoneme"] == phoneme]
            confidence_ellipse(
                phoneme_df["F2_lobanov"],
                phoneme_df["F1_lobanov"],
                ax,
                confidence=0.95,
            )

    ax.set_title("French oral vowels: Lobanov-normalised F1/F2 centroids")
    ax.set_xlabel("F2 Lobanov")
    ax.set_ylabel("F1 Lobanov")
    ax.invert_yaxis()
    ax.legend(title="Group")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(figures_dir / "vowel_chart_lobanov_centroids.png", dpi=300)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, figures_dir: Path) -> None:
    main = df[df["main_oral_vowel"].astype(bool)].copy()
    vowels = sorted(main["phoneme"].unique())

    for formant in ["F1_lobanov", "F2_lobanov"]:
        fig, ax = plt.subplots(figsize=(12, 7))

        data = [main.loc[main["phoneme"] == vowel, formant].dropna() for vowel in vowels]

        ax.boxplot(data, tick_labels=vowels, showfliers=False)
        ax.set_title(f"{formant} distribution by vowel")
        ax.set_xlabel("Vowel")
        ax.set_ylabel(formant)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(figures_dir / f"boxplot_{formant}_by_vowel.png", dpi=300)
        plt.close(fig)


def plot_group_boxplots(df: pd.DataFrame, figures_dir: Path) -> None:
    main = df[df["main_oral_vowel"].astype(bool)].copy()
    main["group"] = main["l1_status"] + "/" + main["gender"].str.upper()

    for formant in ["F1_lobanov", "F2_lobanov"]:
        fig, ax = plt.subplots(figsize=(14, 7))

        labels = []
        data = []

        for phoneme in sorted(main["phoneme"].unique()):
            for group in sorted(main["group"].unique()):
                subset = main[(main["phoneme"] == phoneme) & (main["group"] == group)]
                if len(subset) == 0:
                    continue
                labels.append(f"{phoneme}\n{group}")
                data.append(subset[formant].dropna())

        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"{formant} by vowel and speaker group")
        ax.set_ylabel(formant)
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(figures_dir / f"group_boxplot_{formant}.png", dpi=300)
        plt.close(fig)


def variance_decomposition(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """
    Approximate F1 variance decomposition by phoneme.

    This avoids impossible negative residual variance by reporting:
    - total variance across all tokens
    - between-speaker variance of speaker means
    - mean within-speaker variance
    - proportion of variance associated with between-speaker differences

    This is descriptive, not a full random-effects model.
    """
    main = df[df["main_oral_vowel"].astype(bool)].copy()

    rows = []

    for phoneme, ph_df in main.groupby("phoneme"):
        total_var = ph_df["F1_lobanov"].var()

        speaker_means = ph_df.groupby("speaker_id")["F1_lobanov"].mean()
        speaker_vars = ph_df.groupby("speaker_id")["F1_lobanov"].var().dropna()

        inter_speaker_var = speaker_means.var()
        intra_speaker_var = speaker_vars.mean() if len(speaker_vars) else np.nan

        inter_speaker_ratio = (
            inter_speaker_var / total_var
            if pd.notna(total_var) and total_var > 0
            else np.nan
        )

        intra_speaker_ratio = (
            intra_speaker_var / total_var
            if pd.notna(total_var) and total_var > 0
            else np.nan
        )

        rows.append(
            {
                "phoneme": phoneme,
                "n": len(ph_df),
                "F1_total_variance": total_var,
                "F1_inter_speaker_variance": inter_speaker_var,
                "F1_intra_speaker_variance": intra_speaker_var,
                "F1_inter_speaker_ratio": inter_speaker_ratio,
                "F1_intra_speaker_ratio": intra_speaker_ratio,
            }
        )

    table = pd.DataFrame(rows).sort_values(
        "F1_inter_speaker_variance",
        ascending=False,
    )

    table.to_csv(tables_dir / "variance_decomposition_F1.csv", index=False)
    return table

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    ensure_dirs(args.figures_dir, args.tables_dir)

    df = pd.read_csv(args.input)

    descriptive = make_descriptive_table(df, args.tables_dir)
    variance_table = variance_decomposition(df, args.tables_dir)

    plot_vowel_chart(df, args.figures_dir)
    plot_boxplots(df, args.figures_dir)
    plot_group_boxplots(df, args.figures_dir)

    print("\n=== DESCRIPTIVE ACOUSTIC ANALYSIS ===")
    print(f"Input rows: {len(df)}")
    print(f"Main analysis rows: {df['main_oral_vowel'].sum()}")

    print("\nDescriptive table saved:")
    print(args.tables_dir / "descriptive_acoustic_by_vowel_group.csv")

    print("\nVariance decomposition saved:")
    print(args.tables_dir / "variance_decomposition_F1.csv")

    print("\nFigures saved to:")
    print(args.figures_dir)

    print("\nTop vowels by descriptive inter-speaker F1 variance:")
    print(variance_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()