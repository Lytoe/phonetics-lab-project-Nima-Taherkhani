# File: src/phonetics_lab/qc_acoustics.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def flag_formant_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["F1_valid"] = df["F1_mid"].between(150, 1200)
    df["F2_valid"] = df["F2_mid"].between(500, 3500)
    df["F3_valid"] = df["F3_mid"].between(1200, 4500)

    # Basic phonetic sanity check: F1 should normally be below F2.
    df["formant_order_valid"] = df["F1_mid"] < df["F2_mid"]

    df["formant_valid"] = (
        df["F1_valid"]
        & df["F2_valid"]
        & df["F3_valid"]
        & df["formant_order_valid"]
    )

    df["formant_quality_reason"] = ""

    df.loc[~df["F1_valid"], "formant_quality_reason"] += "F1_out_of_range;"
    df.loc[~df["F2_valid"], "formant_quality_reason"] += "F2_out_of_range;"
    df.loc[~df["F3_valid"], "formant_quality_reason"] += "F3_out_of_range;"
    df.loc[~df["formant_order_valid"], "formant_quality_reason"] += "F1_not_below_F2;"

    for col in ["F1_mid", "F2_mid", "F3_mid"]:
        df.loc[~df["formant_valid"], col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    clean = flag_formant_quality(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False, encoding="utf-8")

    report = (
        clean.groupby(["phoneme", "l1_status", "gender"])
        .agg(
            n=("token_id", "count"),
            invalid_formants=("formant_valid", lambda x: int((~x).sum())),
            invalid_rate=("formant_valid", lambda x: float((~x).mean())),
            f0_missing_rate=("f0_mean", lambda x: float(x.isna().mean())),
        )
        .reset_index()
        .sort_values(["phoneme", "l1_status", "gender"])
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False, encoding="utf-8")

    print("\n=== ACOUSTIC QC SUMMARY ===")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(clean)}")
    print(f"Invalid formant rate: {(~clean['formant_valid']).mean():.4f}")
    print(f"Saved cleaned file to: {args.out}")
    print(f"Saved QC report to: {args.report}")

    print("\nWorst phoneme/group invalid rates:")
    print(report.sort_values("invalid_rate", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()