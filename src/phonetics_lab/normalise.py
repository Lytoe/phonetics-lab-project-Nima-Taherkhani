from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

VOWELS = {"i", "e", "ɛ", "E", "a", "y", "ø", "œ", "u", "o", "ɔ", "O", "ə"}


def lobanov(df: pd.DataFrame, value_col: str) -> pd.Series:
    vowel_mask = df["phoneme"].isin(VOWELS)
    stats = (
        df.loc[vowel_mask]
        .groupby("speaker_id")[value_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": f"{value_col}_speaker_mean", "std": f"{value_col}_speaker_sd"})
    )
    joined = df.join(stats, on="speaker_id")
    return (joined[value_col] - joined[f"{value_col}_speaker_mean"]) / joined[f"{value_col}_speaker_sd"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.acoustic)
    df["F1_lobanov"] = lobanov(df, "F1")
    df["F2_lobanov"] = lobanov(df, "F2")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
