# File: src/phonetics_lab/extract_acoustics.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import parselmouth


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def get_max_formant(gender: str, max_formant_female: float, max_formant_male: float) -> float:
    """
    Corpus metadata uses 'f' and 'm'.
    Female speakers: 5000 Hz
    Male speakers: 4500 Hz
    """
    gender_clean = str(gender).strip().lower()

    if gender_clean == "m":
        return max_formant_male

    return max_formant_female


def safe_make_formant(
    sound: parselmouth.Sound,
    max_formant: float,
    n_formants: int,
) -> Optional[parselmouth.Formant]:
    try:
        return sound.to_formant_burg(
            max_number_of_formants=n_formants,
            maximum_formant=max_formant,
        )
    except Exception:
        return None


def safe_formant_value(
    formant: Optional[parselmouth.Formant],
    time: float,
    formant_number: int,
) -> float:
    if formant is None:
        return np.nan

    try:
        value = formant.get_value_at_time(formant_number, time)

        if value is None or np.isnan(value):
            return np.nan

        return float(value)
    except Exception:
        return np.nan


def safe_extract_segment(
    sound: parselmouth.Sound,
    onset: float,
    offset: float,
) -> Optional[parselmouth.Sound]:
    try:
        if offset <= onset:
            return None

        return sound.extract_part(
            from_time=onset,
            to_time=offset,
            preserve_times=False,
        )
    except Exception:
        return None


def safe_f0_mean(segment: Optional[parselmouth.Sound]) -> float:
    if segment is None:
        return np.nan

    try:
        pitch = segment.to_pitch()
        values = pitch.selected_array["frequency"]
        voiced = values[values > 0]

        if len(voiced) == 0:
            return np.nan

        return float(np.mean(voiced))
    except Exception:
        return np.nan


def safe_rms_energy(segment: Optional[parselmouth.Sound]) -> float:
    if segment is None:
        return np.nan

    try:
        values = segment.values

        if values.size == 0:
            return np.nan

        return float(np.sqrt(np.mean(values**2)))
    except Exception:
        return np.nan


def safe_spectral_centre_of_gravity(segment: Optional[parselmouth.Sound]) -> float:
    if segment is None:
        return np.nan

    try:
        spectrum = segment.to_spectrum()
        value = spectrum.get_centre_of_gravity(power=2.0)

        if value is None or np.isnan(value):
            return np.nan

        return float(value)
    except Exception:
        return np.nan


def add_missing_acoustic_row(row: pd.Series, reason: str) -> dict:
    out = row.to_dict()
    out.update(
        {
            "midpoint": np.nan,
            "t25": np.nan,
            "t75": np.nan,
            "F1_mid": np.nan,
            "F2_mid": np.nan,
            "F3_mid": np.nan,
            "F1_25": np.nan,
            "F2_25": np.nan,
            "F1_75": np.nan,
            "F2_75": np.nan,
            "f0_mean": np.nan,
            "energy_rms": np.nan,
            "spectral_cog": np.nan,
            "acoustic_error": reason,
        }
    )
    return out


def extract_features_for_token(
    row: pd.Series,
    sound: parselmouth.Sound,
    formant: Optional[parselmouth.Formant],
) -> dict:
    onset = safe_float(row["onset"])
    offset = safe_float(row["offset"])
    duration_ms = safe_float(row["duration_ms"])

    if np.isnan(onset) or np.isnan(offset) or offset <= onset:
        return add_missing_acoustic_row(row, reason="invalid_time_bounds")

    midpoint = onset + (offset - onset) / 2
    t25 = onset + (offset - onset) * 0.25
    t75 = onset + (offset - onset) * 0.75

    segment = safe_extract_segment(sound, onset, offset)

    is_oral_vowel = to_bool(row.get("is_oral_vowel", False))
    is_long_vowel = is_oral_vowel and duration_ms > 80

    out = row.to_dict()

    out.update(
        {
            "midpoint": midpoint,
            "t25": t25 if is_long_vowel else np.nan,
            "t75": t75 if is_long_vowel else np.nan,

            # Midpoint formants: meaningful for oral vowels.
            "F1_mid": safe_formant_value(formant, midpoint, 1) if is_oral_vowel else np.nan,
            "F2_mid": safe_formant_value(formant, midpoint, 2) if is_oral_vowel else np.nan,
            "F3_mid": safe_formant_value(formant, midpoint, 3) if is_oral_vowel else np.nan,

            # Trajectory formants for long vowels > 80 ms.
            "F1_25": safe_formant_value(formant, t25, 1) if is_long_vowel else np.nan,
            "F2_25": safe_formant_value(formant, t25, 2) if is_long_vowel else np.nan,
            "F1_75": safe_formant_value(formant, t75, 1) if is_long_vowel else np.nan,
            "F2_75": safe_formant_value(formant, t75, 2) if is_long_vowel else np.nan,

            # General signal features.
            "f0_mean": safe_f0_mean(segment),
            "energy_rms": safe_rms_energy(segment),
            "spectral_cog": safe_spectral_centre_of_gravity(segment),
            "acoustic_error": "",
        }
    )

    return out


def extract_acoustics(
    tokens_path: Path,
    out_path: Path,
    n_formants: int,
    max_formant_female: float,
    max_formant_male: float,
    vowels_only: bool,
) -> pd.DataFrame:
    tokens = pd.read_csv(tokens_path)

    if vowels_only:
        tokens = tokens[tokens["is_oral_vowel"].apply(to_bool)].copy()

    out_rows = []

    required_columns = {"wav_path", "onset", "offset", "duration_ms", "gender"}
    missing_columns = required_columns - set(tokens.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in token table: {missing_columns}")

    grouped = tokens.groupby("wav_path", sort=False)

    total_files = len(grouped)
    processed_files = 0

    for wav_path_str, group in grouped:
        processed_files += 1

        if processed_files % 50 == 0:
            print(f"Processed {processed_files}/{total_files} WAV files...")

        wav_path = Path(wav_path_str)

        if not wav_path.exists():
            for _, row in group.iterrows():
                out_rows.append(add_missing_acoustic_row(row, reason="missing_wav"))
            continue

        try:
            sound = parselmouth.Sound(str(wav_path))
        except Exception:
            for _, row in group.iterrows():
                out_rows.append(add_missing_acoustic_row(row, reason="wav_load_failed"))
            continue

        first_row = group.iloc[0]
        max_formant = get_max_formant(
            gender=first_row.get("gender", ""),
            max_formant_female=max_formant_female,
            max_formant_male=max_formant_male,
        )

        formant = safe_make_formant(
            sound=sound,
            max_formant=max_formant,
            n_formants=n_formants,
        )

        for _, row in group.iterrows():
            out_rows.append(
                extract_features_for_token(
                    row=row,
                    sound=sound,
                    formant=formant,
                )
            )

    out_df = pd.DataFrame(out_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)

    # Kept for compatibility with old Snakefile commands.
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    parser.add_argument("--n-formants", type=int, default=5)
    parser.add_argument("--max-formant-female", type=float, default=5000)
    parser.add_argument("--max-formant-male", type=float, default=4500)

    parser.add_argument(
        "--include-all-tokens",
        action="store_true",
        help="By default, extract formants only for oral vowels. Use this to keep all phoneme tokens.",
    )

    args = parser.parse_args()

    out_df = extract_acoustics(
        tokens_path=args.tokens,
        out_path=args.out,
        n_formants=args.n_formants,
        max_formant_female=args.max_formant_female,
        max_formant_male=args.max_formant_male,
        vowels_only=not args.include_all_tokens,
    )

    print("\n=== ACOUSTIC EXTRACTION SUMMARY ===")
    print(f"Rows written: {len(out_df)}")
    print(f"Output: {args.out}")

    if not out_df.empty:
        print("\nMissing values:")
        print(out_df[["F1_mid", "F2_mid", "F3_mid", "f0_mean", "energy_rms", "spectral_cog"]].isna().mean())

        print("\nRows by L1/gender:")
        print(out_df.groupby(["l1_status", "gender"]).size())

        print("\nRows by vowel:")
        print(out_df["phoneme"].value_counts().head(30))


if __name__ == "__main__":
    main()