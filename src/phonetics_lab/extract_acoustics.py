from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth


def find_wav(raw_dir: Path, file_stem: str) -> Path | None:
    matches = list(raw_dir.rglob(f"{file_stem}.wav")) + list(raw_dir.rglob(f"{file_stem}.WAV"))
    return matches[0] if matches else None


def safe_formant(sound: parselmouth.Sound, time: float, formant_number: int, max_formant: float, n_formants: int) -> float:
    try:
        formant = sound.to_formant_burg(maximum_formant=max_formant, max_number_of_formants=n_formants)
        value = formant.get_value_at_time(formant_number, time)
        return float(value) if value and not np.isnan(value) else np.nan
    except Exception:
        return np.nan


def safe_f0_mean(segment: parselmouth.Sound) -> float:
    try:
        pitch = segment.to_pitch()
        values = pitch.selected_array["frequency"]
        voiced = values[values > 0]
        return float(np.mean(voiced)) if len(voiced) else np.nan
    except Exception:
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-formants", type=int, default=5)
    parser.add_argument("--max-formant-female", type=float, default=5000)
    parser.add_argument("--max-formant-male", type=float, default=4500)
    args = parser.parse_args()

    df = pd.read_csv(args.tokens)
    raw_dir = Path(args.raw_dir)
    out_rows = []

    for _, row in df.iterrows():
        wav_path = find_wav(raw_dir, row["file_stem"])
        if wav_path is None:
            out = row.to_dict()
            out.update({"wav_path": pd.NA, "F1": np.nan, "F2": np.nan, "F3": np.nan, "f0_mean": np.nan})
            out_rows.append(out)
            continue

        sound = parselmouth.Sound(str(wav_path))
        onset, offset = float(row["onset"]), float(row["offset"])
        midpoint = onset + (offset - onset) / 2
        gender = str(row.get("gender", "")).upper()
        max_formant = args.max_formant_male if gender == "M" else args.max_formant_female
        segment = sound.extract_part(from_time=onset, to_time=offset, preserve_times=True)

        out = row.to_dict()
        out.update(
            {
                "wav_path": str(wav_path),
                "midpoint": midpoint,
                "F1": safe_formant(sound, midpoint, 1, max_formant, args.n_formants),
                "F2": safe_formant(sound, midpoint, 2, max_formant, args.n_formants),
                "F3": safe_formant(sound, midpoint, 3, max_formant, args.n_formants),
                "f0_mean": safe_f0_mean(segment),
                "energy": float(segment.get_energy()),
            }
        )
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
