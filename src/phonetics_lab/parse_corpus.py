from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from textgrid import TextGrid


def find_files(raw_dir: Path, suffix: str) -> list[Path]:
    return sorted(raw_dir.rglob(f"*{suffix}"))


def parse_textgrid(path: Path) -> list[dict]:
    """Parse phoneme intervals from a TextGrid.

    This is intentionally defensive because corpus tier names may differ.
    After inspecting the real corpus, set the exact tier name if needed.
    """
    tg = TextGrid.fromFile(str(path))
    candidate_names = {"phoneme", "phonemes", "phones", "phone", "segments"}
    tier = None
    for t in tg.tiers:
        if t.name.lower() in candidate_names:
            tier = t
            break
    if tier is None:
        # fallback: use the interval tier with the most intervals
        interval_tiers = [t for t in tg.tiers if hasattr(t, "intervals")]
        tier = max(interval_tiers, key=lambda t: len(t.intervals))

    rows = []
    for interval in tier.intervals:
        label = interval.mark.strip()
        if not label:
            continue
        rows.append(
            {
                "textgrid_path": str(path),
                "file_stem": path.stem,
                "phoneme": label,
                "onset": float(interval.minTime),
                "offset": float(interval.maxTime),
                "duration_ms": (float(interval.maxTime) - float(interval.minTime)) * 1000,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    rows = []
    for tg_path in find_files(raw_dir, ".TextGrid"):
        rows.extend(parse_textgrid(tg_path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No phoneme intervals found. Check TextGrid files and tier names.")

    # TODO after corpus inspection: derive speaker_id, sentence_id, repetition from filenames/metadata.csv
    df["speaker_id"] = df["file_stem"].str.extract(r"([^_\-]+)", expand=False)
    df["sentence_id"] = df["file_stem"]
    df["repetition"] = 1
    df["l1_status"] = pd.NA
    df["gender"] = pd.NA

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
