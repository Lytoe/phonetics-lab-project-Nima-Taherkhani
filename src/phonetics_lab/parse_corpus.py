# File: src/phonetics_lab/parse_corpus.py

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


FILENAME_RE = re.compile(
    r"(?P<speaker>[a-z0-9]+)_(?P<l1>fra|rus)_list(?P<list_id>\d+)_FRcorp(?P<sentence_id>\d+)",
    re.IGNORECASE,
)

INTERVAL_RE = re.compile(
    r"intervals\s*\[\d+\]:\s*"
    r"xmin\s*=\s*(?P<xmin>[-+0-9.eE]+)\s*"
    r"xmax\s*=\s*(?P<xmax>[-+0-9.eE]+)\s*"
    r'text\s*=\s*"(?P<label>[^"]*)"',
    re.DOTALL,
)

SILENCE_LABELS = {"", "sil", "sp", "spn", "#", "pause"}
MALFORMED_PREFIXES = ("ding",)

ORAL_VOWELS = {
    "i", "e", "ɛ", "a", "ɑ", "y", "u", "o", "ø", "œ", "ə"
}

NASAL_VOWELS = {
    "ɑ̃", "ɛ̃", "œ̃", "ɔ̃"
}

VOWEL_VARIANT_MAP = {
    "aː": "a",
    "a̰": "a",
    "ɑ̃ː": "ɑ̃",
    "ɑ̰̃": "ɑ̃",
    "ɛ̰": "ɛ",
    "ə̰": "ə",
    "ø̰": "ø",
    "i̥": "i",
    "ɪ": "i",
    "y̥": "y",
    "ʉ": "u",
    "ɨ": "i",
    "u:": "u",
}

CONSONANT_VARIANT_MAP = {
    "ʀ": "ʁ",
    "ʀ̥": "ʁ",
    "ʁ̥": "ʁ",
    "ʁ̞": "ʁ",
    "ʒ̥": "ʒ",
    "ʒ̞": "ʒ",
    "ʒʲ": "ʒ",
    "ɡ̥": "ɡ",
    "b̥": "b",
    "d̥": "d",
    "v̥": "v",
    "l̥": "l",
    "lʲ": "l",
    "sʲ": "s",
    "tʲ": "t",
    "pʰ": "p",
    "sː": "s",
    "ʃː": "ʃ",
}


def load_metadata(raw_dir: Path) -> Dict[str, dict]:
    """
    Load speaker-level metadata from metadata_RUFR.csv.

    Expected columns after semicolon parsing:
    processed by; spk; L1; Age; Gender; FR level; RU level; Duration
    """
    metadata_path = raw_dir / "metadata_RUFR.csv"

    if not metadata_path.exists():
        print("WARNING: metadata_RUFR.csv not found. Gender will be missing.")
        return {}

    df = pd.read_csv(metadata_path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]

    if "spk" not in df.columns:
        raise ValueError(
            f"metadata_RUFR.csv has no 'spk' column. Columns: {list(df.columns)}"
        )

    df["speaker_id"] = df["spk"].astype(str).str.strip().str.upper()

    metadata = {}

    for _, row in df.iterrows():
        speaker_id = row["speaker_id"]

        metadata[speaker_id] = {
            "metadata_l1": row.get("L1"),
            "age": row.get("Age"),
            "gender": row.get("Gender"),
            "fr_level": row.get("FR level"),
            "ru_level": row.get("RU level"),
            "duration_metadata": row.get("Duration"),
        }

    return metadata


def parse_filename(textgrid_path: Path) -> Optional[dict]:
    match = FILENAME_RE.search(textgrid_path.stem)

    if not match:
        return None

    data = match.groupdict()
    l1_code = data["l1"].lower()

    return {
        "filename_speaker": data["speaker"].upper(),
        "l1_code": l1_code,
        "l1_status": "L1" if l1_code == "fra" else "L2",
        "native_language": "French" if l1_code == "fra" else "Russian",
        "list_id": int(data["list_id"]),
        "repetition": int(data["list_id"]),
        "sentence_id": f"FRcorp{int(data['sentence_id'])}",
        "sentence_number": int(data["sentence_id"]),
    }


def extract_tier_block(textgrid_text: str, tier_name: str) -> str:
    """
    Extract the block corresponding to a named TextGrid tier.
    Works for standard long-format Praat TextGrid files.
    """
    item_blocks = re.split(r"\n\s*item\s*\[\d+\]:", textgrid_text)

    for block in item_blocks:
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)

        if name_match and name_match.group(1).strip().lower() == tier_name.lower():
            return block

    raise ValueError(f"Tier '{tier_name}' not found.")


def parse_phone_intervals(textgrid_path: Path, tier_name: str = "phones") -> List[dict]:
    text = textgrid_path.read_text(encoding="utf-8", errors="ignore")
    tier_block = extract_tier_block(text, tier_name)

    intervals = []

    for match in INTERVAL_RE.finditer(tier_block):
        onset = float(match.group("xmin"))
        offset = float(match.group("xmax"))
        label = match.group("label").strip()

        if label.lower() in SILENCE_LABELS:
            continue

        intervals.append(
            {
                "phoneme": label,
                "onset": onset,
                "offset": offset,
                "duration_s": offset - onset,
                "duration_ms": (offset - onset) * 1000,
            }
        )

    return intervals


def is_malformed_label(label: str) -> bool:
    clean = str(label).strip().lower()
    return any(clean.startswith(prefix) for prefix in MALFORMED_PREFIXES)


def clean_phoneme_label(label: str) -> str:
    """
    Keep the raw corpus label, but produce a broader analysis label.
    We preserve meaningful IPA symbols and normalize obvious variants.
    """
    label = str(label).strip()

    if label in VOWEL_VARIANT_MAP:
        return VOWEL_VARIANT_MAP[label]

    if label in CONSONANT_VARIANT_MAP:
        return CONSONANT_VARIANT_MAP[label]

    return label


def has_nasal_tilde(label: str) -> bool:
    decomposed = unicodedata.normalize("NFD", str(label))
    return "\u0303" in decomposed


def classify_phone(label: str) -> dict:
    raw = str(label).strip()
    clean = clean_phoneme_label(raw)

    is_malformed = is_malformed_label(raw)
    is_nasal_vowel = clean in NASAL_VOWELS or has_nasal_tilde(clean)
    is_oral_vowel = clean in ORAL_VOWELS and not is_nasal_vowel
    is_vowel = is_oral_vowel or is_nasal_vowel

    return {
        "phoneme_raw": raw,
        "phoneme_clean": clean,
        "is_malformed": is_malformed,
        "is_vowel": is_vowel,
        "is_oral_vowel": is_oral_vowel,
        "is_nasal_vowel": is_nasal_vowel,
    }


def build_phoneme_table(raw_dir: Path) -> pd.DataFrame:
    metadata = load_metadata(raw_dir)

    rows = []
    unmatched_files = []
    failed_textgrids = []
    missing_wavs = []
    excluded_malformed = []

    textgrid_files = sorted(raw_dir.rglob("*.TextGrid"))

    for textgrid_path in textgrid_files:
        filename_info = parse_filename(textgrid_path)

        if filename_info is None:
            unmatched_files.append(str(textgrid_path))
            continue

        speaker_id = textgrid_path.parent.name.upper()
        wav_path = textgrid_path.with_suffix(".wav")

        if not wav_path.exists():
            missing_wavs.append(str(wav_path))
            continue

        speaker_meta = metadata.get(speaker_id, {})

        try:
            intervals = parse_phone_intervals(textgrid_path, tier_name="phones")
        except Exception as exc:
            failed_textgrids.append((str(textgrid_path), str(exc)))
            continue

        for phone_index, interval in enumerate(intervals):
            phone_info = classify_phone(interval["phoneme"])

            if phone_info["is_malformed"]:
                excluded_malformed.append(
                    {
                        "speaker_id": speaker_id,
                        "textgrid_path": str(textgrid_path),
                        "phoneme_raw": phone_info["phoneme_raw"],
                        "onset": interval["onset"],
                        "offset": interval["offset"],
                    }
                )
                continue

            token_id = (
                f"{speaker_id}_"
                f"{filename_info['sentence_id']}_"
                f"rep{filename_info['repetition']}_"
                f"phone{phone_index:04d}"
            )

            rows.append(
                {
                    "token_id": token_id,
                    "speaker_id": speaker_id,
                    "filename_speaker": filename_info["filename_speaker"],
                    "sentence_id": filename_info["sentence_id"],
                    "sentence_number": filename_info["sentence_number"],
                    "repetition": filename_info["repetition"],
                    "list_id": filename_info["list_id"],
                    "l1_code": filename_info["l1_code"],
                    "l1_status": filename_info["l1_status"],
                    "native_language": filename_info["native_language"],
                    "gender": speaker_meta.get("gender"),
                    "age": speaker_meta.get("age"),
                    "fr_level": speaker_meta.get("fr_level"),
                    "ru_level": speaker_meta.get("ru_level"),
                    "phoneme": phone_info["phoneme_clean"],
                    "phoneme_raw": phone_info["phoneme_raw"],
                    "is_vowel": phone_info["is_vowel"],
                    "is_oral_vowel": phone_info["is_oral_vowel"],
                    "is_nasal_vowel": phone_info["is_nasal_vowel"],
                    "onset": interval["onset"],
                    "offset": interval["offset"],
                    "duration_s": interval["duration_s"],
                    "duration_ms": interval["duration_ms"],
                    "textgrid_path": str(textgrid_path),
                    "wav_path": str(wav_path),
                }
            )

    df = pd.DataFrame(rows)

    print("\n=== PARSE SUMMARY ===")
    print(f"TextGrid files found: {len(textgrid_files)}")
    print(f"Parsed phoneme tokens: {len(df)}")
    print(f"Excluded malformed labels: {len(excluded_malformed)}")
    print(f"Unmatched filenames: {len(unmatched_files)}")
    print(f"Failed TextGrids: {len(failed_textgrids)}")
    print(f"Missing WAV files: {len(missing_wavs)}")

    if unmatched_files:
        print("\nUnmatched examples:")
        for item in unmatched_files[:10]:
            print(item)

    if failed_textgrids:
        print("\nFailed TextGrid examples:")
        for path, error in failed_textgrids[:10]:
            print(path, "=>", error)

    if missing_wavs:
        print("\nMissing WAV examples:")
        for item in missing_wavs[:10]:
            print(item)

    if excluded_malformed:
        print("\nExcluded malformed label examples:")
        for item in excluded_malformed[:10]:
            print(item)

    if not df.empty:
        print("\nSpeaker counts:")
        print(df.groupby(["speaker_id", "l1_status", "gender"]).size())

        print("\nMost common cleaned phoneme labels:")
        print(df["phoneme"].value_counts().head(30))

        print("\nMost common raw phoneme labels:")
        print(df["phoneme_raw"].value_counts().head(30))

        print("\nVowel counts:")
        print(df[["is_vowel", "is_oral_vowel", "is_nasal_vowel"]].sum())

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("data/interim/phoneme_tokens.csv"),
    )
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_phoneme_table(args.raw_dir)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"\nSaved phoneme table to: {args.out_csv}")


if __name__ == "__main__":
    main()