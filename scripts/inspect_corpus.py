# File: scripts/inspect_corpus.py

from pathlib import Path
import csv
import re
from collections import Counter, defaultdict

RAW_DIR = Path("data/raw")


def list_file_counts():
    print("\n=== FILE COUNTS ===")
    counts = Counter()
    speaker_counts = defaultdict(Counter)

    for path in RAW_DIR.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            counts[ext] += 1
            speaker = path.relative_to(RAW_DIR).parts[0]
            speaker_counts[speaker][ext] += 1

    print("Global:", dict(counts))
    print("\nBy speaker:")
    for speaker, c in sorted(speaker_counts.items()):
        print(speaker, dict(c))


def inspect_csv_headers(limit=20):
    print("\n=== CSV HEADERS ===")
    csv_files = list(RAW_DIR.rglob("*.csv"))[:limit]

    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            print(f"\n{csv_path}")
            print(header)
        except Exception as e:
            print(f"\n{csv_path}")
            print(f"ERROR: {e}")


def extract_textgrid_tier_names(textgrid_path: Path):
    text = textgrid_path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r'name\s*=\s*"([^"]+)"', text)
    return names


def inspect_textgrid_tiers(limit=12):
    print("\n=== TEXTGRID TIER NAMES ===")
    tg_files = list(RAW_DIR.rglob("*.TextGrid"))[:limit]

    for tg_path in tg_files:
        try:
            names = extract_textgrid_tier_names(tg_path)
            print(f"\n{tg_path}")
            print(names)
        except Exception as e:
            print(f"\n{tg_path}")
            print(f"ERROR: {e}")


def inspect_filename_patterns():
    print("\n=== FILENAME PATTERNS ===")
    pattern = re.compile(
        r"(?P<speaker>[a-z]+)_(?P<l1>fra|rus)_list(?P<list_id>\d+)_FRcorp(?P<sentence>\d+)",
        re.IGNORECASE,
    )

    examples = []
    unmatched = []

    for path in RAW_DIR.rglob("*.TextGrid"):
        match = pattern.search(path.stem)
        if match:
            examples.append(match.groupdict())
        else:
            unmatched.append(str(path))

    print("Matched examples:")
    for item in examples[:10]:
        print(item)

    print(f"\nTotal matched TextGrids: {len(examples)}")
    print(f"Total unmatched TextGrids: {len(unmatched)}")

    if unmatched:
        print("\nUnmatched examples:")
        for item in unmatched[:10]:
            print(item)


if __name__ == "__main__":
    list_file_counts()
    inspect_filename_patterns()
    inspect_csv_headers()
    inspect_textgrid_tiers()