# File: src/phonetics_lab/acoustic_classifier.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


FEATURES = ["F1_lobanov", "F2_lobanov"]


def get_main_data(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["main_oral_vowel"].astype(bool)].copy()
    main = main.dropna(subset=FEATURES + ["phoneme", "speaker_id"])
    return main


def compute_centroids(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby("phoneme")[FEATURES]
        .mean()
        .reset_index()
    )


def predict_nearest_centroid(test_df: pd.DataFrame, centroids: pd.DataFrame) -> list[str]:
    centroid_labels = centroids["phoneme"].tolist()
    centroid_values = centroids[FEATURES].values

    predictions = []

    for _, row in test_df.iterrows():
        x = row[FEATURES].values.astype(float)
        distances = np.linalg.norm(centroid_values - x, axis=1)
        predictions.append(centroid_labels[int(np.argmin(distances))])

    return predictions


def leave_one_speaker_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    speakers = sorted(df["speaker_id"].unique())

    for speaker in speakers:
        train = df[df["speaker_id"] != speaker]
        test = df[df["speaker_id"] == speaker]

        centroids = compute_centroids(train)
        predictions = predict_nearest_centroid(test, centroids)

        fold = test[["token_id", "speaker_id", "l1_status", "gender", "phoneme"]].copy()
        fold["predicted_phoneme"] = predictions
        fold["correct"] = fold["phoneme"] == fold["predicted_phoneme"]

        rows.append(fold)

    return pd.concat(rows, ignore_index=True)


def save_confusion_matrix(predictions: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    labels = sorted(predictions["phoneme"].unique())

    cm = confusion_matrix(
        predictions["phoneme"],
        predictions["predicted_phoneme"],
        labels=labels,
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(tables_dir / "acoustic_classifier_confusion_matrix.csv", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted vowel")
    ax.set_ylabel("True vowel")
    ax.set_title("Acoustic nearest-centroid classifier confusion matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(figures_dir / "acoustic_classifier_confusion_matrix.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    main = get_main_data(df)

    predictions = leave_one_speaker_out(main)

    predictions.to_csv(
        args.tables_dir / "acoustic_classifier_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    accuracy = accuracy_score(predictions["phoneme"], predictions["predicted_phoneme"])
    macro_f1 = f1_score(
        predictions["phoneme"],
        predictions["predicted_phoneme"],
        average="macro",
    )

    report = classification_report(
        predictions["phoneme"],
        predictions["predicted_phoneme"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(args.tables_dir / "acoustic_classifier_report.csv", encoding="utf-8")

    group_accuracy = (
        predictions.groupby("l1_status")
        .agg(
            n=("token_id", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    group_accuracy.to_csv(
        args.tables_dir / "acoustic_classifier_accuracy_by_l1.csv",
        index=False,
        encoding="utf-8",
    )

    save_confusion_matrix(predictions, args.figures_dir, args.tables_dir)

    print("\n=== ACOUSTIC NEAREST-CENTROID CLASSIFIER ===")
    print(f"Rows evaluated: {len(predictions)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nAccuracy by L1 status:")
    print(group_accuracy.to_string(index=False))

    print("\nPer-class report:")
    print(report_df.round(3).to_string())


if __name__ == "__main__":
    main()