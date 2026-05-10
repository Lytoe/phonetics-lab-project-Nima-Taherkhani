# File: src/phonetics_lab/extract_neural_whisper.py

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes"}


def load_audio(wav_path: Path, sample_rate: int) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)

        if len(audio) == 0:
            return None

        return audio.astype(np.float32)
    except Exception:
        return None


def extract_wav_hidden_states(
    audio: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperModel,
    layer: int,
    device: torch.device,
    sample_rate: int,
) -> np.ndarray:
    """
    Run Whisper once on the full WAV and return one hidden-state matrix:
    shape = [n_frames, hidden_dim]
    """
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        outputs = model.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states

    if layer >= len(hidden_states):
        raise ValueError(
            f"Requested layer {layer}, but model returned {len(hidden_states)} hidden-state tensors."
        )

    return hidden_states[layer].squeeze(0).detach().cpu().numpy().astype(np.float32)


def pool_interval_embedding(
    hidden: np.ndarray,
    onset: float,
    offset: float,
    audio_duration: float,
) -> np.ndarray:
    """
    Pool Whisper frames overlapping a phoneme interval.

    We map the phoneme interval onto the hidden-state frame axis by proportional time.
    This is much faster than running Whisper separately for every phoneme.
    """
    n_frames = hidden.shape[0]

    start_idx = int(np.floor((onset / audio_duration) * n_frames))
    end_idx = int(np.ceil((offset / audio_duration) * n_frames))

    start_idx = max(0, min(start_idx, n_frames - 1))
    end_idx = max(start_idx + 1, min(end_idx, n_frames))

    return hidden[start_idx:end_idx].mean(axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)

    # Debug mode: process only the first N WAV files.
    parser.add_argument("--max-wavs", type=int, default=20)

    # Full mode: process all WAV files in the token table.
    parser.add_argument("--full-run", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.tokens)

    # For now, neural extraction uses the same main oral vowel subset as the acoustic analysis.
    df = df[df["main_oral_vowel"].apply(to_bool)].copy()
    df = df.dropna(subset=["wav_path", "onset", "offset"])

    wav_paths = sorted(df["wav_path"].unique())

    if not args.full_run:
        wav_paths = wav_paths[: args.max_wavs]
        df = df[df["wav_path"].isin(wav_paths)].copy()
        print(f"DEBUG MODE: extracting vowels from first {len(wav_paths)} WAV files.")
    else:
        print(f"FULL MODE: extracting vowels from {len(wav_paths)} WAV files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Layer: {args.layer}")
    print(f"Tokens to extract: {len(df)}")

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    embeddings = []
    kept_rows = []
    failed_wavs = []
    failed_tokens = []

    grouped = df.groupby("wav_path", sort=False)

    for wav_path_str, group in tqdm(grouped, total=len(grouped)):
        wav_path = Path(wav_path_str)

        audio = load_audio(
            wav_path=wav_path,
            sample_rate=args.sample_rate,
        )

        if audio is None:
            failed_wavs.append(wav_path_str)
            continue

        audio_duration = len(audio) / args.sample_rate

        try:
            hidden = extract_wav_hidden_states(
                audio=audio,
                processor=processor,
                model=model,
                layer=args.layer,
                device=device,
                sample_rate=args.sample_rate,
            )
        except Exception:
            failed_wavs.append(wav_path_str)
            continue

        for _, row in group.iterrows():
            try:
                emb = pool_interval_embedding(
                    hidden=hidden,
                    onset=float(row["onset"]),
                    offset=float(row["offset"]),
                    audio_duration=audio_duration,
                )
            except Exception:
                failed_tokens.append(row["token_id"])
                continue

            embeddings.append(emb)
            kept_rows.append(row.to_dict())

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    embeddings_array = np.vstack(embeddings)
    metadata = pd.DataFrame(kept_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out,
        embeddings=embeddings_array,
        token_ids=metadata["token_id"].values,
        phonemes=metadata["phoneme"].values,
        speakers=metadata["speaker_id"].values,
        l1_status=metadata["l1_status"].values,
        gender=metadata["gender"].values,
        layer=np.array([args.layer]),
        model_name=np.array([args.model_name]),
    )

    metadata_out = args.out.with_suffix(".metadata.csv")
    metadata.to_csv(metadata_out, index=False, encoding="utf-8")

    print("\n=== WHISPER EXTRACTION SUMMARY ===")
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Saved embeddings: {args.out}")
    print(f"Saved metadata: {metadata_out}")
    print(f"Failed WAVs: {len(failed_wavs)}")
    print(f"Failed tokens: {len(failed_tokens)}")

    if failed_wavs:
        print("\nFailed WAV examples:")
        print(failed_wavs[:10])

    if failed_tokens:
        print("\nFailed token examples:")
        print(failed_tokens[:10])


if __name__ == "__main__":
    main()