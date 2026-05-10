# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

M1/M2 Advanced Statistics research project.

## Goal
Compare classical acoustic features and neural speech embeddings on the Russian–French Interference Corpus.

## Pipeline stages
1. `parse_corpus` -> phoneme token table
2. `extract_acoustics` -> formants, f0, duration, energy
3. `extract_neural_whisper` -> Whisper embeddings
4. `extract_neural_xlsr` -> XLS-R embeddings
5. `normalise` -> Lobanov + PCA/UMAP
6. `analyse` -> statistics, figures, tables, answers

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run pipeline
```bash
snakemake --cores 1
```

## Data
Put the ORTOLANG corpus files in `data/raw/`.
Expected content: WAV files, TextGrid files, and metadata CSV.
