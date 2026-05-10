# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

M1 Final projet for stats for textual data course 

This project analyses the Russian–French Interference Corpus by comparing classical acoustic phonetic features with preliminary neural speech representations.

The strongest completed part of the project is the acoustic analysis pipeline. A preliminary Whisper-small neural extraction stage is also implemented and validated.

---

## Submission Status

### Completed

- Corpus inspection
- TextGrid parsing from the `phones` tier
- Phoneme-token table generation
- Malformed label filtering
- Acoustic feature extraction with Praat/Parselmouth
- Acoustic quality control
- Lobanov normalisation
- Descriptive acoustic statistics
- Acoustic vowel-space visualisations
- L1/L2 acoustic statistical tests with FDR correction
- Effect-size reporting
- Residual gender tests after Lobanov normalisation
- Acoustic distance matrices
- Bootstrap confidence intervals for selected vowel pairs
- Acoustic nearest-centroid classifier with leave-one-speaker-out cross-validation
- Whisper-small layer 4 embedding extraction

### Partially Completed / Future Work

- Whisper upper-layer extraction
- XLS-R extraction
- PCA/UMAP visualisation of full neural representations
- Mantel comparison between acoustic and neural representational similarity matrices
- Linear mixed-effects models
- ROPE classification
- Hierarchical clustering and ARI evaluation

The final written report is available at:

```text
report/final_report.md