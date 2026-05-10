# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

## 1. Overview

This project investigates how French phoneme categories, especially oral vowels, are represented in classical acoustic features and in neural speech representations. The corpus is the Russian–French Interference Corpus, containing French sentence productions by native French speakers and Russian L1 learners of French.

The implemented pipeline covers corpus parsing, acoustic extraction, acoustic quality control, Lobanov normalisation, descriptive statistics, L1/L2 tests, residual gender tests, acoustic distance matrices, an acoustic nearest-centroid classifier, and Whisper neural representation extraction.

Due to hardware and time constraints, the submitted neural branch is partial but no longer limited to a single layer: Whisper-small layers 4 and 10 were fully extracted for the valid oral-vowel token set. The full XLS-R branch, neural PCA/UMAP analysis, mixed-effects models, ROPE classification, and hierarchical clustering remain incomplete. All reported numeric results below are based on generated project outputs, not simulated values.
---

## 2. Pipeline Summary

Implemented stages:

1. `parse_corpus.py`
2. `extract_acoustics.py`
3. `qc_acoustics.py`
4. `normalise.py`
5. `analyse_descriptive.py`
6. `statistical_tests_acoustic.py`
7. `gender_tests_acoustic.py`
8. `acoustic_distances.py`
9. `acoustic_classifier.py`
10. `extract_neural_whisper.py`

The project follows the required staged structure: parse corpus, extract acoustic features, extract neural representations, normalise features, and analyse the resulting representations.

---

## 3. Corpus Parsing

The parser read Praat TextGrid files and extracted the `phones` tier.

Parsing results:

| Item | Value |
|---|---:|
| TextGrid files found | 1482 |
| Valid phoneme tokens after cleaning | 22596 |
| Malformed labels excluded | 323 |
| Missing WAV files | 0 |
| Failed TextGrids | 0 |
| Oral vowel tokens | 8319 |
| Nasal vowel tokens | 215 |

Malformed labels such as `ding... d` were excluded before acoustic extraction. Raw IPA labels were preserved in `phoneme_raw`, while broader analysis labels were stored in `phoneme`.

---

## 4. Acoustic Feature Extraction

For oral vowels, the project extracted:

- F1, F2, F3 at vowel midpoint
- F1/F2 at 25% and 75% for vowels longer than 80 ms
- f0 mean
- RMS energy
- spectral centre of gravity

LPC parameters followed the assignment instructions:

| Speaker gender | Max formant |
|---|---:|
| Female | 5000 Hz |
| Male | 4500 Hz |

---

## 5. Acoustic Quality Control

The initial acoustic extraction produced 8319 oral-vowel rows. A quality-control step flagged invalid formants using broad phonetic plausibility constraints:

- F1 between 150 and 1200 Hz
- F2 between 500 and 3500 Hz
- F3 between 1200 and 4500 Hz
- F1 < F2

QC results:

| Item | Value |
|---|---:|
| Input vowel rows | 8319 |
| Invalid formant rows | 56 |
| Invalid formant rate | 0.67% |
| Main valid vowel rows after QC | 8262 |

This prevented obvious formant-tracking failures from contaminating the statistical analysis.

---

## 6. Lobanov Normalisation

Lobanov normalisation was applied speaker-by-speaker using only valid oral vowel tokens:

\[
F^* = \frac{F - \bar{F}_{speaker}}{SD(F_{speaker})}
\]

Validation showed speaker-level normalised means near 0 and standard deviations near 1 for F1 and F2, confirming correct implementation.

The main analysis vowels were:

| Vowel | Valid tokens |
|---|---:|
| /a/ | 3085 |
| /i/ | 1870 |
| /ɑ/ | 1006 |
| /u/ | 710 |
| /y/ | 532 |
| /ɛ/ | 348 |
| /e/ | 241 |
| /ø/ | 229 |
| /ə/ | 127 |
| /o/ | 114 |

The vowel /œ/ had only one token and was excluded from the main statistical analysis.

---

## 7. Descriptive Acoustic Results

The descriptive speaker-level F1 variability summary showed the highest inter-speaker variability for:

| Vowel | Inter-speaker F1 variance |
|---|---:|
| /ə/ | 0.2045 |
| /ɑ/ | 0.1802 |
| /y/ | 0.0852 |
| /ɛ/ | 0.0644 |
| /ø/ | 0.0580 |

Interpretation should be cautious for /ə/ because it has relatively few tokens and is contextually variable. The more reliable high-variability candidates are /ɑ/ and /y/.

Generated figures include:

- Lobanov vowel chart
- F1 boxplots by vowel
- F2 boxplots by vowel
- Grouped F1/F2 boxplots
- Acoustic classifier confusion matrix

---

## 8. Acoustic L1/L2 Statistical Tests

For each oral vowel and each formant dimension, L1 vs L2 differences were tested after Lobanov normalisation. Shapiro-Wilk and Levene tests were used for assumption checking. Depending on assumptions, Mann-Whitney U tests or t-tests were applied. P-values were corrected using Benjamini-Hochberg FDR.

Significant acoustic contrasts after FDR correction:

| Vowel | Feature | L2 - L1 difference | Cohen's d | Rank-biserial | Effect size |
|---|---|---:|---:|---:|---|
| /a/ | F1 | 0.0006 | 0.0008 | 0.0560 | negligible |
| /i/ | F2 | 0.0509 | 0.0976 | 0.2946 | small |
| /u/ | F1 | -0.2061 | -0.5432 | -0.3736 | medium |
| /u/ | F2 | 0.2065 | 0.4105 | 0.2322 | small |
| /y/ | F1 | -0.2742 | -0.3975 | -0.2166 | small |
| /y/ | F2 | -0.1786 | -0.3526 | -0.1793 | small |
| /ø/ | F1 | 0.1362 | 0.4632 | 0.2006 | small |
| /ɑ/ | F1 | 0.0709 | 0.1577 | 0.1170 | small |
| /ɑ/ | F2 | 0.0688 | 0.2962 | 0.2094 | small |
| /ɛ/ | F1 | -0.1819 | -0.4776 | -0.2941 | small |

The strongest acoustic L1/L2 contrast was /u/ F1, with a medium effect size. The /a/ F1 contrast was statistically significant but practically negligible, showing why p-values alone are insufficient.

---

## 9. Residual Gender Effects

Gender was tested at the speaker level after Lobanov normalisation.

Result:

> No residual gender effect survived FDR correction.

This suggests that Lobanov normalisation effectively reduced the expected anatomical gender differences in formant frequencies.

---

## 10. Acoustic Distance Matrices

Pairwise acoustic distances were computed between vowel centroids in Lobanov-normalised F1/F2 space.

Selected bootstrap confidence intervals:

| Pair | Euclidean distance | 95% CI | Mahalanobis distance | 95% CI |
|---|---:|---|---:|---|
| /e/–/ɛ/ | 0.309 | [0.212, 0.439] | 0.676 | [0.463, 0.935] |
| /y/–/u/ | 1.492 | [1.332, 1.654] | 1.659 | [1.559, 1.752] |
| /ø/–/e/ | 0.643 | [0.493, 0.801] | 1.226 | [0.933, 1.516] |
| /ɑ/–/a/ | 0.720 | [0.583, 0.861] | 1.088 | [0.978, 1.227] |

The /y/–/u/ distance is large, consistent with the front/back distinction between front rounded /y/ and back rounded /u/. The /e/–/ɛ/ distance is smaller, consistent with their close position in the French vowel system.

---

## 11. Acoustic Nearest-Centroid Classifier

A nearest-centroid classifier was evaluated using leave-one-speaker-out cross-validation.

Results:

| Metric | Value |
|---|---:|
| Overall accuracy | 69.27% |
| Macro F1 | 52.69% |
| L1 accuracy | 71.34% |
| L2 accuracy | 67.41% |

Per-class performance was strongest for /i/, /u/, /a/, and /ɑ/. Performance was weaker for sparse or overlapping vowels such as /ə/, /o/, /e/, /ɛ/, and /ø/.

This shows that normalised F1/F2 captures much of the French oral vowel structure but does not perfectly separate all vowel categories.

---

## 12. Whisper Neural Representations

Whisper-small embeddings were extracted for the same valid oral-vowel token set used in the acoustic analysis. Two encoder layers were extracted: layer 4 as a lower/intermediate representation and layer 10 as a higher-layer representation.

Results:

| Representation | Shape | Failed WAVs | Failed tokens |
|---|---:|---:|---:|
| Whisper-small layer 4 | (8262, 768) | 0 | 0 |
| Whisper-small layer 10 | (8262, 768) | 0 | 0 |

This confirms that each valid vowel token has a 768-dimensional neural representation at two different Whisper encoder depths. The extraction was performed by processing each WAV file once and pooling hidden-state frames over the phoneme interval, which is much more efficient than running Whisper separately for each phoneme token.

The full downstream neural analysis — PCA/UMAP, neural L1/L2 tests, neural distance matrices, Mantel tests, and neural classifiers — was not completed before submission. However, the extracted Whisper files provide the necessary foundation for those analyses.

---

# Answers to Project Questions

## Q1. Explain what PCA and UMAP can be used for and how the two approaches differ.

PCA and UMAP are dimensionality-reduction methods. PCA is linear: it finds orthogonal directions of maximum variance and is useful for interpretable global structure. UMAP is nonlinear: it attempts to preserve local neighbourhood relations and is often better for visualising clusters in high-dimensional data. PCA is more transparent and deterministic; UMAP is more flexible but more sensitive to hyperparameters.

## Q2. Which phonemes exhibit the greatest inter-speaker variability in the acoustic space? Is this reflected in the neural representation space?

In the acoustic space, /ə/, /ɑ/, /y/, /ɛ/, and /ø/ showed the largest descriptive inter-speaker F1 variability. However, /ə/ has relatively few tokens, so /ɑ/ and /y/ are more reliable candidates. Neural reflection was not fully analysed yet. However, Whisper-small layers 4 and 10 were successfully extracted, so the next step would be to test whether the same high-variability vowels also show larger dispersion or weaker clustering in Whisper embedding space.

## Q3. In the UMAP projection, do Whisper and XLS-R representations form clearly separable phoneme clusters? Are the clusters aligned with the vowel trapezoid structure?

UMAP projections were not completed in the submitted version. This remains future work. However, the required Whisper input data now exists for two encoder layers: layer 4 and layer 10, both with shape `(8262, 768)`. The next analysis step would be to apply PCA and UMAP to these two matrices, then compare whether lower and higher Whisper layers produce separable vowel clusters and whether those clusters align with the expected French vowel trapezoid.

## Q4. What is the Mantel correlation between the acoustic RSM and the Whisper RSM? Between acoustic RSM and XLS-R RSM?

Mantel correlations were not computed in the submitted version because the full neural distance-analysis stage was not completed. Acoustic distance matrices were computed and are ready for comparison with Whisper/XLS-R RSMs.

## Q5. After FDR correction, for which vowels does the L1/L2 difference persist in acoustic and neural representations?

In acoustic features, significant corrected contrasts were:

- /a/ F1, negligible effect
- /i/ F2, small effect
- /u/ F1 and F2, strongest effect on F1
- /y/ F1 and F2
- /ø/ F1
- /ɑ/ F1 and F2
- /ɛ/ F1

Neural L1/L2 tests were not completed. However, Whisper-small layers 4 and 10 were extracted successfully, so neural L1/L2 testing can be performed next by comparing L1 and L2 centroids or PCA components within each vowel and each layer.

## Q6. Which distance structure best captures the phonological distances expected from the IPA vowel trapezoid?

Among completed representations, the acoustic F1/F2 structure captures a plausible vowel-space geometry: /i/ appears high/front, /u/ high/back, /a/ low, and /e/–/ɛ/ are close. Therefore, the acoustic distance matrix currently provides the best available approximation of the expected vowel trapezoid.

## Q7. Which representation type yields the highest phoneme identification accuracy?

Only the acoustic classifier was completed. It achieved 69.27% accuracy and 52.69% macro F1 using leave-one-speaker-out cross-validation. Neural classifier results remain future work. Since Whisper-small layers 4 and 10 were successfully extracted, the next step would be to run the same nearest-centroid or logistic classifier on Whisper PCA components and compare acoustic vs neural classification accuracy.

## Q8. What is the ICC for F1 of /a/? Does it differ from the ICC on Whisper PC1 for /a/?

Mixed-effects ICC models were not completed. Descriptively, /a/ had low inter-speaker F1 variance relative to its high within-speaker variability, suggesting that /a/ F1 was not strongly speaker-specific after normalisation.

## Q9. Is the L1 × Gender interaction significant in the acoustic model? Is the same interaction significant in the Whisper model?

Linear mixed-effects interaction models were not completed. However, speaker-level residual gender tests after Lobanov normalisation showed no robust gender effect after FDR correction. The same L1 × Gender interaction was not tested in Whisper space, although the extracted layer-4 and layer-10 Whisper embeddings make that analysis possible in a future mixed-effects or PCA-based model.

## Q10. Which representation type yields the highest marginal R² for the L1/L2 fixed effect?

Marginal R² comparisons were not completed because the mixed-effects models were not completed.

## Q11. Identify at least one phoneme where the L1/L2 contrast is statistically significant but falls within the acoustic ROPE.

The strongest candidate is /a/ F1. It was statistically significant after FDR correction, but the L2-L1 difference was only 0.0006 Lobanov units with a negligible effect size. This is likely practically equivalent despite statistical significance.

## Q12. For which representation type is a larger proportion of contrasts classified as non-equivalent?

ROPE classification was not fully implemented. Based on effect sizes alone, only a small subset of acoustic contrasts appear practically meaningful, especially /u/ F1.

## Q13. Are there phonemes for which acoustic and neural ROPE classifications disagree?

Not evaluated in the submitted version because neural ROPE analysis was not completed.

## Q14. Which representation type yields the highest ARI against front/back and high/mid/low vowel distinctions?

Hierarchical clustering and ARI evaluation were not completed.

## Q15. In speaker clustering, which factor — L1/L2 status or gender — is more strongly recovered?

Speaker clustering was not completed. Acoustic residual gender tests suggest that gender was not strongly preserved after Lobanov normalisation, but this does not replace a full clustering analysis.

## Q16. Are any phonemes systematically misclassified across all representation types?

Only the acoustic classifier was completed. In the acoustic classifier, /ə/, /o/, /e/, /ɛ/, and /ø/ were relatively weak classes, likely due to sparse data and overlapping acoustic spaces. Cross-representation systematic misclassification could not be assessed without neural classifiers.

---

## 13. Limitations

This submission is strongest on the acoustic branch. The acoustic pipeline is complete and includes parsing, acoustic extraction, quality control, Lobanov normalisation, descriptive statistics, L1/L2 tests, residual gender tests, acoustic distance matrices, and an acoustic classifier.

The neural branch now includes completed Whisper-small layer 4 and layer 10 extraction for the valid oral-vowel tokens. Both layers produced `(8262, 768)` embedding matrices with zero failed WAVs and zero failed tokens. However, the downstream neural analyses were not completed before submission: neural PCA/UMAP, neural L1/L2 tests, neural distance matrices, Mantel tests, neural classifiers, XLS-R extraction, mixed-effects models, ROPE classification, and hierarchical clustering remain incomplete.

The main computational limitation was CPU-only neural extraction. The optimized Whisper extractor processes each WAV once and pools hidden states over phoneme intervals, making the method scalable, but full execution of all required neural analyses was not completed before submission.

---

## 14. Conclusion

The completed acoustic analysis shows that Lobanov-normalised F1/F2 features recover a plausible French vowel space, support moderate vowel identification accuracy, and reveal several L1/L2 differences. The strongest practically meaningful L1/L2 contrast was /u/ F1. Gender effects were not robust after Lobanov normalisation.

The project codebase provides a reproducible foundation for extending the analysis from the completed Whisper layer-4 and layer-10 extractions toward full neural PCA/UMAP, XLS-R, Mantel comparison, ROPE, and clustering analyses.