# Impact of Training Data Size on Text Classification Models

<p align="center">
  <strong>An Empirical Study of Learning Curves, Data Efficiency, and Statistical Significance</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Research-Machine%20Learning-2563eb" alt="Machine Learning Research">
  <img src="https://img.shields.io/badge/Task-Text%20Classification-7c3aed" alt="Text Classification">
  <img src="https://img.shields.io/badge/Dataset-20%20Newsgroups-f59e0b" alt="20 Newsgroups">
  <img src="https://img.shields.io/badge/Experiments-90-0f766e" alt="90 experiments">
  <img src="https://img.shields.io/badge/INDIACom--2027-Submitted%20%7C%20Under%20Review-orange" alt="INDIACom 2027 submitted under review">
</p>

**Authors:** Shailender and Rinky Ahuja  
**Affiliation:** School of Engineering and Technology, Sushant University, Gurugram, Haryana, India.  
**Conference:** INDIACom-2027  
**Paper ID:** 953  
**Submission date:** 25 November 2026

> **Submission status:** The paper has been received by INDIACom-2027 and is under the conference review process. Submission does not mean acceptance or publication.

---

## Overview

How much labelled training data does a text-classification model actually need?

Adding more labelled examples can improve a model, but collecting and annotating data also takes time and resources. This research investigates how classification performance changes as the training set grows, and when the additional benefit begins to diminish.

The study compares three classical machine-learning classifiers—**Multinomial Naive Bayes (MNB), Logistic Regression (LR), and Linear Support Vector Machine (SVM)**—on five categories from the 20 Newsgroups dataset.

Each model is evaluated at six training-data proportions (10%, 20%, 40%, 60%, 80%, and 100%), using five random seeds. The experiments use a fixed official test partition, TF-IDF features fitted only on the relevant training subset, multiple evaluation metrics, and non-parametric statistical tests.

The central question is not simply which classifier scores highest. It is how much labelled data is useful, how the algorithms respond to additional examples, and whether late-stage improvements are statistically detectable.

## Research at a Glance

| Item | Description |
|---|---|
| Research area | Machine learning, text classification, data efficiency |
| Dataset | Five-category subset of 20 Newsgroups |
| Classifiers | Multinomial Naive Bayes, Logistic Regression, Linear SVM |
| Training sizes | 10%, 20%, 40%, 60%, 80%, 100% |
| Random seeds | 42, 43, 44, 45, 46 |
| Experimental runs | 90 (3 models × 6 sizes × 5 seeds) |
| Primary metric | Macro-F1 |
| Other metrics | Accuracy, Precision, Recall, Weighted-F1 |
| Statistical methods | Shapiro–Wilk, Levene, Friedman, Nemenyi |

## Key Findings

### 1. Linear SVM achieved the highest Macro-F1 at every training size

The mean Macro-F1 scores across the five random seeds are shown below.

| Training data | Multinomial NB | Logistic Regression | Linear SVM |
|---:|---:|---:|---:|
| 10% | 0.7330 | 0.7412 | **0.7595** |
| 20% | 0.7816 | 0.7823 | **0.7933** |
| 40% | 0.8146 | 0.8116 | **0.8179** |
| 60% | 0.8260 | 0.8228 | **0.8299** |
| 80% | 0.8342 | 0.8335 | **0.8412** |
| 100% | 0.8383 | 0.8354 | **0.8453** |

Linear SVM recorded a mean Macro-F1 of **0.7595 at 10%** and **0.8453 at 100%**.

### 2. The models benefited differently from additional data

Although Linear SVM achieved the highest Macro-F1 at every training size, Multinomial Naive Bayes recorded the largest absolute improvement from 10% to 100%.

| Model | Macro-F1 improvement |
|---|---:|
| Multinomial Naive Bayes | +0.1053 |
| Logistic Regression | +0.0942 |
| Linear SVM | +0.0858 |

This highlights an important distinction: the model with the strongest overall performance is not necessarily the model that gains the most from additional training data.

### 3. Improvements flattened beyond approximately 60%

For Linear SVM, mean Macro-F1 increased from **0.8299 at 60%** to **0.8453 at 100%**. The Nemenyi post-hoc comparison found no statistically significant difference between these two training sizes (**p = 0.756**).

This suggests diminishing returns for this particular dataset, feature representation, and model configuration. It should not be interpreted as a universal rule that 60% of training data is sufficient for every text-classification task.

### 4. Statistical tests supported differences across training sizes

The Friedman test was significant for all three classifiers:

| Model | Friedman p-value |
|---|---:|
| Multinomial Naive Bayes | 0.0001 |
| Logistic Regression | 0.0002 |
| Linear SVM | 0.0002 |

The Shapiro–Wilk and Levene tests indicated that the normality and equal-variance assumptions were not satisfied, motivating the use of non-parametric analysis.

---

## Dataset

The experiments use the official train/test partitions of the **20 Newsgroups** corpus, restricted to five categories:

- `alt.atheism`
- `comp.graphics`
- `comp.sys.mac.hardware`
- `rec.autos`
- `sci.space`

| Partition | Documents |
|---|---:|
| Training | 2,829 |
| Testing | 1,883 |
| Total | 4,712 |

The test partition is kept fixed and is not used to fit the vectorizer, tune the models, or select training subsets.

The notebook loads the dataset through scikit-learn's `fetch_20newsgroups` utility and removes headers, footers, and quoted text.

## Methodology

The experimental pipeline follows these stages:

1. **Dataset acquisition:** Load the five selected 20 Newsgroups categories using the official train/test split.
2. **Data partitioning:** Keep the official test partition isolated and fixed.
3. **Stratified subsampling:** Draw 10%, 20%, 40%, 60%, 80%, and 100% training subsets while preserving class proportions.
4. **Feature extraction:** Fit TF-IDF on each training subset only; transform the fixed test set using that fitted vectorizer.
5. **Model training:** Train Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
6. **Evaluation:** Calculate Accuracy, Precision, Recall, Macro-F1, and Weighted-F1.
7. **Statistical analysis:** Apply assumption checks, Friedman tests, and Nemenyi post-hoc comparisons.

### Text representation

The notebook uses TF-IDF with:

- Unigrams and bigrams
- English stop-word removal
- L2 normalization
- Training-subset-only vocabulary and IDF fitting

### Models

| Classifier | Learning approach |
|---|---|
| Multinomial Naive Bayes | Generative probabilistic classifier |
| Logistic Regression | Discriminative probabilistic classifier |
| Linear SVM | Margin-based classifier |

The model configurations are kept fixed across training-data sizes so that the experiment focuses on the effect of training-set size rather than size-specific hyperparameter tuning.

### Repeated experiments

The experiment uses five subsampling seeds: `42`, `43`, `44`, `45`, and `46`.

The design contains 90 model/size/seed runs. At 100% training data, all seeds use the same complete training partition, so those runs coincide; consequently, there are 75 distinct configurations.

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives that are identified |
| Macro-F1 | Unweighted average of per-class F1 scores |
| Weighted-F1 | Average of per-class F1 scores weighted by class support |

Macro-F1 is the primary metric because it gives equal importance to each of the five categories.

## Statistical Analysis

The analysis uses the following tests:

| Test | Role in the study |
|---|---|
| Shapiro–Wilk | Checks the normality of Macro-F1 scores |
| Levene | Checks equality of variances |
| Friedman | Tests for differences across training-data sizes |
| Nemenyi | Performs post-hoc pairwise comparisons |

The significance level for post-hoc comparisons is **α = 0.05**. The results are interpreted within the limits of five random seeds and the selected dataset subset.

## Repository Contents

| File or path | Description |
|---|---|
| [`ResearchPaperCode.ipynb`](./ResearchPaperCode.ipynb) | Jupyter notebook containing the experiment code, model evaluation, statistical analysis, tables, and plots |
| `README.md` | Project overview, methodology, findings, and reproduction instructions |

The repository currently includes the research notebook. Add the final paper and any supplementary files here if you decide to publish them in this repository.

## Reproducing the Experiments

### Requirements

- Python 3
- Jupyter Notebook or Google Colab
- Internet access for the initial 20 Newsgroups dataset download

### 1. Clone the repository

```bash
git clone https://github.com/shlndra/ResearchPaper.git
cd ResearchPaper
```

### 2. Install the dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

### 3. Run the notebook

Open `ResearchPaperCode.ipynb` in Jupyter Notebook or upload it to Google Colab, then run the cells in order.

The notebook fetches the 20 Newsgroups data through scikit-learn. Separate manual dataset files are not required for the notebook's standard workflow.

**Reproducibility note:** The notebook uses fixed subsampling seeds (42–46). Exact numerical results can vary with library versions and execution environments; the paper reports the experimental configuration and results used in the study.

## INDIACom-2027 Submission

The paper has been submitted to **INDIACom-2027: 21st INDIACom and 14th International Conference on Computing for Sustainable Global Development**, organized by Bharati Vidyapeeth's Institute of Computer Applications and Management (BVICAM), New Delhi, India.

| Submission detail | Information |
|---|---|
| Paper title | Impact of Training Data Size on Text Classification Models |
| Paper ID | **953** |
| Date received | 25 November 2026 |
| Conference dates | 7–9 April 2027 |
| IEEE Conference ID | 74162 |
| Current status | Received; under review |

The conference receipt email confirms receipt of the submission and describes a review process involving an initial technical-suitability, scope, and plagiarism assessment, followed by double-blind expert review for qualifying papers.

The conference information supplied for this submission lists **Scopus indexing: Yes**. This is conference-level information and does not constitute confirmation that this individual paper has been accepted, published, or indexed. Publication and inclusion remain subject to the conference's review and publication process.

## Research Integrity Reports

The following results are from the DrillBit reports generated for this paper on **25 November 2026** (Submission ID: **7772264**).

| Check | Reported result |
|---|---:|
| Similarity | **6%** |
| AI-pattern detection | **0.00%** |
| Words flagged with AI-pattern signal | **0** |

The similarity report states that quotations and references/bibliography were excluded from the similarity calculation. The AI report records no AI-matched blocks.

These figures represent the results of the named software reports at the time of scanning. Similarity scores and AI-detection results are tool-specific and should not be treated as independent proof of originality or authorship.

## Limitations and Future Work

The findings are limited to one corpus, five selected categories, TF-IDF features, and three classical classifiers. The approximate saturation point should not be transferred directly to other domains, datasets, or model families.

Future work identified in the paper includes:

- Comparing transformer models such as BERT with the classical baselines.
- Evaluating word embeddings and character n-gram representations.
- Applying active learning to reduce annotation requirements.
- Extending the experiments to imbalanced, multilingual, and multi-label datasets.
- Studying learning curves across additional text domains.

## Citation

**Submitted-paper citation:**

```text
Shailender and R. Ahuja, "Impact of Training Data Size on Text
Classification Models," submitted to INDIACom-2027, 21st INDIACom
and 14th International Conference on Computing for Sustainable
Global Development, BVICAM, New Delhi, India, 2026.
```

Please update this citation with the final proceedings details only if the paper is accepted and published.

## Contact

**Shailender**  
School of Engineering and Technology, Sushant University  
Gurugram, Haryana, India

Email: [reachoutshailender@gmail.com](mailto:reachoutshailender@gmail.com)

---

## Acknowledgement

We acknowledge Sushant University and the organizers of INDIACom-2027 for providing the academic submission and review platform for this research.
