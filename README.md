# LSA vs LDA Topic Modeling Techniques Comparison

## Project Overview

This project analyzes unstructured municipal complaint texts using two different Natural Language Processing (NLP) topic modeling pipelines,
with the goal of extracting the most frequent topics reported by residents of the municipal. 
The extracted topics aim to help municipal decision-makers understand recurrent concerns and prioritize their responses accordingly.

The dataset used in this project is the Civic and Municipality Complaint System dataset (Taj, n.d.), obtained from Kaggle. 
It contains approximately 1,100 reported municipal issues across 26 columns. Only the `issue_description` column is used, 
which contains the unstructured complaint text.

**Two pipelines are implemented and compared:**

- **Pipeline 1:** Bag-of-Words (`CountVectorizer`) + Latent Semantic Analysis (`TruncatedSVD`)
- **Pipeline 2:** TF-IDF (`TfidfVectorizer`) + Latent Dirichlet Allocation (`LatentDirichletAllocation`)
- 
**Note!**
  
Both pipelines use `gensim.models.CoherenceModel` to determine the optimal number of topics (k) based on evidence,
rather than choosing it arbitrarily.

---

## Repository Structure

| File | Description |
|---|---|
| `Pipeline_1.ipynb` | Notebook(Jupyter) implementing CountVectorizer + LSA pipeline |
| `Pipeline_2.ipynb` | Notebook(Jupyter) implementing TfidfVectorizer + LDA pipeline |
| `municipal_complaints_training_dataset.csv` | Raw input dataset from Kaggle |
| `preprocessed_complaints.csv` | Cleaned text output produced by Pipeline 1, shared with Pipeline 2 |
| `requirements.txt` | All Python dependencies needed to run both pipelines |

---

## How to Run

1. Clone or download this repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Open `Pipeline_1.ipynb` in Jupyter Notebook or JupyterLab and run all cells top to bottom
4. Open `Pipeline_2.ipynb` and run all cells top to bottom

---

## Methodology

### Shared Preprocessing (Pipeline 1, Step 3)
Both pipelines use the same cleaned text, preprocessed once in Pipeline 1:
- Lowercasing
- Hyphen-to-space conversion
- Punctuation removal
- Tokenization (`nltk.word_tokenize`)
- Stopword removal (retaining "no", "not", "non")
- Lemmatization (`nltk.stem.WordNetLemmatizer`)

### Pipeline 1 — CountVectorizer + LSA
- Builds a Bag-of-Words document-term matrix (447 documents × 747 terms)
- Applies `TruncatedSVD` for Latent Semantic Analysis
- Uses `gensim.models.CoherenceModel` (c_v) to search for optimal k (range: k=2 to k=30)
- Optimal k found: **k=2** (coherence score: 0.707)

### Pipeline 2 — TfidfVectorizer + LDA
- Builds a TF-IDF document-term matrix (447 documents × 747 terms)
- Applies `LatentDirichletAllocation` for topic modeling
- Uses `gensim.models.CoherenceModel` (c_v) to search for optimal k (range: k=2 to k=10,
  restricted based on corpus size to ensure statistical stability)
- Optimal k found: **k=4** (coherence score: 0.373)

---

## Results Summary

### Pipeline 1 (LSA) — k=2

| Rank | Topic | Prevalence | Top Words |
|---|---|---|---|
| 1 | Topic 0 | 83.00% | please, water, near, main, emergency, road, immediate, crew, risk, safety |
| 2 | Topic 1 | 17.00% | water, please, pipe, im, concerned, pooling, look, safety, really, frustrating |

The two topics overlap heavily in vocabulary, with "water" and "please" dominating both. This
suggests the Bag-of-Words + LSA combination does not statistically support separating the complaints
into fine-grained topics on this corpus size.

### Pipeline 2 (LDA) — k=4

| Rank | Topic | Prevalence | Interpreted Label |
|---|---|---|---|
| 1 | Topic 1 | 34.45% | Public Safety and Noise Disturbances |
| 2 | Topic 2 | 29.75% | Road Infrastructure and Water Leaks |
| 3 | Topic 0 | 19.24% | Illegal Dumping and Water Infrastructure |
| 4 | Topic 3 | 16.55% | Waste Management and Public Spaces |

Pipeline 2 produced clearly more distinct and interpretable topics, with a more balanced prevalence
distribution across all four topics. This demonstrates that TF-IDF weighting combined with LDA
provides more actionable results for this dataset than the Bag-of-Words + LSA approach.

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- `pandas`, `scikit-learn` — data handling and modeling
- `nltk` — text preprocessing
- `gensim` — coherence scoring
- `matplotlib` — visualization

---

## Dataset Reference

Taj, W. (n.d.). *Civil and municipality complaint system dataset*. Kaggle.
https://www.kaggle.com/datasets/wajahattaj/civil-and-municipality-complaint-system-dataset

---

**Author: Moaaz KH Abu Khalaf**

**Course: Project: Data Analysis (DLBDSEDA02)**

**Tutor's Name: Prof.Dr. Frank Passing**

**IU International University of Applied Sciences Germany**
