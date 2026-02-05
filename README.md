# 🎬 IMDb Movie Review Sentiment Analysis

![Sentiment Analysis Concept](https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1000&auto=format&fit=crop)
*(Automating content moderation for the Film Junky Union)*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP_Processing-150458?style=for-the-badge&logo=nltk&logoColor=white)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-Lemmatization-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-FFCC00?style=for-the-badge)](https://lightgbm.readthedocs.io/)

</div>

---

## 🛡️ Context & Business Value

**The Problem:**
The **Film Junky Union**, a community for classic movie enthusiasts, is overwhelmed by the volume of user-submitted content. Manual moderation is slow and inconsistent. They need an automated system to flag negative reviews instantly to maintain community standards.

**The Solution:**
I built an NLP pipeline that automatically flags negative movie reviews. By combining TF-IDF vectorization with linear models, the system achieved an F1-score of **0.88**. This result beat the client's target of 0.85, delivering a reliable tool for scalable moderation.

---

## 📐 Technical Objectives

* **Text Preprocessing:** Implement and compare two distinct NLP pipelines: one using `NLTK` for standard tokenization and another using `spaCy` for advanced lemmatization.
* **Vectorization:** Transform unstructured text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** to capture keyword importance while filtering out common stop words.
* **Model Benchmarking:** Evaluate Linear models (Logistic Regression) against Gradient Boosting methods (LightGBM) to determine the best balance between training speed and classification accuracy.
* **Performance Metric:** Optimize for **F1-Score** to ensure a balance between precision (avoiding false positives) and recall (catching all negative reviews).

---

## 🗃️ The Data Schema

The dataset consists of IMDb movie reviews labeled by sentiment.

| Feature | Description |
| :--- | :--- |
| **`review`** | Unstructured text of the user's movie critique. |
| **`pos`** | Binary target variable: `0` for Negative, `1` for Positive. |
| **`ds_part`** | Metadata indicating train/test split allocation. |

*Source: Andrew L. Maas et al., ACL 2011.*

---

## 🔭 Key Findings & EDA

* **Balanced Classes:** The dataset is split evenly between positive and negative reviews, eliminating the need for resampling techniques like SMOTE.
* **Vectorization Impact:** TF-IDF successfully isolated sentiment-bearing words (e.g., "breathtaking" vs. "waste") from neutral structural text.

---

## ⚖️ Model Comparison

I ran a competitive benchmark to find the most effective classifier. Surprisingly, the simpler linear model outperformed the complex boosting algorithm for this high-dimensional text data.

| Model | Preprocessing | Test Accuracy | Test F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **NLTK** | **0.88** | **0.88** | **0.95** |
| **Logistic Regression** | **spaCy** | **0.88** | **0.88** | **0.95** |
| **LightGBM** | spaCy | 0.86 | 0.86 | 0.93 |
| **Dummy (Baseline)** | None | 0.50 | 0.00 | 0.50 |

---

## ✅ Final Evaluation

The champion model (Logistic Regression with NLTK) was stress-tested against custom, tricky reviews to verify its "human intuition."

| Review Snippet | True Sentiment | LogReg Prediction | LightGBM Prediction |
| :--- | :--- | :--- | :--- |
| *"I did not simply like it..."* | **Negative** | **0.16 (Neg)** | 0.57 (Pos) ❌ |
| *"What a rotten attempt..."* | **Negative** | **0.05 (Neg)** | 0.30 (Neg) |
| *"I was really fascinated..."* | **Positive** | **0.57 (Pos)** | 0.60 (Pos) |

> **Strategic Insight:** LightGBM struggled with nuanced phrasing (e.g., "did not simply like it"), classifying it incorrectly as positive. Logistic Regression correctly captured the negation, proving it is robust for real-world text patterns.

---

## 🧭 Recommendations

1.  **Deploy Logistic Regression:** It is faster, more interpretable, and outperformed LightGBM by 2% in F1-score.
2.  **Use NLTK Preprocessing:** Since NLTK and spaCy yielded identical results, NLTK is recommended for its lighter computational footprint compared to spaCy's heavy NLP pipeline.
3.  **Avoid Boosting for Sparse Data:** Gradient boosting models like LightGBM often struggle with the high-dimensionality and sparsity of TF-IDF matrices compared to linear models.

---

## 🧬 Tools & Technologies

* **Language:** Python 3.12
* **NLP:** NLTK, spaCy, TF-IDF
* **Modeling:** Scikit-Learn (Logistic Regression), LightGBM
* **Viz:** Matplotlib, Seaborn