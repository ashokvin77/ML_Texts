# Sentiment Analysis for Movie Reviews

## Overview  
This project is developed for the **Film Junky Union**, a community of classic movie enthusiasts, to automatically detect negative movie reviews. The goal is to train a model that classifies IMDb movie reviews as positive or negative, achieving an **F1 score of at least 0.85**. The dataset consists of labeled IMDb reviews with polarity (positive/negative) indicators.

## Objectives  

- Load and preprocess IMDB movie review data for sentiment analysis  
- Conduct exploratory data analysis to understand class distribution and text patterns  
- Implement text preprocessing and vectorization techniques 
- Train and evaluate multiple classification models:
  - Logistic Regression with NLTK preprocessing
  - Logistic Regression with spaCy lemmatization
  - LightGBM Classifier 
- Test the models on a test dataset and user-written reviews  
- Compare model performance and analyze differences in classification results 
- Present findings and recommend the best model for deployment  

## Data Description  

The dataset (`imdb_reviews.tsv`) contains the following key columns:

- `review`: Text of the movie review  
- `pos`: Target variable (0 = negative, 1 = positive)  
- `ds_part`: Indicates train/test split 

The dataset was provided by Andrew L. Maas et al. (2011) from "Learning Word Vectors for Sentiment Analysis" presented at the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)

## Key Findings

### EDA Insights
- The dataset is **well-balanced**, with nearly equal positive and negative reviews.
- No significant class imbalance issues were detected.

### Model Performance Comparison

| Model                          | Test Accuracy | Test F1 | ROC AUC |
|--------------------------------|---------------|---------|---------|
| DummyClassifier (Baseline)     | 0.50          | 0.00    | 0.50    |
| **NLTK + TF-IDF + Logistic Regression** | 0.88       | **0.88** | 0.95    |
| **spaCy + TF-IDF + Logistic Regression** | 0.88    | **0.88** | 0.95    |
| spaCy + TF-IDF + LightGBM      | 0.86          | 0.86    | 0.93    |

### Key Conclusions
- Both Logistic Regression models **exceeded the target F1 score** (0.88 vs required 0.85).
- Despite being a more advanced algorithm, **LightGBM underperformed** compared to Logistic Regression.
- Logistic Regression models showed **better alignment with human intuition** when tested on custom reviews.

## Custom Review Classification Examples

| Review Excerpt | Intuitive Sentiment | Model 1 (LR) Prob | Model 2 (LR) Prob | Model 3 (LGBM) Prob |
|----------------|---------------------|-------------------|-------------------|---------------------|
| "I did not simply like it..." | Negative | 0.16 | 0.20 | 0.57 |
| "I was really fascinated..." | Positive | 0.57 | 0.61 | 0.60 |
| "What a rotten attempt..." | Negative | 0.05 | 0.03 | 0.30 |

*Note: Probabilities closer to 0 = more negative, closer to 1 = more positive*

## Recommendations
- **Deploy Logistic Regression model** (either NLTK or spaCy version) for production use.
- The model achieves **88% accuracy** in classifying sentiment while maintaining interpretability.
- LightGBM, while powerful, **is not recommended** for this specific text classification task due to lower performance and inconsistent probability outputs.

## Tools and Technologies  

- Python 3.12.10  
- pandas 2.2.3  
- NumPy 2.2.5  
- scikit-learn 1.6.1  
- NLTK 3.9.1  
- spaCy 3.8.7  
- LightGBM 4.6.0  
- matplotlib 3.10.1  
- seaborn 0.13.2 

