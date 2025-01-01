# IMDB-Sentiment-Analysis-Using-Logistic-Regression-and-Naive-Bayes

This project demonstrates sentiment analysis on the IMDB dataset using text preprocessing, TF-IDF vectorization, and machine learning models (Logistic Regression and Naïve Bayes).

## Features
- Preprocessing text data (removing stopwords, punctuation, etc.).
- Converting text into numerical representation using TF-IDF.
- Training and evaluating two models: Logistic Regression and Naïve Bayes.
- Comparison of model performance through accuracy, classification report, and confusion matrix.
- Saving and loading the trained model for future predictions.

## Dataset
The IMDB Dataset can be found on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn nltk matplotlib seaborn joblib
   ```
3. Download the dataset and place it as `IMDB Dataset.csv` in the project directory.

## Run the Script
Execute the script:
```bash
python sentiment_analysis.py
```

## Results
- Logistic Regression Accuracy: 89%
- Naïve Bayes Accuracy: 86%

## Outputs
- Trained Logistic Regression model (`logistic_regression_model.pkl`)
- TF-IDF vectorizer (`tfidf_vectorizer.pkl`)
