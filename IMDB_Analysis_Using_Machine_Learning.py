# Import necessary libraries
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('/content/IMDB Dataset.csv')

# Fill missing values in the 'review' column
df['review'] = df['review'].fillna("")

# Text preprocessing function
def process_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    processed_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    return " ".join(processed_tokens)

# Preprocess the reviews
df['processed_review'] = df['review'].apply(process_text)

# Prepare data for training
X = df['processed_review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text data to numerical format using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)
y_pred_lr = log_reg.predict(X_test_tfidf)

# Naïve Bayes model
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

# Evaluation
log_reg_accuracy = accuracy_score(y_test, y_pred_lr)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

print("Logistic Regression Evaluation:")
print("Accuracy:", log_reg_accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

print("Naïve Bayes Evaluation:")
print("Accuracy:", nb_accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# Accuracy comparison plot
plt.figure(figsize=(8, 6))
sns.barplot(x=['Logistic Regression', 'Naïve Bayes'], y=[log_reg_accuracy, nb_accuracy], palette='viridis')
plt.title('Accuracy Comparison: Logistic Regression vs Naïve Bayes')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()

# Save the Logistic Regression model and TF-IDF vectorizer
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")


#Testing Purpose For Any Sentences

import joblib
import nltk
import string
from nltk.corpus import stopwords

# Download stopwords if not done previously
nltk.download('stopwords')
nltk.download('punkt')

# Load the saved Logistic Regression model and TF-IDF vectorizer
log_reg = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess the text (same as during training)
def process_text(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punctuations = string.punctuation

    processed_tokens = [
        word for word in tokens
        if word not in stop_words and word not in punctuations and word.isalnum()
    ]
    return " ".join(processed_tokens)

# Example text to test the model
text_1 = "The movie was super good"

# Preprocess the input text
processed_text_1 = process_text(text_1)

# Convert the processed text into TF-IDF features
processed_text_1_tfidf = tfidf_vectorizer.transform([processed_text_1])

# Predict the sentiment using the Logistic Regression model
sentiment_prediction = log_reg.predict(processed_text_1_tfidf)

# Output the predicted sentiment
print(f"Predicted sentiment: {'Positive' if sentiment_prediction[0] == 1 else 'Negative'}")


