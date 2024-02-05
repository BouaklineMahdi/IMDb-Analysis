# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess data


def load_data(folder):
    data = []
    labels = []
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for category in ['pos', 'neg']:
        path = os.path.join(folder, category)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # Tokenization and stemming
                words = [stemmer.stem(word) for word in word_tokenize(
                    text.lower()) if word.isalnum()]
                # Remove stopwords
                words = [word for word in words if word not in stop_words]
                data.append(' '.join(words))
                labels.append(1 if category == 'pos' else 0)

    return data, labels


# Load the IMDb dataset
train_data, train_labels = load_data('aclImdb_v1/aclImdb/train')
test_data, test_labels = load_data('aclImdb_v1/aclImdb/test')

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, train_labels, test_size=0.2, random_state=42)

# Build and train the Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
val_predictions = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("Classification Report:\n", classification_report(y_val, val_predictions))

# Evaluate the model on the test set
test_predictions = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(test_labels, test_predictions))
print("Classification Report:\n", classification_report(
    test_labels, test_predictions))
