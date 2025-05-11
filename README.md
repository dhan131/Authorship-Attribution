BASIC EXAMPLE USING NLP

import nltk
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download NLTK data if not already
nltk.download('gutenberg')

# Sample authors and texts
authors = {
    'austen': gutenberg.raw('austen-emma.txt'),
    'shakespeare': gutenberg.raw('shakespeare-hamlet.txt'),
    'bible': gutenberg.raw('bible-kjv.txt')
}

# Create dataset
texts, labels = [], []
for author, text in authors.items():
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    texts.extend(chunks)
    labels.extend([author] * len(chunks))

# Preprocess and vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
