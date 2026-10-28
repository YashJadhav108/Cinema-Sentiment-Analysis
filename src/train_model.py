import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("IMDB Dataset.csv")
print(df.head())
#NLP
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)        # remove HTML tags
    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"[^a-zA-Z ]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

df["clean_review"] = df["review"].apply(clean_text)


y = df["sentiment"]        # positive / negative
X_text = df["clean_review"]


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X_text).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Model and TFIDF saved successfully!")
