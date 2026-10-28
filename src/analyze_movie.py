import requests
import urllib.parse
import pickle
import re
import os

TMDB_KEY = os.environ["TMDB_API_KEY"]  # set this env var; never commit your key

model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def search_movie_tmdb(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={urllib.parse.quote(movie_name)}"
    data = requests.get(url).json()

    if not data.get("results"):
        return None, None

    movie = data["results"][0]
    tmdb_id = movie["id"]
    return tmdb_id, movie["title"]

def get_imdb_id_from_tmdb(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/external_ids?api_key={TMDB_KEY}"
    data = requests.get(url).json()

    imdb_id = data.get("imdb_id")
    return imdb_id

def get_reviews_tmdb(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews?api_key={TMDB_KEY}"
    data = requests.get(url).json()

    reviews = []
    for r in data.get("results", []):
        reviews.append(r["content"])

    return reviews


def analyze_reviews(reviews):
    sentiments = []
    for r in reviews:
        clean = clean_text(r)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]
        sentiments.append(pred)
    return sentiments


def final_verdict(sentiments):
    total = len(sentiments)
    if total == 0:
        return "No reviews found", 0, 0, 0

    pos = sentiments.count("positive")
    neg = sentiments.count("negative")

    pct = (pos / total) * 100

    if pct >= 60:
        verdict = "🔥 POSITIVE MOVIE"
    elif pct >= 40:
        verdict = "😐 MIXED MOVIE"
    else:
        verdict = "👎 NEGATIVE MOVIE"

    return verdict, pos, neg, total

movie_name = input("Enter movie name: ")

tmdb_id, title = search_movie_tmdb(movie_name)

if not tmdb_id:
    print("Movie not found!")
    exit()

print(f"Movie found: {title}")
print(f"TMDB ID: {tmdb_id}")

imdb_id = get_imdb_id_from_tmdb(tmdb_id)
print(f"IMDB ID: {imdb_id}")

reviews = get_reviews_tmdb(tmdb_id)
print(f"Fetched {len(reviews)} reviews...")

sentiments = analyze_reviews(reviews)
verdict, pos, neg, total = final_verdict(sentiments)

print("\n=========================")
print("Movie:", title)
print("IMDB ID:", imdb_id)
print("=========================")
print("Positive:", pos)
print("Negative:", neg)
print("Total Reviews:", total)
print("=========================")
print("Final Verdict:", verdict)
print("=========================")
