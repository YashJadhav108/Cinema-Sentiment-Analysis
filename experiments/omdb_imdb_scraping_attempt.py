import requests
from bs4 import BeautifulSoup
import re
import pickle
import urllib.parse
import os


model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
'''
def get_movie_id(movie_name):
    query = urllib.parse.quote(movie_name)
    url = f"https://www.imdb.com/find/?q={query}&s=tt&ttype=ft&ref_=fn_ft"
    
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    # first movie result
    result = soup.find("td", class_="result_text")

    if not result:
        return None

    link = result.find("a")["href"]  # /title/tt1375666/
    movie_id = link.split("/")[2]

    return movie_id
'''
def get_movie_id(movie_name):
    api_key = os.environ["OMDB_API_KEY"]
    url = f"http://www.omdbapi.com/?t={urllib.parse.quote(movie_name)}&apikey={api_key}"

    data = requests.get(url).json()

    if data.get("Response") == "False":
        return None

    return data.get("imdbID")



def get_imdb_reviews(movie_id, limit=20):
    url = f"https://www.imdb.com/title/{movie_id}/reviews"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    review_divs = soup.find_all("div", class_="text", limit=limit)

    reviews = [div.get_text().strip() for div in review_divs]

    return reviews
def analyze_reviews(reviews):
    sentiments = []

    for text in reviews:
        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiments.append(pred)

    return sentiments

def final_verdict(sentiments):
    total = len(sentiments)

    if total == 0:
        return "No reviews found", 0, 0, 0

    pos = sentiments.count("positive")
    neg = sentiments.count("negative")

    positive_ratio = (pos / total) * 100

    if positive_ratio >= 60:
        verdict = "🔥 POSITIVE MOVIE"
    elif positive_ratio >= 40:
        verdict = "😐 MIXED MOVIE"
    else:
        verdict = "👎 NEGATIVE MOVIE"

    return verdict, pos, neg, total
