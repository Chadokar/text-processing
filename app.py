from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

CORS(app)

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


stemmer = nltk.PorterStemmer()

# Preprocess the text of the queries


def preprocess_queries(queries):
    # Remove punctuation, stop words, and spelling errors
    queries = [query.lower().strip() for query in queries]
    # text = re.sub(r'[^\w\s]', '', text)

    # Lemmatize the text
    queries = [stemmer.stem(query) for query in queries]

    return queries


def tokenizer(text):
    tfidfVectorizer = TfidfVectorizer()
    query_embeddings = tfidfVectorizer.fit_transform(text)
    return query_embeddings


def cluster_transform(text, query_embeddings):
    tfidfVectorizer = TfidfVectorizer()
    # Cluster the questions
    num_clusters = 10  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(query_embeddings)

    # Find the cluster centroids
    cluster_centers = kmeans.cluster_centers_
    text["cluster"] = kmeans.labels_

    # Select representative questions from each cluster
    representative_questions = []

    for cluster_id in range(num_clusters):
        cluster_data = text[text["cluster"] == cluster_id]

        # Choose a representative question based on the highest similarity to the centroid
        centroid = kmeans.cluster_centers_[cluster_id]

        # Calculate the cosine similarity between each question and the centroid
        similarities = cosine_similarity(
            tfidfVectorizer.transform(
                cluster_data["question_text"]), [centroid]
        )

        # Select the question with the highest similarity score as the representative question
        representative_question = cluster_data.iloc[similarities.argmax()][
            "question_text"
        ]
        representative_questions.append(representative_question)
        print(representative_question)
        # representative_questions[f'Cluster {cluster_id + 1}'] = representative_question
    return representative_questions


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]
        # queries = preprocess_queries(data)
        result = []
        for query in data:
            query_embeddings = tokenizer(query)
            representative_data = cluster_transform(query_embeddings)
            result.append(representative_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "_main_":
    app.run(debug=True)
