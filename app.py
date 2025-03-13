from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie dataset
movies_data = pd.read_csv('movies.csv')

# Select important features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Fill missing values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director'] + ' '

# Convert text data into numerical feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity scores
similarity = cosine_similarity(feature_vectors)


# Function to get movie recommendations
def recommend_movies(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        return ["No close match found. Try another movie."]

    close_match = find_close_match[0]
    
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:6]):  # Top 5 recommendations
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)

    return recommended_movies


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = recommend_movies(movie_name)
    return render_template("index.html", movies=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
