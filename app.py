import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Helper Functions
# ----------------------------
def convert(obj):
    """Extract name values from JSON-like string"""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


def convert3(obj):
    """Extract top 3 cast names"""
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


def fetch_director(obj):
    """Extract director name"""
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


def remove_space(L):
    """Remove spaces in words for better vector matching"""
    return [i.replace(" ", "") for i in L]


# ----------------------------
# Load Dataset
# ----------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')

# Keep only useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
movies.dropna(inplace=True)

# Convert JSON-like columns into list format
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

# Split overview into list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from words
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)

# Create tags column by combining all features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Keep only title + tags
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags list into a string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# ----------------------------
# Vectorization and Similarity
# ----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)


# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(movie):
    if movie not in new_df['title'].values:
        print("\n❌ Movie not found in dataset! Try another name.\n")
        return

    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    print("\n✅ Recommended Movies:\n")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# ----------------------------
# Main Program
# ----------------------------
if __name__ == "__main__":
    print("=====================================")
    print(" 🎬 AI MOVIE RECOMMENDATION SYSTEM ")
    print("=====================================")

    while True:
        movie_name = input("\nEnter a movie name (or type exit to stop): ")

        if movie_name.lower() == "exit":
            print("\nThank you for using Movie Recommendation System!")
            break

        recommend(movie_name)