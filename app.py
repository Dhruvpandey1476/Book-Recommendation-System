import streamlit as st
import pickle
import pandas as pd
import requests

# ------------------------------
# Load saved model
# ------------------------------

with open("book_recommendation_model.pkl","rb") as f:
    model = pickle.load(f)

tfidf_vectorizer = model["tfidf_vectorizer"]
tfidf_matrix = model["tfidf_matrix"]
cosine_sim = model["cosine_sim"]
data = model["data"]

# ------------------------------
# Fetch Book Poster from OpenLibrary
# ------------------------------

@st.cache_data
def fetch_poster(book_title):

    url = f"https://openlibrary.org/search.json?title={book_title}"

    response = requests.get(url)
    data_json = response.json()

    try:
        cover_id = data_json["docs"][0]["cover_i"]
        poster = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    except:
        poster = "https://via.placeholder.com/150"

    return poster

# ------------------------------
# Recommendation Function
# ------------------------------

def recommend(book):

    index = data[data['title'] == book].index[0]

    distances = cosine_sim[index]

    books_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    recommended_books = []
    recommended_posters = []

    for i in books_list:

        title = data.iloc[i[0]]['title']

        recommended_books.append(title)
        recommended_posters.append(fetch_poster(title))

    return recommended_books, recommended_posters


# ------------------------------
# Streamlit UI
# ------------------------------

st.title("📚 Book Recommendation System")

selected_book = st.selectbox(
    "Select a Book",
    data['title'].values
)

if st.button("Recommend"):

    names, posters = recommend(selected_book)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.write(names[0])

    with col2:
        st.image(posters[1])
        st.write(names[1])

    with col3:
        st.image(posters[2])
        st.write(names[2])

    with col4:
        st.image(posters[3])
        st.write(names[3])

    with col5:
        st.image(posters[4])
        st.write(names[4])